"""Evaluates retrieval and answer quality on evals/dataset.jsonl.

Run (Postgres must be up; uses your .env for the LLM):
    uv run python -m evals.make_corpus      # once
    uv run python -m evals.run              # full run with LLM-as-judge
    uv run python -m evals.run --no-judge   # retrieval metrics + refusals only (cheaper)

Metrics
- hit@1 / hit@k / MRR: did retrieval put the right file + page in the top results (no LLM needed)
- correctness: does the answer contain the reference facts (LLM judge, 0 / 0.5 / 1)
- faithfulness: is every claim supported by the retrieved sources, i.e. no hallucination (LLM judge)
- refusal accuracy: for questions the documents can't answer, did it say "I don't know"
- citation rate / invalid citations: answers that cite sources, and citations to sources that don't exist
- latency and cost per question

Results are written to evals/results/<timestamp>.json and evals/results/latest.md."""

import argparse
import asyncio
import json
import re
import statistics
import time
from datetime import datetime
from pathlib import Path

from app.config import settings
from app.db import close_db, get_pool, init_db
from app.ingest import ingest_document
from app.llm import Answer, build_context, cited_tags, client, is_no_answer, stream_answer
from app.retrieval import retrieve
from app.storage import get_storage

ROOT = Path(__file__).parent
CORPUS, RESULTS = ROOT / "corpus", ROOT / "results"
OWNER = "eval-runner"

JUDGE_PROMPT = """You grade answers from a document question-answering system.

<question>{question}</question>
<reference_answer>{reference}</reference_answer>
<sources_given_to_the_system>
{sources}
</sources_given_to_the_system>
<system_answer>{answer}</system_answer>

Score two things:
- correctness: 1 if the system answer contains the key facts of the reference answer, 0.5 if partially, 0 if wrong
  or missing. Saying "I don't know" scores 0.
- faithfulness: 1 if every factual claim in the system answer is supported by the sources, 0.5 if mostly,
  0 if it states things the sources don't support. An "I don't know" answer scores 1.

Reply with JSON only: {{"correctness": <0|0.5|1>, "faithfulness": <0|0.5|1>, "reason": "<one sentence>"}}"""


async def ingest_corpus(reuse: bool) -> None:
    files = sorted(CORPUS.glob("*.pdf"))
    if not files:
        raise SystemExit("No corpus found. Run: uv run python -m evals.make_corpus")
    async with get_pool().connection() as conn:
        cur = await conn.execute("SELECT filename FROM documents WHERE owner_id = %s AND status = 'ready'", (OWNER,))
        ready = {r["filename"] for r in await cur.fetchall()}
    if reuse and ready == {f.name for f in files}:
        print(f"Reusing {len(ready)} already-ingested documents")
        return

    async with get_pool().connection() as conn:
        await conn.execute("DELETE FROM documents WHERE owner_id = %s", (OWNER,))
    for f in files:
        async with get_pool().connection() as conn:
            cur = await conn.execute(
                "INSERT INTO documents (owner_id, filename, status) VALUES (%s, %s, 'processing') RETURNING id",
                (OWNER, f.name),
            )
            doc_id = (await cur.fetchone())["id"]
            key = f"{OWNER}/{doc_id}.pdf"
            await get_storage().save(key, f.read_bytes(), "application/pdf")
            await conn.execute("UPDATE documents SET storage_key = %s WHERE id = %s", (key, doc_id))
        t0 = time.perf_counter()
        try:
            await ingest_document(doc_id)
            print(f"Ingested {f.name} in {time.perf_counter() - t0:.1f}s")
        except Exception as e:  # e.g. the scanned notice when no OCR engine is available
            async with get_pool().connection() as conn:
                await conn.execute("UPDATE documents SET status = 'failed', error = %s WHERE id = %s", (str(e), doc_id))
            print(f"WARNING: could not ingest {f.name}: {e}")


async def judge(item: dict, chunks: list[dict], answer: str, model: str) -> dict:
    prompt = JUDGE_PROMPT.format(
        question=item["question"], reference=item["answer"], sources=build_context(chunks), answer=answer
    )
    resp = await client().chat.completions.create(
        model=model,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    text = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(re.sub(r"^```(json)?|```$", "", text.strip()).strip())
        return {
            "correctness": float(data["correctness"]),
            "faithfulness": float(data["faithfulness"]),
            "reason": data.get("reason", ""),
        }
    except (ValueError, KeyError, TypeError):
        return {"correctness": None, "faithfulness": None, "reason": f"unparseable judge output: {text[:200]}"}


async def evaluate(item: dict, use_judge: bool, judge_model: str, sem: asyncio.Semaphore) -> dict:
    async with sem:
        t0 = time.perf_counter()
        chunks = await retrieve(OWNER, item["question"])
        answer = Answer()
        if chunks:
            async for _ in stream_answer(item["question"], chunks, answer=answer):
                pass
        else:
            answer.text = "I don't know based on the uploaded documents."
        latency = time.perf_counter() - t0

        result = {
            "id": item["id"],
            "question": item["question"],
            "reference": item["answer"],
            "answer": answer.text,
            "retrieved": [{"file": c["filename"], "page": c["page"]} for c in chunks],
            "latency_s": round(latency, 2),
            "cost_usd": answer.usage.cost_usd,
            "invalid_citations": sorted(cited_tags(answer.text) - {f"S{i}" for i in range(1, len(chunks) + 1)}),
            "cited": bool(cited_tags(answer.text)),
        }
        if item["answer"] is None:  # the documents don't contain the answer
            result["refused"] = is_no_answer(answer.text)
            return result

        target = (item["file"], item["page"])
        ranks = [i for i, c in enumerate(chunks, 1) if (c["filename"], c["page"]) == target]
        result["rank"] = ranks[0] if ranks else None
        if use_judge:
            result.update(await judge(item, chunks, answer.text, judge_model))
        return result


def mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 3) if values else None


def summarize(results: list[dict]) -> dict:
    answerable = [r for r in results if r["reference"] is not None]
    negatives = [r for r in results if r["reference"] is None]
    latencies = sorted(r["latency_s"] for r in results)
    return {
        "questions": len(results),
        "hit@1": mean([r["rank"] == 1 for r in answerable]),
        f"hit@{settings.top_k}": mean([r["rank"] is not None for r in answerable]),
        "mrr": mean([1 / r["rank"] if r["rank"] else 0 for r in answerable]),
        "correctness": mean([r.get("correctness") for r in answerable]),
        "faithfulness": mean([r.get("faithfulness") for r in answerable]),
        "refusal_accuracy": mean([r["refused"] for r in negatives]),
        "citation_rate": mean([r["cited"] for r in answerable]),
        "answers_with_invalid_citations": sum(bool(r["invalid_citations"]) for r in results),
        "latency_p50_s": latencies[len(latencies) // 2],
        "latency_p95_s": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))],
        "total_cost_usd": round(sum(r["cost_usd"] for r in results), 5),
    }


def write_report(summary: dict, results: list[dict], config: dict) -> Path:
    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (RESULTS / f"{stamp}.json").write_text(
        json.dumps({"config": config, "summary": summary, "results": results}, indent=2, default=str)
    )
    lines = [
        f"# Eval results ({datetime.now():%Y-%m-%d %H:%M})",
        "",
        "Config: " + ", ".join(f"`{k}={v}`" for k, v in config.items()),
        "",
        "| Metric | Value |",
        "|---|---|",
        *(f"| {k} | {v} |" for k, v in summary.items()),
        "",
        "## Misses",
        "",
        "| id | rank | correctness | faithfulness | answer |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        bad = (r["reference"] is None and not r.get("refused")) or (
            r["reference"] is not None
            and (r["rank"] is None or (r.get("correctness") is not None and r["correctness"] < 1))
        )
        if bad:
            short = r["answer"].replace("\n", " ").replace("|", "/")[:120]
            lines.append(
                f"| {r['id']} | {r.get('rank')} | {r.get('correctness')} | {r.get('faithfulness')} | {short} |"
            )
    path = RESULTS / "latest.md"
    path.write_text("\n".join(lines) + "\n")
    return path


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-judge", action="store_true", help="skip the LLM judge")
    parser.add_argument(
        "--judge-model", default=settings.chat_model, help="ideally a stronger model than the one judged"
    )
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument("--concurrency", type=int, default=2, help="parallel questions (mind API rate limits)")
    parser.add_argument("--reingest", action="store_true", help="re-ingest the corpus even if already present")
    args = parser.parse_args()
    use_judge = not args.no_judge and settings.llm_provider != "fake"

    items = [json.loads(line) for line in (ROOT / "dataset.jsonl").read_text().splitlines() if line.strip()]
    items = items[: args.limit] if args.limit else items

    await init_db()
    try:
        await ingest_corpus(reuse=not args.reingest)
        sem = asyncio.Semaphore(args.concurrency)
        results = await asyncio.gather(*(evaluate(it, use_judge, args.judge_model, sem) for it in items))
    finally:
        await close_db()

    summary = summarize(results)
    config = {
        "chat_model": settings.chat_model if settings.llm_provider != "fake" else "fake",
        "embed_model": settings.embed_model,
        "reranker": settings.reranker,
        "top_k": settings.top_k,
        "chunk_size": settings.chunk_size,
        "judge": args.judge_model if use_judge else "off",
    }
    path = write_report(summary, results, config)
    print(json.dumps(summary, indent=2))
    print(f"Report: {path}")


if __name__ == "__main__":
    asyncio.run(main())
