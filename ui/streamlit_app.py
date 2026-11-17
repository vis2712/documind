"""DocuMind web UI. Run with:  uv run streamlit run ui/streamlit_app.py

Talks to the FastAPI backend over HTTP only (it never touches the database), like any other client."""

import json
import os

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
STATUS_ICON = {"queued": "⏳", "pending": "⏳", "processing": "⚙️", "ready": "✅", "failed": "❌"}

st.set_page_config(page_title="DocuMind", page_icon="📄", layout="wide")
state = st.session_state
state.setdefault("token", None)
state.setdefault("email", None)
state.setdefault("conversation_id", None)
state.setdefault("messages", [])


# ---------------- API helpers ----------------


def headers() -> dict:
    return {"Authorization": f"Bearer {state.token}"} if state.token else {}


def logout() -> None:
    for key in ("token", "email", "conversation_id"):
        state[key] = None
    state.messages = []


def api(method: str, path: str, **kwargs) -> httpx.Response | None:
    try:
        r = httpx.request(method, API_URL + path, headers=headers(), timeout=60, **kwargs)
    except httpx.HTTPError as e:
        st.error(f"Can't reach the API at {API_URL}: {e}")
        return None
    if r.status_code == 401 and state.token:
        logout()
        st.warning("Your session expired. Please sign in again.")
        st.rerun()
    return r


def error_detail(r: httpx.Response) -> str:
    try:
        detail = r.json().get("detail")
    except ValueError:
        return r.text
    if isinstance(detail, list):  # FastAPI validation errors
        return "; ".join(d.get("msg", "") for d in detail)
    return str(detail)


def stream_chat(question: str, result: dict):
    """Yields answer tokens for st.write_stream; fills `result` with meta, sources and usage."""
    body = {"question": question, "conversation_id": state.conversation_id}
    if state.get("doc_filter"):
        body["document_ids"] = state.doc_filter
    with httpx.stream("POST", f"{API_URL}/chat", json=body, headers=headers(), timeout=120) as r:
        if r.status_code != 200:
            r.read()
            result["error"] = error_detail(r)
            return
        event = None
        for line in r.iter_lines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
                if event == "token":
                    yield data
                else:
                    result[event] = data


# ---------------- rendering ----------------


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            ocr = " · OCR" if s.get("is_ocr") else ""
            st.markdown(f"**[{s['tag']}] {s['filename']} · page {s['page']}**{ocr} · score {s['score']}")
            st.caption(s["content"][:600] + ("…" if len(s["content"]) > 600 else ""))


def render_usage(usage: dict | None) -> None:
    if usage:
        st.caption(
            f"⏱ {usage['latency_ms'] / 1000:.1f}s (first token {(usage.get('first_token_ms') or 0) / 1000:.1f}s)"
            f" · 🔤 {usage['prompt_tokens']} in / {usage['completion_tokens']} out"
            f" · 💲{usage['cost_usd']:.5f} · {usage.get('model', '')}"
        )


def load_conversation(conversation_id: str) -> None:
    r = api("GET", f"/conversations/{conversation_id}")
    if r is not None and r.status_code == 200:
        state.conversation_id = conversation_id
        state.messages = r.json()["messages"]


# ---------------- views ----------------


def login_view() -> None:
    st.title("📄 DocuMind")
    st.write("Upload PDFs or scans, ask questions, get answers with page-level citations.")
    sign_in, register = st.tabs(["Sign in", "Create account"])
    with sign_in, st.form("login"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Sign in", type="primary"):
            r = api("POST", "/auth/login", data={"username": email, "password": password})
            if r is not None and r.status_code == 200:
                state.token, state.email = r.json()["access_token"], email
                st.rerun()
            elif r is not None:
                st.error(error_detail(r))
    with register, st.form("register"):
        email = st.text_input("Email", key="reg_email")
        password = st.text_input("Password (8+ characters)", type="password", key="reg_password")
        if st.form_submit_button("Create account"):
            r = api("POST", "/auth/register", json={"email": email, "password": password})
            if r is not None and r.status_code == 201:
                st.success("Account created. Sign in on the first tab.")
            elif r is not None:
                st.error(error_detail(r))


@st.fragment(run_every=3)  # refreshes on its own so ingestion progress shows up live
def documents_panel() -> None:
    r = api("GET", "/documents")
    docs = r.json() if r is not None and r.status_code == 200 else []
    if not docs:
        st.caption("No documents yet. Upload a PDF or image above.")
        return
    for d in docs:
        cols = st.columns([0.8, 0.2])
        meta = f"{d['num_pages']} pages · {d['num_chunks']} chunks" if d["status"] == "ready" else d["status"]
        if d.get("ocr_pages"):
            meta += f" · {d['ocr_pages']} OCR"
        cols[0].markdown(f"{STATUS_ICON.get(d['status'], '•')} **{d['filename']}**  \n{meta}")
        if d["status"] == "failed":
            cols[0].caption(f":red[{(d['error'] or '')[:160]}]")
        if cols[1].button("🗑", key=f"del-{d['id']}", help="Delete document"):
            api("DELETE", f"/documents/{d['id']}")
            st.rerun(scope="fragment")

    ready = {d["id"]: d["filename"] for d in docs if d["status"] == "ready"}
    state.doc_filter = st.multiselect(
        "Search only in (optional)",
        options=list(ready),
        format_func=lambda i: ready.get(i, i),
        default=[i for i in state.get("doc_filter") or [] if i in ready],
        placeholder="All documents",
    )


def sidebar() -> None:
    with st.sidebar:
        st.markdown(f"Signed in as **{state.email}**")
        if st.button("Sign out"):
            logout()
            st.rerun()

        st.subheader("Chats")
        if st.button("➕ New chat", use_container_width=True):
            state.conversation_id, state.messages = None, []
            st.rerun()
        r = api("GET", "/conversations")
        for c in (r.json() if r is not None and r.status_code == 200 else [])[:15]:
            current = c["id"] == state.conversation_id
            if st.button(("▶ " if current else "") + c["title"][:40], key=f"conv-{c['id']}", use_container_width=True):
                load_conversation(c["id"])
                st.rerun()

        st.subheader("Documents")
        with st.form("upload", clear_on_submit=True):
            files = st.file_uploader("PDF, PNG or JPG", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True)
            if st.form_submit_button("Upload", use_container_width=True) and files:
                for f in files:
                    r = api("POST", "/documents", files={"file": (f.name, f.getvalue(), f.type)})
                    if r is not None and r.status_code != 202:
                        st.error(f"{f.name}: {error_detail(r)}")
        documents_panel()


def chat_view() -> None:
    for m in state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                render_sources(m.get("sources") or [])
                render_usage(m.get("usage"))

    question = st.chat_input("Ask about your documents…")
    if not question:
        return
    state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        result: dict = {}
        text = st.write_stream(stream_chat(question, result))
        if "error" in result:
            st.error(result["error"] if isinstance(result["error"], str) else result["error"].get("message"))
            return
        meta, done = result.get("meta", {}), result.get("done", {})
        if meta.get("search_query") and meta["search_query"] != question:
            st.caption(f"🔎 searched for: {meta['search_query']}")
        render_sources(result.get("sources", []))
        render_usage(done.get("usage"))
        if done.get("invalid_citations"):
            st.warning(f"The answer cited sources that don't exist: {', '.join(done['invalid_citations'])}")
    state.conversation_id = meta.get("conversation_id", state.conversation_id)
    state.messages.append(
        {"role": "assistant", "content": text, "sources": result.get("sources"), "usage": done.get("usage")}
    )


def usage_view() -> None:
    r = api("GET", "/stats")
    if r is None or r.status_code != 200:
        return
    docs, answers = r.json()["documents"], r.json()["answers"]
    c = st.columns(4)
    c[0].metric("Documents ready", f"{docs['ready']} / {docs['documents']}")
    c[1].metric("Pages indexed", docs["pages"])
    c[2].metric("Chunks", docs["chunks"])
    c[3].metric("Questions answered", answers["answers"])
    c = st.columns(4)
    c[0].metric("Avg latency", f"{answers['avg_latency_ms'] / 1000:.2f}s")
    c[1].metric("p95 latency", f"{answers['p95_latency_ms'] / 1000:.2f}s")
    c[2].metric("Tokens (in / out)", f"{answers['prompt_tokens']:,} / {answers['completion_tokens']:,}")
    c[3].metric("Total cost", f"${answers['cost_usd']:.4f}")


if not state.token:
    login_view()
else:
    sidebar()
    chat_tab, usage_tab = st.tabs(["💬 Chat", "📊 Usage"])
    with chat_tab:
        chat_view()
    with usage_tab:
        usage_view()
