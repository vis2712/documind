from app.ingest import chunk_text
from app.retrieval import keyword_query


def test_short_text_is_one_chunk():
    assert chunk_text("Hello world.", size=100, overlap=10) == ["Hello world."]


def test_chunks_respect_size():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 40 for i in range(30))
    chunks = chunk_text(text, size=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)


def test_chunks_overlap():
    text = " ".join(f"w{i}" for i in range(500))
    chunks = chunk_text(text, size=200, overlap=40)
    for a, b in zip(chunks, chunks[1:]):
        assert a.split()[-1] in b.split()  # last word of a chunk reappears at the start of the next


def test_no_text_is_lost():
    text = " ".join(f"w{i}" for i in range(500))
    joined = " ".join(chunk_text(text, size=200, overlap=40)).split()
    assert set(joined) == set(text.split())


def test_keyword_query_uses_or():
    assert keyword_query("What is the GST total?") == "What or is or the or GST or total"
