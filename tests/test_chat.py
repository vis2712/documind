import pytest

from tests.helpers import chat, make_pdf, new_user, upload

HANDBOOK = [
    "Leave policy\nEmployees get 24 days of annual leave per year.\nSick leave is 12 days per year.",
    "Travel policy\nFlights under 6 hours are booked in economy class.\nThe daily meal allowance is 1200 rupees.",
]
INVOICE = ["Invoice KL-0147\nPayment terms: Net 30 days.\nGST total: 62100 rupees."]


@pytest.fixture(scope="module")
def alice(client):
    headers = new_user(client)
    handbook = upload(client, headers, make_pdf(HANDBOOK), "handbook.pdf")
    invoice = upload(client, headers, make_pdf(INVOICE), "invoice.pdf")
    assert handbook["status"] == invoice["status"] == "ready"
    return {"headers": headers, "handbook": handbook, "invoice": invoice}


def test_answer_streams_with_citations_and_usage(client, alice):
    out = chat(client, alice["headers"], "What is the daily meal allowance?")

    assert out["meta"]["conversation_id"]
    top = out["sources"][0]
    assert (top["filename"], top["page"]) == ("handbook.pdf", 2)
    assert "[S1]" in out["tokens"]
    assert out["done"]["invalid_citations"] == []
    usage = out["done"]["usage"]
    assert usage["latency_ms"] >= 0 and usage["prompt_tokens"] > 0


def test_follow_up_uses_conversation_history(client, alice):
    first = chat(client, alice["headers"], "How many days of annual leave do employees get?")
    conv_id = first["meta"]["conversation_id"]

    follow_up = chat(client, alice["headers"], "And sick leave?", conversation_id=conv_id)
    assert follow_up["meta"]["conversation_id"] == conv_id
    # The search query was rewritten using the earlier question.
    assert "annual leave" in follow_up["meta"]["search_query"]

    conv = client.get(f"/conversations/{conv_id}", headers=alice["headers"]).json()
    assert [m["role"] for m in conv["messages"]] == ["user", "assistant", "user", "assistant"]
    assert conv["messages"][1]["sources"] and conv["messages"][1]["usage"]
    assert conv_id in [c["id"] for c in client.get("/conversations", headers=alice["headers"]).json()]


def test_search_can_be_limited_to_selected_documents(client, alice):
    out = chat(client, alice["headers"], "What is the GST total?", document_ids=[alice["handbook"]["id"]])
    assert {s["filename"] for s in out["sources"]} == {"handbook.pdf"}


def test_other_users_cannot_see_conversations_or_chunks(client, alice):
    conv_id = chat(client, alice["headers"], "What are the payment terms?")["meta"]["conversation_id"]
    bob = new_user(client)

    assert client.get(f"/conversations/{conv_id}", headers=bob).status_code == 404
    assert client.post("/chat", headers=bob, json={"question": "hi", "conversation_id": conv_id}).status_code == 404
    assert client.delete(f"/conversations/{conv_id}", headers=bob).status_code == 404

    out = chat(client, bob, "What are the payment terms?")
    assert out["sources"] == []
    assert out["tokens"].startswith("I don't know")


def test_stats_and_delete_conversation(client, alice):
    conv_id = chat(client, alice["headers"], "What is the GST total?")["meta"]["conversation_id"]
    stats = client.get("/stats", headers=alice["headers"]).json()
    assert stats["documents"]["ready"] == 2
    assert stats["answers"]["answers"] >= 1
    assert stats["answers"]["p95_latency_ms"] >= 0

    assert client.delete(f"/conversations/{conv_id}", headers=alice["headers"]).status_code == 204
    assert client.get(f"/conversations/{conv_id}", headers=alice["headers"]).status_code == 404
