"""API tests with a fake assistant: no models, no network, no API key needed."""
import logging
import types

import pytest
from fastapi.testclient import TestClient

from api import VERSION, create_app
from service_utils import JsonFormatter, get_logger

ANSWERED = {
    "answer": "Yes, you can travel abroad and return [1].",
    "answered": True,
    "best_similarity": 0.61,
    "sources": [{"n": 1, "title": "Graduate visa", "url": "https://www.gov.uk/graduate-visa", "updated": "2025-01-01"}],
    "retrieve_ms": 120,
    "total_ms": 2200,
    "contexts": ["Graduate visa\n\nYou can travel abroad and return to the UK."],
}
REFUSED = {
    "answer": "I don't know based on the GOV.UK pages I have indexed.",
    "answered": False,
    "best_similarity": 0.2,
    "sources": [],
    "retrieve_ms": 50,
    "total_ms": 55,
    "contexts": [],
}
QUESTION = "Can I travel abroad on a Graduate visa?"


class FakeAssistant:
    model = "fake-model"

    def __init__(self, *results, error=None):
        self.results = list(results) or [ANSWERED]
        self.error = error
        self.calls = []
        self.retriever = types.SimpleNamespace(chunks={"1-0": {}, "1-1": {}})

    def ask(self, question, mode="rerank"):
        self.calls.append((question, mode))
        if self.error:
            raise self.error
        result = self.results[0] if len(self.results) == 1 else self.results.pop(0)
        return dict(result)


@pytest.fixture
def make_client():
    clients = []

    def _make(assistant=None, **kwargs):
        kwargs.setdefault("rate_limit_per_min", 1000)
        kwargs.setdefault("daily_limit", 100_000)
        fake = assistant or FakeAssistant()
        client = TestClient(create_app(lambda: fake, **kwargs))
        client.__enter__()  # runs the startup (lifespan) code
        clients.append(client)
        return client

    yield _make
    for c in clients:
        c.__exit__(None, None, None)


def ask(client, question=QUESTION, **extra):
    return client.post("/ask", json={"question": question, **extra})


# ---------- basic endpoints ----------
def test_health_reports_model_and_chunk_count(make_client):
    data = make_client().get("/health").json()
    assert data == {"status": "ok", "version": VERSION, "model": "fake-model", "chunks": 2}


def test_root_points_to_docs(make_client):
    assert make_client().get("/").json()["docs"] == "/docs"


# ---------- /ask ----------
def test_ask_returns_cited_answer_and_request_id(make_client):
    resp = ask(make_client())
    assert resp.status_code == 200
    data = resp.json()
    assert data["answered"] is True
    assert data["sources"][0]["url"] == "https://www.gov.uk/graduate-visa"
    assert data["total_ms"] == 2200
    assert len(data["request_id"]) == 12
    assert resp.headers["X-Request-ID"] == data["request_id"]


def test_ask_refusal_has_no_sources(make_client):
    data = ask(make_client(FakeAssistant(REFUSED)), "Who won the cricket world cup?").json()
    assert data["answered"] is False
    assert data["sources"] == []


def test_question_is_trimmed_before_use(make_client):
    fake = FakeAssistant()
    ask(make_client(fake), "   Can I work on a Graduate visa?   ")
    assert fake.calls[0][0] == "Can I work on a Graduate visa?"


def test_mode_is_forwarded_and_defaults_to_the_assistants_default(make_client):
    fake = FakeAssistant()
    client = make_client(fake)
    ask(client, mode="hybrid")
    ask(client)
    assert [mode for _, mode in fake.calls] == ["hybrid", "rerank"]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": ""},
        {"question": "  "},
        {"question": "ab"},
        {"question": "x" * 501},
        {"question": QUESTION, "mode": "nope"},
        {"question": 123},
    ],
)
def test_invalid_requests_are_rejected_before_reaching_the_assistant(make_client, payload):
    fake = FakeAssistant()
    resp = make_client(fake).post("/ask", json=payload)
    assert resp.status_code == 422
    assert fake.calls == []


def test_assistant_failure_returns_503_without_leaking_details(make_client):
    fake = FakeAssistant(error=RuntimeError("secret-key-abc quota exceeded"))
    client = make_client(fake)
    resp = ask(client)
    assert resp.status_code == 503
    assert "secret" not in resp.text and "quota" not in resp.text
    assert "X-Request-ID" in resp.headers
    assert client.get("/metrics").json()["errors"] == 1


# ---------- /metrics ----------
def test_metrics_reflect_traffic(make_client):
    fake = FakeAssistant(dict(ANSWERED, total_ms=1000), dict(ANSWERED, total_ms=3000), REFUSED)
    client = make_client(fake)
    for _ in range(3):
        ask(client)
    m = client.get("/metrics").json()
    assert (m["requests_total"], m["answered"], m["refused"], m["errors"]) == (3, 2, 1, 0)
    assert (m["latency_window"], m["latency_p50_ms"], m["latency_p95_ms"]) == (2, 2000, 2900)


# ---------- rate limiting ----------
def test_per_minute_limit_returns_429_with_retry_after(make_client):
    fake = FakeAssistant()
    client = make_client(fake, rate_limit_per_min=2)
    assert [ask(client).status_code for _ in range(3)] == [200, 200, 429]
    assert ask(client).headers["Retry-After"] == "60"
    assert len(fake.calls) == 2  # the limited requests never reached the assistant


def test_daily_limit_returns_429(make_client):
    client = make_client(daily_limit=1)
    assert ask(client).status_code == 200
    resp = ask(client)
    assert resp.status_code == 429 and "Daily" in resp.json()["detail"]


def test_forged_forwarded_header_cannot_bypass_the_limit_by_default(make_client):
    client = make_client(rate_limit_per_min=1)
    assert client.post("/ask", json={"question": QUESTION}, headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 200
    assert client.post("/ask", json={"question": QUESTION}, headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 429


def test_behind_a_trusted_proxy_clients_are_told_apart_by_the_last_entry(make_client):
    client = make_client(rate_limit_per_min=1, trust_proxy=True)

    def post(forwarded):
        return client.post("/ask", json={"question": QUESTION}, headers={"X-Forwarded-For": forwarded}).status_code

    assert post("1.1.1.1") == 200
    assert post("2.2.2.2") == 200  # a different client
    assert post("forged, 1.1.1.1") == 429  # forged prefix ignored: still client 1.1.1.1


# ---------- logging ----------
class ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.setFormatter(JsonFormatter())
        self.lines = []

    def emit(self, record):
        self.lines.append(self.format(record))


def captured_logs(make_client, **kwargs):
    handler, logger = ListHandler(), get_logger()
    logger.addHandler(handler)
    try:
        ask(make_client(**kwargs), "Can I travel abroad on a Graduate visa?")
    finally:
        logger.removeHandler(handler)
    return "\n".join(handler.lines)


def test_question_text_is_not_logged_by_default(make_client):
    logs = captured_logs(make_client)
    assert "Graduate visa?" not in logs  # privacy: only length and hash are logged
    assert '"q_hash"' in logs and '"request_id"' in logs


def test_question_text_is_logged_when_enabled(make_client):
    assert "Can I travel abroad on a Graduate visa?" in captured_logs(make_client, log_questions=True)
