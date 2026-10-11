"""Tests for the answer logic in generate.py, using fake retriever and fake Gemini client.

No models are loaded and no API calls are made.
"""
import types
from contextlib import contextmanager

import generate

GRAD = "https://www.gov.uk/graduate-visa"
STUDENT = "https://www.gov.uk/student-visa"


def hit(cid, title, url, text):
    return {"id": cid, "title": title, "url": url, "updated": "2025-01-01", "text": text, "similarity": 0.6}


HITS = [
    hit("1-2", "Graduate visa", GRAD, "Graduate visa\n\nYou can travel abroad and return to the UK."),
    hit("1-4", "Graduate visa", GRAD, "Graduate visa\n\nYou can work in most jobs."),
    hit("9-1", "Student visa", STUDENT, "Student visa\n\nYou can study here."),
]


class FakeRetriever:
    def __init__(self, top_sim, hits):
        self.top_sim, self.hits = top_sim, hits

    def top_similarity(self, query):
        return self.top_sim

    def search(self, query, k, mode):
        return self.hits


class FakeClient:
    def __init__(self, text):
        self.text, self.prompts = text, []
        self.models = types.SimpleNamespace(generate_content=self._generate)

    def _generate(self, model, contents, config):
        self.prompts.append(contents)
        return types.SimpleNamespace(text=self.text)


def make_assistant(top_sim, text="Yes, you can travel [1].", hits=HITS):
    a = generate.RagAssistant.__new__(generate.RagAssistant)  # skip __init__: no models, no API key
    a.retriever = FakeRetriever(top_sim, hits)
    a.client = FakeClient(text)
    return a


@contextmanager
def patched(name, value):
    old = getattr(generate, name)
    setattr(generate, name, value)
    try:
        yield
    finally:
        setattr(generate, name, old)


def test_gate_refuses_without_calling_the_llm():
    with patched("MIN_SIM", 0.45):
        a = make_assistant(top_sim=0.30)
        r = a.ask("Who won the cricket world cup?")
    assert r["answered"] is False
    assert r["answer"] == generate.NO_ANSWER
    assert r["sources"] == [] and r["contexts"] == []
    assert a.client.prompts == []  # no LLM call, so no cost and no latency


def test_gate_threshold_is_inclusive():
    with patched("MIN_SIM", 0.45):
        a = make_assistant(top_sim=0.45)
        r = a.ask("Can I travel on a Graduate visa?")
    assert r["answered"] is True
    assert len(a.client.prompts) == 1


def test_sources_are_numbered_per_page_and_match_the_prompt():
    with patched("MIN_SIM", 0.45):
        r = make_assistant(top_sim=0.7, text="Yes [1]. Study too [2].").ask("q")
    assert [s["url"] for s in r["sources"]] == [GRAD, STUDENT]
    assert [s["n"] for s in r["sources"]] == [1, 2]
    assert len(r["contexts"]) == 2  # one merged body per page


def test_prompt_contains_numbered_pages_and_the_question():
    with patched("MIN_SIM", 0.45):
        a = make_assistant(top_sim=0.7)
        a.ask("Can I travel on a Graduate visa?")
    prompt = a.client.prompts[0]
    assert f"[1] Graduate visa ({GRAD})" in prompt
    assert f"[2] Student visa ({STUDENT})" in prompt
    assert prompt.rstrip().endswith("Question: Can I travel on a Graduate visa?")


def test_chunks_of_one_page_are_merged_in_document_order():
    hits = [HITS[1], HITS[0], HITS[2]]  # 1-4 retrieved before 1-2
    with patched("MIN_SIM", 0.45):
        r = make_assistant(top_sim=0.7, hits=hits).ask("q")
    body = r["contexts"][0]
    assert body.index("travel abroad") < body.index("most jobs")
    assert "Graduate visa\n\n" not in body  # title prefix stripped


def test_only_cited_pages_are_returned():
    with patched("MIN_SIM", 0.45):
        r = make_assistant(top_sim=0.7, text="Yes [1].").ask("q")
    assert [s["n"] for s in r["sources"]] == [1]


def test_no_citations_falls_back_to_all_pages():
    with patched("MIN_SIM", 0.45):
        r = make_assistant(top_sim=0.7, text="Yes.").ask("q")
    assert [s["n"] for s in r["sources"]] == [1, 2]


def test_comma_separated_citations_are_parsed():
    with patched("MIN_SIM", 0.45):
        r = make_assistant(top_sim=0.7, text="Yes [1, 2].").ask("q")
    assert [s["n"] for s in r["sources"]] == [1, 2]


def test_model_saying_i_dont_know_counts_as_not_answered():
    with patched("MIN_SIM", 0.45):
        r = make_assistant(top_sim=0.7, text=generate.NO_ANSWER).ask("Can I work full-time?")
    assert r["answered"] is False
    assert r["sources"] == []  # no citations for a non-answer
    assert len(r["contexts"]) == 2  # but the LLM was used, which matters for latency stats


def test_empty_model_response_is_treated_as_no_answer():
    with patched("MIN_SIM", 0.45):
        r = make_assistant(top_sim=0.7, text=None).ask("Can I travel on a Graduate visa?")
    assert r["answered"] is False
    assert r["answer"] == generate.NO_ANSWER


def test_system_prompt_uses_the_exact_refusal_sentence():
    # The code detects refusals by this exact sentence, so the prompt must contain it.
    assert generate.NO_ANSWER in generate.SYSTEM_PROMPT