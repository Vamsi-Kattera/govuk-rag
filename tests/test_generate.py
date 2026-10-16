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


def test_answer_without_any_citation_is_treated_as_a_refusal():
    with patched("MIN_SIM", 0.45):
        r = make_assistant(top_sim=0.7, text="The sources do not contain information about the US.").ask("q")
    assert r["answered"] is False
    assert r["answer"] == generate.NO_ANSWER
    assert r["sources"] == []


def test_citation_to_a_source_that_does_not_exist_does_not_count():
    with patched("MIN_SIM", 0.45):
        r = make_assistant(top_sim=0.7, text="Yes [7].").ask("q")  # only 2 pages were given
    assert r["answered"] is False and r["sources"] == []


def test_prompt_has_no_copyable_example_gap_sentence():
    assert "full-time" not in generate.SYSTEM_PROMPT


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


def test_neighbouring_chunks_are_added_for_top_hits():
    pool = {h["id"]: h for h in HITS}
    pool["1-1"] = hit("1-1", "Graduate visa", GRAD, "Graduate visa\n\nYou can work or look for work at any skill level.")
    a = make_assistant(top_sim=0.7, hits=[HITS[0]])  # only 1-2 retrieved
    a.retriever.chunks = pool
    with patched("MIN_SIM", 0.45):
        r = a.ask("Can I work full-time?")
    body = r["contexts"][0]
    assert "work or look for work" in body  # pulled in from chunk 1-1
    assert body.index("work or look for work") < body.index("travel abroad")


def test_neighbours_can_be_switched_off():
    pool = {"1-1": hit("1-1", "Graduate visa", GRAD, "Graduate visa\n\nWORK-LIST"), "1-2": HITS[0]}
    a = make_assistant(top_sim=0.7, hits=[HITS[0]])
    a.retriever.chunks = pool
    with patched("MIN_SIM", 0.45), patched("NEIGHBOURS", 0):
        r = a.ask("q")
    assert "WORK-LIST" not in r["contexts"][0]


class Flaky503:
    """Fake client that fails with a 503 a set number of times, then answers."""
    def __init__(self, failures):
        self.failures, self.calls = failures, 0
        self.models = types.SimpleNamespace(generate_content=self._generate)

    def _generate(self, model, contents, config):
        self.calls += 1
        if self.calls <= self.failures:
            err = RuntimeError("503 UNAVAILABLE")
            err.code = 503
            raise err
        return types.SimpleNamespace(text="Yes [1].")


def test_temporary_503_is_retried_once():
    a = make_assistant(top_sim=0.7)
    a.client = Flaky503(failures=1)
    with patched("MIN_SIM", 0.45), patched("RETRY_DELAY", 0), patched("FALLBACK_MODEL", ""):
        r = a.ask("q")
    assert r["answered"] is True and a.client.calls == 2


def test_persistent_503_is_raised_after_two_attempts():
    a = make_assistant(top_sim=0.7)
    a.client = Flaky503(failures=99)
    with patched("MIN_SIM", 0.45), patched("RETRY_DELAY", 0), patched("FALLBACK_MODEL", ""):
        try:
            a.ask("q")
            raised = False
        except RuntimeError:
            raised = True
    assert raised and a.client.calls == 2


def test_non_temporary_errors_are_not_retried():
    a = make_assistant(top_sim=0.7)
    a.client = Flaky503(failures=99)
    orig = a.client._generate

    def auth_error(model, contents, config):
        a.client.calls += 1
        err = RuntimeError("401")
        err.code = 401
        raise err

    a.client.models.generate_content = auth_error
    with patched("MIN_SIM", 0.45), patched("RETRY_DELAY", 0), patched("FALLBACK_MODEL", ""):
        try:
            a.ask("q")
        except RuntimeError:
            pass
    assert a.client.calls == 1


def test_timeouts_are_retried_like_503s():
    class ReadTimeout(Exception):
        pass

    calls = []

    def gen(model, contents, config):
        calls.append(model)
        if len(calls) == 1:
            raise ReadTimeout("The read operation timed out")
        return types.SimpleNamespace(text="Yes [1].")

    a = make_assistant(top_sim=0.7)
    a.client.models.generate_content = gen
    with patched("MIN_SIM", 0.45), patched("RETRY_DELAY", 0), patched("FALLBACK_MODEL", ""):
        r = a.ask("q")
    assert r["answered"] is True and len(calls) == 2


def test_overlap_between_adjacent_chunks_is_not_repeated():
    t1 = "Graduate visa\n\nYou can: work in most jobs\nlook for work\ntravel abroad and return to the UK"
    t2 = "Graduate visa\n\nlook for work\ntravel abroad and return to the UK\ndo voluntary work"
    hits = [hit("1-1", "Graduate visa", GRAD, t1), hit("1-2", "Graduate visa", GRAD, t2)]
    with patched("MIN_SIM", 0.45):
        r = make_assistant(top_sim=0.7, hits=hits).ask("q")
    body = r["contexts"][0]
    assert body.count("travel abroad and return to the UK") == 1
    assert "work in most jobs" in body and "do voluntary work" in body


def test_non_adjacent_chunks_get_a_gap_marker():
    hits = [hit("1-1", "Graduate visa", GRAD, "Graduate visa\n\nA"), hit("1-4", "Graduate visa", GRAD, "Graduate visa\n\nB")]
    with patched("MIN_SIM", 0.45):
        r = make_assistant(top_sim=0.7, hits=hits).ask("q")
    assert "A\n\n[...]\n\nB" in r["contexts"][0]


def test_prompt_allows_partial_answers_but_keeps_exact_refusal():
    assert "partly" in generate.SYSTEM_PROMPT
    assert generate.NO_ANSWER in generate.SYSTEM_PROMPT


def test_fallback_model_is_tried_after_the_first_model_keeps_failing():
    a = make_assistant(top_sim=0.7)
    models = []

    def gen(model, contents, config):
        models.append(model)
        if model == "primary":
            err = RuntimeError("503")
            err.code = 503
            raise err
        return types.SimpleNamespace(text="Yes [1].")

    a.client.models.generate_content = gen
    with patched("MIN_SIM", 0.45), patched("RETRY_DELAY", 0), patched("MODEL", "primary"), patched("FALLBACK_MODEL", "backup"):
        r = a.ask("q")
    assert r["answered"] is True
    assert models == ["primary", "primary", "backup"]


def test_section_style_citations_count_as_the_source_number():
    # The model sometimes copies section numbering from the source text: [1.a] means source 1.
    with patched("MIN_SIM", 0.45):
        r = make_assistant(top_sim=0.7, text="Rules apply [1.a]. Also [0.e, 0.g] and [2.b].").ask("q")
    assert r["answered"] is True
    assert [s["n"] for s in r["sources"]] == [1, 2]  # 0 is not a real source


def test_brackets_with_ordinary_text_are_not_citations():
    assert generate.RagAssistant._cited("See [the 2024 guide] and [1 week].") == set()
    assert generate.RagAssistant._cited("Yes [1][3] and [2, 4].") == {1, 2, 3, 4}


def test_prompt_says_to_cite_source_numbers_not_section_numbers():
    assert "Never cite section or paragraph numbers" in generate.SYSTEM_PROMPT


def test_prompt_forbids_unneeded_gap_remarks():
    assert "do not add remarks about what the" in generate.SYSTEM_PROMPT