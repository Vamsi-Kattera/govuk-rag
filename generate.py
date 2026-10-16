"""Day 3: retrieve -> gate -> Gemini answer with citations.

Setup (PowerShell):   $env:GEMINI_API_KEY="your-key"
Run from repo root:   python generate.py "Can I work full-time on a Graduate visa?"
"""
import os
import re
import sys
import time

from google import genai
from google.genai import types

# Defaults match what was evaluated on Day 4 (see eval/results.md).
MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
MIN_SIM = float(os.getenv("MIN_SIM", "0.45"))  # chosen from the Day 4 gate sweep
TOP_K = int(os.getenv("TOP_K", "5"))
NEIGHBOURS = int(os.getenv("NEIGHBOURS", "1"))  # also include chunks N before/after each of the top 2 hits (0 = off)
EXPAND_TOP = 2
FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "")  # optional second model if the first is overloaded
RETRY_DELAY = 2.0  # seconds between attempts
THINKING_BUDGET = os.getenv("THINKING_BUDGET")  # e.g. "0" turns off model "thinking" (faster); unset = model default
LLM_TIMEOUT_MS = int(os.getenv("LLM_TIMEOUT_MS", "60000"))
PROMPT_VERSION = "v7-gaps-only-when-needed"  # bump when the prompt or context layout changes (keys the eval cache)
NO_ANSWER = "I don't know based on the GOV.UK pages I have indexed."

SYSTEM_PROMPT = f"""You are an assistant that answers questions about UK government guidance
(visas, right to work, employment rights, benefits) using ONLY the numbered sources provided.

Rules:
- Use only facts stated in the sources. Never use outside knowledge.
- Cite every claim with the number of the source it came from, in square brackets containing only
  that number, e.g. [1] or [2][3]. Never cite section or paragraph numbers that appear inside a source.
- If the sources answer the question fully, answer it and stop: do not add remarks about what the
  sources do not cover.
- If the sources answer it only partly, give that part first, then add one short sentence naming the
  part of the question they do not answer. Never say the sources do not mention something when they
  say something relevant to it.
- If the sources contain nothing relevant to the question, reply with exactly this sentence and
  nothing else (no explanation, no citations): {NO_ANSWER}
- Be concise and plain-English. Use short bullet points for conditions or steps.
- End with one line: "This is general guidance, not legal advice."
"""


class RagAssistant:
    last_raw = ""  # the model's answer before the cite-or-abstain rule (for debugging)

    def __init__(self) -> None:
        # Imported here so tests can load this module without torch or chromadb installed.
        from retriever import HybridRetriever

        self.model = MODEL
        self.retriever = HybridRetriever()
        opts = {"timeout": LLM_TIMEOUT_MS}
        if hasattr(types, "HttpRetryOptions"):  # one attempt per call: we do our own short retry below
            opts["retry_options"] = types.HttpRetryOptions(attempts=1)
        self.client = genai.Client(
            api_key=os.environ["GEMINI_API_KEY"], http_options=types.HttpOptions(**opts)
        )

    def _generate(self, prompt: str):
        """Call Gemini; on a temporary overload (5xx) retry once, then try the fallback model."""
        kwargs = {}
        if THINKING_BUDGET not in (None, ""):
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=int(THINKING_BUDGET))
        config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.1, **kwargs)
        models = [MODEL] + ([FALLBACK_MODEL] if FALLBACK_MODEL and FALLBACK_MODEL != MODEL else [])
        last = None
        for model in models:
            for attempt in range(2):
                try:
                    return self.client.models.generate_content(model=model, contents=prompt, config=config)
                except Exception as e:  # noqa: BLE001
                    last = e
                    temporary = getattr(e, "code", None) in (500, 502, 503, 504) or "Timeout" in type(e).__name__
                    if not temporary:
                        raise  # auth errors, quota, bad request: retrying will not help
                    if attempt == 0:
                        time.sleep(RETRY_DELAY)
        raise last

    def _expand(self, hits: list[dict]) -> list[dict]:
        """Add the chunks just before/after the best hits: lists and sentences often straddle a chunk boundary."""
        pool = getattr(self.retriever, "chunks", None) or {}
        if NEIGHBOURS <= 0 or not pool:
            return hits
        out = list(hits)
        for h in hits[:EXPAND_TOP]:
            doc, _, n = str(h["id"]).rpartition("-")
            if not n.isdigit():
                continue
            for d in range(1, NEIGHBOURS + 1):
                for m in (int(n) - d, int(n) + d):
                    c = pool.get(f"{doc}-{m}")
                    if c:
                        out.append(c)
        return out

    @staticmethod
    def _group_by_page(hits: list[dict]) -> list[dict]:
        """Merge chunks of the same page (document order), pages in first-appearance order."""
        pages: dict[str, dict] = {}
        seen_ids: set = set()
        for h in hits:
            if h.get("id") in seen_ids:
                continue
            seen_ids.add(h.get("id"))
            p = pages.setdefault(
                h["url"],
                {"title": h["title"], "url": h["url"], "updated": h.get("updated"), "chunks": []},
            )
            try:
                order = int(str(h["id"]).rsplit("-", 1)[-1])
            except (KeyError, ValueError):
                order = 0
            body = h["text"].removeprefix(h["title"] + "\n\n")
            p["chunks"].append((order, body))
        out = []
        for p in pages.values():
            p["body"] = RagAssistant._merge_chunks(sorted(p["chunks"], key=lambda c: c[0]))
            out.append(p)
        return out

    @staticmethod
    def _merge_chunks(chunks: list[tuple[int, str]]) -> str:
        """Join chunks in order. Adjacent chunks share an overlap (from the splitter): drop the repeat.
        Chunks that are not adjacent are separated by a gap marker."""
        text, prev_order = "", None
        for order, body in chunks:
            if prev_order is None:
                text = body
            elif order == prev_order:
                pass
            elif order == prev_order + 1:
                for k in range(min(len(text), len(body), 600), 19, -1):
                    if text.endswith(body[:k]):
                        body = body[k:]
                        break
                text = text + "\n" + body.lstrip("\n")
            else:
                text = text + "\n\n[...]\n\n" + body
            prev_order = order
        return text

    @staticmethod
    def _build_context(pages: list[dict]) -> str:
        blocks = [f"[{n}] {p['title']} ({p['url']})\n{p['body']}" for n, p in enumerate(pages, start=1)]
        return "\n\n---\n\n".join(blocks)

    _CITE = re.compile(r"\s*\d+(?:\.\w+)?(?:\s*[,;]\s*\d+(?:\.\w+)?)*\s*")

    @staticmethod
    def _cited(answer: str) -> set[int]:
        """Source numbers cited in the answer: [1], [1, 2], and section-style [1.a] (counts as source 1)."""
        nums: set[int] = set()
        for group in re.findall(r"\[([^\[\]]{1,40})\]", answer):
            if RagAssistant._CITE.fullmatch(group):
                nums.update(int(x) for x in re.findall(r"(\d+)(?:\.\w+)?", group))
        return nums

    def ask(self, question: str, mode: str = os.getenv("RETRIEVAL_MODE", "rerank")) -> dict:
        t0 = time.perf_counter()
        hits = self.retriever.search(question, k=TOP_K, mode=mode)
        t_retrieve = time.perf_counter() - t0

        # "I don't know" gate: is ANY chunk semantically close to the question?
        # Uses the top vector similarity, so the gate does not depend on retrieval mode.
        best = self.retriever.top_similarity(question)
        if best < MIN_SIM:
            return {
                "answer": NO_ANSWER,
                "answered": False,
                "contexts": [],
                "best_similarity": round(best, 3),
                "sources": [],
                "retrieve_ms": round(t_retrieve * 1000),
                "total_ms": round((time.perf_counter() - t0) * 1000),
            }

        pages = self._group_by_page(self._expand(hits))
        prompt = f"Sources:\n\n{self._build_context(pages)}\n\nQuestion: {question}"
        resp = self._generate(prompt)
        answer = (resp.text or "").strip()
        self.last_raw = answer
        if not answer:  # blocked or empty response: treat it as "no answer", never as an answer
            answer = NO_ANSWER

        # Cite or abstain: an answer must cite at least one source that exists. A paraphrased refusal
        # ("the sources do not contain...") or an uncited answer is treated as a refusal.
        valid = {n for n in self._cited(answer) if 1 <= n <= len(pages)}
        if NO_ANSWER not in answer and not valid:
            answer = NO_ANSWER
        answered = NO_ANSWER not in answer
        sources = [
            {"n": n, "title": p["title"], "url": p["url"], "updated": p["updated"]}
            for n, p in enumerate(pages, start=1)
            if n in valid
        ]

        return {
            "answer": answer,
            "answered": answered,
            "contexts": [p["body"] for p in pages],
            "best_similarity": round(best, 3),
            "sources": sources if answered else [],
            "retrieve_ms": round(t_retrieve * 1000),
            "total_ms": round((time.perf_counter() - t0) * 1000),
        }


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Can I work full-time on a Graduate visa?"
    assistant = RagAssistant()
    result = assistant.ask(q)
    print(f"\nQ: {q}\n\n{result['answer']}\n")
    for s in result["sources"]:
        print(f"[{s['n']}] {s['title']} - {s['url']}")
    if os.getenv("SHOW_RAW"):  # what the model really wrote, before the cite-or-abstain rule
        print(f"\n--- raw model answer ---\n{assistant.last_raw}\n------------------------")
    if os.getenv("SHOW_CONTEXT"):  # debugging: print the full text of every retrieved chunk
        for n, c in enumerate(result["contexts"], start=1):
            print(f"\n--- source text [{n}] ---\n{c}")
    print(
        f"\nanswered={result['answered']}  best_sim={result['best_similarity']}  "
        f"retrieve={result['retrieve_ms']}ms  total={result['total_ms']}ms"
    )