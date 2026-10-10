"""Day 3: retrieve -> gate -> Gemini answer with citations.

Setup (PowerShell):   $env:GEMINI_API_KEY="your-key"
Run from repo root:   python generate.py "Can I work full-time on a Graduate visa?"
"""
import os
import sys
import time

from google import genai
from google.genai import types

# Defaults match what was evaluated on Day 4 (see eval/results.md).
MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
MIN_SIM = float(os.getenv("MIN_SIM", "0.45"))  # chosen from the Day 4 gate sweep
TOP_K = int(os.getenv("TOP_K", "5"))
NO_ANSWER = "I don't know based on the GOV.UK pages I have indexed."

SYSTEM_PROMPT = f"""You are an assistant that answers questions about UK government guidance
(visas, right to work, employment rights, benefits) using ONLY the numbered sources provided.

Rules:
- Use only facts stated in the sources. Never use outside knowledge.
- Cite every claim with its source number in square brackets, e.g. [1] or [2][3].
- If the sources do not contain the answer, reply exactly: {NO_ANSWER}
- Be concise and plain-English. Use short bullet points for conditions or steps.
- End with one line: "This is general guidance, not legal advice."
"""


class RagAssistant:
    def __init__(self) -> None:
        # Imported here so tests can load this module without torch or chromadb installed.
        from retriever import HybridRetriever

        self.model = MODEL
        self.retriever = HybridRetriever()
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    @staticmethod
    def _build_context(hits: list[dict]) -> str:
        blocks = []
        for n, h in enumerate(hits, start=1):
            blocks.append(f"[{n}] {h['title']} ({h['url']})\n{h['text']}")
        return "\n\n---\n\n".join(blocks)

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

        prompt = f"Sources:\n\n{self._build_context(hits)}\n\nQuestion: {question}"
        resp = self.client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT, temperature=0.1
            ),
        )
        answer = (resp.text or "").strip()
        if not answer:  # blocked or empty response: treat it as "no answer", never as an answer
            answer = NO_ANSWER

        seen, sources = set(), []
        for n, h in enumerate(hits, start=1):
            if h["url"] not in seen:
                seen.add(h["url"])
                sources.append(
                    {"n": n, "title": h["title"], "url": h["url"], "updated": h["updated"]}
                )

        answered = NO_ANSWER not in answer
        return {
            "answer": answer,
            "answered": answered,
            "contexts": [h["text"] for h in hits],
            "best_similarity": round(best, 3),
            "sources": sources if answered else [],
            "retrieve_ms": round(t_retrieve * 1000),
            "total_ms": round((time.perf_counter() - t0) * 1000),
        }


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Can I work full-time on a Graduate visa?"
    result = RagAssistant().ask(q)
    print(f"\nQ: {q}\n\n{result['answer']}\n")
    for s in result["sources"]:
        print(f"[{s['n']}] {s['title']} - {s['url']}")
    if os.getenv("SHOW_CONTEXT"):  # debugging: print the full text of every retrieved chunk
        for n, c in enumerate(result["contexts"], start=1):
            print(f"\n--- source text [{n}] ---\n{c}")
    print(
        f"\nanswered={result['answered']}  best_sim={result['best_similarity']}  "
        f"retrieve={result['retrieve_ms']}ms  total={result['total_ms']}ms"
    )