"""Day 4a: draft an evaluation question set.

Samples chunks from different pages, asks Gemini to write one realistic question
and a reference answer for each, then adds hand-written seed and out-of-scope questions.

Progress is saved after every question (eval/generated_cache.json), so if you hit the
API quota you can simply run it again later and it resumes where it stopped.

Run (repo root):  python make_questions.py
Output:           eval/questions_draft.json
-> REVIEW IT BY HAND, delete weak questions, then save it as eval/questions.json
"""
import json
import random
import time
from collections import defaultdict
from pathlib import Path

from gemini_utils import call_json

CHUNKS_FILE = Path("data/chunks.jsonl")
CACHE = Path("eval/generated_cache.json")
OUT = Path("eval/questions_draft.json")
N_GENERATED = 40   # draft more than you need; you will delete the weak ones
MIN_CHARS = 700    # skip short, low-content chunks
SLEEP = 6.5        # seconds between calls (free-tier rate limit)
SEED = 7

SYSTEM = """You write evaluation questions for a Q&A assistant over GOV.UK guidance.
Given one passage, write ONE question a real person in the UK might ask that the passage
fully answers, plus a short reference answer (1-3 sentences) using only facts in the passage.
Rules:
- Natural wording, as a user would type it. Never say "the passage", "the text" or "according to".
- Name the specific visa, benefit or scheme in the question when the answer depends on it,
  so the question is not ambiguous across similar pages.
- Prefer concrete facts: eligibility, amounts, deadlines, what you can or cannot do.
Return JSON: {"question": "...", "expected_answer": "..."}"""

# Hand-written cases. The first one is the real failure you found on Day 3.
SEED_IN_SCOPE = [
    {
        "question": "can i go outside uk on my graduate visa for a trip?",
        "expected_answer": "Yes. A Graduate visa allows you to travel abroad and return to the UK.",
        "gold_url": "https://www.gov.uk/graduate-visa",
        "gold_chunk": "1-2",  # the chunk your Day 3 debug run showed contains "travel abroad"
    },
    {
        "question": "Can I work full-time on a Graduate visa?",
        "expected_answer": "A Graduate visa lets you work in most jobs and be self-employed.",
        "gold_url": "https://www.gov.uk/graduate-visa",
    },
]

# Clearly unrelated, plus "near-scope" questions that sound official but are not UK guidance.
OUT_OF_SCOPE = [
    "Who won the cricket world cup?",
    "How do I bake sourdough bread?",
    "What is the capital of France?",
    "Which laptop is best for programming?",
    "Explain how photosynthesis works.",
    "How do I fix a flat bicycle tyre?",
    "Write me a poem about autumn.",
    "How do I apply for a US work visa?",
    "What is the minimum wage in Germany?",
    "How much does a Canadian work permit cost?",
]


def load_cache() -> list[dict]:
    if CACHE.exists():
        return json.loads(CACHE.read_text(encoding="utf-8"))
    return []


def main() -> None:
    random.seed(SEED)
    OUT.parent.mkdir(parents=True, exist_ok=True)

    by_page = defaultdict(list)
    for line in CHUNKS_FILE.open(encoding="utf-8"):
        c = json.loads(line)
        if len(c["text"]) >= MIN_CHARS:
            by_page[c["url"]].append(c)

    urls = list(by_page)
    random.shuffle(urls)  # one chunk per page keeps the set diverse

    generated = load_cache()
    done = {g["gold_url"] for g in generated}
    if generated:
        print(f"Resuming: {len(generated)} questions already saved in {CACHE}")

    for url in urls:
        if len(generated) >= N_GENERATED:
            break
        if url in done:
            continue
        chunk = random.choice(by_page[url])
        prompt = f"Page title: {chunk['title']}\n\nPassage:\n{chunk['text'][:2500]}"
        try:
            out = call_json(prompt, SYSTEM)
        except Exception as e:  # noqa: BLE001
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print("\nAPI quota reached. Stopping here; run this script again later to resume.")
                break
            print(f"skip {url}: {e}")
            continue
        if not isinstance(out, dict) or "question" not in out or "expected_answer" not in out:
            print(f"skip {url}: unexpected output")
            continue
        generated.append(
            {
                "question": out["question"].strip(),
                "expected_answer": out["expected_answer"].strip(),
                "gold_url": url,
                "gold_chunk": chunk["id"],
            }
        )
        CACHE.write_text(json.dumps(generated, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[{len(generated)}] {out['question']}")
        time.sleep(SLEEP)

    questions = []
    for item in SEED_IN_SCOPE + generated:
        questions.append({"type": "in_scope", **item})
    for q in OUT_OF_SCOPE:
        questions.append(
            {"type": "out_of_scope", "question": q, "expected_answer": None, "gold_url": None}
        )
    for i, q in enumerate(questions, start=1):
        q["id"] = f"q{i:02d}"

    OUT.write_text(json.dumps(questions, indent=2, ensure_ascii=False), encoding="utf-8")
    n_in = len(SEED_IN_SCOPE) + len(generated)
    print(f"\nWrote {len(questions)} questions ({n_in} in-scope, {len(OUT_OF_SCOPE)} out-of-scope) to {OUT}")
    print("Next: review it by hand, delete weak ones, save as eval/questions.json")


if __name__ == "__main__":
    main()