"""One-off helper: turn a hand-reviewed list of questions into the format run_eval.py needs.

Run (repo root):  python fix_questions.py

Reads eval/questions.json (a list of questions with question / expected_answer / gold_url),
adds the missing `type` and `id` fields, adds the 2 seed questions and the 10 out-of-scope
questions if they are not there yet, and saves the result back to eval/questions.json.
Your original file is kept as eval/questions_backup.json. No API calls are made.
"""
import json
import shutil
from pathlib import Path

from make_questions import OUT_OF_SCOPE, SEED_IN_SCOPE

FILE = Path("eval/questions.json")
BACKUP = Path("eval/questions_backup.json")


def main() -> None:
    items = json.loads(FILE.read_text(encoding="utf-8"))
    if not BACKUP.exists():
        shutil.copy(FILE, BACKUP)

    questions, seen = [], set()

    def add(q: dict) -> None:
        key = q["question"].strip().lower()
        if key not in seen:  # skip duplicates
            seen.add(key)
            questions.append(q)

    for q in SEED_IN_SCOPE + items:
        add({**q, "type": q.get("type", "in_scope")})
    for text in OUT_OF_SCOPE:
        add({"type": "out_of_scope", "question": text, "expected_answer": None, "gold_url": None})

    for i, q in enumerate(questions, start=1):
        q["id"] = f"q{i:02d}"

    problems = [
        q["id"] for q in questions
        if q["type"] == "in_scope" and not (q.get("expected_answer") and q.get("gold_url"))
    ]
    FILE.write_text(json.dumps(questions, indent=2, ensure_ascii=False), encoding="utf-8")

    n_in = sum(q["type"] == "in_scope" for q in questions)
    print(f"Saved {len(questions)} questions to {FILE}: {n_in} in-scope, {len(questions) - n_in} out-of-scope")
    print(f"Your original is kept in {BACKUP}")
    if problems:
        print(f"WARNING: these in-scope questions lack an expected_answer or gold_url: {problems}")


if __name__ == "__main__":
    main()
