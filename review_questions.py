"""Day 4c: review the drafted questions one by one, quickly.

Run (repo root):  python review_questions.py

For each question you see the question, the reference answer, and the source text it was
written from, so you can check the answer in seconds.
    Enter = keep     d = drop     e = edit the reference answer     q = save and quit

Progress is saved after every question. Run it again to continue where you stopped.
Output: eval/questions.json  (kept in-scope questions + all out-of-scope questions)
"""
import json
import sys
from pathlib import Path

DRAFT = Path("eval/questions_draft.json")
STATE = Path("eval/review_state.json")
OUT = Path("eval/questions.json")
CHUNKS = Path("data/chunks.jsonl")
SHOW_CHARS = 1500


def save(questions: list[dict], state: dict) -> int:
    STATE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    kept = []
    for q in questions:
        if q["type"] == "out_of_scope":
            kept.append(q)
        elif state.get(q["id"], {}).get("action") == "keep":
            kept.append({**q, "expected_answer": state[q["id"]]["expected_answer"]})
    OUT.write_text(json.dumps(kept, indent=2, ensure_ascii=False), encoding="utf-8")
    return sum(1 for q in kept if q["type"] == "in_scope")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles
    if not DRAFT.exists():
        raise SystemExit(
            f"{DRAFT} not found. Run  python make_questions.py  first.\n"
            "(It makes no API calls if eval/generated_cache.json already holds your questions.)"
        )
    questions = json.loads(DRAFT.read_text(encoding="utf-8"))

    chunks = {}
    for line in CHUNKS.open(encoding="utf-8"):
        c = json.loads(line)
        chunks[c["id"]] = c["text"]

    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    todo = [q for q in questions if q["type"] == "in_scope" and q["id"] not in state]
    print(f"{len(todo)} questions left to review ({len(state)} already done)\n")

    for n, q in enumerate(todo, 1):
        print("=" * 72)
        print(f"[{n}/{len(todo)}]  {q['id']}")
        print(f"QUESTION   : {q['question']}")
        print(f"REF ANSWER : {q['expected_answer']}")
        print(f"PAGE       : {q['gold_url']}")
        text = chunks.get(q.get("gold_chunk") or "", "")
        if text:
            print("\n--- source text the question was written from ---")
            print(text[:SHOW_CHARS] + (" ..." if len(text) > SHOW_CHARS else ""))
        else:
            print("\n(no source text stored for this one: check the answer on the page itself)")

        while True:
            choice = input("\n[Enter]=keep  d=drop  e=edit answer  q=save and quit > ").strip().lower()
            if choice == "":
                state[q["id"]] = {"action": "keep", "expected_answer": q["expected_answer"]}
                break
            if choice == "d":
                state[q["id"]] = {"action": "drop"}
                break
            if choice == "e":
                new = input("New reference answer: ").strip()
                if new:
                    state[q["id"]] = {"action": "keep", "expected_answer": new}
                    break
            if choice == "q":
                n_in = save(questions, state)
                print(f"\nSaved. {n_in} in-scope questions kept so far in {OUT}")
                return
        save(questions, state)

    n_in = save(questions, state)
    n_out = sum(1 for q in questions if q["type"] == "out_of_scope")
    print(f"\nDone. {n_in} in-scope + {n_out} out-of-scope questions saved to {OUT}")


if __name__ == "__main__":
    main()