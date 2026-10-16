"""Print eval answers for hand-checking the LLM judge.

    python show_answers.py q08 q35 q50          # specific questions
    python show_answers.py --sample 6           # 6 random in-scope answers the judge marked correct
Reads eval/results.json (written by run_eval.py --answers rerank).
"""
import json
import random
import sys
from pathlib import Path

data = json.loads(Path("eval/results.json").read_text(encoding="utf-8"))
rows = next(iter(data["details"].values()))
by_id = {r["id"]: r for r in rows}

args = sys.argv[1:]
if args[:1] == ["--sample"]:
    n = int(args[1]) if len(args) > 1 else 6
    pool = [r for r in rows if r["type"] == "in_scope" and r.get("correct")]
    chosen = random.sample(pool, min(n, len(pool)))
else:
    chosen = [by_id[i] for i in args if i in by_id]

for r in chosen:
    print("=" * 78)
    print(f"{r['id']} [{r['type']}]  answered={r['answered']}  correct={r.get('correct')}  faithful={r.get('faithful')}")
    print(f"Q: {r['question']}\n")
    print(r["answer"])
    if r.get("reason"):
        print(f"\nJudge said: {r['reason']}")
