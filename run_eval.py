"""Day 4b: evaluate retrieval, the "I don't know" gate, and answer quality.

Usage (repo root):
    python run_eval.py                      # retrieval + gate sweep (no LLM calls, free)
    python run_eval.py --answers rerank     # + accuracy / hallucination for one mode
    python run_eval.py --answers hybrid,rerank

Needs eval/questions.json (see make_questions.py).
Writes eval/results.json (per-question detail) and eval/results.md (tables for your README).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from gemini_utils import MODEL, call_json, retry
from generate import MIN_SIM, PROMPT_VERSION, RagAssistant
from retriever import HybridRetriever

QUESTIONS_FILE = Path("eval/questions.json")
RESULTS_JSON = Path("eval/results.json")
RESULTS_MD = Path("eval/results.md")
MODES = ["vector", "bm25", "hybrid", "rerank"]
K = 5
THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]

JUDGE_SYSTEM = """You are a strict grader for a question-answering system.
You receive a question, a reference answer, the system's answer, and the sources the system was given.
Return JSON: {"correct": true or false, "faithful": true or false, "reason": "one short sentence"}
- correct: the system answer states the key facts of the reference answer and does not contradict it.
  Extra detail is fine if it is also faithful.
- faithful: every factual claim in the system answer is supported by the sources.
  A claim that goes beyond the sources is NOT faithful, even if it is true in the real world."""


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.0f}%"


# ---------- 1. retrieval ----------
def retrieval_eval(retriever: HybridRetriever, in_scope: list[dict]) -> tuple[dict, dict]:
    table, misses = {}, {}
    for mode in MODES:
        ranks, missed, chunk_hits, n_chunk = [], [], 0, 0
        for q in in_scope:
            hits = retriever.search(q["question"], k=K, mode=mode)
            urls = [h["url"] for h in hits]
            rank = urls.index(q["gold_url"]) + 1 if q["gold_url"] in urls else None
            ranks.append(rank)
            if rank is None:
                missed.append(q["question"])
            if q.get("gold_chunk"):  # stricter: the exact chunk the question came from
                n_chunk += 1
                chunk_hits += q["gold_chunk"] in [h["id"] for h in hits]
        n = len(ranks)
        table[mode] = {
            "hit@1": sum(r == 1 for r in ranks) / n,
            "hit@3": sum(r is not None and r <= 3 for r in ranks) / n,
            "hit@5": sum(r is not None for r in ranks) / n,
            "mrr@5": sum(1 / r for r in ranks if r) / n,
            "chunk@5": chunk_hits / n_chunk if n_chunk else None,
        }
        misses[mode] = missed
        print(f"  {mode:<7} hit@1={pct(table[mode]['hit@1'])} hit@5={pct(table[mode]['hit@5'])}")
    return table, misses


# ---------- 2. "I don't know" gate ----------
def gate_sweep(retriever: HybridRetriever, questions: list[dict]) -> tuple[list[dict], dict]:
    scored = [(q["type"], q["question"], retriever.top_similarity(q["question"])) for q in questions]
    in_s = [s for t, _, s in scored if t == "in_scope"]
    out_s = [s for t, _, s in scored if t == "out_of_scope"]
    top_oos = sorted(((s, qu) for t, qu, s in scored if t == "out_of_scope"), reverse=True)[:5]
    summary = {
        "in_scope_median": float(np.median(in_s)) if in_s else None,
        "in_scope_min": float(min(in_s)) if in_s else None,
        "out_scope_median": float(np.median(out_s)) if out_s else None,
        "out_scope_max": float(max(out_s)) if out_s else None,
        "top_out_of_scope": [[round(s, 2), qu] for s, qu in top_oos],
    }
    rows = []
    for t in THRESHOLDS:
        rows.append(
            {
                "threshold": t,
                "in_scope_answered": sum(s >= t for s in in_s) / len(in_s) if in_s else None,
                "out_scope_refused": sum(s < t for s in out_s) / len(out_s) if out_s else None,
            }
        )
    return rows, summary


# ---------- 3. answer quality ----------
def judge(q: dict, res: dict) -> dict:
    sources = "\n\n---\n\n".join(f"[{i}] {c}" for i, c in enumerate(res["contexts"], 1))
    prompt = (
        f"Question: {q['question']}\n\nReference answer: {q['expected_answer']}\n\n"
        f"System answer: {res['answer']}\n\nSources given to the system:\n{sources}"
    )
    return call_json(prompt, JUDGE_SYSTEM)


def answer_eval(assistant: RagAssistant, mode: str, questions: list[dict], sleep: float) -> tuple[dict, list[dict]]:
    assistant.retriever.search("warm up", k=1, mode=mode)  # load models before timing
    details, latencies = [], []
    # Progress is cached per mode/model/gate, so a quota error never loses finished questions.
    # Delete eval/answers_cache_* if you change the prompt or retrieval code and want a fresh run.
    cache_file = Path(f"eval/answers_cache_{mode}_{MODEL}_{MIN_SIM}_{PROMPT_VERSION}.json")
    cache = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {}
    for i, q in enumerate(questions, 1):
        if q["question"] in cache:  # finished in an earlier run
            row = cache[q["question"]]
            details.append(row)
            if row.get("llm_used"):
                latencies.append(row["total_ms"])
            print(f"  [{i}/{len(questions)}] {q['id']} (cached)")
            continue
        res = retry(lambda: assistant.ask(q["question"], mode=mode))
        if res["contexts"]:  # the LLM was called
            latencies.append(res["total_ms"])
            time.sleep(sleep)
        row = {
            "id": q["id"],
            "type": q["type"],
            "question": q["question"],
            "answer": res["answer"],
            "answered": res["answered"],
            "total_ms": res["total_ms"],
            "llm_used": bool(res["contexts"]),
        }
        if q["type"] == "in_scope" and res["answered"]:
            verdict = retry(lambda: judge(q, res))
            row.update(
                correct=bool(verdict.get("correct")),
                faithful=bool(verdict.get("faithful")),
                reason=verdict.get("reason", ""),
            )
            time.sleep(sleep)
        elif q["type"] == "in_scope":
            row.update(correct=False, faithful=None, reason="refused to answer")
        details.append(row)
        cache[q["question"]] = row
        cache_file.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  [{i}/{len(questions)}] {q['id']} answered={row['answered']} correct={row.get('correct')}")

    ins = [r for r in details if r["type"] == "in_scope"]
    oos = [r for r in details if r["type"] == "out_of_scope"]
    answered = [r for r in ins if r["answered"]]
    metrics = {
        "n_in_scope": len(ins),
        "n_out_of_scope": len(oos),
        "accuracy": sum(r["correct"] for r in ins) / len(ins) if ins else None,
        "false_refusal_rate": (len(ins) - len(answered)) / len(ins) if ins else None,
        "hallucination_rate": sum(not r["faithful"] for r in answered) / len(answered) if answered else None,
        "oos_refusal_rate": sum(not r["answered"] for r in oos) / len(oos) if oos else None,
        "latency_p50_ms": int(np.percentile(latencies, 50)) if latencies else None,
        "latency_p95_ms": int(np.percentile(latencies, 95)) if latencies else None,
    }
    return metrics, details


# ---------- report ----------
def build_report(n_in, n_out, retrieval, gate_rows, gate_summary, answers) -> str:
    L = [
        "# Evaluation results",
        "",
        f"{n_in} in-scope and {n_out} out-of-scope questions. "
        f"Gate threshold in use: MIN_SIM={MIN_SIM}. Answer and judge model: {MODEL}.",
        "",
        "## Retrieval (top-5 chunks)",
        "",
        "Hit@k = a chunk from the right page is in the top k. "
        "Chunk@5 = the exact chunk the question was written from is in the top 5 (stricter).",
        "",
        "| Mode | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Chunk@5 |",
        "|---|---|---|---|---|---|",
    ]
    for mode, m in retrieval.items():
        L.append(
            f"| {mode} | {pct(m['hit@1'])} | {pct(m['hit@3'])} | {pct(m['hit@5'])} "
            f"| {m['mrr@5']:.2f} | {pct(m['chunk@5'])} |"
        )

    L += ["", "## 'I don't know' gate (top vector similarity)", ""]
    if gate_summary["in_scope_median"] is not None and gate_summary["out_scope_median"] is not None:
        L.append(
            f"In-scope median similarity {gate_summary['in_scope_median']:.2f} (min {gate_summary['in_scope_min']:.2f}); "
            f"out-of-scope median {gate_summary['out_scope_median']:.2f} (max {gate_summary['out_scope_max']:.2f})."
        )
        L.append("")
    L += ["| Threshold | In-scope answered | Out-of-scope refused |", "|---|---|---|"]
    for r in gate_rows:
        L.append(f"| {r['threshold']:.2f} | {pct(r['in_scope_answered'])} | {pct(r['out_scope_refused'])} |")

    if answers:
        L += [
            "",
            "## Answer quality (LLM-judged)",
            "",
            "| Mode | Accuracy | Hallucination | False refusals | Out-of-scope refused | p50 latency | p95 latency |",
            "|---|---|---|---|---|---|---|",
        ]
        for mode, m in answers.items():
            L.append(
                f"| {mode} | {pct(m['accuracy'])} | {pct(m['hallucination_rate'])} | {pct(m['false_refusal_rate'])} "
                f"| {pct(m['oos_refusal_rate'])} | {m['latency_p50_ms']} ms | {m['latency_p95_ms']} ms |"
            )
        L += [
            "",
            "Accuracy = correct answers / all in-scope questions (a refusal counts as wrong). "
            "Hallucination = answered questions with at least one claim not supported by the retrieved sources. "
            "Judge: Gemini, spot-checked by hand.",
        ]
    return "\n".join(L) + "\n"


def validate_questions(questions) -> None:
    """Fail early with a readable message if eval/questions.json is not in the expected shape."""
    problems = []
    if not isinstance(questions, list):
        problems.append("the file must contain a JSON list of questions")
    else:
        for i, q in enumerate(questions, 1):
            kind = q.get("type")
            need = ["id", "type", "question"]
            if kind == "in_scope":
                need += ["expected_answer", "gold_url"]
            elif kind != "out_of_scope":
                problems.append(f"entry #{i} has type={kind!r}; keys present: {sorted(q)}")
                continue
            missing = [k for k in need if not q.get(k)]
            if missing:
                problems.append(f"entry #{i} ({q.get('id', 'no id')}) is missing: {missing}")
    if problems:
        shown = "\n  ".join(problems[:5])
        raise SystemExit(
            f"{QUESTIONS_FILE} is not in the expected format ({len(problems)} problems):\n  {shown}\n\n"
            "This file should be produced by review_questions.py (not copied from another file).\n"
            "Fix: delete eval/questions.json, then run:  python review_questions.py"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", default="", help="comma-separated modes to run answer eval on")
    ap.add_argument("--sleep", type=float, default=6.5, help="seconds between Gemini calls")
    args = ap.parse_args()

    if not QUESTIONS_FILE.exists():
        raise SystemExit(
            f"{QUESTIONS_FILE} not found.\n"
            "1) python make_questions.py   (creates eval/questions_draft.json)\n"
            "2) review and edit the draft by hand\n"
            "3) save it as eval/questions.json, then run this script again"
        )
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    validate_questions(questions)
    in_scope = [q for q in questions if q["type"] == "in_scope"]
    n_out = len(questions) - len(in_scope)
    print(f"{len(in_scope)} in-scope, {n_out} out-of-scope questions")

    answer_modes = [m.strip() for m in args.answers.split(",") if m.strip()]
    assistant = RagAssistant() if answer_modes else None  # needs GEMINI_API_KEY
    retriever = assistant.retriever if assistant else HybridRetriever()

    print("\nRetrieval:")
    retrieval, misses = retrieval_eval(retriever, in_scope)
    gate_rows, gate_summary = gate_sweep(retriever, questions)
    print("\nOut-of-scope questions closest to the indexed pages (the gate struggles with these):")
    for s, qu in gate_summary["top_out_of_scope"]:
        print(f"  {s:.2f}  {qu}")

    answers, details = {}, {}
    for mode in answer_modes:
        print(f"\nAnswer eval: {mode}")
        answers[mode], details[mode] = answer_eval(assistant, mode, questions, args.sleep)

    report = build_report(len(in_scope), n_out, retrieval, gate_rows, gate_summary, answers)
    RESULTS_MD.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_MD.write_text(report, encoding="utf-8")
    RESULTS_JSON.write_text(
        json.dumps(
            {"retrieval": retrieval, "retrieval_misses": misses, "gate": gate_rows,
             "answers": answers, "details": details},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print("\n" + report)

    for mode, rows in details.items():
        bad = [r for r in rows if r["type"] == "in_scope" and (not r.get("correct") or r.get("faithful") is False)]
        if bad:
            print(f"--- Review these ({mode}) ---")
            for r in bad[:10]:
                print(f"{r['id']}: {r['question']}\n    -> {r.get('reason', '')}")
    print(f"\nSaved {RESULTS_MD} and {RESULTS_JSON}")


if __name__ == "__main__":
    main()