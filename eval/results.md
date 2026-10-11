# Evaluation results

42 in-scope and 10 out-of-scope questions. Gate threshold in use: MIN_SIM=0.45. Answer and judge model: gemini-flash-lite-latest.

## Retrieval (top-5 chunks)

Hit@k = a chunk from the right page is in the top k. Chunk@5 = the exact chunk the question was written from is in the top 5 (stricter).

| Mode | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Chunk@5 |
|---|---|---|---|---|---|
| vector | 74% | 93% | 95% | 0.83 | 90% |
| bm25 | 67% | 98% | 98% | 0.80 | 88% |
| hybrid | 86% | 95% | 98% | 0.91 | 93% |
| rerank | 86% | 95% | 98% | 0.90 | 98% |

## 'I don't know' gate (top vector similarity)

In-scope median similarity 0.69 (min 0.49); out-of-scope median 0.21 (max 0.57).

| Threshold | In-scope answered | Out-of-scope refused |
|---|---|---|
| 0.20 | 100% | 50% |
| 0.25 | 100% | 60% |
| 0.30 | 100% | 70% |
| 0.35 | 100% | 70% |
| 0.40 | 100% | 70% |
| 0.45 | 100% | 80% |
| 0.50 | 98% | 90% |
| 0.55 | 93% | 90% |

## Answer quality (LLM-judged)

| Mode | Accuracy | Hallucination | False refusals | Out-of-scope refused | p50 latency | p95 latency |
|---|---|---|---|---|---|---|
| rerank | 95% | 2% | 2% | 100% | 2619 ms | 2978 ms |

Accuracy = correct answers / all in-scope questions (a refusal counts as wrong). Hallucination = answered questions with at least one claim not supported by the retrieved sources. Judge: Gemini, spot-checked by hand.
