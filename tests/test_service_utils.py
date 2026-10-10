"""Tests for the dependency-free helpers: percentiles, metrics, rate limiting, JSON logs."""
import json
import logging

from service_utils import JsonFormatter, Metrics, RateLimiter, percentile


def close(a, b, tol=1e-9):
    return abs(a - b) < tol


# ---------- percentile ----------
def test_percentile_matches_linear_interpolation():
    values = list(range(1, 101))
    assert close(percentile(values, 50), 50.5)
    assert close(percentile(values, 95), 95.05)
    assert close(percentile(values, 0), 1)
    assert close(percentile(values, 100), 100)


def test_percentile_edge_cases():
    assert percentile([], 50) is None
    assert percentile([7], 95) == 7
    assert close(percentile([1000, 3000], 95), 2900)  # 1000 + 0.95 * 2000


# ---------- metrics ----------
def test_metrics_counts_and_latency_only_for_llm_calls():
    m = Metrics()
    m.record(answered=True, total_ms=1000, llm_used=True)
    m.record(answered=True, total_ms=3000, llm_used=True)
    m.record(answered=False, total_ms=50, llm_used=False)  # refused by the gate: no LLM call
    m.record_error()
    s = m.snapshot()
    assert (s["requests_total"], s["answered"], s["refused"], s["errors"]) == (4, 2, 1, 1)
    assert s["latency_window"] == 2
    assert (s["latency_p50_ms"], s["latency_p95_ms"]) == (2000, 2900)


def test_metrics_empty_snapshot_has_no_latency():
    s = Metrics().snapshot()
    assert s["requests_total"] == 0
    assert s["latency_p50_ms"] is None and s["latency_p95_ms"] is None


def test_metrics_window_keeps_only_recent_latencies():
    m = Metrics(window=3)
    for ms in (10, 20, 30, 40, 50):
        m.record(True, ms, llm_used=True)
    assert m.snapshot()["latency_window"] == 3
    assert m.snapshot()["latency_p50_ms"] == 40  # median of 30, 40, 50


# ---------- rate limiter ----------
def test_per_minute_limit_and_window_expiry():
    rl = RateLimiter(per_minute=2, daily=1000)
    assert rl.check("a", now=1000.0) is None
    assert rl.check("a", now=1001.0) is None
    assert rl.check("a", now=1002.0) == "minute"
    assert rl.check("a", now=1061.5) is None  # the first hit is now more than 60 s old


def test_limits_are_per_client():
    rl = RateLimiter(per_minute=1, daily=1000)
    assert rl.check("a", now=1000.0) is None
    assert rl.check("b", now=1000.0) is None
    assert rl.check("a", now=1001.0) == "minute"


def test_daily_cap_is_global_and_resets_next_day():
    rl = RateLimiter(per_minute=100, daily=2)
    day1 = 1_700_000_000.0
    assert rl.check("a", now=day1) is None
    assert rl.check("b", now=day1 + 1) is None
    assert rl.check("c", now=day1 + 2) == "daily"
    assert rl.check("c", now=day1 + 86_400) is None  # next UTC day


def test_rejected_requests_do_not_use_up_the_daily_cap():
    rl = RateLimiter(per_minute=1, daily=2)
    assert rl.check("a", now=1_700_000_000.0) is None
    for i in range(5):  # rejected by the per-minute limit
        assert rl.check("a", now=1_700_000_001.0 + i) == "minute"
    assert rl.check("b", now=1_700_000_010.0) is None  # still one slot left today


# ---------- JSON logging ----------
def _record(msg="ask", fields=None, exc_info=None):
    rec = logging.LogRecord("govuk_rag", logging.INFO, __file__, 1, msg, None, exc_info)
    if fields is not None:
        rec.fields = fields
    return rec


def test_json_formatter_outputs_valid_json_with_fields():
    line = JsonFormatter().format(_record(fields={"request_id": "abc", "total_ms": 2100}))
    data = json.loads(line)
    assert data["event"] == "ask" and data["level"] == "INFO"
    assert data["request_id"] == "abc" and data["total_ms"] == 2100
    assert data["ts"].endswith("Z")


def test_json_formatter_includes_exception_text():
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        line = JsonFormatter().format(_record(msg="ask_failed", exc_info=sys.exc_info()))
    assert "ValueError: boom" in json.loads(line)["exception"]
