"""Small, dependency-free helpers for the API: JSON logging, metrics, rate limiting."""
import json
import logging
import os
import threading
import time
from collections import defaultdict, deque


# ---------- structured logging ----------
class JsonFormatter(logging.Formatter):
    """One JSON object per log line, easy to grep and to ship to a log service."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def get_logger(name: str = "govuk_rag") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
        logger.propagate = False
    return logger


# ---------- metrics ----------
def percentile(values: list[float], p: float) -> float | None:
    """Linear-interpolation percentile (same method as numpy's default)."""
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _round(x: float | None) -> int | None:
    return None if x is None else round(x)


class Metrics:
    """In-memory counters plus a rolling latency window. Thread-safe."""

    def __init__(self, window: int = 500) -> None:
        self._lock = threading.Lock()
        self._started = time.time()
        self.total = self.answered = self.refused = self.errors = 0
        self._latencies: deque[int] = deque(maxlen=window)  # only requests that called the LLM

    def record(self, answered: bool, total_ms: int, llm_used: bool) -> None:
        with self._lock:
            self.total += 1
            if answered:
                self.answered += 1
            else:
                self.refused += 1
            if llm_used:
                self._latencies.append(total_ms)

    def record_error(self) -> None:
        with self._lock:
            self.total += 1
            self.errors += 1

    def snapshot(self) -> dict:
        with self._lock:
            lat = list(self._latencies)
            return {
                "uptime_s": int(time.time() - self._started),
                "requests_total": self.total,
                "answered": self.answered,
                "refused": self.refused,
                "errors": self.errors,
                "latency_window": len(lat),
                "latency_p50_ms": _round(percentile(lat, 50)),
                "latency_p95_ms": _round(percentile(lat, 95)),
            }


# ---------- rate limiting ----------
class RateLimiter:
    """Per-client requests per minute, plus one global daily cap (protects your LLM quota)."""

    def __init__(self, per_minute: int, daily: int) -> None:
        self.per_minute = per_minute
        self.daily = daily
        self._lock = threading.Lock()
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._day = ""
        self._day_count = 0

    def check(self, key: str, now: float | None = None) -> str | None:
        """Return None if the request is allowed, otherwise 'minute' or 'daily'."""
        now = time.time() if now is None else now
        with self._lock:
            today = time.strftime("%Y-%m-%d", time.gmtime(now))
            if today != self._day:
                self._day, self._day_count = today, 0
            if self._day_count >= self.daily:
                return "daily"
            hits = self._hits[key]
            while hits and now - hits[0] >= 60:
                hits.popleft()
            if len(hits) >= self.per_minute:
                return "minute"
            hits.append(now)
            self._day_count += 1
            if len(self._hits) > 5000:  # drop idle clients so memory stays bounded
                self._hits = defaultdict(deque, {k: v for k, v in self._hits.items() if v})
            return None
