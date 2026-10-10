"""Small shared helpers: Gemini calls with retries and JSON output."""
import json
import os
import time

from google import genai
from google.genai import types

MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
_client = None


def client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


def retry(fn, tries: int = 4, delay: float = 20.0):
    """Call fn(). On failure (e.g. a 429 rate limit) wait, then try again."""
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            # A per-day quota will not clear in a minute, so do not wait and retry
            text = str(e)
            if "PerDay" in text or "NOT_FOUND" in text or "API_KEY" in text or attempt == tries - 1:
                raise
            print(f"  retrying in {delay:.0f}s ({type(e).__name__}: {str(e)[:100]})")
            time.sleep(delay)
            delay *= 2


def call_json(prompt: str, system: str | None = None) -> dict:
    def _call():
        resp = client().models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )
        return json.loads(resp.text)

    return retry(_call)