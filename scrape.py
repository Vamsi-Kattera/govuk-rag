"""Day 1a: collect GOV.UK pages using the public Search + Content APIs.

Run from the repo root:  python ingest/scrape.py
Output: data/raw/pages.jsonl  (one JSON object per page)
"""
import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE = "https://www.gov.uk"
OUT_DIR = Path("data/raw")
OUT_FILE = OUT_DIR / "pages.jsonl"
HEADERS = {"User-Agent": "govuk-rag-portfolio-project (learning project)"}

# Topics the assistant will cover. Edit freely.
QUERIES = [
    "graduate visa",
    "skilled worker visa",
    "student visa work rules",
    "right to work check",
    "employer right to work",
    "national minimum wage",
    "statutory sick pay",
    "holiday entitlement",
    "redundancy pay",
    "maternity pay and leave",
    "employment contracts",
    "national insurance number",
    "pay tax on income",
    "tenancy rights renting",
    "settle in the uk",
]
PER_QUERY = 15
MAX_PAGES = 200


def search_links(query: str, count: int) -> list[dict]:
    r = requests.get(
        f"{BASE}/api/search.json",
        params={"q": query, "count": count, "fields": ["title", "link", "format"]},
        headers=HEADERS,
        timeout=30,
    )
    r.raise_for_status()
    return r.json().get("results", [])


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text("\n", strip=True)
    return re.sub(r"\n{3,}", "\n\n", text)


def fetch_page(path: str) -> dict | None:
    r = requests.get(f"{BASE}/api/content{path}", headers=HEADERS, timeout=30)
    if r.status_code != 200:
        return None
    data = r.json()
    details = data.get("details", {}) or {}

    # Multi-part guides store content in details.parts; others in details.body
    if details.get("parts"):
        text = "\n\n".join(
            f"{p.get('title', '')}\n{html_to_text(p.get('body', ''))}"
            for p in details["parts"]
        )
    elif isinstance(details.get("body"), str):
        text = html_to_text(details["body"])
    else:
        return None

    if len(text) < 300:  # skip near-empty pages
        return None

    return {
        "url": f"{BASE}{path}",
        "title": data.get("title", ""),
        "updated": data.get("public_updated_at") or data.get("updated_at"),
        "format": data.get("document_type", ""),
        "text": text,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Gather unique page paths from all queries
    paths: dict[str, str] = {}
    for q in QUERIES:
        for item in search_links(q, PER_QUERY):
            link = item.get("link", "")
            if link.startswith("/") and link not in paths:
                paths[link] = item.get("title", "")
        time.sleep(0.3)
    print(f"Found {len(paths)} candidate pages")

    # 2. Fetch each page, keep the usable ones
    saved = 0
    with OUT_FILE.open("w", encoding="utf-8") as f:
        for path in list(paths)[:MAX_PAGES]:
            page = fetch_page(path)
            if page:
                f.write(json.dumps(page, ensure_ascii=False) + "\n")
                saved += 1
                print(f"[{saved}] {page['title']}")
            time.sleep(0.3)  # be polite to the API
    print(f"Saved {saved} pages to {OUT_FILE}")


if __name__ == "__main__":
    main()
