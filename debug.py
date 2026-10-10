"""Debug retrieval: what chunks come back, and do they contain a keyword?

Usage (repo root):
    python debug.py "can i go outside uk on my graduate visa for a trip?" travel
"""
import json
import sys
from pathlib import Path

from retriever import HybridRetriever

question = sys.argv[1]
keyword = sys.argv[2].lower() if len(sys.argv) > 2 else ""

# 1. Is the keyword anywhere in the scraped Graduate visa page?
print("=== Raw page check ===")
for line in Path("data/raw/pages.jsonl").open(encoding="utf-8"):
    page = json.loads(line)
    if page["url"].endswith("/graduate-visa"):
        text = page["text"].lower()
        print(f"{page['url']}  length={len(text)} chars  '{keyword}' present: {keyword in text}")
        if keyword in text:
            i = text.index(keyword)
            print("  context:", page["text"][max(0, i - 150): i + 250].replace("\n", " "))

# 2. What does each retrieval mode return, and does it contain the keyword?
r = HybridRetriever()
for mode in ("vector", "bm25", "hybrid", "rerank"):
    print(f"\n=== {mode.upper()} top 5 ===")
    for h in r.search(question, k=5, mode=mode):
        has = keyword in h["text"].lower() if keyword else False
        print(f"- [{'HAS' if has else '   '} '{keyword}'] {h['title']} | chunk {h['id']} | sim {h['similarity']}")