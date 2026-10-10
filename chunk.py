"""Day 1b: split scraped pages into overlapping chunks with metadata.

Run from the repo root:  python ingest/chunk.py
Input:  data/raw/pages.jsonl
Output: data/chunks.jsonl
"""
import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

IN_FILE = Path("data/raw/pages.jsonl")
OUT_FILE = Path("data/chunks.jsonl")

# ~2000 characters is roughly 400-500 tokens. You will tune this on Day 4.
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200


def main() -> None:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )

    n_pages = n_chunks = 0
    with IN_FILE.open(encoding="utf-8") as fin, OUT_FILE.open("w", encoding="utf-8") as fout:
        for line in fin:
            page = json.loads(line)
            n_pages += 1
            for i, piece in enumerate(splitter.split_text(page["text"])):
                chunk = {
                    "id": f"{n_pages}-{i}",
                    # Title is prepended so the embedding knows the topic
                    "text": f"{page['title']}\n\n{piece}",
                    "title": page["title"],
                    "url": page["url"],
                    "updated": page["updated"],
                    "chunk_index": i,
                }
                fout.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                n_chunks += 1

    print(f"{n_pages} pages -> {n_chunks} chunks ({n_chunks / max(n_pages, 1):.1f} per page)")
    print(f"Saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
