"""Day 2a: embed all chunks and store them in a persistent Chroma collection.

Run from the repo root:  python index.py
Input:  data/chunks.jsonl
Output: data/chroma/  (vector index on disk)
"""
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = Path("data/chunks.jsonl")
CHROMA_DIR = "data/chroma"
COLLECTION = "govuk"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH = 64


def main() -> None:
    chunks = [json.loads(line) for line in CHUNKS_FILE.open(encoding="utf-8")]
    print(f"Loaded {len(chunks)} chunks")

    model = SentenceTransformer(MODEL_NAME)  # first run downloads ~90MB
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Start fresh so re-running never creates duplicates
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    for i in range(0, len(chunks), BATCH):
        batch = chunks[i : i + BATCH]
        embeddings = model.encode(
            [c["text"] for c in batch], normalize_embeddings=True
        ).tolist()
        col.add(
            ids=[c["id"] for c in batch],
            embeddings=embeddings,
            documents=[c["text"] for c in batch],
            metadatas=[
                {
                    "title": c["title"],
                    "url": c["url"],
                    "updated": c["updated"] or "",
                    "chunk_index": c["chunk_index"],
                }
                for c in batch
            ],
        )
        print(f"Indexed {min(i + BATCH, len(chunks))}/{len(chunks)}")

    print(f"Done. Collection '{COLLECTION}' has {col.count()} vectors in {CHROMA_DIR}")


if __name__ == "__main__":
    main()
