"""Day 2b: hybrid retriever = Chroma vector search + BM25, merged with
Reciprocal Rank Fusion (RRF).

Try it from the repo root:
    python retriever.py "Can I work full-time on a Graduate visa?"
"""
import json
import re
import sys
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

CHUNKS_FILE = Path("data/chunks.jsonl")
CHROMA_DIR = "data/chroma"
COLLECTION = "govuk"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RRF_K = 60  # standard RRF constant
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # ~90MB, runs on CPU


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class HybridRetriever:
    def __init__(self) -> None:
        self.chunks = {}
        for line in CHUNKS_FILE.open(encoding="utf-8"):
            c = json.loads(line)
            self.chunks[c["id"]] = c
        self.ids = list(self.chunks)

        self.bm25 = BM25Okapi([tokenize(self.chunks[i]["text"]) for i in self.ids])
        self.model = SentenceTransformer(MODEL_NAME)
        self.col = chromadb.PersistentClient(path=CHROMA_DIR).get_collection(COLLECTION)
        self.reranker = None  # loaded lazily, only when mode="rerank"

    def _rerank(self, query: str, candidate_ids: list[str]) -> list[str]:
        if self.reranker is None:
            self.reranker = CrossEncoder(RERANK_MODEL)
        pairs = [(query, self.chunks[cid]["text"]) for cid in candidate_ids]
        scores = self.reranker.predict(pairs)
        order = sorted(range(len(candidate_ids)), key=lambda i: scores[i], reverse=True)
        return [candidate_ids[i] for i in order]

    def _vector(self, query: str, n: int):
        emb = self.model.encode([query], normalize_embeddings=True).tolist()
        res = self.col.query(query_embeddings=emb, n_results=n)
        # cosine distance -> similarity
        return list(zip(res["ids"][0], [1 - d for d in res["distances"][0]]))

    def _bm25(self, query: str, n: int):
        scores = self.bm25.get_scores(tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return [(self.ids[i], float(scores[i])) for i in top]

    def top_similarity(self, query: str) -> float:
        """Cosine similarity of the single closest chunk. Used by the 'I don't know' gate."""
        return self._vector(query, 1)[0][1]

    def search(self, query: str, k: int = 5, mode: str = "hybrid", pool: int = 20):
        """mode: 'vector' | 'bm25' | 'hybrid'. Returns a list of dicts."""
        vec = self._vector(query, pool)
        kw = self._bm25(query, pool)
        sim = dict(vec)  # cosine similarity per chunk id, used for the "I don't know" gate

        if mode == "vector":
            ranked = [i for i, _ in vec]
        elif mode == "bm25":
            ranked = [i for i, _ in kw]
        else:
            fused: dict[str, float] = {}
            for results in (vec, kw):
                for rank, (cid, _) in enumerate(results):
                    fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
            ranked = sorted(fused, key=fused.get, reverse=True)
            if mode == "rerank":
                # hybrid candidate pool, re-scored by a cross-encoder
                ranked = self._rerank(query, ranked[:pool])

        out = []
        for cid in ranked[:k]:
            c = self.chunks[cid]
            out.append(
                {
                    "id": cid,
                    "title": c["title"],
                    "url": c["url"],
                    "updated": c["updated"],
                    "text": c["text"],
                    # None if this chunk was not in the vector top-N
                    "similarity": sim.get(cid),
                }
            )
        return out


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Can I work full-time on a Graduate visa?"
    r = HybridRetriever()
    for mode in ("vector", "bm25", "hybrid", "rerank"):
        print(f"\n=== {mode.upper()} ===")
        for hit in r.search(q, k=3, mode=mode):
            sim = f"{hit['similarity']:.2f}" if hit["similarity"] is not None else "n/a"
            print(f"- {hit['title']}  (sim {sim})\n  {hit['url']}")