"""
Dense passage retriever over the SciQ-derived course corpus.

Uses a sentence-transformers bi-encoder with a normalised inner-product index
implemented in numpy (the corpus is small enough that an exact brute-force
search costs only a few milliseconds per query, so an ANN library is not
required). Embeddings are cached to disk so that every experiment condition
shares the same index.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
INDEX = ROOT / "data" / "index"

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
# bge models are trained with an asymmetric query instruction for retrieval
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Retriever:
    def __init__(self, model_name: str = DEFAULT_MODEL, corpus_path: Path | None = None,
                 index_dir: Path | None = None, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.corpus_path = Path(corpus_path or PROC / "corpus.jsonl")
        self.index_dir = Path(index_dir or INDEX)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.docs = [json.loads(line) for line in self.corpus_path.open(encoding="utf-8")]
        self.doc_ids = [d["doc_id"] for d in self.docs]
        self.texts = [d["text"] for d in self.docs]
        self.id2pos = {d: i for i, d in enumerate(self.doc_ids)}

        tag = model_name.replace("/", "__")
        # The cache must be invalidated whenever the corpus changes, otherwise the
        # embedding matrix and the document list fall out of alignment.
        fp = hashlib.md5(
            ("|".join(self.doc_ids) + f"|{model_name}").encode("utf-8")
        ).hexdigest()[:12]
        emb_path = self.index_dir / f"{tag}_{len(self.doc_ids)}_{fp}.npy"

        if emb_path.exists():
            self.emb = np.load(emb_path)
            assert self.emb.shape[0] == len(self.doc_ids), "stale index cache"
        else:
            model = SentenceTransformer(model_name, device=device)
            self.emb = model.encode(
                self.texts,
                batch_size=128,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True,
            ).astype(np.float32)
            np.save(emb_path, self.emb)

        self.model = SentenceTransformer(model_name, device=device)

    # ------------------------------------------------------------------ #
    def encode_queries(self, queries: list[str], batch_size: int = 128) -> np.ndarray:
        prompts = [QUERY_INSTRUCTION + q for q in queries]
        return self.model.encode(
            prompts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(queries) > 200,
        ).astype(np.float32)

    def search(self, query: str, k: int = 5, exclude: set[str] | None = None) -> list[dict]:
        """Return top-k documents as dicts with doc_id, text, score."""
        return self.search_batch([query], k=k, exclude=exclude)[0]

    def search_batch(self, queries: list[str], k: int = 5,
                     exclude: set[str] | None = None) -> list[list[dict]]:
        q = self.encode_queries(queries)
        scores = q @ self.emb.T  # (n_q, n_docs), cosine similarity
        out: list[list[dict]] = []
        for row in scores:
            order = np.argsort(-row)
            hits = []
            for j in order:
                doc_id = self.doc_ids[j]
                if exclude and doc_id in exclude:
                    continue
                hits.append({"doc_id": doc_id, "text": self.texts[j], "score": float(row[j])})
                if len(hits) >= k:
                    break
            out.append(hits)
        return out


def retrieval_metrics(ranked_ids: list[list[str]], gold_ids: list[str], ks=(1, 3, 5, 10)) -> dict:
    """Recall@k and MRR for a batch of queries."""
    res = {}
    for k in ks:
        hit = sum(1 for ids, g in zip(ranked_ids, gold_ids) if g in ids[:k])
        res[f"recall@{k}"] = round(hit / len(gold_ids), 4)
    rr = 0.0
    for ids, g in zip(ranked_ids, gold_ids):
        if g in ids:
            rr += 1.0 / (ids.index(g) + 1)
    res["mrr"] = round(rr / len(gold_ids), 4)
    return res
