"""Widen the stored retrieval context on the evaluation split from 5 to 8 documents.

The evaluation harness slices `retrieved_docs[:k]`, so a k = 8 condition was
silently identical to k = 5 because only five documents had been stored. This
script re-runs the same retriever at depth 8 and rewrites only the two document
lists, leaving prompts, targets and flags untouched. The first five hits are
asserted to be unchanged, so every result already computed at k <= 5 stays valid.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build_raft_data import cap_passage  # noqa: E402
from retriever import Retriever  # noqa: E402

PROC = ROOT / "data" / "processed"
DEPTH = 8


def main() -> None:
    corpus = [json.loads(l) for l in (PROC / "corpus.jsonl").open(encoding="utf-8")]
    id2text = {c["doc_id"]: c["text"] for c in corpus}
    ret = Retriever(corpus_path=PROC / "corpus.jsonl", device="cpu")

    for split in ("val", "test"):
        path = PROC / f"raft_{split}_mixed.jsonl"
        rows = [json.loads(l) for l in path.open(encoding="utf-8")]
        if rows and len(rows[0]["retrieved_docs"]) >= DEPTH:
            print(f"{split}: already at depth {len(rows[0]['retrieved_docs'])}, skipped")
            continue

        hits = ret.search_batch([r["question"] for r in rows], k=DEPTH)
        changed = 0
        for row, hs in zip(rows, hits):
            old = [d["doc_id"] for d in row["retrieved_docs"]]
            ids = [h["doc_id"] for h in hs]
            if ids[: len(old)] != old:
                # Cosine ties can reorder two equally scored documents between
                # runs. The stored order is kept so that every k <= 5 result
                # already on disk stays reproducible, and the newly needed
                # documents are appended after it.
                changed += 1
                ids = old + [d for d in ids if d not in old]
            gold = row["gold_doc_id"]
            row["retrieved_docs"] = [{"doc_id": d, "text": cap_passage(id2text[d])}
                                     for d in ids]
            row["oracle_docs"] = [{"doc_id": d, "text": cap_passage(id2text[d])}
                                  for d in [gold] + [d for d in ids if d != gold]][:DEPTH]
        shutil.copy2(path, path.with_suffix(".jsonl.k5.bak"))
        with path.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{split}: {len(rows)} rows rewritten at depth {DEPTH}, "
              f"{changed} kept their original top-5 order (tie-break) "
              f"(backup: {path.name}.k5.bak)")


if __name__ == "__main__":
    main()
