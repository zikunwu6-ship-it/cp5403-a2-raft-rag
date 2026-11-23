"""
Build the retrieval corpus from SciQ support passages.

SciQ (Welbl et al., 2017) provides, for each question, a `support` passage that
was extracted from science textbooks / educational web material. We treat the
union of unique support passages across all splits as our document corpus. This
gives a realistic "course reading pack" of ~13.7k short expository passages.

Outputs:
  data/processed/corpus.jsonl   {doc_id, text, n_tokens, split_origin}
"""
import json
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# SciQ contains a small number of questions whose `support` field is empty or a
# fragment. Such questions cannot be answered from the corpus by construction,
# so both the corpus and the evaluation sets use this same threshold.
MIN_PASSAGE_WORDS = 5


def normalise(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main() -> None:
    ds = load_dataset("allenai/sciq")

    seen: dict[str, dict] = {}
    per_split = Counter()

    for split in ("train", "validation", "test"):
        for row in ds[split]:
            passage = normalise(row["support"])
            if len(passage.split()) < MIN_PASSAGE_WORDS:  # drop empty / degenerate passages
                continue
            key = passage.lower()
            if key in seen:
                seen[key]["splits"].add(split)
                continue
            seen[key] = {"text": passage, "splits": {split}}
            per_split[split] += 1

    corpus = []
    for i, (_, rec) in enumerate(sorted(seen.items(), key=lambda kv: kv[1]["text"])):
        corpus.append(
            {
                "doc_id": f"p{i:05d}",
                "text": rec["text"],
                "n_tokens": len(rec["text"].split()),
                "split_origin": sorted(rec["splits"]),
            }
        )

    with open(OUT / "corpus.jsonl", "w", encoding="utf-8") as f:
        for rec in corpus:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    lengths = [c["n_tokens"] for c in corpus]
    stats = {
        "n_passages": len(corpus),
        "unique_passages_per_split": dict(per_split),
        "tokens_mean": round(sum(lengths) / len(lengths), 1),
        "tokens_min": min(lengths),
        "tokens_max": max(lengths),
        "tokens_total": sum(lengths),
    }
    (OUT / "corpus_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
