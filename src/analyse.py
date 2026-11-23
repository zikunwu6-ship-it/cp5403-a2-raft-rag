"""
Aggregate evaluation results into report tables, figures and a single summary
JSON. Also recomputes retrieval metrics on the evaluation set so that the
retrieval ceiling quoted in the report is measured rather than assumed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

COND_LABEL = {
    "base_closed": "C1 Base, closed-book",
    "base_rag": "C2 Base + RAG",
    "base_oracle": "C2b Base, oracle context",
    "raft_closed": "C3 RAFT, closed-book",
    "raft_rag": "C4 RAFT + RAG",
    "raft_oracle": "C5 RAFT, oracle context",
    "sft_gold": "A1 SFT control (gold only)",
    "raft_nocot": "A2 RAFT without CoT",
    "raft_random": "A3 RAFT, random distractors",
}


def load(cond: str) -> dict | None:
    p = RES / f"{cond}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.1f}"


def retrieval_metrics() -> dict:
    """Recall@k and MRR of the retriever on the evaluation question set."""
    rows = [json.loads(l) for l in (PROC / "raft_test_mixed.jsonl").open(encoding="utf-8")]
    sys.path.insert(0, str(ROOT / "src"))
    from retriever import Retriever

    ret = Retriever(device="cpu")
    ks = (1, 3, 5, 10)
    hits = ret.search_batch([r["question"] for r in rows], k=max(ks))
    ranked = [[h["doc_id"] for h in hs] for hs in hits]
    golds = [r["gold_doc_id"] for r in rows]

    out = {}
    for k in ks:
        out[f"recall@{k}"] = round(
            sum(1 for ids, g in zip(ranked, golds) if g in ids[:k]) / len(golds), 4)
    rr = sum(1.0 / (ids.index(g) + 1) for ids, g in zip(ranked, golds) if g in ids)
    out["mrr"] = round(rr / len(golds), 4)
    out["n"] = len(golds)
    return out


def main() -> None:
    conds = list(COND_LABEL)
    data = {c: load(c) for c in conds}
    data = {k: v for k, v in data.items() if v}

    ret = retrieval_metrics()
    print("retrieval:", ret)

    # ---- main table ----------------------------------------------------- #
    order_main = ["base_closed", "base_rag", "raft_closed", "raft_rag", "raft_oracle"]
    lines = ["| Condition | Contains | EM | F1 | Quote rate | Grounded | Fabricated | Gold cited | Verified correct | Tokens | Latency (s) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in order_main:
        if c not in data:
            continue
        o = data[c]["overall"]
        lines.append(
            f"| {COND_LABEL[c]} | {pct(o['contains'])} | {pct(o['em'])} | {pct(o['f1'])} | "
            f"{pct(o['quote_rate'])} | {pct(o['grounded_rate'])} | {pct(o['fabricated_rate'])} | "
            f"{pct(o['gold_cited_rate'])} | {pct(o['verified_correct'])} | "
            f"{o['avg_out_tokens']:.0f} | {o['avg_latency_s']:.2f} |")
    table_main = "\n".join(lines)

    # ---- ablation table -------------------------------------------------- #
    order_abl = ["raft_rag", "sft_gold", "raft_nocot", "raft_random"]
    lines = ["| Condition | Contains | Verified correct | Gold cited | Distractor cited | Fabricated |",
             "|---|---|---|---|---|---|"]
    for c in order_abl:
        if c not in data:
            continue
        o = data[c]["overall"]
        lines.append(f"| {COND_LABEL[c]} | {pct(o['contains'])} | {pct(o['verified_correct'])} | "
                     f"{pct(o['gold_cited_rate'])} | {pct(o['distractor_cited_rate'])} | "
                     f"{pct(o['fabricated_rate'])} |")
    table_abl = "\n".join(lines)

    # ---- gold-present / gold-absent split -------------------------------- #
    lines = ["| Condition | n (gold retrieved) | Contains | n (gold missing) | Contains |",
             "|---|---|---|---|---|"]
    for c in ["base_rag", "raft_rag", "raft_oracle"]:
        if c not in data:
            continue
        gp, ga = data[c]["gold_present_subset"], data[c]["gold_absent_subset"]
        lines.append(f"| {COND_LABEL[c]} | {gp.get('n', 0)} | {pct(gp.get('contains'))} | "
                     f"{ga.get('n', 0)} | {pct(ga.get('contains'))} |")
    table_subset = "\n".join(lines)

    # ---- top-k sweep ----------------------------------------------------- #
    lines = ["| k | Contains | Grounded | Gold cited | Verified correct | Fabricated | Tokens |",
             "|---|---|---|---|---|---|---|"]
    for f in sorted(RES.glob("topk_*.json"), key=lambda p: int(p.stem.split("_")[-1])):
        d = json.loads(f.read_text(encoding="utf-8"))
        o = d["overall"]
        lines.append(f"| {d['k']} | {pct(o['contains'])} | {pct(o['grounded_rate'])} | "
                     f"{pct(o['gold_cited_rate'])} | {pct(o['verified_correct'])} | "
                     f"{pct(o['fabricated_rate'])} | {o['avg_out_tokens']:.0f} |")
    table_topk = "\n".join(lines)

    # ---- profile sweep --------------------------------------------------- #
    lines = ["| Profile | Contains | Avg output tokens | Verified correct |", "|---|---|---|---|"]
    for f in sorted(RES.glob("profile_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        o = d["overall"]
        lines.append(f"| {d['profile']} | {pct(o['contains'])} | {o['avg_out_tokens']:.0f} | "
                     f"{pct(o['verified_correct'])} |")
    table_profile = "\n".join(lines)

    summary = {
        "retrieval": ret,
        "conditions": {c: {"overall": d["overall"],
                           "gold_present": d["gold_present_subset"],
                           "gold_absent": d["gold_absent_subset"]}
                       for c, d in data.items()},
    }
    (RES / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (RES / "tables.md").write_text(
        "\n\n".join(["## Main comparison", table_main,
                     "## Ablations", table_abl,
                     "## Retrieval success split", table_subset,
                     "## Top-k sweep", table_topk,
                     "## Learner profile", table_profile]),
        encoding="utf-8")

    print(table_main)
    print()
    print(table_abl)
    print()
    print(table_subset)
    print()
    print(table_topk)
    print()
    print(table_profile)


if __name__ == "__main__":
    main()
