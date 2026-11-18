"""
Generate the figures used in the report and the presentation.

Reads results/*.json produced by src/evaluate.py. Uses a restrained palette and
no chart junk: the figures have to survive being projected and being printed in a
Word document at 12 cm wide.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# palette: one neutral for baselines, one accent for the treatment
BASE = "#8A8F98"
ACCENT = "#1F6FEB"
ACCENT2 = "#0E9F6E"
WARN = "#C2410C"
GRID = "#D8DEE6"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": "#5A6270",
    "axes.labelcolor": "#1F2328",
    "text.color": "#1F2328",
    "xtick.color": "#1F2328",
    "ytick.color": "#1F2328",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
})


def load(cond):
    p = RES / f"{cond}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _pct(v):
    return 100 * v


def fig_main():
    conds = ["base_closed", "base_rag", "raft_closed", "raft_rag", "raft_oracle"]
    labels = ["C1\nBase\nclosed", "C2\nBase\n+ RAG", "C3\nRAFT\nclosed",
              "C4\nRAFT\n+ RAG", "C5\nRAFT\noracle"]
    data = [(c, load(c)) for c in conds]
    data = [(c, d) for c, d in data if d]
    if not data:
        return
    acc = [_pct(d["overall"]["contains"]) for _, d in data]
    ver = [_pct(d["overall"]["verified_correct"]) for _, d in data]
    x = np.arange(len(data))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    b1 = ax.bar(x - w / 2, acc, w, label="Answer accuracy (contains)", color=BASE)
    b2 = ax.bar(x + w / 2, ver, w, label="Verified correct", color=ACCENT)
    for bars in (b1, b2):
        for r in bars:
            ax.annotate(f"{r.get_height():.0f}", (r.get_x() + r.get_width() / 2, r.get_height()),
                        ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[conds.index(c)] for c, _ in data], fontsize=7.5)
    ax.set_ylabel("% of 500 questions")
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "fig_main.png", bbox_inches="tight")
    plt.close(fig)


def fig_citation():
    c2, c4 = load("base_rag"), load("raft_rag")
    if not (c2 and c4):
        return
    keys = ["quote_rate", "grounded_rate", "fabricated_rate", "verified_correct"]
    labels = ["Quotes\na span", "Quote is\nverbatim real", "Quote is\nfabricated", "Verified\ncorrect"]
    a = [_pct(c2["overall"][k]) for k in keys]
    b = [_pct(c4["overall"][k]) for k in keys]
    x = np.arange(len(keys))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    r1 = ax.bar(x - w / 2, a, w, label="C2 Base + RAG", color=BASE)
    r2 = ax.bar(x + w / 2, b, w, label="C4 RAFT + RAG", color=ACCENT)
    for bars in (r1, r2):
        for r in bars:
            ax.annotate(f"{r.get_height():.0f}", (r.get_x() + r.get_width() / 2, r.get_height()),
                        ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("% of responses")
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig_citation.png", bbox_inches="tight")
    plt.close(fig)


def fig_ablation():
    conds = ["raft_rag", "sft_gold", "raft_nocot", "raft_random"]
    labels = ["C4\nfull RAFT", "A1\ngold-only SFT", "A2\nno CoT", "A3\nrandom distractors"]
    data = [(c, load(c)) for c in conds]
    data = [(c, d) for c, d in data if d]
    if len(data) < 2:
        return
    acc = [_pct(d["overall"]["contains"]) for _, d in data]
    ver = [_pct(d["overall"]["verified_correct"]) for _, d in data]
    x = np.arange(len(data))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    ax.bar(x - w / 2, acc, w, label="Answer accuracy", color=BASE)
    ax.bar(x + w / 2, ver, w, label="Verified correct", color=ACCENT)
    for i, (a, v) in enumerate(zip(acc, ver)):
        ax.annotate(f"{a:.0f}", (i - w / 2, a), ha="center", va="bottom", fontsize=7)
        ax.annotate(f"{v:.0f}", (i + w / 2, v), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[conds.index(c)] for c, _ in data], fontsize=7.5)
    ax.set_ylabel("%")
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig_ablation.png", bbox_inches="tight")
    plt.close(fig)


def fig_retrieval_split():
    pairs = [("base_rag", "C2 Base + RAG"), ("raft_rag", "C4 RAFT + RAG")]
    rows = [(lab, load(c)) for c, lab in pairs]
    rows = [(lab, d) for lab, d in rows if d]
    if len(rows) < 2:
        return
    x = np.arange(len(rows))
    w = 0.38
    present = [_pct(d["gold_present_subset"].get("contains", 0)) for _, d in rows]
    missing = [_pct(d["gold_absent_subset"].get("contains", 0)) for _, d in rows]
    fig, ax = plt.subplots(figsize=(5.2, 2.7))
    ax.bar(x - w / 2, present, w, label="gold passage retrieved", color=ACCENT2)
    ax.bar(x + w / 2, missing, w, label="gold passage not retrieved", color=WARN)
    for i, (p, m) in enumerate(zip(present, missing)):
        ax.annotate(f"{p:.0f}", (i - w / 2, p), ha="center", va="bottom", fontsize=7)
        ax.annotate(f"{m:.0f}", (i + w / 2, m), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for lab, _ in rows], fontsize=8)
    ax.set_ylabel("Answer accuracy (%)")
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig_retrieval_split.png", bbox_inches="tight")
    plt.close(fig)


def fig_topk():
    files = sorted(RES.glob("topk_*.json"), key=lambda p: int(p.stem.split("_")[-1]))
    if not files:
        return
    ks, acc = [], []
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        ks.append(d["k"])
        acc.append(_pct(d["overall"]["contains"]))
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    ax.plot(ks, acc, marker="o", color=ACCENT, linewidth=1.6, markersize=5)
    for k, a in zip(ks, acc):
        ax.annotate(f"{a:.0f}", (k, a), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=7)
    ax.set_xlabel("passages retrieved (k)")
    ax.set_ylabel("Answer accuracy (%)")
    ax.set_xticks(ks)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "fig_topk.png", bbox_inches="tight")
    plt.close(fig)


def fig_profile():
    files = sorted(RES.glob("profile_*.json"))
    if not files:
        return
    order = {"novice": 0, "intermediate": 1, "advanced": 2}
    rows = []
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        rows.append((d["profile"], d))
    rows.sort(key=lambda r: order.get(r[0], 9))
    labels = [p for p, _ in rows]
    tok = [d["overall"]["avg_out_tokens"] for _, d in rows]
    acc = [_pct(d["overall"]["contains"]) for _, d in rows]
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    bars = ax.bar(labels, tok, color=ACCENT, width=0.55)
    for r, a in zip(bars, acc):
        ax.annotate(f"{r.get_height():.0f} tok\n{a:.0f}% acc",
                    (r.get_x() + r.get_width() / 2, r.get_height()),
                    ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("mean output tokens")
    ax.set_ylim(0, max(tok) * 1.35)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "fig_profile.png", bbox_inches="tight")
    plt.close(fig)


def fig_loss():
    p = ROOT / "outputs" / "trainer" / "raft_mixed" / "log_history.json"
    if not p.exists():
        return
    hist = json.loads(p.read_text(encoding="utf-8"))
    tr = [(h["step"], h["loss"]) for h in hist if "loss" in h]
    ev = [(h["step"], h["eval_loss"]) for h in hist if "eval_loss" in h]
    if not tr:
        return
    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    ax.plot([s for s, _ in tr], [l for _, l in tr], color=ACCENT, linewidth=1.0,
            alpha=0.55, label="training loss")
    if len(tr) > 8:
        win = max(3, len(tr) // 12)
        sm = np.convolve([l for _, l in tr], np.ones(win) / win, mode="valid")
        ax.plot([s for s, _ in tr][win - 1:], sm, color=ACCENT, linewidth=2.0,
                label=f"moving average (w={win})")
    if ev:
        ax.plot([s for s, _ in ev], [l for _, l in ev], color=WARN, marker="o",
                linewidth=1.4, markersize=4, label="validation loss")
    ax.set_xlabel("optimiser step")
    ax.set_ylabel("loss")
    ax.legend(frameon=False, fontsize=7)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "fig_loss.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    for fn in (fig_main, fig_citation, fig_ablation, fig_retrieval_split,
               fig_topk, fig_profile, fig_loss):
        try:
            fn()
            print("ok", fn.__name__)
        except Exception as e:  # noqa: BLE001
            print("skip", fn.__name__, e)
    print("figures ->", FIG)
