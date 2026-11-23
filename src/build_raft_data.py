"""
Build RAFT-style training / validation / test data.

Follows Zhang et al. (2024, "RAFT: Adapting Language Model to Domain Specific
RAG") with two adaptations required by the educational setting:

  1. Distractors are *retrieved* hard negatives (top-scoring non-gold passages)
     mixed with random passages, instead of purely random passages. This makes
     the distractor-robustness problem realistic: in a real course corpus the
     wrong passages a retriever returns are topically adjacent, not random.
  2. A learner-profile block is included in every prompt and the chain-of-thought
     target is rendered at a depth appropriate to that profile. The profile is
     held constant across the main comparison and varied only in the dedicated
     personalisation experiment.

Answer format (identical at train and inference time):
    ##Reason: <reasoning, with ##begin_quote## ... ##end_quote## spans>
    ##Answer: <answer>

A quote marker is emitted ONLY when the quoted span exists verbatim in the
document. This keeps the citation-grounding metric meaningful: any quote the
model produces can be checked by substring search.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

MIN_PASSAGE_WORDS = 5
# Passages are capped at a sentence boundary before they are put in the prompt.
# Without this, ~11% of prompts exceeded the 1024-token budget and were truncated
# from the right - which cut off the assistant header and, during training, the
# supervision target itself.
MAX_PASSAGE_CHARS = 800
N_DOCS = 5                 # 1 gold + 4 distractors, as in the RAFT paper
P_KEEP_GOLD = 0.8          # 80% of samples keep the gold document
N_HARD_DISTRACTORS = 2     # of the 4 distractors, how many are retrieved

PROFILES = {
    "novice": (
        "Prior knowledge: beginner - has not studied this topic before.\n"
        "Goal: understand the underlying concept well enough to explain it.\n"
        "Preferred depth: step by step, define any technical term you use, "
        "and say why the alternatives do not fit."
    ),
    "intermediate": (
        "Prior knowledge: some background from earlier study.\n"
        "Goal: confirm understanding and correct misconceptions.\n"
        "Preferred depth: a short chain of reasoning with the key evidence."
    ),
    "advanced": (
        "Prior knowledge: strong background, revising for an exam.\n"
        "Goal: quickly verify the discriminating detail.\n"
        "Preferred depth: terse - state the evidence and the answer, no definitions."
    ),
}

# --------------------------------------------------------------------------- #
# Prompt templates (shared by training, evaluation and the demo application)
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = (
    "You are AcademaTutor, a study-support assistant for adult learners studying "
    "online. You answer questions from course passages, you cite the exact text "
    "you relied on, and you adapt your explanation to the learner's profile."
)

FORMAT_SPEC = (
    "Reply strictly in this format, with no other text:\n"
    "##Reason: <your reasoning; wrap any quoted passage text in "
    "##begin_quote## ... ##end_quote##>\n"
    "##Answer: <the answer, as a short phrase>"
)


def cap_passage(text: str, limit: int = MAX_PASSAGE_CHARS) -> str:
    """Trim a passage to `limit` characters, preferring a sentence boundary."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    for sep in (". ", "? ", "! "):
        i = cut.rfind(sep)
        if i > limit * 0.5:
            return cut[: i + 1].strip()
    return cut.rsplit(" ", 1)[0].strip() + " ..."


def render_parts(question: str, docs: list[dict], profile_key: str) -> dict:
    """Return the prompt as chat messages.

    An instruction-tuned base model must be prompted through its chat template;
    feeding it a flat string makes it continue the passages instead of answering.
    `docs == []` selects the closed-book condition.
    """
    parts = ["<learner_profile>", PROFILES[profile_key], "</learner_profile>", ""]
    if docs:
        parts.append("<course_passages>")
        for i, d in enumerate(docs, 1):
            parts.append(f"[{i}] {d['text']}")
        parts.append("</course_passages>")
        parts.append("")
        parts.append(
            "Some of the passages above are irrelevant to the question. Use only the "
            "passage(s) that actually contain the evidence, and ignore the rest."
        )
    else:
        parts.append("<course_passages>")
        parts.append("(none available)")
        parts.append("</course_passages>")
        parts.append("")
        parts.append(
            "No course passages are available. Answer from your own knowledge and say "
            "that you are doing so."
        )
    parts += ["", f"<question>\n{question}\n</question>", "", FORMAT_SPEC]
    return {"system": SYSTEM_PROMPT, "user": "\n".join(parts)}


def render_prompt(question: str, docs: list[dict], profile_key: str) -> str:
    """Flat rendering, for inspection and for the report appendix."""
    p = render_parts(question, docs, profile_key)
    return f"[system]\n{p['system']}\n\n[user]\n{p['user']}"


# --------------------------------------------------------------------------- #
# Chain-of-thought target construction
# --------------------------------------------------------------------------- #

_SPLIT_SENTS = re.compile(r"(?<=[.!?])\s+")


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SPLIT_SENTS.split(text) if s.strip()]


def _content_tokens(s: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 2]


def locate_evidence(passage: str, answer: str) -> tuple[str | None, bool]:
    """Return (evidence_sentence, is_verbatim).

    is_verbatim=True means `answer` appears literally inside the returned
    sentence, so the sentence can be quoted safely.
    """
    sents = sentences(passage)
    if not sents:
        return None, False
    a = answer.lower().strip()
    for s in sents:
        if a and a in s.lower():
            return s, True
    # fall back to the sentence with the largest token overlap with the answer
    at = set(_content_tokens(answer))
    if not at:
        return sents[0], False
    best, best_ov = sents[0], -1.0
    for s in sents:
        st = set(_content_tokens(s))
        ov = len(at & st) / len(at)
        if ov > best_ov:
            best, best_ov = s, ov
    return best, False


REJECT_PHRASES = [
    "The other passages discuss unrelated material and give no evidence for this question.",
    "Passages {others} do not address this question, so I set them aside.",
    "The remaining passages are about other topics and cannot support an answer here.",
    "Nothing in passages {others} bears on the question, so they are ignored.",
]


def build_cot(question, answer, docs, gold_idx, profile_key, rng) -> str:
    """Render the RAFT chain-of-thought target."""
    n = len(docs)
    others = ", ".join(str(i) for i in range(1, n + 1) if i != gold_idx) or "none"
    rej = rng.choice(REJECT_PHRASES).format(others=others)

    if gold_idx is None:
        # No gold document present: answer from parametric knowledge.
        openers = [
            "None of the passages contains information about this question; they cover "
            "unrelated topics.",
            "The passages provided do not contain the evidence needed here.",
            "I cannot find support for this question in the passages above.",
        ]
        return (
            f"##Reason: {rng.choice(openers)} Answering from general knowledge, "
            f"{answer} is the correct response. {rej}\n##Answer: {answer}"
        )

    ev, verbatim = locate_evidence(docs[gold_idx - 1]["text"], answer)

    if verbatim:
        lead = rng.choice([
            f"Passage [{gold_idx}] addresses the question directly",
            f"The evidence is in passage [{gold_idx}]",
            f"Passage [{gold_idx}] is the relevant source",
        ])
        tail = rng.choice([
            f"so the answer is {answer}.",
            f"which identifies the answer as {answer}.",
            f"giving {answer} as the answer.",
        ])
        return (
            f"##Reason: {lead}: ##begin_quote## {ev} ##end_quote## "
            f"This states the relationship the question asks about, {tail} {rej}\n"
            f"##Answer: {answer}"
        )

    # Gold present but the answer is not literally quotable - do NOT fabricate a
    # quote marker, otherwise the grounding metric would be measuring nothing.
    if profile_key == "advanced":
        return (
            f"##Reason: Passage [{gold_idx}] is the only relevant source; it describes the "
            f"mechanism the question asks about, which yields {answer}. {rej}\n"
            f"##Answer: {answer}"
        )
    return (
        f"##Reason: Passage [{gold_idx}] is the only one that discusses this topic. It "
        f"explains the underlying process rather than naming the answer outright, and "
        f"following that explanation leads to {answer}. {rej}\n"
        f"##Answer: {answer}"
    )


# --------------------------------------------------------------------------- #

def load_corpus() -> tuple[list[dict], dict[str, str]]:
    docs = [json.loads(l) for l in (PROC / "corpus.jsonl").open(encoding="utf-8")]
    text2id = {d["text"].lower(): d["doc_id"] for d in docs}
    return docs, text2id


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--distractor-mode", default="mixed",
                    choices=["mixed", "random", "hard", "gold_only"])
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from datasets import load_dataset

    rng = random.Random(args.seed)
    docs, text2id = load_corpus()
    id2text = {d["doc_id"]: d["text"] for d in docs}
    all_ids = [d["doc_id"] for d in docs]

    ds = load_dataset("allenai/sciq")

    def clean(row):
        return re.sub(r"\s+", " ", row["support"]).strip()

    def collect(split):
        out = []
        for row in ds[split]:
            sup = clean(row)
            if len(sup.split()) < MIN_PASSAGE_WORDS:
                continue  # unanswerable from the corpus by construction
            gid = text2id.get(sup.lower())
            if gid is None:
                continue
            out.append({
                "question": row["question"].strip(),
                "answer": row["correct_answer"].strip(),
                "gold_doc_id": gid,
                "options": [row["correct_answer"], row["distractor1"],
                            row["distractor2"], row["distractor3"]],
                "split": split,
            })
        return out

    train_pool = collect("train")
    val_pool = collect("validation")
    test_pool = collect("test")
    rng.shuffle(train_pool)

    # ---- retrieve hard negatives for the training pool -------------------- #
    from retriever import Retriever

    ret = Retriever()
    need_retrieval = args.distractor_mode in ("mixed", "hard")

    hard_neg: dict[str, list[str]] = {}
    if need_retrieval:
        pool = train_pool[: args.n_train + 400]
        hits = ret.search_batch([q["question"] for q in pool], k=25)
        for q, hs in zip(pool, hits):
            hard_neg[q["question"]] = [h["doc_id"] for h in hs if h["doc_id"] != q["gold_doc_id"]][:20]
        print(f"retrieved hard negatives for {len(hard_neg)} training questions")

    def make_distractors(q, n, used):
        chosen = []
        if args.distractor_mode in ("mixed", "hard") and hard_neg.get(q["question"]):
            cand = [d for d in hard_neg[q["question"]] if d not in used]
            n_hard = n if args.distractor_mode == "hard" else min(N_HARD_DISTRACTORS, n)
            chosen += rng.sample(cand, min(n_hard, len(cand)))
        while len(chosen) < n:
            d = rng.choice(all_ids)
            if d in used or d in chosen:
                continue
            chosen.append(d)
        return chosen

    def build(q, keep_gold):
        used = {q["gold_doc_id"]}
        n_dist = N_DOCS - 1 if keep_gold else N_DOCS
        dist_ids = make_distractors(q, n_dist, used)
        entries = [{"doc_id": d, "text": cap_passage(id2text[d])} for d in dist_ids]
        if keep_gold:
            entries.insert(rng.randint(0, len(entries)),
                           {"doc_id": q["gold_doc_id"],
                            "text": cap_passage(id2text[q["gold_doc_id"]])})
            gold_idx = next(i for i, e in enumerate(entries, 1) if e["doc_id"] == q["gold_doc_id"])
        else:
            gold_idx = None
        profile = rng.choice(list(PROFILES))
        parts = render_parts(q["question"], entries, profile)
        return {
            "question": q["question"],
            "answer": q["answer"],
            "options": q["options"],
            "gold_doc_id": q["gold_doc_id"],
            "profile": profile,
            "gold_present": keep_gold,
            "docs": entries,
            "system": parts["system"],
            "user": parts["user"],
            "prompt": render_prompt(q["question"], entries, profile),
            "target": build_cot(q["question"], q["answer"], entries, gold_idx, profile, rng),
        }

    # ---- training set ----------------------------------------------------- #
    if args.distractor_mode == "gold_only":
        train_rows = []
        for q in train_pool[: args.n_train]:
            profile = rng.choice(list(PROFILES))
            entries = [{"doc_id": q["gold_doc_id"],
                        "text": cap_passage(id2text[q["gold_doc_id"]])}]
            parts = render_parts(q["question"], entries, profile)
            train_rows.append({
                "question": q["question"], "answer": q["answer"], "options": q["options"],
                "gold_doc_id": q["gold_doc_id"], "profile": profile, "gold_present": True,
                "docs": entries,
                "system": parts["system"], "user": parts["user"],
                "prompt": render_prompt(q["question"], entries, profile),
                "target": build_cot(q["question"], q["answer"], entries, 1, profile, rng),
            })
    else:
        train_rows = []
        for q in train_pool[: args.n_train]:
            train_rows.append(build(q, rng.random() < P_KEEP_GOLD))

    # ---- validation / test -------------------------------------------------- #
    # Two document sets are stored per evaluation item:
    #   retrieved_docs -> the real top-k the retriever returns (gold may be absent)
    #   oracle_docs    -> gold forced into slot 1, followed by the retrieved non-gold
    # This lets the evaluation harness separate "end-to-end with real retrieval"
    # from "oracle context", which matters because retrieval failure caps accuracy.
    def eval_rows(pool, n):
        rows = []
        sample = pool[:n] if len(pool) <= n else rng.sample(pool, n)
        hits = ret.search_batch([q["question"] for q in sample], k=N_DOCS)
        for q, hs in zip(sample, hits):
            retrieved_ids = [h["doc_id"] for h in hs]
            gold_in = q["gold_doc_id"] in retrieved_ids
            oracle_ids = [q["gold_doc_id"]] + [d for d in retrieved_ids
                                               if d != q["gold_doc_id"]]
            retrieved = [{"doc_id": d, "text": cap_passage(id2text[d])} for d in retrieved_ids]
            oracle = [{"doc_id": d, "text": cap_passage(id2text[d])} for d in oracle_ids[:N_DOCS]]
            parts = render_parts(q["question"], retrieved, "intermediate")
            rows.append({
                "question": q["question"], "answer": q["answer"], "options": q["options"],
                "gold_doc_id": q["gold_doc_id"], "profile": "intermediate",
                "gold_retrieved": gold_in,
                "retrieved_docs": retrieved,
                "oracle_docs": oracle,
                "docs": retrieved,
                "system": parts["system"], "user": parts["user"],
                "prompt": render_prompt(q["question"], retrieved, "intermediate"),
                "target": build_cot(
                    q["question"], q["answer"], retrieved,
                    next((i for i, e in enumerate(retrieved, 1)
                          if e["doc_id"] == q["gold_doc_id"]), None),
                    "intermediate", rng),
            })
        return rows

    val_rows = eval_rows(val_pool, 200)
    test_rows = eval_rows(test_pool, args.n_test)

    tag = args.distractor_mode
    def dump(rows, name):
        p = PROC / f"{name}_{tag}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {p}  ({len(rows)} rows)")

    dump(train_rows, "raft_train")
    dump(val_rows, "raft_val")
    dump(test_rows, "raft_test")

    if tag == "mixed":
        # canonical test file used by every condition
        dump(test_rows, "raft_test")

    stats = {
        "distractor_mode": tag,
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "n_test": len(test_rows),
        "gold_present_rate_train": round(sum(r["gold_present"] for r in train_rows) / len(train_rows), 3),
        "gold_retrieved_rate_test": round(sum(r["gold_retrieved"] for r in test_rows) / len(test_rows), 3),
        "gold_retrieved_rate_val": round(sum(r["gold_retrieved"] for r in val_rows) / len(val_rows), 3),
        "avg_docs_per_train_sample": round(sum(len(r["docs"]) for r in train_rows) / len(train_rows), 2),
        "avg_target_tokens": round(sum(len(r["target"].split()) for r in train_rows) / len(train_rows), 1),
        "avg_prompt_tokens": round(sum(len(r["prompt"].split()) for r in train_rows) / len(train_rows), 1),
        "pool_sizes": {"train": len(train_pool), "val": len(val_pool), "test": len(test_pool)},
    }
    (PROC / f"data_stats_{tag}.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
