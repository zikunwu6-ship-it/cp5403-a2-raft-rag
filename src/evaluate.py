"""
Evaluation harness for the AcademaTutor experiment.

Every condition is scored with the same parser and the same metrics so that the
comparison is like-for-like:

  answer accuracy   normalised exact match and token-level F1 against the SciQ
                    gold answer string
  quote_rate        share of responses that cite at least one span
  grounded_rate     share of cited spans that appear VERBATIM in a supplied
                    document (i.e. not fabricated)
  gold_cited_rate   share of responses that cite the gold document
                    (only meaningful when the gold document was supplied)
  distractor_cited  share of responses that quote a non-gold document
  abstain_rate      share of responses that declare the context insufficient
  tokens / latency  cost of the condition

Usage:
  python src/evaluate.py --adapter outputs/trainer/raft_mixed/final \
      --context retrieved --out results/raft_rag.json
"""
from __future__ import annotations

import argparse
import json
import re
import string
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_raft_data import PROFILES, render_parts  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

ABSTAIN_PAT = re.compile(
    r"\b(no information|cannot be determined|not (?:found|provided|available|enough)|"
    r"insufficient|none of the passages|do not contain|does not contain|unable to)\b",
    re.I,
)
QUOTE_PAT = re.compile(r"##begin_quote##(.*?)##end_quote##", re.S)
ANSWER_PAT = re.compile(r"##Answer:\s*(.+?)(?:\n|$)", re.S)


# --------------------------------------------------------------------------- #
# Text normalisation and answer scoring (SQuAD-style)
# --------------------------------------------------------------------------- #
def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = {}
    for t in p:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in g:
        if common.get(t, 0) > 0:
            common[t] -= 1
            overlap += 1
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(g)
    return 2 * prec * rec / (prec + rec)


_STRIP_MARKERS = re.compile(r"##begin_quote##|##end_quote##|##\s*Reason\s*:?|##\s*Answer\s*:?|##")


def parse_output(text: str) -> dict:
    """Extract the answer, tolerating imperfect format compliance.

    A 1.5B instruction-tuned model does not always emit the requested
    `##Answer:` field. Scoring only well-formatted responses would confound
    answer quality with format compliance, so when the field is missing we fall
    back to the last sentence of the response. `format_ok` records which path
    was taken and is reported separately.
    """
    m = ANSWER_PAT.search(text)
    format_ok = m is not None
    if format_ok:
        answer = m.group(1).strip().split("##")[0].strip()
    else:
        cleaned = _STRIP_MARKERS.sub(" ", text)
        cleaned = re.sub(r"\[\d+\]", " ", cleaned)
        sents = [x.strip() for x in re.split(r"(?<=[.!?])\s+", cleaned) if x.strip()]
        answer = sents[-1] if sents else cleaned.strip()
    answer = answer.strip().strip('"').strip("'").rstrip(".").strip()
    quotes = [q.strip() for q in QUOTE_PAT.findall(text) if q.strip()]
    return {
        "answer": answer,
        "quotes": quotes,
        "format_ok": format_ok,
        "has_reason": "##Reason" in text,
        "abstained": bool(ABSTAIN_PAT.search(text.split("##Answer")[0])),
    }


def contains_gold(pred: str, gold: str) -> float:
    """Lenient accuracy: does the gold answer appear as a token sequence?

    Insensitive to how verbose the response is, so it does not reward the
    terse output style that RAFT training happens to produce.
    """
    p = " " + normalize(pred) + " "
    g = " " + normalize(gold) + " "
    return float(g.strip() != "" and g in p)


def _norm_ws(s: str) -> str:
    s = s.replace("\u2014", "-").replace("\u2013", "-").replace("\u2019", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    return " ".join(s.lower().split())


def quote_is_grounded(quote: str, docs: list[dict]) -> tuple[bool, bool]:
    """Return (grounded_in_any_doc, grounded_in_gold)."""
    q = _norm_ws(quote).strip(" .\"'")
    if len(q) < 8:
        return False, False
    in_any = in_gold = False
    for i, d in enumerate(docs):
        if q in _norm_ws(d["text"]):
            in_any = True
            if i == 0 and d.get("is_gold"):
                in_gold = True
    return in_any, in_gold


# --------------------------------------------------------------------------- #
def build_inputs(row: dict, context: str, k: int, profile: str) -> tuple[dict, list[dict]]:
    """Return (chat messages, documents actually shown to the model)."""
    if context == "closed":
        return render_parts(row["question"], [], profile), []
    if context == "oracle":
        docs = [{**d, "is_gold": d["doc_id"] == row["gold_doc_id"]}
                for d in row["oracle_docs"]][:k]
    elif context == "retrieved":
        docs = [{**d, "is_gold": d["doc_id"] == row["gold_doc_id"]}
                for d in row["retrieved_docs"]][:k]
    else:
        raise ValueError(context)
    return render_parts(row["question"], docs, profile), docs


def apply_template(tok, parts: dict) -> str:
    return tok.apply_chat_template(
        [{"role": "system", "content": parts["system"]},
         {"role": "user", "content": parts["user"]}],
        tokenize=False, add_generation_prompt=True,
    )


def load_model(base: str, adapter: str | None):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        base, quantization_config=bnb, device_map={"": 0},
        torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok


@torch.inference_mode()
def generate_batch(model, tok, prompts: list[str], max_new_tokens: int) -> list[str]:
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
              max_length=1152).to(model.device)
    out = model.generate(
        **enc, max_new_tokens=max_new_tokens, do_sample=False,
        num_beams=1, repetition_penalty=1.05,
        pad_token_id=tok.pad_token_id,
    )
    gen = out[:, enc["input_ids"].shape[1]:]
    return tok.batch_decode(gen, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--condition", required=True, help="label for this run")
    ap.add_argument("--context", default="retrieved", choices=["closed", "retrieved", "oracle"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--profile", default="intermediate", choices=list(PROFILES))
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=140)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump", action="store_true", help="store per-example predictions")
    args = ap.parse_args()

    rows = [json.loads(l) for l in (PROC / f"raft_{args.split}_mixed.jsonl").open(encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]

    model, tok = load_model(args.base, args.adapter)

    records, latencies = [], []
    t_start = time.time()
    for i in range(0, len(rows), args.batch_size):
        chunk = rows[i: i + args.batch_size]
        built = [build_inputs(r, args.context, args.k, args.profile) for r in chunk]
        prompts = [apply_template(tok, b[0]) for b in built]
        t0 = time.time()
        outs = generate_batch(model, tok, prompts, args.max_new_tokens)
        dt = time.time() - t0
        latencies += [dt / len(chunk)] * len(chunk)

        for row, (_, docs), out in zip(chunk, built, outs):
            p = parse_output(out)
            grounded, in_gold = (False, False)
            if p["quotes"]:
                flags = [quote_is_grounded(q, docs) for q in p["quotes"]]
                grounded = all(f[0] for f in flags)
                in_gold = any(f[1] for f in flags)
            gold_present = any(d["doc_id"] == row["gold_doc_id"] for d in docs) if docs else False
            em = float(normalize(p["answer"]) == normalize(row["answer"]))
            has_quote = bool(p["quotes"])
            contains = contains_gold(p["answer"], row["answer"])
            # A citation is "fabricated" when the model emits a quote that appears
            # in none of the passages it was given (including the closed-book
            # condition, where any quote is necessarily fabricated). It is a
            # "distractor citation" only when the quote is real but comes from a
            # non-gold passage.
            fabricated = has_quote and not grounded
            distractor_cited = has_quote and grounded and not in_gold
            records.append({
                "question": row["question"],
                "gold": row["answer"],
                "pred": p["answer"],
                "raw": out if args.dump else None,
                "em": em,
                "f1": token_f1(p["answer"], row["answer"]),
                "contains": contains,
                "format_ok": p["format_ok"],
                "has_quote": has_quote,
                "grounded": grounded,
                "gold_cited": in_gold,
                "fabricated": fabricated,
                "distractor_cited": distractor_cited,
                "verified_correct": bool(contains and in_gold),
                "abstained": p["abstained"],
                "gold_present": gold_present,
                "gold_retrieved": row.get("gold_retrieved", None),
                "n_out_tokens": len(out.split()),
            })
    total_time = time.time() - t_start

    def agg(rs):
        if not rs:
            return {}
        n = len(rs)
        return {
            "n": n,
            "em": round(sum(r["em"] for r in rs) / n, 4),
            "f1": round(sum(r["f1"] for r in rs) / n, 4),
            "contains": round(sum(r["contains"] for r in rs) / n, 4),
            "format_rate": round(sum(r["format_ok"] for r in rs) / n, 4),
            "quote_rate": round(sum(r["has_quote"] for r in rs) / n, 4),
            "grounded_rate": round(sum(r["grounded"] for r in rs) / n, 4),
            "gold_cited_rate": round(sum(r["gold_cited"] for r in rs) / n, 4),
            "verified_correct": round(sum(r["verified_correct"] for r in rs) / n, 4),
            "fabricated_rate": round(sum(r["fabricated"] for r in rs) / n, 4),
            "distractor_cited_rate": round(sum(r["distractor_cited"] for r in rs) / n, 4),
            "abstain_rate": round(sum(r["abstained"] for r in rs) / n, 4),
            "avg_out_tokens": round(sum(r["n_out_tokens"] for r in rs) / n, 1),
            "avg_latency_s": round(sum(latencies[:n]) / n, 3),
        }

    with_gold = [r for r in records if r["gold_present"]]
    without = [r for r in records if not r["gold_present"]]

    result = {
        "condition": args.condition,
        "adapter": args.adapter,
        "base": args.base,
        "context": args.context,
        "k": args.k,
        "profile": args.profile,
        "split": args.split,
        "overall": agg(records),
        "gold_present_subset": agg(with_gold),
        "gold_absent_subset": agg(without),
        "wall_clock_s": round(total_time, 1),
        "samples_per_s": round(len(records) / total_time, 2),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")

    if args.dump:
        d = Path(args.out).with_suffix(".preds.jsonl")
        with d.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(json.dumps({k: v for k, v in result.items()
                      if k in ("condition", "context", "k", "overall",
                               "gold_present_subset", "gold_absent_subset",
                               "wall_clock_s")}, indent=2))


if __name__ == "__main__":
    main()
