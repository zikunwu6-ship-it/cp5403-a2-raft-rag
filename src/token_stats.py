"""
Compute real token statistics for the training data.

`build_raft_data.py` records `avg_prompt_tokens` from a whitespace split, which is
a *word* count, not a token count. The report quotes these figures, so they are
recomputed here with the actual tokenizer and written to token_stats.json.
"""
import json
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def main():
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    rows = [json.loads(l) for l in (PROC / "raft_train_mixed.jsonl").open(encoding="utf-8")]

    prompt_lens, target_lens = [], []
    for r in rows:
        templated = tok.apply_chat_template(
            [{"role": "system", "content": r["system"]},
             {"role": "user", "content": r["user"]}],
            tokenize=False, add_generation_prompt=True,
        )
        prompt_lens.append(len(tok(templated, add_special_tokens=False)["input_ids"]))
        target_lens.append(len(tok(r["target"] + tok.eos_token, add_special_tokens=False)["input_ids"]))

    prompt_lens.sort()
    n = len(prompt_lens)
    stats = {
        "n": n,
        "prompt_tokens_mean": round(sum(prompt_lens) / n, 1),
        "prompt_tokens_p50": prompt_lens[n // 2],
        "prompt_tokens_p95": prompt_lens[int(n * 0.95)],
        "prompt_tokens_max": prompt_lens[-1],
        "target_tokens_mean": round(sum(target_lens) / n, 1),
        "target_tokens_max": max(target_lens),
        "prompt_share_of_sequence_pct": round(
            100 * sum(prompt_lens) / (sum(prompt_lens) + sum(target_lens)), 1),
        "n_over_1024": sum(1 for x in prompt_lens if x > 1024),
    }
    (PROC / "token_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
