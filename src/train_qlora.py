"""
QLoRA supervised fine-tuning of an open-weights LLM on RAFT-format data.

Memory plan for an 8 GB consumer GPU:
  * base weights in 4-bit NF4 with double quantisation (~1.0 GB for 1.5B)
  * LoRA adapters only (r=16) - the base model is frozen
  * gradient checkpointing + paged AdamW 8-bit
  * batch 2 x grad-accum 8, max sequence length 1152

Loss is computed on the target tokens only; the prompt tokens are masked to -100.
That matters here because ~90% of each sequence is the instruction + passages,
and training on those tokens would dilute the signal we actually care about.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data-tag", default="mixed")
    ap.add_argument("--run-name", default="raft_mixed")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1152)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--answer-only", action="store_true",
                    help="ablation: train on the final answer only, no chain-of-thought")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = OUT / "trainer" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # Loaded separately: the evaluation files carry retrieval metadata
    # (retrieved_docs / oracle_docs) that the training files do not, so the two
    # schemas cannot be merged into a single DatasetDict.
    train_ds = load_dataset("json", data_files=str(PROC / f"raft_train_{args.data_tag}.jsonl"))["train"]
    val_ds = load_dataset("json", data_files=str(PROC / f"raft_val_{args.data_tag}.jsonl"))["train"]

    def to_features(ex):
        # The base model is instruction-tuned, so the prompt must go through its
        # chat template; a flat string makes it continue the passages instead of
        # answering the question.
        templated = tok.apply_chat_template(
            [{"role": "system", "content": ex["system"]},
             {"role": "user", "content": ex["user"]}],
            tokenize=False, add_generation_prompt=True,
        )
        prompt_ids = tok(templated, add_special_tokens=False)["input_ids"]
        if args.answer_only:
            target = f"##Answer: {ex['answer']}"
        else:
            target = ex["target"]
        target_ids = tok(target + tok.eos_token, add_special_tokens=False)["input_ids"]
        input_ids = (prompt_ids + target_ids)[: args.max_len]
        labels = ([-100] * len(prompt_ids) + target_ids)[: args.max_len]
        return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}

    train_ds = train_ds.map(to_features, remove_columns=train_ds.column_names, desc="tokenising train")
    val_ds = val_ds.map(to_features, remove_columns=val_ds.column_names, desc="tokenising val")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        pad = tok.pad_token_id
        input_ids, labels, attn = [], [], []
        for b in batch:
            n = maxlen - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad] * n)
            labels.append(b["labels"] + [-100] * n)
            attn.append(b["attention_mask"] + [0] * n)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }

    targs = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        bf16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        report_to="none",
        seed=args.seed,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate,
    )
    trainer.train()

    final = out_dir / "final"
    model.save_pretrained(final)
    tok.save_pretrained(final)

    hist = trainer.state.log_history
    (out_dir / "log_history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")
    meta = {
        "base_model": args.model,
        "data_tag": args.data_tag,
        "run_name": args.run_name,
        "answer_only": args.answer_only,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch": args.batch_size * args.grad_accum,
        "max_len": args.max_len,
        "n_train": len(train_ds),
        "train_loss_final": next((h["loss"] for h in reversed(hist) if "loss" in h), None),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)
        if torch.cuda.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
