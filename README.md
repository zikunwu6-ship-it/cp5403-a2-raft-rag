# AcademaTutor — RAFT + RAG for citation-grounded study support

CP5403 Assessment 2 (LLM Project). Individual work.

A small open-weights language model (Qwen2.5-1.5B-Instruct) is adapted with
**Retrieval-Augmented Fine-Tuning (RAFT)** so that it can answer a learner's
question from a course corpus while (a) citing the exact passage it relied on,
(b) ignoring retrieved passages that look relevant but are not, and (c) adapting
the depth of its explanation to a learner profile.

## Why this problem

Adult learners studying online need answers they can **check**. A vanilla RAG
pipeline returns a fluent answer with no verifiable provenance, and it is
sensitive to the wrong passage being retrieved — a failure mode that is
especially damaging in formative study support, where a confident wrong answer
teaches the wrong thing. RAFT (Zhang et al., 2024) addresses exactly this: train
the model to quote the supporting span verbatim and to discard distractors.

The learner-profile conditioning connects the implementation to *academagogy*
(Addanki et al., 2020, 2022), the framework for personalising adult online
learning that motivates the application.

## Pipeline

```
SciQ (Welbl et al., 2017)
   │
   ├── support passages ──► corpus (12,231 passages) ──► bge-small-en-v1.5 index
   │                                                        │
   └── questions ──► RAFT data builder ◄─────── top-k retrieved ──┘
                        │  1 gold + 4 distractors (2 retrieved-hard, 2 random)
                        │  P = 0.8 keep gold, 0.2 gold absent
                        ▼
                QLoRA SFT (4-bit NF4, r=16) ──► adapter
                        │
                        ▼
        evaluation: Base / Base+RAG / RAFT / RAFT+RAG
```

## What the experiments found

500 held-out SciQ questions, one RTX 4060 Laptop (8 GB), Qwen2.5-1.5B-Instruct
with QLoRA (r=16), one epoch over 3,000 samples.

| ID | Condition | Contains | Verified correct | Fabricated |
|---|---|---|---|---|
| C1 | Base, closed-book | 42.6% | 0.0% | 39.6% |
| C2 | Base + RAG | 69.2% | 1.4% | 5.0% |
| C3 | RAFT, closed-book | 34.0% | 0.0% | 0.6% |
| C4 | RAFT + RAG | 72.4% | 40.4% | 0.0% |
| C5 | RAFT, oracle context | 74.6% | 57.6% | 0.0% |
| A1 | SFT control (gold only) | 76.0% | 50.0% | 0.8% |
| A2 | RAFT without CoT | 78.6% | 0.0% | 0.0% |
| A3 | RAFT, random distractors | 76.6% | 48.2% | 1.0% |

*Verified correct* = the answer is right **and** supported by a quotation that
occurs verbatim in the gold passage.

Four findings, in order of how much they changed the project:

1. **Prompting for citations does not produce them.** The prompted baseline emits
   quotation marks in 8.6% of answers and only 3.6% are real; with no passages at
   all it still quotes 39.6% of the time. Fine-tuning takes fabrication to 0.0%.
2. **The chain-of-thought targets are what buy citations.** Removing them (A2)
   gives the highest raw accuracy in the study and zero verifiability.
3. **The hard-negative distractors — the proposed adaptation — did not help.**
   Gold-only fine-tuning (A1) and random distractors (A3) both beat the full
   recipe on verified correctness (50.0% and 48.2% against 40.4%). This is
   reported as the result rather than buried.
4. **The ceiling is source selection, not grounding.** The model grounds 99.6% of
   its quotations but cites the gold passage in only 47.8%, and when the gold
   passage is missing it quotes a distractor instead of declining. Verified
   correctness is therefore a lower bound: SciQ has one gold passage per question
   and the corpus often contains other passages that support the same answer.

The learner-profile channel did not work either: novice and advanced profiles
produce 53.8 and 53.9 output tokens. Reported as a negative result.

## Repository layout

| Path | Purpose |
|---|---|
| `src/build_corpus.py` | Builds the retrieval corpus from SciQ support passages |
| `src/build_raft_data.py` | Builds RAFT train/val/test data (prompts, CoT targets, distractor sampling) |
| `src/retriever.py` | Dense bi-encoder retriever + Recall@k / MRR |
| `src/train_qlora.py` | QLoRA supervised fine-tuning |
| `src/evaluate.py` | Evaluation harness (accuracy, citation grounding, abstention, cost) |
| `src/analyse.py` | Aggregates results into report tables and figures |
| `data/processed/` | Generated datasets (regenerable, not committed) |
| `results/` | Metrics as JSON + per-example predictions |
| `outputs/trainer/` | LoRA adapters and training logs |

## Reproducing

```bash
python -m venv .venv && ./.venv/Scripts/activate     # Windows
pip install -r requirements.txt

python src/build_corpus.py
python src/build_raft_data.py --distractor-mode mixed --n-train 3000 --n-test 500
python src/build_raft_data.py --distractor-mode random --n-train 3000 --n-test 500
python src/build_raft_data.py --distractor-mode gold_only --n-train 3000 --n-test 500

python src/train_qlora.py --data-tag mixed --run-name raft_mixed --epochs 1
python src/train_qlora.py --data-tag gold_only --run-name sft_gold --epochs 1
python src/train_qlora.py --data-tag mixed --run-name raft_nocot --epochs 1 --answer-only

python src/evaluate.py --condition raft_rag --context retrieved \
    --adapter outputs/trainer/raft_mixed/final --out results/raft_rag.json --dump
```

## Hardware

Single consumer GPU, 8 GB VRAM (RTX 4060 Laptop). Training uses 4-bit NF4
quantisation, LoRA adapters, gradient checkpointing and paged 8-bit AdamW.
Peak training memory ≈ 7 GB at batch 2 × grad-accum 8.

## Data and licensing

* SciQ — public, released for research use (Welbl et al., 2017).
* Qwen2.5-1.5B-Instruct — Apache 2.0.
* bge-small-en-v1.5 — MIT.

## AI use

Generative AI was used in this project. See `docs/AI_use_report.docx` and
`docs/prompt_history.docx` for the declaration and the full prompt log.
