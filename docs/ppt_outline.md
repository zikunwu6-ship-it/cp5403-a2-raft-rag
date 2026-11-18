# Presentation outline — 15 minutes

Target: 15 min + Q&A. 17 slides. Spoken word budget ≈ 2,100 words (≈ 140 wpm).

| # | Slide | Time | Cumulative |
|---|---|---|---|
| 1 | Title | 0:15 | 0:15 |
| 2 | The learner, not the model | 1:15 | 1:30 |
| 3 | Why RAG alone is not enough | 1:25 | 2:55 |
| 4 | Research question and contributions | 0:45 | 3:40 |
| 5 | Data: SciQ and the course corpus | 1:10 | 4:50 |
| 6 | System architecture | 1:00 | 5:50 |
| 7 | RAFT data construction | 1:15 | 7:05 |
| 8 | Learner-profile conditioning | 0:45 | 7:50 |
| 9 | Training setup | 0:45 | 8:35 |
| 10 | Evaluation design | 0:45 | 9:20 |
| 11 | Main results | 1:25 | 10:45 |
| 12 | Citation quality — the key finding | 1:05 | 11:50 |
| 13 | What causes the gain (ablations) | 1:00 | 12:50 |
| 14 | Retrieval failure is the ceiling | 0:30 | 13:20 |
| 15 | Limitations | 0:40 | 14:00 |
| 16 | Reflection on peer feedback | 0:45 | 14:45 |
| 17 | Conclusion / questions | 0:15 | 15:00 |

The budget sums to exactly 15:00. Slide 17 is the handover to questions, so its
0:15 is a beat, not a section — if you are running late, take the time from
slide 15 rather than from the results.

---

## Slide 1 — Title

**AcademaTutor**
*Citation-grounded study support for adult online learners: adapting RAFT to a course corpus*

CP5403 Assessment 2 · Zikun Wu · 15115787 · November 2026

> Notes: 20 seconds. State the one-line claim: "I built a study assistant that
> shows its evidence, and I measured whether the evidence is real."

---

## Slide 2 — The learner, not the model

**Visual:** two-column contrast.

Left — *Adult online learner*: asynchronous, time-poor, no tutor in the room,
studying at 11pm.

Right — *What they need*: an answer they can **check**, at a depth that matches
what they already know.

**Key line on slide:** Addanki's *academagogy* — engagement depends on the system
responding to the individual learner, not on delivering more content.

> Notes: Frame the problem in the assessor's own research territory without
> overclaiming. Cite Addanki et al. (2020, 2022). Say plainly: this is the
> motivation, not something I claim to have solved.

---

## Slide 3 — Why RAG alone is not enough

**Visual:** three failure modes as three small panels.

1. **Retrieval is wrong, not random.** Dense retrievers return *topically* close
   passages. That is exactly when models get misled (Shi et al., 2023).
2. **No provenance.** A fluent answer the learner cannot check is not study
   support.
3. **Fabricated citation.** Measured in this project: the prompted baseline emits
   quotation marks in {{C2_QUOTE}} of answers, and only {{C2_GROUNDED}} of those
   quotes actually exist in the supplied passages.

> Notes: Point 3 is the hook — it is a measured number from my own baseline, not
> a literature claim. It sets up why "just prompt it to cite" fails.

---

## Slide 4 — Research question and contributions

**RQ:** Can RAFT turn a RAG study assistant from *fluent* into *checkable*, on a
consumer GPU, without losing accuracy?

Three contributions:

1. RAFT adapted so distractors are **retrieved hard negatives**, matching real
   retriever behaviour.
2. A **learner-profile channel** connecting the implementation to academagogy,
   evaluated as a controlled variable.
3. An evaluation that separates accuracy from citation quality, with **verbatim
   grounding verification** and ablations.

---

## Slide 5 — Data: SciQ and the course corpus

**Visual:** pipeline numbers as a horizontal flow.

SciQ (Welbl et al., 2017) · 13.7k science exam questions · **each has a gold
support passage** → this is what makes citation measurable.

| | |
|---|---|
| Corpus | {{N_PASSAGES}} unique passages, {{TOTAL_TOKENS}} tokens, mean {{MEAN_TOKENS}} |
| Excluded | {{EMPTY_RATE}} of test items have an empty support passage → unanswerable by construction |
| Train / eval | {{N_TRAIN}} RAFT samples / {{N_TEST}} held-out questions |

> Notes: Be explicit about the exclusion — it is a judgement call and I should own
> it. Also flag the limitation: corpus and gold evidence are the same passages, so
> recall here is an upper bound.

---

## Slide 6 — System architecture

**Visual:** block diagram.

```
question ──► bge-small-en-v1.5 ──► top-5 passages ─┐
                                                   ├──► Qwen2.5-1.5B + LoRA ──►
learner profile ───────────────────────────────────┘        (RAFT-tuned)
                                                              │
                                          ##Reason: quote + reject distractors
                                          ##Answer: <phrase>
```

> Notes: Emphasise the two inputs — evidence and learner. The output format is
> not cosmetic; the quote markers are what makes grounding checkable by substring.

---

## Slide 7 — RAFT data construction

**Visual:** one worked example, gold passage highlighted among four distractors.

- 1 gold + 4 distractors per sample
- P = 0.8 keep gold · 0.2 gold absent → teaches "answer from knowledge / abstain"
- **Adaptation 1:** 2 of the 4 distractors are *retrieved* hard negatives, not random
- **Adaptation 2:** CoT targets built programmatically; a quote marker is emitted
  **only when the span is verbatim** in the document

> Notes: Adaptation 2 is the integrity-critical decision. If the training target
> contained fake quotes, my grounding metric would be measuring my template. Own
> the cost: templated traces are less diverse than GPT-4 traces — that is the
> biggest weakness in the project and it is on the limitations slide.

---

## Slide 8 — Learner-profile conditioning

**Visual:** same question, three outputs, different depth.

| Profile | Instruction | Effect |
|---|---|---|
| Novice | step by step, define terms | longest |
| Intermediate | short chain of reasoning | baseline |
| Advanced | terse, discriminating detail only | shortest |

Held **constant** across the main comparison, varied only in the profile experiment.

> Notes: This is the academagogy link, and it is thin: the profile is *declared*,
> not inferred. Say now that the sweep later found no usable effect, so this slide
> is a design description, not a claim of a working feature.

---

## Slide 9 — Training setup

QLoRA, 4-bit NF4 + double quant · LoRA r=16, α=32, all linear layers
· {{TRAINABLE}} trainable params ({{TRAINABLE_PCT}})
· lr 2e-4 cosine · 1 epoch · effective batch 16 · max len 1152
· loss on target tokens only (prompt masked)
· **8 GB RTX 4060 Laptop**, peak ≈ 7 GB

> Notes: Mention the two bugs caught during verification — missing chat template
> (model continued the passages) and an eval set that forced gold into rank 1.
> Shows the numbers were checked, not assumed.

---

## Slide 10 — Evaluation design

**Four conditions**, identical prompts and parser:

C1 Base closed-book · C2 Base+RAG · C3 RAFT closed-book · **C4 RAFT+RAG**

Plus C5 oracle, and ablations A1 (gold-only SFT), A2 (no CoT), A3 (random
distractors).

**Metrics:** Contains / EM / F1 · quote rate · grounded rate · fabricated rate ·
gold-cited rate · **verified correct** (right *and* checkable) · tokens · latency.

> Notes: "Verified correct" is the metric that matters for the application. Define
> it before showing the table so the audience reads the right column.

---

## Slide 11 — Main results

{{TABLE_MAIN_SLIDE}}

**Say:** accuracy {{C1_CONTAINS}} → {{C2_CONTAINS}} → {{C4_CONTAINS}}.
Fine-tuning without retrieval (C3) helps, but less than retrieval does.

---

## Slide 12 — Citation quality — the key finding

**Visual:** grouped bar chart, fabricated vs grounded, C2 vs C4.

| | C2 Base+RAG | C4 RAFT+RAG |
|---|---|---|
| Quotes something | {{C2_QUOTE}} | {{C4_QUOTE}} |
| ...and it is verbatim real | {{C2_GROUNDED}} | {{C4_GROUNDED}} |
| Verified correct | {{C2_VERIFIED}} | {{C4_VERIFIED}} |

**Say:** prompting for citations does not produce citations. Training does.

> Notes: This is the slide to slow down on. It is the result that justifies the
> whole approach.

---

## Slide 13 — What causes the gain, and what does not

{{TABLE_ABLATION_SLIDE}}

**Say:** this is the slide where the argument changed. Two things fall out.
First, the chain-of-thought targets are what buy the citations: remove them and
accuracy goes *up* to {{A2_CONTAINS}} but nothing is quoted. Second, the hard
negatives do not buy discrimination. Verified correctness is {{A1_VERIFIED}} with
gold-only training, {{A3_VERIFIED}} with random distractors, and {{C4_VERIFIED}}
with my retrieved hard negatives. The adaptation I proposed was the worst of the
three, and I am reporting that rather than burying it.

> Notes: be ready for "so was RAFT wrong?" The answer is no — RAFT's own ablation
> varies distractor *presence*, not distractor *hardness*, and my A1 changes three
> things at once. What this shows is that the hard-negative variant is not a free
> improvement at 1.5B parameters on this corpus. Say that plainly.

---

## Slide 14 — Source selection is the ceiling

**Visual:** two-bar comparison split by whether gold was retrieved.

| | gold retrieved | gold missing |
|---|---|---|
| C2 | {{C2_CONTAINS_GOLD}} | {{C2_CONTAINS_MISS}} |
| C4 | {{C4_CONTAINS_GOLD}} | {{C4_CONTAINS_MISS}} |

**Say:** fabrication is solved — zero fabricated citations, every quote is real.
What is not solved is choosing *which* passage to quote: gold is cited in only
{{C4_GOLD_CITED}} of answers. And when the gold passage is missing the model quotes
a distractor rather than declining, so the abstention the recipe was supposed to
teach did not transfer. This is the most useful negative result in the project,
and it is only visible because the quote is verbatim and therefore checkable.

---

## Slide 15 — Limitations

- Templated CoT targets, not GPT-4-generated → discourse structure may be learned
- Corpus = gold passages → Recall@k is an upper bound
- One dataset, one model, one seed → no confidence intervals
- Profile conditioning did not work: no verbosity effect, so it needs depth-varied targets
- No human evaluation of whether explanations actually help learners

---

## Slide 16 — Reflection on peer feedback

> **TO BE COMPLETED BY THE STUDENT WITH THEIR ACTUAL WEEK 9/10 FEEDBACK.**
> Do not invent feedback. Use the four-row structure below.

| Feedback received | What I changed | Why | Effect |
|---|---|---|---|
| [peer comment 1] | [change] | [reason] | [what improved] |
| [peer comment 2] | [change] | [reason] | [what improved] |
| [peer comment 3] | [change] | [reason] | [what improved] |

> Notes: The rubric weights this 10% and asks specifically for *how you responded*,
> *what you changed*, *why*, and *how it improved the project*. Keep it concrete and
> avoid "the feedback was useful" — name the change.

---

## Slide 17 — Conclusion

RAFT makes a small model's answers checkable: verified correctness
{{C2_VERIFIED}} → {{C4_VERIFIED}} on {{N_TEST}} held-out questions, at
{{C4_LATENCY}} s/query on a laptop GPU.

Next: better retrieval, a corpus that looks like a real course site, and human
evaluation of whether profile-conditioned explanations help.

**Questions.**

---

## Anticipated questions (not on slides)

1. *Why not a 7B model?* 8 GB VRAM; a 7B at 4-bit leaves no room for batch or
   gradient checkpointing at usable throughput. 1.5B was the largest model that
   trained and evaluated inside the time budget.
2. *Why is your Recall@5 only {{RECALL5}}?* Short passages, single-vector dense
   retrieval, no reranking. A cross-encoder reranker is the obvious next step and
   was deliberately left out to keep the retrieval component fixed across
   conditions.
3. *Isn't the templated CoT cheating?* It constrains what the model can learn. It
   cannot learn to reason in ways the template does not express. What it does
   learn — which passage to trust and to quote verbatim — is what the evaluation
   measures.
4. *How is "contains" not gameable?* A verbose answer that mentions many things
   could contain the gold string by accident. That is why EM and F1 are reported
   alongside, and why output length is reported: C4 is *shorter*, not longer.
5. *Does the profile channel actually work?* Partially — see slide 8 and the
   numbers in the report. Depth shifts; accuracy does not degrade.
