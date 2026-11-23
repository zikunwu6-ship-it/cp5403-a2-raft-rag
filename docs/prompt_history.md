# CP5403 Assessment 2 — Prompt history

**Student:** Zikun Wu · **Student ID:** 15115787
**AI tool:** WorkBuddy AI assistant (DeepSeek-V4.1-Flash)
**Session date:** 18 November 2026
**Purpose of submission:** Required record of generative-AI use for the project
report, code and presentation.

---

## How to read this document

This log records every substantive interaction that contributed to a deliverable.
For each entry it gives the purpose, the prompt used, the output produced, and —
most importantly — **how the output was checked**. Prompts are reproduced in
condensed form: the originals contain long inline file paths, code blocks and
result dumps that would obscure the record.

Entries marked **[correction]** record a case where the AI output was wrong and
was caught during verification. These are included deliberately. They are the
clearest evidence of what was actually verified rather than accepted.

---

## 1. Requirements extraction

**Purpose:** establish exactly what must be submitted and how it is marked.

**Prompt (condensed):** *"Here are the submission requirements for Assessment 2
[assessment brief pasted]. Read the assessment document, the PDF brief and the
AI-use template, and tell me what the deliverables are, the word and time limits,
and the marking weights."*

**Output:** A table of four deliverables (report 30%, presentation 20%, AI-use
template, prompt history), the 3,000–4,000 word and 15-minute limits, and the
rubric criteria broken down by weight.

**Verification:** The weights were checked against the PDF rubric page by page.
The rubric was found to weight *reflection on peer feedback* at 10% of the
presentation, which is not obvious from the brief — this changed the slide plan.

---

## 2. Workspace and environment audit

**Purpose:** find out what already existed before writing anything.

**Prompt (condensed):** *"List everything in the project folder, including hidden
files, and report the GPU, VRAM, installed library versions and disk space."*

**Output:** The project directory contained only empty `data/`, `src/`,
`scripts/`, `results/` folders and a `.venv`. GPU: RTX 4060 Laptop, 8 GB VRAM.
torch 2.14.1+cu126, transformers 5.23.0, peft 0.21.3, bitsandbytes 0.50.2.

**Verification:** Confirmed by direct import test inside the venv, not by reading
`pip list` — `pip` and `python` resolved to different environments, which would
have produced a misleading answer.

**Consequence:** The 8 GB VRAM ceiling set the model size (≤1.5B at 4-bit) and the
batch/gradient-accumulation configuration for the whole project.

---

## 3. Framing the project against the subject's research

**Purpose:** choose a target problem that is defensible rather than generic.

**Prompt (condensed):** *"This is the Google Scholar profile of the subject
coordinator. Retrieve the publication list, identify the research themes, and
propose how a RAFT + RAG project could sit inside that territory without
overclaiming."*

**Output:** The profile resolves to Kranthi Addanki (JCU), whose work centres on
*academagogy* — personalising adult online learner engagement — plus digital twins
in higher education and GenAI governance. Proposed framing: a citation-grounded
study assistant whose explanation depth adapts to a declared learner profile.

**Verification:** The Scholar page was blocked by reCAPTCHA through the normal
fetch path; the profile was retrieved by direct HTTP request and the publication
list parsed from the returned HTML. Titles, venues and years were read from that
page, and the two most load-bearing references were then checked independently
before being cited.

**Note on integrity:** the framing uses the assessor's published work as
*motivation*. The project does not claim to implement academagogy, and §6.3 of the
report states explicitly how narrow the connection is.

---

## 4. Dataset selection

**Purpose:** find a corpus where "did the model cite real evidence?" is
measurable, not a matter of opinion.

**Prompt (condensed):** *"Compare candidate datasets for a RAFT project in an
education setting — SciQ, OpenBookQA, PubMedQA, HotpotQA, Natural Questions. The
requirement is that each question carries a gold evidence passage, because
otherwise citation quality cannot be scored. Recommend one and justify it."*

**Output:** SciQ recommended: 13.7k crowdsourced science exam questions, each with
a *support* passage and four options, in an educational domain.

**Verification:** The dataset was downloaded and inspected directly. Two properties
that the documentation does not advertise were found and had to be handled:
(a) 116 of 1,000 test items have an **empty** support passage, and (b) the answer
string appears verbatim in a gold sentence for only 91.6% of items. Both facts
changed the design (see entries 6 and 8).

---

## 5. Corpus and retriever

**Purpose:** build the retrieval index and establish the retrieval ceiling before
building anything on top of it.

**Prompt (condensed):** *"Build the corpus from unique SciQ support passages,
deduplicate, drop degenerate passages, and write a bi-encoder retriever with
Recall@k and MRR. Use exact search, not ANN, so retrieval recall is not a
confound."*

**Output:** `src/build_corpus.py`, `src/retriever.py`. Corpus: 12,231 passages,
930k tokens, mean 76.6 tokens. Initial probe on 300 questions: Recall@1 = 0.513,
Recall@5 = 0.757, MRR = 0.614.

**Verification:** The first probe reported 12% of gold passages missing from the
corpus. Investigation showed this was caused by the minimum-length filter
discarding short support passages, not by a retrieval problem. The filter was
relaxed and the evaluation pool restricted to answerable questions, giving
Recall@5 = 0.862 on the final set.

---

## 6. Verifying the RAFT recipe against the primary source

**Purpose:** do not implement a method from memory.

**Prompt (condensed):** *"Fetch the RAFT paper and extract the exact data
construction: how many distractors, what fraction of samples keep the gold
document, what the model is trained to do when the gold document is absent, and
the chain-of-thought answer format."*

**Output:** 1 gold + 4 distractors; P = 0.8 keeps gold, 0.2 supplies distractors
only; CoT format `##Reason: ... ##begin_quote## ... ##end_quote## ... ##Answer:`.

**Verification:** The paper's HTML version was read rather than relying on the
abstract. The `##begin_quote##` / `##end_quote##` convention was adopted
specifically because it makes every citation checkable by substring search — this
decision is what the whole evaluation rests on.

---

## 7. RAFT data builder

**Purpose:** generate training data with a controlled distractor distribution.

**Prompt (condensed):** *"Implement the RAFT data builder. Two deviations from the
paper, both deliberate: two of the four distractors must be retrieved hard
negatives rather than random, and the chain-of-thought must be built
programmatically from the gold evidence. Emit a quote marker only when the quoted
span actually occurs in the document."*

**Output:** `src/build_raft_data.py`. 3,000 training samples, 500 evaluation
questions, gold-present rate 0.799 in training (matching P = 0.8), 5 documents per
sample.

**Verification:** Sampled training rows were inspected by hand. One was traced end
to end: the quoted sentence was confirmed to exist verbatim in the cited passage,
and the distractors were confirmed to be topically adjacent (two about blood
cells, one about quartz, one about tundra) rather than random.

---

## 8. **[correction]** Stale retrieval index

**Purpose:** diagnose an implausible result.

**Prompt (condensed):** *"The evaluation reports that the gold passage is
retrieved only 0.8% of the time, but an earlier probe measured Recall@5 = 0.757.
Find the bug."*

**Output:** The embedding cache was keyed only on the model name
(`bge-small-en-v1.5.npy`). After the corpus was rebuilt from 12,128 to 12,231
passages, the stale matrix was reused and silently misaligned against the document
list — every similarity score was compared to the wrong passage.

**Fix:** the cache key now includes the document count and an MD5 fingerprint of
the document IDs, with an assertion on the matrix shape.

**Verification:** Re-running the same evaluation gave 0.862, consistent with the
independent probe.

---

## 9. **[correction]** Evaluation set leaked the answer

**Purpose:** audit the evaluation design before trusting any number from it.

**Prompt (condensed):** *"Review the evaluation data construction for anything
that would make the RAG condition look better than it is."*

**Output:** The evaluation rows were built as `[gold_doc] + retrieved`, which
placed the gold passage at rank 1 by construction. The RAG condition was therefore
being evaluated with a perfect retriever.

**Fix:** evaluation rows now store two document sets — the real top-5 retrieved
(gold may be absent) and an oracle set with the gold forced in — and the harness
selects between them by condition. Retrieval failure is now measured rather than
hidden.

**Consequence:** This produced the most interesting analysis in the report
(§5.3): accuracy split by whether the gold passage was retrieved.

---

## 10. QLoRA training script

**Purpose:** fit fine-tuning into 8 GB of VRAM.

**Prompt (condensed):** *"Write a QLoRA training script: 4-bit NF4 with double
quantisation, LoRA r=16 alpha=32 on all linear layers, gradient checkpointing,
paged 8-bit AdamW, loss on target tokens only with the prompt masked to -100.
Report trainable parameter count and save a metadata file with the full
configuration."*

**Output:** `src/train_qlora.py`. 18,464,768 trainable parameters, 1.18% of the
model.

**Verification:** A 48-example smoke run was executed before the real run. Peak
memory and step time were measured, which is how the 2-epoch budget was sized.

---

## 11. **[correction]** Missing chat template

**Purpose:** explain why baseline answers were incoherent.

**Prompt (condensed):** *"The base model's outputs are continuations of the
passages rather than answers, and it repeats the same sentence. Diagnose."*

**Output:** The prompt was being passed as a flat string to an instruction-tuned
model. Qwen2.5-Instruct expects `<|im_start|>system ... <|im_start|>assistant`,
so the model was performing document continuation.

**Fix:** prompts are now rendered through `apply_chat_template` with
`add_generation_prompt=True`, in training and inference alike.

**Consequence:** Every baseline number produced before this fix was discarded. The
first-pass figures (EM ≈ 0.22) were artefacts and do not appear in the report.

---

## 12. **[correction]** Silent truncation of the supervision target

**Purpose:** explain why one example in each batch produced garbage.

**Prompt (condensed):** *"Exactly one item per batch produces a passage
continuation while the rest are fine. Find out why."*

**Output:** Prompt length was measured across the dataset: mean 719 tokens, p95
1,175, max 2,164. At `max_length=1024` the longest prompts were truncated from the
right, cutting off the `<|im_start|>assistant` header at inference and the
supervision target at training time.

**Fix:** passages are capped at 800 characters at a sentence boundary before
rendering, and the sequence budget was raised to 1,152. Post-fix: mean 616 tokens,
max 1,086 in training, 1,013 in evaluation, zero truncated.

**Verification:** Length distributions were re-measured after the fix, and the
batch anomaly disappeared.

---

## 13. Metric design

**Purpose:** make "the answer is trustworthy" measurable.

**Prompt (condensed):** *"Design metrics that separate answer accuracy from
citation quality. Grounding must be verified by substring search, not by an LLM
judge. Flag any metric that would be confounded by the treatment."*

**Output:** Contains / EM / F1 for accuracy; quote rate, grounded rate, fabricated
rate and gold-cited rate for citations; and a composite *verified correct* (right
answer **and** a real quotation from the gold passage).

**Verification — a metric that was rejected:** an earlier `distractor_cited` metric
counted "quoted something that is not the gold passage", which misfired in the
closed-book condition where no gold passage exists, reporting 39.6% distractor
citations for a condition with no passages at all. It was replaced by
`fabricated_rate` (quoted a span present in no supplied passage) with
`distractor_cited` restricted to grounded quotes.

**Verification — a confound that was controlled:** exact match penalises verbose
answers, and RAFT training happens to make answers terser, which would have
inflated the treatment's apparent advantage. *Contains* was adopted as the
headline accuracy metric for that reason, with EM and F1 reported alongside and
output length reported as a check.

---

## 14. Baseline evaluation

**Purpose:** establish C1 and C2 before any fine-tuning.

**Prompt (condensed):** *"Run the closed-book and retrieved-context conditions for
the untuned model on all 500 evaluation questions, and dump per-example
predictions so the outputs can be inspected rather than only the aggregate."*

**Output:** C1 contains 42.6%, C2 contains 69.2%. C2 quotes a span in 8.6% of
responses and only 3.6% of responses contain exclusively verbatim quotations;
verified correct 1.4%.

**Verification:** Per-example predictions were read, not just the summary. This is
how the closed-book fabrication behaviour was found — the model emits quotation
marks in 39.6% of closed-book answers where there is nothing to quote. That
observation became a load-bearing argument in §1 and slide 3.

---

## 15. Fine-tuning

**Purpose:** train the RAFT adapter.

**Prompt (condensed):** *"Launch the main RAFT training run in the background and
report step time and peak memory so the remaining budget can be planned."*

**Output:** 1 epoch over 3,000 samples, effective batch 16, 188 optimizer steps,
12–16 s per step, ≈45 minutes, peak 6.87 GB. Training loss 1.94 → 0.077;
validation loss 0.117 → 0.092, logged every 50 steps.

**Verification:** Training loss and eval loss were read from the saved
`log_history.json` and checked for divergence before any evaluation was run on the
adapter. Peak memory and step count were written to `meta.json` by the script
rather than estimated afterwards.

**Correction made at this step:** the run was originally configured for 2 epochs.
The first fifth of the first epoch dropped loss from 1.94 to 0.24, and validation
loss tracked it exactly, which means the templated targets are close to trivially
fittable. A second pass would have added nothing except over-fitting, so the run
was restarted at 1 epoch and the change was propagated to the report (§4.4, §6.2),
the deck and the outline. Validation logging was also moved from every 100 steps
to every 50, because 188 total steps would otherwise have produced a single
validation point and an unusable loss figure.

---

## 16. Report drafting

**Purpose:** turn measurements into an argument.

**Prompt (condensed):** *"Draft the report to the required structure in 3,000–4,000
words for an academic audience. Every number must be injected from the result
files rather than typed. State the limitations plainly, including the ones that
weaken my own result. Use APA 7th."*

**Output:** `docs/report_en.md`, with numeric placeholders resolved from
`results/*.json` by a script so that no figure in the prose can drift from the
measurement.

**Verification:** Reference metadata was checked against arXiv and publisher
records for every citation, because the rubric fails the report outright for a
fabricated reference. Three claims in the first draft were removed because the
experiments did not support them.

---

## 17. Presentation

**Purpose:** fit the project into 15 minutes.

**Prompt (condensed):** *"Build a 17-slide deck with a per-slide time budget
totalling 15 minutes, speaker notes, and a dedicated reflection slide. The
reflection must be left as a structured placeholder — I will fill it with the
feedback I actually received, so do not invent any."*

**Output:** `docs/ppt_outline.md` and the slide deck.

**Verification:** The time budget was summed explicitly. The reflection slide
carries a visible instruction not to fabricate feedback, and remains to be
completed by the student.

---

## 18. **[correction]** The k = 8 sweep condition was a duplicate of k = 5

**Purpose:** check that the retrieval-depth sweep actually varied retrieval depth.

**Prompt (condensed):** *"Read back the evaluation harness and tell me what the
k = 8 condition actually feeds the model."*

**Output:** The harness slices `retrieved_docs[:k]`, but the evaluation file stored
only five retrieved documents, so `k = 8` was silently identical to `k = 5`. The
sweep would have reported a flat result at the top end and I would have written a
sentence explaining a difference that did not exist.

**Fix:** `src/extend_retrieval.py` re-runs the same retriever at depth 8 and
rewrites only the two document lists. The stored top-5 order is preserved — one
row in 500 had a cosine tie reordered between runs, and rather than accept a
silently different prompt for that row, the original order is kept and the new
documents appended after it. Every result already computed at k ≤ 5 therefore
remains exactly reproducible, and the backup of the pre-change file is kept.

---

## Summary of corrections

| # | Defect | How it was caught | Effect on results |
|---|---|---|---|
| 8 | Stale retrieval index cache | Implausible 0.8% retrieval rate | Retrieval metrics corrected to 0.862 |
| 9 | Evaluation set forced gold to rank 1 | Design audit before trusting results | RAG condition no longer given a perfect retriever; produced the retrieval-failure analysis |
| 11 | Chat template not applied | Incoherent, repetitive outputs | All pre-fix baseline numbers discarded |
| 12 | Silent truncation of prompts | One bad item per batch | Passage cap + longer sequence budget |
| 13 | `distractor_cited` metric misfired | Closed-book reported 39.6% distractor citations with no passages present | Metric redefined; `fabricated_rate` added |
| 15 | 2-epoch schedule with no justification | Loss fell 1.94 → 0.24 in the first fifth of epoch 1, eval loss tracking it | Re-run at 1 epoch; hyperparameters and prose updated together |
| 18 | `k = 8` sweep was a duplicate of `k = 5` | Read back what the harness actually feeds the model | Retrieval context widened to 8; k ≤ 5 results preserved bit-for-bit |

---

## Statement

I confirm that this log is an accurate record of the generative-AI use in this
assessment. All experimental results were produced by code executed on my own
hardware. I have read and verified the final report and presentation, and I can
explain and defend every design decision and every line of the submitted code.
