# CP5403 — Assessment 2: AI use reporting

**Student:** Zikun Wu · **Student ID:** 15115787 · **Date:** 18 November 2026

**Tool used:** WorkBuddy AI assistant (DeepSeek-V4.1-Flash) — used for all entries
below. No image-generation tool was used.

---

## Part A — Declaration of AI-generated material

Generative artificial intelligence was used in the preparation of this assessment
item. The subject permits unrestricted use of GenAI provided that it is
acknowledged. The scope of use is:

- **Text:** AI was used to draft and restructure prose. All technical claims,
  design decisions and interpretations were reviewed and, where necessary,
  corrected by me.
- **Code:** AI was used to write the data-construction, training and evaluation
  code from specifications I supplied, and to diagnose runtime errors. I ran every
  script, and I can explain every component.
- **Results:** **No result, metric, table or figure in the report was generated or
  estimated by an AI system.** All numbers come from code executed on my own
  hardware against the public datasets described in the report.
- **Literature:** AI was used to locate primary sources and check reference
  metadata. Every reference in the report was verified against the publisher or
  arXiv record; none were generated without verification.

**What I verified personally.** Three defects were found and corrected during
verification, and they are recorded here because they changed the results:

1. The first training and inference pipeline passed a flat prompt string to an
   instruction-tuned model without applying its chat template. The model continued
   the passages instead of answering. All baseline numbers produced before this
   fix were discarded and regenerated.
2. The first evaluation set placed the gold passage at rank 1 by construction,
   which made retrieval appear perfect. It was rebuilt so that the retrieved
   context is the real retriever output, with a separate oracle condition.
3. The retrieval index cache was keyed only on the model name, so it silently
   survived a corpus rebuild and misaligned embeddings against documents. It is
   now keyed on a corpus fingerprint.

---

## Part B — AI use by category

| AI tool | Description of use | Used (Y/N) | Section of assessment |
|---|---|---|---|
| WorkBuddy (DeepSeek-V4.1-Flash) | Idea brainstorming — decomposing the brief into a project plan, choosing between RAFT-only, RAG-only and RAFT+RAG designs, and deciding which ablations were worth the compute budget | **Y** | Entire work — project scoping; §2, §4.5 |
| WorkBuddy (DeepSeek-V4.1-Flash) | Generating a draft conclusion that I revised — an initial draft restated the results table; I rewrote it to state the limitation and the next step instead | **Y** | §7 Conclusion |
| WorkBuddy (DeepSeek-V4.1-Flash) | Spelling and grammar check and draft editing | **Y** | Entire report |
| WorkBuddy (DeepSeek-V4.1-Flash) | Modifying tone and expression for academic style — converting first-person working notes into formal register, and removing evaluative adjectives that were not supported by the measurements | **Y** | §1, §5, §6 |
| WorkBuddy (DeepSeek-V4.1-Flash) | Comparing and summarising journal articles — used to position RAFT against vanilla RAG, Self-RAG and Chain-of-Note, and to identify which failure modes each addresses | **Y** | §1, §4.2, §6.1 |
| WorkBuddy (DeepSeek-V4.1-Flash) | Generating images | **N** | — (all figures are matplotlib output or native PowerPoint shapes) |
| WorkBuddy (DeepSeek-V4.1-Flash) | Finding relevant literature — locating primary sources for RAFT, RAG, QLoRA, SciQ and the academagogy framework, and retrieving the assessor's published work for the framing | **Y** | §1, §3, References |
| WorkBuddy (DeepSeek-V4.1-Flash) | Reference formatting — converting verified metadata into APA 7th edition | **Y** | References |
| WorkBuddy (DeepSeek-V4.1-Flash) | Themed template creation — the report and slide layout structure | **Y** | Entire report; presentation |
| WorkBuddy (DeepSeek-V4.1-Flash) | Writing implementation code (data pipeline, QLoRA training loop, evaluation harness) from my written specifications, and debugging runtime errors | **Y** | §4; `src/`, `scripts/` |
| WorkBuddy (DeepSeek-V4.1-Flash) | Designing the evaluation metrics, including the verbatim grounding check and the "verified correct" composite | **Y** | §4.6, §5 |
| WorkBuddy (DeepSeek-V4.1-Flash) | Building tables and figures from the measured result files | **Y** | §5; presentation slides 11–14 |

---

## Part C — What the AI did not do

To be explicit about the boundary:

- It did not run the experiments. Every reported number is from a script executed
  on the student's GPU.
- It did not select the topic independently; the direction was set by the student
  and aligned to the subject's published research.
- It did not fabricate any reference. Each of the 20 references was checked
  against arXiv, the publisher, or the author's public profile.
- It did not write the reflection on peer feedback, which reports feedback the
  student actually received.

---

## Part D — Prompt history

The full prompt history is submitted as a separate document
(`Prompt_history.docx`). It records each substantive interaction: the purpose, the
prompt used, the output produced, and how the output was verified or corrected.
