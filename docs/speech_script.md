# Presentation script — 15 minutes

This is a speaking script, not the report. Each slide gets a "Say" block that can be read almost verbatim at about 140 words per minute, followed by delivery cues in Chinese. Every number below is taken from `results/*.json`, so the script and the deck cannot drift apart.

Read the Say blocks as written the first two times you rehearse; after that, keep the numbers and the signposts and let the sentences go. The cues tell you where to point, where to pause, and what to cut if you are running long.

## Timing at a glance

| # | Slide | Time | Cumulative | Words |
|---|---|---|---|---|
| 1 | Title | 0:15 | 0:15 | 39 |
| 2 | The learner, not the model | 1:15 | 1:30 | 153 |
| 3 | Why retrieval alone is not enough | 1:25 | 2:55 | 168 |
| 4 | Research question | 0:45 | 3:40 | 97 |
| 5 | Data | 1:10 | 4:50 | 163 |
| 6 | Architecture | 1:00 | 5:50 | 131 |
| 7 | RAFT data construction | 1:15 | 7:05 | 182 |
| 8 | Learner-profile conditioning | 0:45 | 7:50 | 100 |
| 9 | Training setup | 0:45 | 8:35 | 105 |
| 10 | Evaluation design | 0:45 | 9:20 | 105 |
| 11 | Main results | 1:25 | 10:45 | 166 |
| 12 | Citation quality | 1:05 | 11:50 | 142 |
| 13 | Ablations | 1:00 | 12:50 | 142 |
| 14 | Source selection | 0:30 | 13:20 | 80 |
| 15 | Limitations | 0:40 | 14:00 | 80 |
| 16 | Reflection | 0:45 | 14:45 | 43 (grows to about 105 once filled in) |
| 17 | Conclusion | 0:15 | 15:00 | 36 |

Total spoken words as written: 1,945, which is 130 words per minute across 15 minutes. That is a deliberate pace, not a rushed one, and it leaves room for the pauses and the pointing. Slide 16 is short on purpose: it will grow to about 105 words when the real peer feedback goes in, which is exactly its 45 seconds.

## Three rules for the whole talk

1. Never read a slide aloud. The audience can read faster than you can speak; say what the slide does not say.
2. Every number gets a beat of silence after it. The silence is what makes it land.
3. Negative results are stated in the active voice and without apology: "the ingredient I proposed was the worst of the three." Do not soften them with "unfortunately" — it reads as excuse-making.

## Slide 1 · Title — 0:15 (cumulative 0:15)

**Say**

Good morning, I'm Zikun. I built a study assistant for adult online learners, and my question was not whether a small model can answer questions. It was whether it can show its evidence, and whether that evidence is real.

**中文提示**

- 只做两件事：报名字，抛出“证据是否真实”这个钩子。不要念副标题，不要念学号。
- 说完停半秒再切页，切页时保持沉默。

## Slide 2 · The learner, not the model — 1:15 (cumulative 1:30)

**Say**

Let me start with the learner rather than the model.

The people this is built for study asynchronously. They are time-poor, they are often working at eleven at night, there is no tutor in the room and no cohort to ask, and when feedback finally arrives it comes weeks after the question was asked. For them the useful thing is not more content. It is an answer they can check, at a depth that matches what they already know.

That second half is Addanki's argument, and I want to be careful about how I use it. Academagogy says engagement comes from the system responding to the individual learner, not from delivering more content. I am using that as motivation, not claiming I have solved it. This project touches one narrow slice of it — explanation depth — and later in this talk I will show you that even that slice did not work.

**中文提示**

- 左边讲人，右边讲要求；说到 “check” 时手指右侧第一条。
- 最后一句主动承认画像失败，是给第 8 页埋伏笔，不要删。这一句能让整场从“卖方案”变成“做研究”。

## Slide 3 · Why retrieval alone is not enough — 1:25 (cumulative 2:55)

**Say**

So why is retrieval not enough on its own? Three reasons.

First, retrieval is wrong, not random. A dense retriever returns passages that are topically close to the question, and Shi and colleagues showed in 2023 that topically close but wrong evidence is exactly the condition in which language models are most easily misled — because the wrong passage looks like the right one.

Second, there is no provenance. A fluent answer that the learner cannot check is not study support. It is a plausible-sounding guess with good grammar.

Third — and this one I measured rather than read. I asked the prompted baseline to cite its sources, with five retrieved passages sitting in its context window. It produced a quotation in 8.6 per cent of answers. Only 3.6 per cent of answers contained a quote that actually exists in those passages. So among the answers that did quote, more than half quoted something that is not there. Prompting a model to cite does not make it cite.

**中文提示**

- 三张卡逐一指，每张一句，不要展开。第三个数字是全场钩子，说完停一秒。
- 如果这一页超时，把 Second 压成一句：“没有出处，学生就无法核对，那只是语法正确的猜测。”
- 讲第三个时不要看屏幕，看着评委说 “I measured rather than read”。

## Slide 4 · Research question — 0:45 (cumulative 3:40)

**Say**

That gives me the research question. Can retrieval-augmented fine-tuning turn a retrieval-based study assistant from fluent into checkable, on a consumer GPU, without losing accuracy?

Three contributions. One: I adapt RAFT so that the distractors are retrieved hard negatives — the kind of wrong passage a real retriever actually returns, rather than a random one. Two: a learner-profile channel that connects the implementation to academagogy, and is evaluated as a controlled variable instead of asserted. Three: an evaluation that separates accuracy from citation quality, verifies grounding by verbatim substring match, and ablates what actually causes the gain.

**中文提示**

- 这一页只交代“我要做什么”，一个字都不要提结果。
- 手从左到右扫三张卡，每张一句。第三张说完直接切页，不要停顿解释。

## Slide 5 · Data — 1:10 (cumulative 4:50)

**Say**

The data is SciQ: 13.7 thousand science exam questions, and — this is why I chose it — every question comes with the passage it was written from. Without a gold passage, the question "did the model cite real evidence" cannot be answered at all.

After deduplication the corpus is 12,231 passages, about 930 thousand tokens, mean 76 tokens per passage. I built 3,000 RAFT training samples and held out 500 questions, and the question sets are disjoint across splits.

One judgement call I want to own. 11.6 per cent of SciQ test items have an empty support passage. They cannot be answered from the corpus by construction, so I excluded them and evaluated on the 884 answerable items, of which 500 are held out. And one limitation I will repeat later: here the corpus and the gold evidence are the same set of passages, so the recall I report is an upper bound, not what a real course site would give you.

**中文提示**

- 顺序不能反：先讲“为什么是 SciQ”（自带金标准段落），再报三个数字。
- 右上橙色卡是主动认错，语气放平，不要说成“小问题”或“影响不大”。
- “upper bound” 这个词要说清楚，它是后面第 14 页结论成立的前提。

## Slide 6 · Architecture — 1:00 (cumulative 5:50)

**Say**

Here is the system, and it has two inputs. A question goes through a small dense retriever, bge-small, which returns the top five passages. A learner profile — novice, intermediate or advanced — goes through a profile block. Both are handed to Qwen 2.5 at 1.5 billion parameters, with LoRA adapters trained on RAFT data. The output has two sections: a reason section that quotes the evidence and rejects the distractors, and a short answer.

The format is not cosmetic. The quote markers are what make grounding checkable by substring: I can take the text between the quote marks and ask whether it literally occurs in the passage. That is the entire measurement strategy, and it is why the training targets had to be built the way I will describe next.

**中文提示**

- 顺箭头走一遍即可，不要念方框里的字。重点强调“两个输入”：证据 + 学习者。
- 最后一句是第 7 页的引子，必须说出来，否则第 7 页会显得突兀。

## Slide 7 · RAFT data construction — 1:15 (cumulative 7:05)

**Say**

Each training sample has one gold document that contains the answer, and four distractors. With probability 0.8 the gold document is kept; with probability 0.2 it is removed, which is how RAFT teaches a model to answer from its own knowledge, or to abstain.

I made two deliberate deviations from the paper. First, two of the four distractors are retrieved hard negatives rather than random passages, so the model trains against the noise a real retriever produces instead of generic noise. Second, the chain-of-thought targets are generated programmatically, and a quote marker is emitted only when the span is verbatim in the document. If it is not verbatim, the reasoning is phrased as inference and there is no quote at all.

That second decision is the integrity-critical one. If my training targets contained invented quotes, then my grounding metric would be measuring my template, not my model. The cost is that templated traces are less diverse than GPT-4 traces, and that is the biggest weakness in this project. It is on the limitations slide, and I am not going to hide it.

**中文提示**

- 左边图只指三下：金标准、两条 hard negative、两条随机。不要逐个念编号。
- “integrity-critical” 重读。承认模板化代价时看观众，不要看屏幕。
- 这里不要提前剧透 A1/A2/A3 的结果，留给第 13 页。

## Slide 8 · Learner-profile conditioning — 0:45 (cumulative 7:50)

**Say**

The profile channel is the academagogy link, and it is a thin one. The learner declares a profile, and the instruction changes: a novice gets step-by-step reasoning with terms defined, an intermediate gets a short chain, an advanced learner gets the discriminating detail only. It is held constant across the main comparison and varied only in the personalisation experiment.

Two honest caveats. The profile is declared, not inferred from behaviour. And the sweep found no usable effect — 53.8 tokens for the novice condition, 53.9 for the advanced one — so this is a design description, not a working feature.

**中文提示**

- 这一页是“设计说明”，不是“功能演示”，语气必须克制。
- 数字提前给出，避免第 13 页之后再被问一次。两个数字只差 0.1，说出来反而更有说服力。

## Slide 9 · Training setup — 0:45 (cumulative 8:35)

**Say**

Training runs on one 8 gigabyte laptop GPU: 4-bit NF4 with double quantisation, LoRA rank 16 on all linear layers, 18.5 million trainable parameters — 1.18 per cent of the model — one epoch, 188 steps, peak memory 6.9 gigabytes.

Two bugs were caught during verification, and both matter. The prompt was passed as a flat string, so the model continued the passages instead of answering them. And the evaluation set forced the gold passage to rank one, so retrieval looked perfect when it wasn't. Both were found by reading individual outputs rather than aggregates, and every number produced before the fixes was thrown away.

**中文提示**

- 数字快读，不逐条解释。重点是“两个 bug 是被查出来的”这一句。
- 这句话直接回应评委心里的“你的数字可信吗”，说完看一眼评委，停一秒。

## Slide 10 · Evaluation design — 0:45 (cumulative 9:20)

**Say**

Four conditions share one prompt and one parser: base closed-book, base with retrieval, the fine-tuned model closed-book, and the fine-tuned model with retrieval — that last one is the treatment. Plus an oracle condition where the gold passage is guaranteed to be in context, which removes retrieval failure entirely, and three ablations.

The metric I care about is verified correct: the answer is right, and it is supported by a quote that literally exists in the gold passage. Accuracy alone rewards fluency. This rewards an answer a learner can actually check, and I would ask you to read that column and not the accuracy column.

**中文提示**

- 必须先定义指标再看结果表，顺序反了观众会读错列。
- 手指右侧蓝卡说 “this one”，然后直接切第 11 页，不要在这里报任何数字。

## Slide 11 · Main results — 1:25 (cumulative 10:45)

**Say**

Here is the table. Reading across: the base model closed-book gets 42.6 per cent by containment; give it retrieval and it goes to 69.2. Fine-tuning alone goes the other way — 34.0, below the closed-book baseline.

That surprised me, and the explanation is in the exact-match column. Exact match rose from 28 to 30 per cent while containment fell. The model learned the output format and became terser, so it stopped happening to mention the right string inside a longer sentence. That is a formatting effect, not knowledge.

Together they reach 72.4 per cent. And the oracle condition, where retrieval cannot fail, reaches 74.6 — so even with perfect retrieval there is headroom, but not much.

Now the blue bars, which are the reason the project exists. Verified correctness: 1.4 per cent for the retrieval baseline, 40.4 per cent for the treatment. Same retriever, same five passages, same prompt. The only difference is that one of the two models was trained to quote what it read.

**中文提示**

- 只读三个数：42.6 → 69.2 → 72.4，其余让表自己说话。
- C3 的反常必须解释清楚（格式效应），否则会被当成“微调失败”。
- 蓝色柱是全场重点。说完 “trained to quote what it read” 后停两秒再切页。

## Slide 12 · Citation quality — 1:05 (cumulative 11:50)

**Say**

This is the slide I would like you to remember.

On the left, the prompted baseline: it quotes in 8.6 per cent of answers and grounds 3.6 per cent. On the right, the trained model: it quotes in 99.6 per cent of answers, and 99.6 per cent of answers contain a quote that literally exists in the supplied passages. Fabricated citations go from 5 per cent to zero. Verified correctness goes from 1.4 to 40.4.

The sentence I would put on the slide if I had room is this: prompting for citations does not produce citations. Training does. And what training buys is not better science knowledge — the accuracy gain is modest — it is that the answer becomes auditable. A learner can take the quoted sentence, find it in the passage, and decide for themselves whether to trust the rest.

**中文提示**

- 全场最慢的一页。“prompting does not produce citations” 和 “Training does” 分成两句说，中间停。
- 不要在这一页加新信息，只把两组对比说清楚。左右手指向要明确。
- “auditable” 是这一页的关键词，比 “accurate” 重要。

## Slide 13 · What causes the gain — 1:00 (cumulative 12:50)

**Say**

This is the slide where my argument changed, and I want to be straight about it. Two things fall out of the ablations.

First, the chain-of-thought targets are what buy the citations. Remove them and accuracy goes up to 78.6 per cent — but the model emits 2.4 tokens per answer, which is just the answer phrase, and it quotes nothing at all. So that accuracy is not a better tutor. It is a shorter answer.

Second, the hard negatives do not buy discrimination. Verified correctness is 50.0 per cent with gold-only training, 48.2 with random distractors, and 40.4 with my retrieved hard negatives. The ingredient I proposed was the worst of the three. I am reporting that rather than burying it, and the honest reading is that at 1.5 billion parameters, on this corpus, hard negatives are not a free improvement.

**中文提示**

- 语气要稳，不要辩解。“my ingredient was the worst of the three” 是加分句，慢慢说。
- 2.4 token 这个细节必须说，否则 78.6% 会被误读成“去掉 CoT 更好”。
- 如果被追问 “so was RAFT wrong?”，答案在问答区第一条，不要在这里展开。

## Slide 14 · Source selection is the ceiling — 0:30 (cumulative 13:20)

**Say**

So what is left? Fabrication is solved: zero fabricated citations, every quote real. What is not solved is choosing which passage to quote. The gold passage is cited in only 47.8 per cent of answers. And when the gold passage is missing, the model quotes a distractor 98.6 per cent of the time and abstains 1.4 per cent of the time — the no-gold training did not teach refusal. The bottleneck has moved from making quotes up to choosing sources.

**中文提示**

- 这是“最有用的负结果”。语速可以快，但 98.6% 要重读。
- 时间紧的话，这一页只讲最后一句：“瓶颈从编造引用转移到了选择来源。”

## Slide 15 · Limitations — 0:40 (cumulative 14:00)

**Say**

Five limitations, quickly. Templated reasoning traces, so the model may have learned the template's discourse as much as the discrimination it was meant to learn. Corpus equals gold evidence, so recall is an upper bound. One dataset, one model, one seed, so no confidence intervals — the differences among the ablations are indicative, not significant. Profile conditioning did not work. And there is no human evaluation: I have not shown that a novice actually finds the novice explanation more useful.

**中文提示**

- 逐条点名，一条一句，不解释。第 5 条说完停一下。
- 评委最想听的是最后一条（没有人类评估），不要跳过，也不要用“未来工作”糊过去。

## Slide 16 · Reflection on peer feedback — 0:45 (cumulative 14:45)

**Say**

On peer feedback. In the week 9 seminar, [peer comment 1] came up. What I changed was [specific change], because [the reason], and the effect was [what measurably improved]. The second comment was [peer comment 2]; that one led to [specific change], for the same reason — [the reason] — and it [what improved].

**中文提示**

- 必须用你真实收到的反馈填写，不要编。评分点在于“我改了什么、为什么改、改完哪里变好了”，不在于反馈本身是否中听。
- 三行表格各一句，具体到动作。禁止出现 “the feedback was very useful” 这类空话。
- 现在这一页只有 43 词；填完两条真实反馈后约 105 词，正好 45 秒，不要再多。
- 句式可以直接套：comment → change → because → effect。哪怕只有两条，讲清楚也比凑三条强。

## Slide 17 · Conclusion — 0:15 (cumulative 15:00)

**Say**

To close: RAFT makes a small model's answers checkable. Verified correctness 1.4 to 40.4 per cent, fabricated citations to zero, under a second per query on a laptop. Next: rerank, and test whether it helps a learner. Thank you — questions.

**中文提示**

- 三个绿色数字各指一下，然后立刻说 “questions”，把时间交出去。
- 这一页念完约 20 秒，比预算多 5 秒；超时从第 15 页借，不要压缩结果页。
- 不要补充新内容，也不要道歉（“I ran out of time” 之类一律不说）。

## Anticipated questions

Answer in two sentences, then stop. Long answers invite more questions than they close.

**Q1. Why not a 7B model?**

Eight gigabytes of VRAM. A 7B model at 4-bit leaves no room for a usable batch size or for gradient checkpointing at throughput that fits the time budget. 1.5B was the largest model I could train and evaluate inside eight weeks — and the point of the project is the method, which transfers.

中文要点：显存是硬约束；方法可迁移，规模不是本项目的贡献点。

**Q2. Why is Recall@5 only 85.8 per cent, and why does deeper retrieval make things worse?**

Short passages, single-vector dense retrieval, no reranking — a cross-encoder reranker is the obvious next step and was deliberately left out so that the retrieval component stayed fixed across conditions. On depth: verified correctness falls monotonically as k grows, 47.6 per cent at k=1, 43.8 at k=3, 40.4 at k=5, 37.4 at k=8. More passages means more plausible distractors, and the model's weak point is source selection, not recall.

中文要点：召回不是瓶颈，选源才是。k 越大可核实性越低，这正好印证第 14 页的结论。

**Q3. Isn't the templated chain-of-thought cheating?**

It constrains what the model can learn: it cannot learn to reason in ways the template does not express, and that is a real limitation. But what it does learn — which passage to trust, and to quote it verbatim — is exactly what the evaluation measures. The alternative, GPT-4 traces, would have been more diverse and would have cost me the ability to guarantee that every quoted span is real.

中文要点：承认约束，指出“模板保证引文真实”是设计取舍，不是取巧。

**Q4. Isn't "contains" easy to game by being verbose?**

Yes, which is why exact match and F1 are reported beside it, and why output length is reported too. C4 is shorter than the baseline per answer, not longer — 53.7 tokens against 27.6 for C2 in absolute terms, but the containment gain survives when you switch to exact match, which cannot be gamed by length.

中文要点：所以同时报 EM/F1；EM 从 38.0 涨到 67.8，长度解释不了这个涨幅。

**Q5. Does the profile channel work?**

No. The novice condition produced 53.8 tokens per answer and the advanced condition 53.9 — no verbosity ordering at all, so the channel is reported as not working. My reading is that depth has to be varied in the training targets, not just declared in the prompt; prompting alone does not move a model that was trained on one fixed depth.

中文要点：直接说“没有效果”，然后给出机制解释，不要含糊其辞。

**Q6. If the no-CoT ablation is 78.6 per cent accurate, why not ship that?**

Because it emits 2.4 tokens — just the answer phrase — and quotes nothing. Its accuracy is inflated by being short, and there is nothing for a learner to check. It would be a better exam-answer machine and a worse tutor, which is the opposite of the goal.

中文要点：准确率高是因为答得短，没有可核对的内容；本项目的目标是可核查而非应试。

**Q7. What would you do with another month?**

Rerank before generating, and change the target template so it never names the distractors — the current template teaches the model to discuss them, and that is part of why it cites them. Then a corpus that looks like a real course site, and a small human study on whether the profile-conditioned explanations help.

中文要点：三件事，按优先级说，不要列五条。

## Rescue card — one line per slide

If you lose your place, read the line and the slide will carry you.

| # | One line |
|---|---|
| 1 | I built a study assistant, and I measured whether its evidence is real. |
| 2 | The learner needs an answer they can check, at the right depth. |
| 3 | Prompting a model to cite does not make it cite. |
| 4 | Can RAFT make it checkable without losing accuracy? |
| 5 | SciQ gives every question its evidence, so citation becomes measurable. |
| 6 | Two inputs: the passages, and the learner. |
| 7 | Quotes are emitted only when they are verbatim. |
| 8 | The profile is declared, not inferred — and it did not work. |
| 9 | 8 GB, 188 steps, and two bugs found by reading outputs. |
| 10 | Verified correct: right and checkable. |
| 11 | 42.6, 69.2, 72.4 — and 1.4 to 40.4 on the blue bars. |
| 12 | Prompting does not produce citations. Training does. |
| 13 | CoT buys the citations; hard negatives bought nothing. |
| 14 | Grounding is solved; source selection is not. |
| 15 | Five things this does not establish. |
| 16 | Here is what I changed, and why. |
| 17 | Checkable, at under a second per query. |

## Time control

- Normal run: 15:00 exactly, ending on "questions".
- If you are at 12:00 by slide 11: drop the C3 explanation on slide 11 to one clause, cut slide 8 to two sentences, and take slide 15 down to three limitations. Do not cut slides 11 to 14.
- If you are at 13:00 by slide 12: slide 14 becomes one sentence, slide 16 becomes two rows, slide 15 becomes a list read aloud in eight seconds.
- If you are running early, the only place to expand is slide 13 — explain why RAFT's own ablation varies distractor presence, not distractor hardness.
- Rehearse twice with a timer. The second run should be 30 seconds under, so that nerves do not push you over.

## Before you present

- Title slide, report title pages and the AI-use forms already carry Zikun Wu / 15115787 / github.com/zikunwu6-ship-it.
- Fill slide 16 with feedback you actually received, and update the Say block to match.
- Say the numbers out loud once — "forty point four" reads differently from "40.4", and the spoken form is what the audience hears.
