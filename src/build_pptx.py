"""
Build the 15-minute presentation deck.

16:9, one accent colour, no template stock imagery. Diagrams are drawn with
native shapes so they stay editable in PowerPoint; result charts are the same
matplotlib PNGs used in the report, so the numbers cannot diverge between the two
deliverables.
"""
from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "docs" / "figures"
OUT = ROOT / "docs" / "CP5403_Presentation.pptx"

# ---------------- palette ----------------
INK = RGBColor(0x1A, 0x1D, 0x23)
MUTED = RGBColor(0x5A, 0x62, 0x70)
ACCENT = RGBColor(0x1F, 0x6F, 0xEB)
DEEP = RGBColor(0x14, 0x2B, 0x4A)
GOOD = RGBColor(0x0E, 0x9F, 0x6E)
WARN = RGBColor(0xC2, 0x41, 0x0C)
LIGHT = RGBColor(0xF3, 0xF6, 0xFA)
RULE = RGBColor(0xD8, 0xDE, 0xE6)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

W, H = Inches(13.333), Inches(7.5)
MARGIN = Inches(0.85)

FONT = "Calibri"
SERIF = "Georgia"


def load(name):
    p = RES / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def val(d, f="contains", k="overall"):
    return d[k][f] if d and k in d and f in d[k] else None


def pc(v):
    return "—" if v is None else f"{v*100:.0f}%"


def p1(v):
    return "—" if v is None else f"{v*100:.1f}"


def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def textbox(slide, x, y, w, h, text, size=18, bold=False, color=INK, font=FONT,
            align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, italic=False, spacing=1.0):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0
    # Newlines inside the string start a new paragraph; **segments** are bold.
    for pi, line in enumerate(str(text).split("\n")):
        p = tf.paragraphs[0] if pi == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        for si, seg in enumerate(line.split("**")):
            if not seg:
                continue
            r = p.add_run()
            r.text = seg
            r.font.size = Pt(size)
            r.font.bold = bold or (si % 2 == 1)
            r.font.italic = italic
            r.font.color.rgb = color
            r.font.name = font
    return tb


def bullets(slide, x, y, w, h, items, size=17, color=INK, gap=10, bullet_color=ACCENT):
    """items: list of (text, level) or str."""
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0
    first = True
    for it in items:
        text, level = (it, 0) if isinstance(it, str) else it
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.space_after = Pt(gap)
        p.line_spacing = 1.05
        p.level = level
        marker = "▪  " if level == 0 else "–  "
        rm = p.add_run()
        rm.text = marker
        rm.font.size = Pt(size - (2 if level else 0))
        rm.font.color.rgb = bullet_color
        rm.font.name = FONT
        # inline **bold** segments
        for seg in text.split("**"):
            r = p.add_run()
            r.text = seg
            r.font.size = Pt(size - (2 if level else 0))
            r.font.color.rgb = color if level == 0 else MUTED
            r.font.name = FONT
        # bold alternate segments
        idx = 0
        for run in p.runs[1:]:
            if idx % 2 == 1:
                run.font.bold = True
            idx += 1
    return tb


def slide_header(slide, kicker, title):
    textbox(slide, MARGIN, Inches(0.42), Inches(10), Inches(0.3), kicker.upper(),
            size=10.5, bold=True, color=ACCENT)
    # Long titles wrap; at 27pt a second line runs into the accent rule below, so
    # the size steps down rather than the layout moving.
    size = 27 if len(title) <= 50 else 23
    textbox(slide, MARGIN, Inches(0.70), W - 2 * MARGIN, Inches(0.7), title,
            size=size, bold=True, color=DEEP, font=SERIF)
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, MARGIN, Inches(1.42),
                                 Inches(1.1), Pt(3))
    line.fill.solid()
    line.fill.fore_color.rgb = ACCENT
    line.line.fill.background()
    line.shadow.inherit = False


def footer(slide, idx, total):
    textbox(slide, MARGIN, H - Inches(0.5), Inches(6), Inches(0.25),
            "AcademaTutor · CP5403 Assessment 2", size=9, color=MUTED)
    textbox(slide, W - MARGIN - Inches(1.2), H - Inches(0.5), Inches(1.2), Inches(0.25),
            f"{idx} / {total}", size=9, color=MUTED, align=PP_ALIGN.RIGHT)


def card(slide, x, y, w, h, fill=LIGHT, line=RULE, radius=0.04):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    sh.line.color.rgb = line
    sh.line.width = Pt(0.75)
    sh.shadow.inherit = False
    try:
        sh.adjustments[0] = radius
    except Exception:
        pass
    return sh


def arrow(slide, x1, y1, x2, y2, color=ACCENT, width=1.75):
    from pptx.enum.shapes import MSO_CONNECTOR
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x1, y1, x2, y2)
    c.line.color.rgb = color
    c.line.width = Pt(width)
    ln = c.line._get_or_add_ln()
    from pptx.oxml.ns import qn
    tail = ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"})
    ln.append(tail)
    return c


def table(slide, x, y, w, rows, col_widths=None, font_size=11.5, header_fill=DEEP,
          row_h=Inches(0.34)):
    nrows, ncols = len(rows), len(rows[0])
    shape = slide.shapes.add_table(nrows, ncols, x, y, w, row_h * nrows)
    tbl = shape.table
    if col_widths:
        total = sum(col_widths)
        for i, cw in enumerate(col_widths):
            tbl.columns[i].width = Emu(int(w * cw / total))
    for r in range(nrows):
        tbl.rows[r].height = row_h
        for c in range(ncols):
            cell = tbl.cell(r, c)
            cell.text = ""
            cell.margin_left = Inches(0.08)
            cell.margin_right = Inches(0.06)
            cell.margin_top = Inches(0.02)
            cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
            run = p.add_run()
            run.text = str(rows[r][c])
            run.font.size = Pt(font_size if r else font_size)
            run.font.name = FONT
            run.font.bold = (r == 0)
            run.font.color.rgb = WHITE if r == 0 else INK
            cell.fill.solid()
            cell.fill.fore_color.rgb = header_fill if r == 0 else (
                WHITE if r % 2 else LIGHT)
    return tbl


def picture(slide, path, x, y, w=None, h=None):
    if not Path(path).exists():
        return None
    kw = {}
    if w:
        kw["width"] = w
    if h:
        kw["height"] = h
    return slide.shapes.add_picture(str(path), x, y, **kw)


def kpi(slide, x, y, w, value, label, color=ACCENT, vsize=40):
    textbox(slide, x, y, w, Inches(0.7), value, size=vsize, bold=True, color=color,
            font=SERIF)
    textbox(slide, x, y + Inches(0.62), w, Inches(0.5), label, size=11.5, color=MUTED)


# --------------------------------------------------------------------------- #
def build():
    c1, c2, c3 = load("base_closed"), load("base_rag"), load("raft_closed")
    c4, c5 = load("raft_rag"), load("raft_oracle")
    a1, a2, a3 = load("sft_gold"), load("raft_nocot"), load("raft_random")

    summary = json.loads((RES / "summary.json").read_text(encoding="utf-8")) \
        if (RES / "summary.json").exists() else {}
    ret = summary.get("retrieval", {})
    r5 = ret.get("recall@5")
    miss = None if r5 is None else 1 - r5

    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    total = 17
    idx = 0

    # ---------- 1 title ----------
    idx += 1
    s = blank(prs)
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H)
    bg.fill.solid()
    bg.fill.fore_color.rgb = DEEP
    bg.line.fill.background()
    bg.shadow.inherit = False

    textbox(s, MARGIN, Inches(1.75), Inches(11), Inches(0.4), "CP5403  ·  ASSESSMENT 2  ·  LLM PROJECT",
            size=12, bold=True, color=RGBColor(0x8F, 0xB6, 0xE8))
    textbox(s, MARGIN, Inches(2.25), Inches(10.6), Inches(1.5), "AcademaTutor",
            size=54, bold=True, color=WHITE, font=SERIF)
    textbox(s, MARGIN, Inches(3.45), Inches(10.6), Inches(1.2),
            "Citation-grounded study support for adult online learners",
            size=25, color=RGBColor(0xD8, 0xE4, 0xF5), font=SERIF)
    textbox(s, MARGIN, Inches(4.25), Inches(10.6), Inches(0.8),
            "Adapting Retrieval-Augmented Fine-Tuning to a course corpus",
            size=16, italic=True, color=RGBColor(0x9F, 0xB8, 0xD8))
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, MARGIN, Inches(5.25), Inches(1.6), Pt(3))
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT
    bar.line.fill.background()
    textbox(s, MARGIN, Inches(5.6), Inches(9), Inches(0.4),
            "Zikun Wu  ·  15115787  ·  November 2026", size=13,
            color=RGBColor(0xB8, 0xCB, 0xE3))
    textbox(s, MARGIN, Inches(6.15), Inches(9), Inches(0.4),
            "github.com/zikunwu6-ship-it/cp5403-a2-raft-rag", size=11,
            color=RGBColor(0x7F, 0x9C, 0xC4))

    # ---------- 2 the learner ----------
    idx += 1
    s = blank(prs)
    slide_header(s, "Motivation", "The problem is the learner, not the model")
    card(s, MARGIN, Inches(1.85), Inches(5.3), Inches(3.9), fill=LIGHT)
    textbox(s, MARGIN + Inches(0.3), Inches(2.05), Inches(4.7), Inches(0.35),
            "ADULT ONLINE LEARNER", size=10.5, bold=True, color=ACCENT)
    bullets(s, MARGIN + Inches(0.3), Inches(2.45), Inches(4.7), Inches(3.1), [
        "Asynchronous, time-poor, often studying at night",
        "No tutor in the room, no cohort to ask",
        "Feedback arrives weeks after the question",
        "Needs to **check** an answer, not just receive one",
    ], size=15)

    card(s, Inches(6.55), Inches(1.85), Inches(5.35), Inches(3.9), fill=RGBColor(0xEC, 0xF3, 0xFE))
    textbox(s, Inches(6.85), Inches(2.05), Inches(4.8), Inches(0.35),
            "WHAT THAT IMPLIES FOR AN AI TUTOR", size=10.5, bold=True, color=ACCENT)
    bullets(s, Inches(6.85), Inches(2.45), Inches(4.8), Inches(3.1), [
        "An answer must be **traceable to a passage**",
        "It must resist plausible-but-wrong evidence",
        "Depth must match what the learner already knows",
    ], size=15)

    textbox(s, MARGIN, Inches(6.05), W - 2 * MARGIN, Inches(0.9),
            "Addanki's academagogy: engagement depends on the system responding to the individual "
            "learner, not on delivering more content  (Addanki et al., 2020, 2022).",
            size=13.5, italic=True, color=MUTED)
    footer(s, idx, total)

    # ---------- 3 why RAG alone fails ----------
    idx += 1
    s = blank(prs)
    slide_header(s, "The gap", "Why retrieval alone is not enough")
    items = [
        ("1", "Retrieval is wrong, not random",
         "Dense retrievers return topically close passages — exactly when models are most easily misled (Shi et al., 2023)."),
        ("2", "No provenance",
         "A fluent answer the learner cannot check is not study support; it is a plausible-sounding guess."),
        ("3", "Fabricated citations",
         f"Measured here: the prompted baseline emits quotes in {pc(val(c2,'quote_rate'))} of answers, and only {pc(val(c2,'grounded_rate'))} of responses contain a quote that actually exists."),
    ]
    y = Inches(1.85)
    for num, head, body in items:
        card(s, MARGIN, y, W - 2 * MARGIN, Inches(1.32), fill=WHITE)
        circ = s.shapes.add_shape(MSO_SHAPE.OVAL, MARGIN + Inches(0.28), y + Inches(0.34),
                                  Inches(0.62), Inches(0.62))
        circ.fill.solid()
        circ.fill.fore_color.rgb = ACCENT if num != "3" else WARN
        circ.line.fill.background()
        tf = circ.text_frame
        tf.margin_left = tf.margin_right = 0
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = num
        r.font.size = Pt(20)
        r.font.bold = True
        r.font.color.rgb = WHITE
        r.font.name = SERIF
        textbox(s, MARGIN + Inches(1.1), y + Inches(0.2), Inches(9.6), Inches(0.35),
                head, size=16.5, bold=True, color=DEEP)
        textbox(s, MARGIN + Inches(1.1), y + Inches(0.6), Inches(10.4), Inches(0.65),
                body, size=13, color=MUTED)
        y += Inches(1.52)
    footer(s, idx, total)

    # ---------- 4 RQ ----------
    idx += 1
    s = blank(prs)
    slide_header(s, "Research question", "Can RAFT make a RAG tutor checkable?")
    textbox(s, MARGIN, Inches(1.8), W - 2 * MARGIN, Inches(1.0),
            "On a consumer GPU, without losing accuracy — and with the explanation depth "
            "adapted to the learner.", size=17, color=MUTED, italic=True)
    heads = [
        ("Adaptation", "Distractors are retrieved hard negatives, so training matches a real retriever's failure mode."),
        ("Learner channel", "A profile block conditions explanation depth — the academagogy link, evaluated as a controlled variable."),
        ("Evaluation", "Accuracy separated from citation quality; grounding verified by verbatim substring match; ablations on what causes the gain."),
    ]
    x = MARGIN
    cw = (W - 2 * MARGIN - Inches(0.5)) / 3
    for head, body in heads:
        card(s, x, Inches(2.95), cw, Inches(3.0), fill=LIGHT)
        textbox(s, x + Inches(0.28), Inches(3.2), cw - Inches(0.56), Inches(0.4),
                head, size=15, bold=True, color=ACCENT)
        textbox(s, x + Inches(0.28), Inches(3.68), cw - Inches(0.56), Inches(2.1),
                body, size=13, color=INK)
        x += cw + Inches(0.25)
    footer(s, idx, total)

    # ---------- 5 data ----------
    idx += 1
    s = blank(prs)
    slide_header(s, "Data", "SciQ: questions that come with their evidence")
    textbox(s, MARGIN, Inches(1.75), Inches(6.3), Inches(0.9),
            "Each of 13.7k science exam questions carries the passage it was written from. "
            "Without that, \u201cdid the model cite real evidence?\u201d is unanswerable.",
            size=14.5, color=MUTED)
    kpi(s, MARGIN, Inches(2.85), Inches(2.4), "12,231", "passages in the corpus", ACCENT, 34)
    kpi(s, MARGIN + Inches(2.6), Inches(2.85), Inches(2.4), "930k", "tokens · mean 76", ACCENT, 34)
    kpi(s, MARGIN + Inches(5.2), Inches(2.85), Inches(2.4), "500", "held-out questions", ACCENT, 34)

    card(s, Inches(7.55), Inches(1.75), Inches(4.35), Inches(3.9), fill=RGBColor(0xFD, 0xF3, 0xEC))
    textbox(s, Inches(7.8), Inches(1.95), Inches(3.9), Inches(0.35),
            "A JUDGEMENT CALL", size=10.5, bold=True, color=WARN)
    textbox(s, Inches(7.8), Inches(2.35), Inches(3.9), Inches(3.1),
            "11.6% of SciQ test items have an empty support passage. They cannot be answered "
            "from the corpus by construction, so they are excluded from the evaluation pool "
            "(884 answerable items).\n\n"
            "Also noted: the corpus and the gold evidence are the same passages, so recall here "
            "is an upper bound.", size=12.5, color=INK)

    textbox(s, MARGIN, Inches(6.0), Inches(6.3), Inches(0.8),
            "3,000 RAFT training samples · 500 evaluation questions · questions disjoint across splits",
            size=12, color=MUTED)
    footer(s, idx, total)

    # ---------- 6 architecture ----------
    idx += 1
    s = blank(prs)
    slide_header(s, "System", "Architecture")

    def box(x, y, w, h, label, fill=LIGHT, color=INK, size=12.5, bold=True):
        sh = card(s, x, y, w, h, fill=fill)
        tf = sh.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = Inches(0.08)
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = label
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = color
        r.font.name = FONT
        return sh

    # three rows: question -> retriever -> passages -> model
    #             corpus feeds the retriever
    #             profile -> profile block -> model
    box(MARGIN, Inches(1.95), Inches(2.1), Inches(0.85), "Learner question")
    box(Inches(3.55), Inches(1.95), Inches(2.6), Inches(0.85), "Retriever\nbge-small-en-v1.5")
    box(Inches(6.75), Inches(1.95), Inches(2.15), Inches(0.85), "top-5 passages")

    box(Inches(3.55), Inches(3.05), Inches(2.6), Inches(0.8), "Corpus\n12,231 passages",
        fill=RGBColor(0xE4, 0xEA, 0xF2), color=MUTED, size=12)

    box(MARGIN, Inches(4.15), Inches(2.1), Inches(0.85), "Learner profile")
    box(Inches(3.55), Inches(4.15), Inches(2.6), Inches(0.85), "profile block")

    box(Inches(9.95), Inches(1.95), Inches(2.55), Inches(3.05),
        "Qwen2.5-1.5B\n+ LoRA\n(RAFT-tuned)", fill=RGBColor(0xE7, 0xF0, 0xFE),
        color=DEEP, size=13.5)

    box(Inches(7.6), Inches(5.35), Inches(4.9), Inches(1.05),
        "##Reason:  quote the evidence, reject the distractors\n##Answer:  <short phrase>",
        fill=RGBColor(0xF7, 0xF9, 0xFC), size=11.5)

    arrow(s, MARGIN + Inches(2.1), Inches(2.37), Inches(3.55), Inches(2.37))
    arrow(s, Inches(4.85), Inches(3.05), Inches(4.85), Inches(2.8), color=MUTED)
    arrow(s, MARGIN + Inches(2.1), Inches(4.57), Inches(3.55), Inches(4.57))
    arrow(s, Inches(6.15), Inches(2.37), Inches(6.75), Inches(2.37))
    arrow(s, Inches(8.9), Inches(2.37), Inches(9.95), Inches(2.6))
    arrow(s, Inches(6.15), Inches(4.57), Inches(9.95), Inches(4.2))
    arrow(s, Inches(11.22), Inches(5.0), Inches(11.22), Inches(5.35))

    textbox(s, MARGIN, Inches(6.62), W - 2 * MARGIN, Inches(0.6),
            "The output format is not cosmetic: the quote markers are what make grounding checkable by substring.",
            size=13, italic=True, color=MUTED)
    footer(s, idx, total)

    # ---------- 7 RAFT data ----------
    idx += 1
    s = blank(prs)
    slide_header(s, "Method", "RAFT data construction")
    textbox(s, MARGIN, Inches(1.72), Inches(6.0), Inches(0.4),
            "1 gold document + 4 distractors · P = 0.8 keeps gold", size=14, bold=True, color=DEEP)
    y = Inches(2.25)
    gold = card(s, MARGIN, y, Inches(6.0), Inches(0.72), fill=RGBColor(0xE6,0xF7,0xEE),
                line=GOOD)
    textbox(s, MARGIN + Inches(0.2), y + Inches(0.2), Inches(5.6), Inches(0.35),
            "[1]  GOLD — contains the answer", size=12.5, bold=True, color=GOOD)
    y += Inches(0.82)
    for i in range(2, 6):
        lab = ("[%d]  hard negative — retrieved, topically close" % i) if i <= 3 else \
              ("[%d]  random passage" % i)
        card(s, MARGIN, y, Inches(6.0), Inches(0.62), fill=LIGHT)
        textbox(s, MARGIN + Inches(0.2), y + Inches(0.17), Inches(5.6), Inches(0.3),
                lab, size=11.5, color=MUTED)
        y += Inches(0.7)

    textbox(s, Inches(7.2), Inches(1.72), Inches(4.9), Inches(0.4),
            "Two deliberate deviations", size=14, bold=True, color=DEEP)
    bullets(s, Inches(7.2), Inches(2.2), Inches(4.9), Inches(3.4), [
        "**Retrieved hard negatives** instead of random ones — trains against the noise a real retriever produces, not generic noise.",
        "**Programmatic CoT targets.** A quote marker is emitted only when the span is verbatim in the document. Otherwise the reasoning is phrased as inference, with no quote.",
    ], size=13)
    textbox(s, Inches(7.2), Inches(5.35), Inches(4.9), Inches(1.2),
            "Why the second one matters: if the targets contained invented quotes, the grounding metric "
            "would be measuring the template, not the model.",
            size=12, italic=True, color=WARN)
    footer(s, idx, total)

    # ---------- 8 profile ----------
    idx += 1
    s = blank(prs)
    slide_header(s, "Method", "Learner-profile conditioning")
    textbox(s, MARGIN, Inches(1.72), W - 2 * MARGIN, Inches(0.5),
            "The same question, three declared profiles — held constant across the main comparison, "
            "varied only in the personalisation experiment.", size=13.5, color=MUTED)
    rows = [
        ["Profile", "Instruction", "Expected effect"],
        ["Novice", "step by step, define every term", "longest"],
        ["Intermediate", "short chain of reasoning", "baseline"],
        ["Advanced", "terse; the discriminating detail only", "shortest"],
    ]
    table(s, MARGIN, Inches(2.45), Inches(7.6), rows, col_widths=[1.5, 3.4, 1.6], font_size=12.5,
          row_h=Inches(0.45))
    card(s, Inches(8.9), Inches(2.45), Inches(3.0), Inches(2.0), fill=RGBColor(0xFD,0xF3,0xEC))
    textbox(s, Inches(9.1), Inches(2.62), Inches(2.6), Inches(1.7),
            "Honest scope\n\nThe profile is declared by the learner, not inferred from behaviour.",
            size=11.5, color=WARN)
    textbox(s, MARGIN, Inches(4.85), W - 2 * MARGIN, Inches(1.2),
            "This is the academagogy link, and it is a thin one. Addanki's framework concerns the whole "
            "learner situation; this implementation addresses one slice — explanation depth — and asks the "
            "learner to state it. The sweep later found no usable effect, so the channel is reported as "
            "not working rather than as a working feature.", size=13.5, color=INK)
    footer(s, idx, total)

    # ---------- 9 training ----------
    idx += 1
    s = blank(prs)
    slide_header(s, "Implementation", "Training setup — 8 GB laptop GPU")
    left = [
        ("Quantisation", "4-bit NF4, double quantisation"),
        ("Adapters", "LoRA r=16, α=32, all linear layers"),
        ("Trainable", "18.5M params — 1.18% of the model"),
        ("Optimiser", "paged AdamW 8-bit, lr 2e-4 cosine"),
        ("Schedule", "1 epoch, effective batch 16, max len 1152"),
        ("Loss", "target tokens only; prompt masked to −100"),
    ]
    y = Inches(1.85)
    for k, v in left:
        textbox(s, MARGIN, y, Inches(2.3), Inches(0.3), k.upper(), size=10, bold=True, color=ACCENT)
        textbox(s, MARGIN + Inches(2.4), y - Inches(0.03), Inches(4.4), Inches(0.35), v,
                size=13, color=INK)
        y += Inches(0.62)
    card(s, Inches(7.6), Inches(1.85), Inches(4.3), Inches(3.5), fill=LIGHT)
    textbox(s, Inches(7.85), Inches(2.05), Inches(3.8), Inches(0.35),
            "TWO BUGS CAUGHT IN VERIFICATION", size=10.5, bold=True, color=WARN)
    bullets(s, Inches(7.85), Inches(2.45), Inches(3.8), Inches(2.8), [
        "Prompt passed as a flat string → model continued the passages instead of answering.",
        "Evaluation set forced the gold passage to rank 1 → retrieval looked perfect.",
    ], size=12)
    textbox(s, MARGIN, Inches(5.75), Inches(6.4), Inches(0.9),
            "Both were found by inspecting per-example outputs, not the aggregate. Every number produced "
            "before the fixes was discarded.", size=12, italic=True, color=MUTED)
    footer(s, idx, total)

    # ---------- 10 evaluation design ----------
    idx += 1
    s = blank(prs)
    slide_header(s, "Evaluation", "Four conditions, one parser, one metric set")
    rows = [
        ["", "Retrieval", "Fine-tuned", "Role"],
        ["C1  Base, closed", "—", "no", "parametric knowledge floor"],
        ["C2  Base + RAG", "top-5", "no", "the baseline everyone ships"],
        ["C3  RAFT, closed", "—", "yes", "domain fine-tuning alone"],
        ["C4  RAFT + RAG", "top-5", "yes", "the treatment"],
        ["C5  RAFT, oracle", "gold + 4", "yes", "retrieval-failure removed"],
    ]
    table(s, MARGIN, Inches(1.8), Inches(7.3), rows, col_widths=[1.9, 1.1, 1.0, 2.6],
          font_size=11.5, row_h=Inches(0.4))
    textbox(s, MARGIN, Inches(4.5), Inches(7.3), Inches(0.4),
            "Ablations: A1 gold-only SFT · A2 no chain-of-thought · A3 random distractors",
            size=11.5, color=MUTED)

    card(s, Inches(8.55), Inches(1.8), Inches(3.4), Inches(4.1), fill=RGBColor(0xE7,0xF0,0xFE))
    textbox(s, Inches(8.8), Inches(1.98), Inches(2.95), Inches(0.35),
            "THE HEADLINE METRIC", size=10.5, bold=True, color=ACCENT)
    textbox(s, Inches(8.8), Inches(2.35), Inches(2.95), Inches(0.5),
            "Verified correct", size=19, bold=True, color=DEEP, font=SERIF)
    textbox(s, Inches(8.8), Inches(2.95), Inches(2.95), Inches(2.6),
            "Answer is right\nAND\nsupported by a quote that literally exists in the gold passage.\n\n"
            "Accuracy alone rewards fluency. This rewards an answer a learner can check.",
            size=12, color=INK)
    footer(s, idx, total)

    # ---------- 11 main results ----------
    idx += 1
    s = blank(prs)
    slide_header(s, "Results", "Accuracy and verified correctness")
    picture(s, FIG / "fig_main.png", MARGIN, Inches(1.8), w=Inches(7.4))
    card(s, Inches(8.6), Inches(1.8), Inches(3.35), Inches(3.4), fill=LIGHT)
    textbox(s, Inches(8.85), Inches(1.98), Inches(2.9), Inches(0.35),
            "READ ACROSS", size=10.5, bold=True, color=ACCENT)
    bullets(s, Inches(8.85), Inches(2.35), Inches(2.9), Inches(2.7), [
        f"RAG alone: **{pc(val(c1))} → {pc(val(c2))}**",
        f"Fine-tuning alone: **{pc(val(c3))}** — below baseline",
        f"Together: **{pc(val(c4))}**",
        f"Oracle ceiling: **{pc(val(c5))}**",
    ], size=12)
    textbox(s, MARGIN, Inches(5.55), W - 2 * MARGIN, Inches(0.9),
            f"Answer accuracy {p1(val(c2))}% → {p1(val(c4))}%. The blue bars are the point: "
            f"verified correctness {p1(val(c2,'verified_correct'))}% → {p1(val(c4,'verified_correct'))}%.",
            size=14, color=INK)
    footer(s, idx, total)

    # ---------- 12 citation quality ----------
    idx += 1
    s = blank(prs)
    slide_header(s, "Results", "The key finding: prompting for citations does not produce them")
    picture(s, FIG / "fig_citation.png", MARGIN, Inches(1.75), w=Inches(7.2))
    card(s, Inches(8.4), Inches(1.75), Inches(3.55), Inches(2.5), fill=RGBColor(0xFD,0xF3,0xEC))
    textbox(s, Inches(8.62), Inches(1.95), Inches(3.1), Inches(0.35),
            "BASELINE, MEASURED", size=10.5, bold=True, color=WARN)
    textbox(s, Inches(8.62), Inches(2.3), Inches(3.1), Inches(1.8),
            f"{pc(val(c2,'quote_rate'))} of answers contain a quote.\n"
            f"Only {pc(val(c2,'grounded_rate'))} contain a quote that actually exists in the passages.",
            size=12.5, color=INK)
    card(s, Inches(8.4), Inches(4.45), Inches(3.55), Inches(1.7), fill=RGBColor(0xE6,0xF7,0xEE))
    textbox(s, Inches(8.62), Inches(4.62), Inches(3.1), Inches(0.35),
            "TREATMENT, MEASURED", size=10.5, bold=True, color=GOOD)
    textbox(s, Inches(8.62), Inches(4.97), Inches(3.1), Inches(1.1),
            f"{pc(val(c4,'grounded_rate'))} grounded · verified correct "
            f"{p1(val(c2,'verified_correct'))}% → {p1(val(c4,'verified_correct'))}%",
            size=12.5, color=INK)
    footer(s, idx, total)

    # ---------- 13 ablations ----------
    idx += 1
    s = blank(prs)
    slide_header(s, "Results", "What causes the gain — and what does not")
    picture(s, FIG / "fig_ablation.png", MARGIN, Inches(1.78), w=Inches(6.6))
    card(s, Inches(7.75), Inches(1.78), Inches(4.2), Inches(3.25), fill=RGBColor(0xFD,0xF3,0xEC))
    textbox(s, Inches(7.98), Inches(1.96), Inches(3.75), Inches(0.35),
            "THE RESULT THAT CHANGED THE ARGUMENT", size=10.5, bold=True, color=WARN)
    textbox(s, Inches(7.98), Inches(2.34), Inches(3.75), Inches(3.3),
            f"**CoT is what buys citations.** Drop it and accuracy rises to "
            f"{p1(val(a2))}%, but only {p1(val(a2,'quote_rate'))}% of answers quote anything.\n\n"
            f"**Hard negatives are not what buys discrimination.** Verified correct: "
            f"gold-only {p1(val(a1,'verified_correct'))}%, random distractors "
            f"{p1(val(a3,'verified_correct'))}%, my retrieved hard negatives "
            f"{p1(val(c4,'verified_correct'))}%.\n\n"
            "The ingredient I proposed was the worst of the three.",
            size=11.5, color=INK)
    footer(s, idx, total)

    # ---------- 14 retrieval failure ----------
    idx += 1
    s = blank(prs)
    slide_header(s, "Results", "The remaining error is source selection, not fabrication")
    picture(s, FIG / "fig_retrieval_split.png", MARGIN, Inches(1.8), w=Inches(6.2))
    card(s, Inches(7.5), Inches(1.8), Inches(4.45), Inches(3.2), fill=RGBColor(0xFD,0xF3,0xEC))
    textbox(s, Inches(7.75), Inches(2.0), Inches(3.95), Inches(0.35),
            "THE HONEST READING", size=10.5, bold=True, color=WARN)
    textbox(s, Inches(7.75), Inches(2.38), Inches(3.95), Inches(3.0),
            f"Fabricated citations: {p1(val(c4,'fabricated_rate'))}%. Every quote is real.\n\n"
            f"But it cites the gold passage in only {p1(val(c4,'gold_cited_rate'))}% of answers "
            f"and some other passage in {p1(val(c4,'distractor_cited_rate'))}%.\n\n"
            f"With the gold passage missing it quotes a distractor in 98.6% of cases and "
            f"abstains in {p1(val(c4,'abstain_rate'))}% — the no-gold training did not teach refusal.",
            size=12, color=INK)
    textbox(s, MARGIN, Inches(5.7), Inches(6.2), Inches(0.9),
            "Grounding is solved. Choosing what to ground on is not. "
            "That is the next thing to fix — and it is checkable, because the quote is verbatim.",
            size=13, italic=True, color=MUTED)
    footer(s, idx, total)

    # ---------- 15 limitations ----------
    idx += 1
    s = blank(prs)
    slide_header(s, "Limitations", "What this project does not establish")
    lims = [
        ("Templated reasoning traces", "Less diverse than GPT-4 traces; the model may have learned the template's discourse as much as the discrimination."),
        ("Corpus = gold evidence", "Recall@k here is an upper bound; a real course site is messier."),
        ("One dataset, one model, one seed", "No confidence intervals. Small differences among ablations are indicative only."),
        ("Profile conditioning failed", "No usable verbosity effect; the channel needs depth-varied training targets."),
        ("No human evaluation", "Whether a novice actually finds the novice explanation more useful is untested."),
    ]
    y = Inches(1.85)
    for head, body in lims:
        bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, MARGIN, y + Inches(0.06), Pt(3), Inches(0.62))
        bar.fill.solid()
        bar.fill.fore_color.rgb = WARN
        bar.line.fill.background()
        textbox(s, MARGIN + Inches(0.2), y, Inches(3.0), Inches(0.35), head,
                size=13.5, bold=True, color=DEEP)
        textbox(s, MARGIN + Inches(3.3), y, Inches(8.3), Inches(0.7), body,
                size=12.5, color=MUTED)
        y += Inches(0.88)
    footer(s, idx, total)

    # ---------- 16 reflection ----------
    idx += 1
    s = blank(prs)
    slide_header(s, "Reflection", "How I responded to peer feedback")
    warn = card(s, MARGIN, Inches(1.75), W - 2 * MARGIN, Inches(0.62),
                fill=RGBColor(0xFD,0xF3,0xEC), line=WARN)
    textbox(s, MARGIN + Inches(0.25), Inches(1.9), W - 2 * MARGIN - Inches(0.5), Inches(0.35),
            "TO BE COMPLETED WITH THE FEEDBACK ACTUALLY RECEIVED IN THE WEEK 9 / 10 SEMINAR",
            size=11.5, bold=True, color=WARN)

    rows = [
        ["Feedback received", "What I changed", "Why", "Effect on the project"],
        ["[peer comment 1]", "[specific change]", "[reasoning]", "[what improved]"],
        ["[peer comment 2]", "[specific change]", "[reasoning]", "[what improved]"],
        ["[peer comment 3]", "[specific change]", "[reasoning]", "[what improved]"],
    ]
    table(s, MARGIN, Inches(2.6), W - 2 * MARGIN, rows,
          col_widths=[3.2, 3.2, 2.6, 3.0], font_size=11.5, row_h=Inches(0.72))
    textbox(s, MARGIN, Inches(5.85), W - 2 * MARGIN, Inches(0.9),
            "The rubric asks for how I responded, what I changed, why, and how it improved the project — "
            "not whether the feedback was agreeable.",
            size=12.5, italic=True, color=MUTED)
    footer(s, idx, total)

    # ---------- 17 conclusion ----------
    idx += 1
    s = blank(prs)
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H)
    bg.fill.solid()
    bg.fill.fore_color.rgb = DEEP
    bg.line.fill.background()
    bg.shadow.inherit = False

    textbox(s, MARGIN, Inches(1.5), Inches(11), Inches(0.4), "CONCLUSION",
            size=12, bold=True, color=RGBColor(0x8F, 0xB6, 0xE8))
    textbox(s, MARGIN, Inches(1.95), Inches(10.8), Inches(1.4),
            "RAFT makes a small model's answers checkable.", size=34, bold=True,
            color=WHITE, font=SERIF)

    kpi(s, MARGIN, Inches(3.5), Inches(3.4), f"{p1(val(c2,'verified_correct'))}% → {p1(val(c4,'verified_correct'))}%",
        "verified correct, 500 held-out questions", RGBColor(0x6E, 0xE7, 0xB7), 30)
    kpi(s, MARGIN + Inches(3.4), Inches(3.5), Inches(3.0), f"{p1(val(c2,'fabricated_rate'))}% → {p1(val(c4,'fabricated_rate'))}%",
        "fabricated citations", RGBColor(0x6E, 0xE7, 0xB7), 30)
    lat = val(c4, "avg_latency_s")
    kpi(s, MARGIN + Inches(6.8), Inches(3.5), Inches(3.0),
        f"{lat:.2f} s" if lat else "— s",
        "per query, laptop GPU", RGBColor(0x6E, 0xE7, 0xB7), 30)

    textbox(s, MARGIN, Inches(5.2), Inches(11), Inches(1.0),
            "Next: rerank before generating, a target template that never names the distractors, "
            "depth-varied profile targets, and human evaluation of whether any of it helps a learner.",
            size=14.5, color=RGBColor(0xC8, 0xD8, 0xEE))
    textbox(s, MARGIN, Inches(6.45), Inches(6), Inches(0.5), "Questions.",
            size=20, bold=True, color=WHITE, font=SERIF)

    prs.save(str(OUT))
    print("wrote", OUT, f"({len(prs.slides)} slides)")


if __name__ == "__main__":
    build()
