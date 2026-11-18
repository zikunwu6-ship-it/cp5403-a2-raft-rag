"""
Markdown -> DOCX converter with academic-report typography.

Written because the deliverable needs exact control: the prose is final and must
be preserved verbatim, numbers must match the result files, and the layout has to
follow a conventional report structure (title page, numbered headings, captioned
tables and figures, hanging-indent references, page numbers).

Supports the subset of markdown actually used in docs/*.md:
  # ## ### headings, paragraphs, **bold**, *italic*, `code`,
  bullet and numbered lists, pipe tables, --- rules, > blockquotes,
  ![caption](path) images, and {{PLACEHOLDER}} substitution.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

BODY_FONT_LATIN = "Times New Roman"
BODY_FONT_CJK = "SimSun"
HEAD_FONT_LATIN = "Times New Roman"
HEAD_FONT_CJK = "SimHei"
CODE_FONT = "Consolas"
ACCENT = RGBColor(0x1F, 0x3A, 0x5F)


# --------------------------------------------------------------------------- #
# low-level helpers
# --------------------------------------------------------------------------- #
def set_run_fonts(run, latin=BODY_FONT_LATIN, cjk=BODY_FONT_CJK, size=None,
                  bold=None, italic=None, color=None):
    run.font.name = latin
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), latin)
    rfonts.set(qn("w:hAnsi"), latin)
    rfonts.set(qn("w:eastAsia"), cjk)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color is not None:
        run.font.color.rgb = color


def shade(cell, hex_fill: str):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tc_pr.append(shd)


def set_cell_margins(cell, top=60, bottom=60, left=90, right=90):
    tc_pr = cell._tc.get_or_add_tcPr()
    mar = OxmlElement("w:tcMar")
    for tag, val in (("top", top), ("bottom", bottom), ("start", left), ("end", right)):
        el = OxmlElement(f"w:{tag}")
        el.set(qn("w:w"), str(val))
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    tc_pr.append(mar)


def table_borders(table, sz=4, color="A6A6A6"):
    tbl_pr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), str(sz))
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), color)
        borders.append(el)
    tbl_pr.append(borders)


def add_page_number_footer(section, prefix=""):
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if prefix:
        r = p.add_run(prefix)
        set_run_fonts(r, size=9, color=RGBColor(0x66, 0x66, 0x66))
    run = p.add_run()
    set_run_fonts(run, size=9, color=RGBColor(0x66, 0x66, 0x66))
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_end)


def hr(paragraph):
    p_pr = paragraph._p.get_or_add_pPr()
    pbdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "BFBFBF")
    pbdr.append(bottom)
    p_pr.append(pbdr)


# --------------------------------------------------------------------------- #
# inline markdown
# --------------------------------------------------------------------------- #
INLINE = re.compile(r"(\*\*.+?\*\*|\*[^*]+?\*|`[^`]+?`|\{\{[A-Z0-9_]+\}\})")


def add_inline(paragraph, text: str, size=10.5, base_bold=False, color=None):
    for piece in INLINE.split(text):
        if not piece:
            continue
        if piece.startswith("**") and piece.endswith("**") and len(piece) > 4:
            r = paragraph.add_run(piece[2:-2])
            set_run_fonts(r, size=size, bold=True, color=color)
        elif piece.startswith("`") and piece.endswith("`") and len(piece) > 2:
            r = paragraph.add_run(piece[1:-1])
            set_run_fonts(r, latin=CODE_FONT, cjk=CODE_FONT, size=size - 1, color=color)
        elif piece.startswith("*") and piece.endswith("*") and len(piece) > 2:
            r = paragraph.add_run(piece[1:-1])
            set_run_fonts(r, size=size, italic=True, bold=base_bold, color=color)
        else:
            r = paragraph.add_run(piece)
            set_run_fonts(r, size=size, bold=base_bold, color=color)


# --------------------------------------------------------------------------- #
# block renderer
# --------------------------------------------------------------------------- #
def style_document(doc: Document):
    st = doc.styles["Normal"]
    st.font.name = BODY_FONT_LATIN
    st.font.size = Pt(10.5)
    rpr = st.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), BODY_FONT_CJK)
    pf = st.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    pf.line_spacing = 1.18
    pf.space_after = Pt(6)


def render(md_path: Path, out_path: Path, meta: dict, placeholders: dict,
           title_page: bool = True):
    text = md_path.read_text(encoding="utf-8")
    for k, v in placeholders.items():
        text = text.replace("{{" + k + "}}", str(v))
    leftover = sorted(set(re.findall(r"\{\{[A-Z0-9_]+\}\}", text)))
    if leftover:
        print("  WARNING unresolved placeholders:", leftover)

    lines = text.split("\n")
    doc = Document()
    style_document(doc)

    sec = doc.sections[0]
    sec.page_height = Cm(29.7)
    sec.page_width = Cm(21.0)
    sec.top_margin = Cm(2.5)
    sec.bottom_margin = Cm(2.5)
    sec.left_margin = Cm(2.5)
    sec.right_margin = Cm(2.5)
    sec.different_first_page_header_footer = title_page
    add_page_number_footer(sec)

    # ---------------- title page ----------------
    if title_page:
        for _ in range(4):
            doc.add_paragraph()
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(meta["title"])
        set_run_fonts(r, latin=HEAD_FONT_LATIN, cjk=HEAD_FONT_CJK, size=19, bold=True, color=ACCENT)
        p.paragraph_format.space_after = Pt(18)

        if meta.get("subtitle"):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(meta["subtitle"])
            set_run_fonts(r, size=12, italic=True)
            p.paragraph_format.space_after = Pt(30)

        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        hr(p)

        for _ in range(2):
            doc.add_paragraph()
        for k, v in meta.get("fields", {}).items():
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.left_indent = Cm(4.4)
            p.paragraph_format.first_line_indent = Cm(-3.3)
            r = p.add_run(f"{k}:  ")
            set_run_fonts(r, size=11, bold=True)
            r = p.add_run(v)
            set_run_fonts(r, size=11)
            p.paragraph_format.space_after = Pt(5)

        doc.add_page_break()

    # ---------------- body ----------------
    i = 0
    n = len(lines)
    first_h1_skipped = False
    while i < n:
        raw = lines[i]
        line = raw.rstrip()

        if not line.strip():
            i += 1
            continue

        # table
        if line.lstrip().startswith("|") and i + 1 < n and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            block = []
            while i < n and lines[i].lstrip().startswith("|"):
                block.append(lines[i].strip())
                i += 1
            rows = [[c.strip() for c in r.strip().strip("|").split("|")] for r in block]
            header, body = rows[0], rows[2:]
            ncol = len(header)
            table = doc.add_table(rows=1, cols=ncol)
            table.alignment = WD_TABLE_ALIGNMENT.CENTER
            table.autofit = True
            table_borders(table)
            for j, h in enumerate(header):
                cell = table.rows[0].cells[j]
                cell.text = ""
                para = cell.paragraphs[0]
                para.paragraph_format.space_after = Pt(0)
                add_inline(para, h, size=8.5, base_bold=True)
                shade(cell, "E8EDF3")
                set_cell_margins(cell)
            # Long result tables break across pages; without this the continuation
            # page shows unlabelled numbers.
            trPr = table.rows[0]._tr.get_or_add_trPr()
            tbl_header = OxmlElement("w:tblHeader")
            tbl_header.set(qn("w:val"), "true")
            trPr.append(tbl_header)
            for row in body:
                cells = table.add_row().cells
                for j in range(ncol):
                    val = row[j] if j < len(row) else ""
                    cells[j].text = ""
                    para = cells[j].paragraphs[0]
                    para.paragraph_format.space_after = Pt(0)
                    add_inline(para, val, size=8.5)
                    set_cell_margins(cells[j])
            doc.add_paragraph().paragraph_format.space_after = Pt(2)
            continue

        # heading
        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            level = len(m.group(1))
            txt = m.group(2).strip()
            if level == 1 and not first_h1_skipped and title_page:
                first_h1_skipped = True
                i += 1
                continue
            if level == 1 and not title_page:
                first_h1_skipped = True
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(16 if level <= 2 else 10)
            p.paragraph_format.space_after = Pt(6)
            p.paragraph_format.keep_with_next = True
            r = p.add_run(txt)
            size = {1: 15, 2: 12.5, 3: 11, 4: 10.5}[level]
            set_run_fonts(r, latin=HEAD_FONT_LATIN, cjk=HEAD_FONT_CJK, size=size,
                          bold=True, color=ACCENT if level <= 2 else None)
            i += 1
            continue

        # horizontal rule
        if re.match(r"^\s*---+\s*$", line):
            p = doc.add_paragraph()
            hr(p)
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(8)
            i += 1
            continue

        # blockquote
        if line.lstrip().startswith(">"):
            block = []
            while i < n and lines[i].lstrip().startswith(">"):
                block.append(lines[i].lstrip()[1:].strip())
                i += 1
            for b in block:
                if not b:
                    continue
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = Cm(0.8)
                p.paragraph_format.space_after = Pt(3)
                add_inline(p, b, size=9.5, color=RGBColor(0x44, 0x4A, 0x52))
                p.runs[0].italic = True
            continue

        # image
        m = re.match(r"^!\[(.*?)\]\((.*?)\)\s*$", line)
        if m:
            caption, src = m.group(1), m.group(2)
            path = (md_path.parent / src).resolve() if not Path(src).is_absolute() else Path(src)
            if path.exists():
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(8)
                p.paragraph_format.space_after = Pt(2)
                p.add_run().add_picture(str(path), width=Cm(14.5))
                if caption:
                    cp = doc.add_paragraph()
                    cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    cp.paragraph_format.space_after = Pt(10)
                    r = cp.add_run(caption)
                    set_run_fonts(r, size=9, italic=True)
            i += 1
            continue

        # lists
        # Lists are rendered with an explicit marker plus a tab stop rather than
        # Word's List Bullet / List Number styles. Those styles carry their own
        # numbering indent with suff="nothing", which makes the marker collide
        # with the first word and breaks the hanging indent.
        def _consume_continuation(start, first_text):
            """A list item may wrap across several source lines; those lines are
            indented and must be folded into the item, otherwise they become
            paragraphs at the page margin and the hanging indent looks broken."""
            j = start
            text = first_text
            while j < n:
                nxt = lines[j]
                if not nxt.strip():
                    break
                if re.match(r"^\s*[-*]\s+", nxt) or re.match(r"^\s*\d+[.)]\s+", nxt):
                    break
                if nxt.startswith("  ") and not nxt.lstrip().startswith(("#", "|", ">", "```")):
                    text = text.rstrip() + " " + nxt.strip()
                    j += 1
                    continue
                break
            return text, j

        m = re.match(r"^(\s*)[-*]\s+(.*)$", raw)
        if m:
            level = len(m.group(1)) // 2
            left = 0.95 + 0.65 * level
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(left)
            p.paragraph_format.first_line_indent = Cm(-0.5)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.tab_stops.add_tab_stop(Cm(left))
            r = p.add_run(("\u2022" if level == 0 else "\u2013") + "\t")
            set_run_fonts(r, size=10.5, color=ACCENT if level == 0 else RGBColor(0x66, 0x6C, 0x76))
            item_text, i = _consume_continuation(i + 1, m.group(2))
            add_inline(p, item_text)
            continue

        m = re.match(r"^(\s*)(\d+)[.)]\s+(.*)$", raw)
        if m:
            level = len(m.group(1)) // 2
            left = 0.95 + 0.65 * level
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(left)
            p.paragraph_format.first_line_indent = Cm(-0.5)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.tab_stops.add_tab_stop(Cm(left))
            r = p.add_run(m.group(2) + ".\t")
            set_run_fonts(r, size=10.5, color=ACCENT)
            item_text, i = _consume_continuation(i + 1, m.group(3))
            add_inline(p, item_text)
            continue

        # code fence
        if line.strip().startswith("```"):
            i += 1
            block = []
            while i < n and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.6)
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(8)
            r = p.add_run("\n".join(block))
            set_run_fonts(r, latin=CODE_FONT, cjk=CODE_FONT, size=8.5)
            continue

        # table / figure caption
        mcap = re.match(r"^\*?(Table|Figure|图|表)\s*\d+[.:]?.*\*?$", line.strip())
        if mcap and len(line.strip()) < 220:
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.keep_with_next = True
            r = p.add_run(line.strip().strip("*"))
            set_run_fonts(r, size=9, bold=True, color=RGBColor(0x33, 0x3A, 0x44))
            i += 1
            continue

        # paragraph (join consecutive non-blank lines)
        block = [line]
        i += 1
        while i < n:
            nxt = lines[i]
            if (not nxt.strip() or nxt.lstrip().startswith(("#", "|", ">", "```", "---"))
                    # A wrapped line can legitimately begin with an italic word
                    # ("*contains*, below the baseline"), so only a real bullet
                    # marker — asterisk or dash followed by a space — ends the
                    # paragraph. Testing the bare character split paragraphs in
                    # the middle of sentences.
                    or re.match(r"^\s*[-*]\s+", nxt)
                    or re.match(r"^\s*\d+[.)]\s", nxt) or re.match(r"^!\[", nxt)):
                break
            block.append(nxt.strip())
            i += 1
        para_text = " ".join(block)
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.first_line_indent = Cm(0.0)
        add_inline(p, para_text)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path))
    print("wrote", out_path)


def references_hanging_indent(path: Path):
    """Post-process: hanging indent for the reference list."""
    doc = Document(str(path))
    in_refs = False
    for p in doc.paragraphs:
        t = p.text.strip()
        if t.lower().startswith("references"):
            in_refs = True
            continue
        if in_refs:
            if t.startswith("Appendix") or t.startswith("附录"):
                break
            if t:
                p.paragraph_format.left_indent = Cm(1.0)
                p.paragraph_format.first_line_indent = Cm(-1.0)
                p.paragraph_format.space_after = Pt(6)
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    doc.save(str(path))
