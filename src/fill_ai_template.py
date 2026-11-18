"""
Fill the university's AI-use reporting template in place.

The assessment requires the *supplied* template to be filled, so this edits a copy
of `AI-use reporting template.docx` rather than inventing a new layout. Cell
formatting is set explicitly because the template's runs do not all inherit the
same font.

Output: docs/AI_use_reporting_template_filled.docx
"""
from __future__ import annotations

import shutil
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT.parent / "AI-use reporting template.docx"
OUT = ROOT / "docs" / "AI_use_reporting_template_filled.docx"

TOOL = "WorkBuddy AI assistant (DeepSeek-V4.1-Flash)"

# (row index, used Y/N, section) for the nine standard categories in the template
ANSWERS = {
    1: ("Y", "Entire work — project scoping; report §2, §4.5"),
    2: ("Y", "§7 Conclusion"),
    3: ("Y", "Entire report"),
    4: ("Y", "§1, §5, §6"),
    5: ("Y", "§1, §4.2, §6.1"),
    6: ("N", "— (all figures are matplotlib output or native PowerPoint shapes)"),
    7: ("Y", "§1, §3; References"),
    8: ("Y", "References"),
    9: ("Y", "Entire report; presentation slides"),
}

EXTRA = [
    ("Writing implementation code (data pipeline, QLoRA training loop, evaluation harness) "
     "from my written specifications, and debugging runtime errors",
     "Y", "§4 Methodology; src/, scripts/"),
    ("Designing the evaluation metrics, including the verbatim grounding check and the "
     "\u201cverified correct\u201d composite",
     "Y", "§4.6, §5"),
    ("Building report tables and presentation figures from the measured result files",
     "Y", "§5; slides 11\u201314"),
    ("Drafting and restructuring report prose to a fixed structure and word budget",
     "Y", "§1\u2013§7"),
]


def set_cell(cell, text, size=9.5, bold=False, color=None):
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    r = p.add_run(text)
    r.font.name = "Calibri"
    r.font.size = Pt(size)
    r.font.bold = bold
    if color:
        r.font.color.rgb = color
    rpr = r._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        from docx.oxml import OxmlElement
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), "SimSun")


def add_para(doc, text, size=10.5, bold=False, space_before=6, space_after=4,
             italic=False, color=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    r = p.add_run(text)
    r.font.name = "Calibri"
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.italic = italic
    if color:
        r.font.color.rgb = color
    return p


def main():
    if not TEMPLATE.exists():
        raise SystemExit(f"template not found: {TEMPLATE}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(TEMPLATE, OUT)

    doc = Document(str(OUT))
    tbl = doc.tables[0]

    # tool name
    set_cell(tbl.cell(1, 0), TOOL, size=9.5, bold=True)

    for row, (used, section) in ANSWERS.items():
        set_cell(tbl.cell(row, 2), used, size=9.5)
        set_cell(tbl.cell(row, 3), section, size=9)

    # rows 10-12 in the template are instruction placeholders: repurpose them,
    # and add rows if there are more additional uses than spare rows.
    spare = [10, 11, 12]
    needed = len(EXTRA)
    while len(spare) < needed:
        tbl.add_row()
        spare.append(len(tbl.rows) - 1)

    for idx, row in enumerate(spare):
        if idx < needed:
            desc, used, section = EXTRA[idx]
            set_cell(tbl.cell(row, 0), TOOL if idx == 0 else "", size=9.5, bold=(idx == 0))
            set_cell(tbl.cell(row, 1), desc, size=9)
            set_cell(tbl.cell(row, 2), used, size=9.5)
            set_cell(tbl.cell(row, 3), section, size=9)
        else:
            for c in range(4):
                set_cell(tbl.cell(row, c), "", size=9)

    # ---- declaration below the table ----
    add_para(doc, "", size=6, space_before=4, space_after=2)
    add_para(doc, "Declaration of AI-Generated Material", size=12, bold=True,
             space_before=10, space_after=6)
    add_para(doc,
             "Generative artificial intelligence was used in the preparation of this assessment "
             "item, as set out in the table above. In accordance with the subject policy, the "
             "scope of that use is declared here and the full prompt history is submitted as a "
             "separate document.", size=10.5, space_after=6)
    add_para(doc,
             "No result, metric, table or figure reported in the project report was generated or "
             "estimated by an AI system. All experimental results were produced by code executed "
             "on the student's own hardware against the public datasets described in the report. "
             "Every reference was verified against the arXiv record or the publisher's page; none "
             "were generated without verification.", size=10.5, space_after=6)
    add_para(doc,
             "Three defects in AI-produced output were identified and corrected during "
             "verification, and are recorded in the prompt history because they changed the "
             "results: (1) an inference pipeline that omitted the model's chat template, which "
             "caused the model to continue the passages instead of answering; (2) an evaluation "
             "set that placed the gold passage at rank 1 by construction, making retrieval appear "
             "perfect; and (3) a retrieval index cache that survived a corpus rebuild and silently "
             "misaligned embeddings against documents. All figures produced before these fixes "
             "were discarded.", size=10.5, space_after=6)
    add_para(doc,
             "I have read and verified the final report and presentation, I can explain every "
             "design decision and every line of the submitted code, and I accept responsibility "
             "for the accuracy of the content.", size=10.5, space_after=10)

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    r = p.add_run("Signed: ____________________________     Date: ____________________")
    r.font.size = Pt(10.5)
    r.font.name = "Calibri"

    doc.save(str(OUT))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
