"""
Builds an MDPI-formatted .docx directly from manuscript_SUBMISSION_draft1.md.

Applies the MDPI manuscript layout: Palatino Linotype throughout, 10 pt justified body,
9 pt abstract/keywords/captions/references, run-in "Abstract:"/"Keywords:" and run-in
back-matter statements, table captions above with three-line (no vertical rule) MDPI
table styling, figure captions below, and an "Article" type label above the title.

Run from project/:
    python scripts/build_mdpi_docx.py
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

SRC = PROJECT_ROOT / "manuscript" / "manuscript_SUBMISSION_draft1.md"
FIGDIR = PROJECT_ROOT / "results" / "figures"
OUT = PROJECT_ROOT.parent / "manuscript_MDPI_FORMATTED.docx"

# Optional override for building a "what changed" copy for reviewer re-review:
# python scripts/build_mdpi_docx.py <src.md> <out.docx>
# In that source, text wrapped in ==like this== renders with a yellow highlight;
# the default clean submission source never contains "==", so this is a no-op there.
if len(sys.argv) >= 2:
    SRC = Path(sys.argv[1])
if len(sys.argv) >= 3:
    OUT = Path(sys.argv[2])

FONT = "Palatino Linotype"
SZ_TITLE, SZ_AUTHOR, SZ_AFFIL = 16, 10, 9
SZ_BODY, SZ_SMALL = 10, 9

# Figure number -> (file, caption). Order matches first mention in the text.
FIGURES = {
    1: ("orig_fig1_architecture.png", "GlucoSense AI five-layer system architecture."),
    2: ("orig_fig2_dataflow.png", "Multi-modal data processing flowchart."),
    3: ("fig4_llm_latency_hist.png",
        "End-to-end narrative generation latency, on-premises DeepSeek V4 Flash (n = 15 cases)."),
    4: ("fig2_rmse_by_horizon.png",
        "DDATSHR leave-one-patient-out RMSE by prediction horizon and model."),
    5: ("fig1_clarke_error_grid.png",
        "Clarke Error Grid for Random Forest 30-minute predictions on DDATSHR "
        "(n = 104,399 held-out predictions; Zone A = 80.7%)."),
    6: ("fig5_diadata_rmse_by_horizon.png",
        "DiaData (non-overlapping, n = 28) leave-one-patient-out RMSE by horizon and model."),
    7: ("fig3_per_subject_rmse.png",
        "Per-patient RMSE, Random Forest, 30-minute horizon, across held-out DDATSHR subjects."),
    8: ("fig6_deep_vs_tree_ensemble.png",
        "DDATSHR: retrained deep-learning ensemble members and the full stacked ensemble (Eq. 2), "
        "compared against the tree-based Random Forest baseline, by horizon."),
}

BACKMATTER = [
    "Author Contributions", "Funding", "Institutional Review Board Statement",
    "Informed Consent Statement", "Data Availability Statement", "Conflicts of Interest",
]


# ---------------------------------------------------------------- doc helpers
def set_cell_border(cell, **kwargs):
    tcPr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    for edge in ("top", "bottom", "left", "right"):
        spec = kwargs.get(edge)
        el = OxmlElement(f"w:{edge}")
        if spec is None:
            el.set(qn("w:val"), "nil")
        else:
            el.set(qn("w:val"), "single")
            el.set(qn("w:sz"), str(spec))
            el.set(qn("w:color"), "000000")
        borders.append(el)
    tcPr.append(borders)


def style_run(run, size=SZ_BODY, bold=False, italic=False):
    run.font.name = FONT
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rFonts.set(qn(a), FONT)
    return run


INLINE = re.compile(r"(\*\*.+?\*\*|\*[^*]+?\*)")


def _add_rich_segment(par, text, size, base_bold, highlighted):
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            r = style_run(par.add_run(part[2:-2]), size, bold=True)
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            r = style_run(par.add_run(part[1:-1]), size, bold=base_bold, italic=True)
        else:
            r = style_run(par.add_run(part), size, bold=base_bold)
        if highlighted:
            r.font.highlight_color = WD_COLOR_INDEX.YELLOW


def add_rich(par, text, size=SZ_BODY, base_bold=False):
    """Adds text to a paragraph, converting **bold** and *italic* into real runs.
    Text wrapped in ==...== (a "reviewer track-changes" convention used only in the
    highlighted-copy source, never in the clean submission source) is rendered with a
    yellow highlight in addition to any nested bold/italic."""
    segments = text.split("==")
    for idx, seg in enumerate(segments):
        if not seg:
            continue
        _add_rich_segment(par, seg, size, base_bold, highlighted=(idx % 2 == 1))
    return par


SUP_MAP = {"¹": "1", "²": "2", "³": "3"}


def add_superscripted(par, text, size, bold=False, lead_superscript=False):
    """Renders affiliation/author markers as true superscript runs. Unicode superscript
    glyphs are converted to plain digits first, so the marker is raised by formatting
    rather than by glyph -- a superscript-formatted '¹' renders as an unreadably small
    mark in Word."""
    def emit(chunk, is_marker):
        r = style_run(par.add_run(chunk), size, bold=(bold and not is_marker))
        if is_marker:
            r.font.superscript = True
        return r

    if lead_superscript:
        m = re.match(r"^([¹²³*])\s*(.*)$", text)
        if m:
            emit(SUP_MAP.get(m.group(1), m.group(1)), True)
            emit(" " + m.group(2), False)
            return par

    for part in re.split(r"([¹²³*]+)", text):
        if not part:
            continue
        if re.fullmatch(r"[¹²³*]+", part):
            emit("".join(SUP_MAP.get(ch, ch) for ch in part), True)
        else:
            emit(part, False)
    return par


def body_par(doc, text, size=SZ_BODY, indent=True, justify=True, space_after=6):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(0)
    pf.space_after = Pt(space_after)
    pf.line_spacing = 1.0
    if justify:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if indent:
        pf.first_line_indent = Cm(0.5)
    add_rich(p, text, size)
    return p


# ---------------------------------------------------------------- md parsing
def parse_markdown(md):
    """Splits the source into a flat list of typed blocks."""
    lines = md.split("\n")
    blocks, i = [], 0
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()

        if not s or s == "---":
            i += 1
            continue

        if s.startswith("|"):  # table
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i].strip())
                i += 1
            blocks.append(("table", rows))
            continue

        m = re.match(r"^(#{1,3})\s+(.*)$", s)
        if m:
            blocks.append((f"h{len(m.group(1))}", m.group(2).strip()))
            i += 1
            continue

        m = re.match(r"^(\d+)\.\s+(.*)$", s)
        if m:
            buf = m.group(2)
            i += 1
            while i < len(lines) and lines[i].strip() and not re.match(
                    r"^(\d+\.|[-*#]|\|)", lines[i].strip()):
                buf += " " + lines[i].strip()
                i += 1
            blocks.append(("ol", (m.group(1), buf)))
            continue

        if s.startswith("- "):
            buf = s[2:]
            i += 1
            while i < len(lines) and lines[i].strip() and not re.match(
                    r"^(\d+\.|[-*#]|\|)", lines[i].strip()):
                buf += " " + lines[i].strip()
                i += 1
            blocks.append(("ul", buf))
            continue

        # equation: indented line ending in (N)
        if re.search(r"\(\d\)\s*$", s) and (ln.startswith("  ") or ln.startswith("\t")):
            blocks.append(("eq", s))
            i += 1
            continue

        buf = s
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(
                r"^(#{1,3}\s|\||\d+\.\s|-\s)", lines[i].strip()) and lines[i].strip() != "---":
            nxt = lines[i]
            if re.search(r"\(\d\)\s*$", nxt.strip()) and (nxt.startswith("  ") or nxt.startswith("\t")):
                break
            buf += " " + nxt.strip()
            i += 1
        blocks.append(("p", buf))
    return blocks


def add_mdpi_table(doc, rows):
    """rows = markdown pipe-table lines. MDPI style: horizontal rules only."""
    grid = []
    for r in rows:
        if re.match(r"^\|[\s:\-|]+\|$", r):  # separator
            continue
        cells = [c.strip() for c in r.strip().strip("|").split("|")]
        grid.append(cells)
    if not grid:
        return
    ncol = max(len(r) for r in grid)
    t = doc.add_table(rows=len(grid), cols=ncol)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = True

    for ri, row in enumerate(grid):
        for ci in range(ncol):
            cell = t.cell(ri, ci)
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if ci else WD_ALIGN_PARAGRAPH.LEFT
            txt = row[ci] if ci < len(row) else ""
            add_rich(p, txt, SZ_SMALL, base_bold=(ri == 0))
            # MDPI: rule above header, below header, below last row only
            top = 8 if ri == 0 else (4 if ri == 1 else None)
            bottom = 8 if ri == len(grid) - 1 else (4 if ri == 0 else None)
            set_cell_border(cell, top=top, bottom=bottom, left=None, right=None)
    return t


def add_figure(doc, num):
    fname, caption = FIGURES[num]
    path = FIGDIR / fname
    if not path.exists():
        print(f"  !! missing figure file: {fname}")
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(str(path), width=Cm(16.8))

    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(10)
    style_run(cap.add_run(f"Figure {num}. "), SZ_SMALL, bold=True)
    add_rich(cap, caption, SZ_SMALL)


# ---------------------------------------------------------------- build
def main():
    md = SRC.read_text(encoding="utf-8")
    blocks = parse_markdown(md)

    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.top_margin = sec.bottom_margin = Cm(1.7)
    sec.left_margin = sec.right_margin = Cm(1.75)

    normal = doc.styles["Normal"]
    normal.font.name = FONT
    normal.font.size = Pt(SZ_BODY)
    rPr = normal.element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rFonts.set(qn(a), FONT)
    if rFonts.getparent() is None:
        rPr.append(rFonts)

    placed_figs = set()
    in_refs = False
    ref_n = 0
    i = 0

    # ---- front matter, consumed manually
    title = blocks[0][1]
    p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(2)
    style_run(p.add_run("Article"), SZ_BODY, bold=True)

    p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(6)
    add_rich(p, title, SZ_TITLE)
    for r in p.runs:
        r.bold = True

    i = 1
    while i < len(blocks) and blocks[i][0] != "h2":
        kind, val = blocks[i]
        txt = val if isinstance(val, str) else ""
        txt = txt.replace("\\*", "*")
        if txt.startswith("**Md. Najmul"):
            q = doc.add_paragraph(); q.paragraph_format.space_after = Pt(2)
            add_superscripted(q, txt.replace("**", ""), SZ_AUTHOR, bold=True)
        elif txt.startswith("¹ Department") or txt.startswith("* Correspondence"):
            q = doc.add_paragraph(); q.paragraph_format.space_after = Pt(2)
            add_superscripted(q, txt, SZ_AFFIL, lead_superscript=True)
        i += 1

    # ---- body
    while i < len(blocks):
        kind, val = blocks[i]

        if kind == "h2":
            head = val
            if head == "Abstract":
                i += 1
                abs_txt = blocks[i][1]
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                p.paragraph_format.space_after = Pt(6)
                style_run(p.add_run("Abstract: "), SZ_SMALL, bold=True)
                add_rich(p, abs_txt, SZ_SMALL)
                i += 1
                continue
            if head == "References":
                in_refs = True
                p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(10)
                p.paragraph_format.space_after = Pt(4)
                style_run(p.add_run("References"), SZ_BODY, bold=True)
                i += 1
                continue
            if head in BACKMATTER:
                i += 1
                txt = blocks[i][1] if i < len(blocks) else ""
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                p.paragraph_format.space_after = Pt(4)
                style_run(p.add_run(f"{head}: "), SZ_SMALL, bold=True)
                add_rich(p, txt, SZ_SMALL)
                i += 1
                continue
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(10)
            p.paragraph_format.space_after = Pt(4)
            style_run(p.add_run(head), SZ_BODY, bold=True)
            i += 1
            continue

        if kind == "h3":
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(3)
            style_run(p.add_run(val), SZ_BODY, bold=True)
            i += 1
            continue

        if kind == "table":
            add_mdpi_table(doc, val)
            doc.add_paragraph().paragraph_format.space_after = Pt(6)
            i += 1
            continue

        if kind == "eq":
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(4)
            style_run(p.add_run(re.sub(r"\s+", " ", val)), SZ_BODY)
            i += 1
            continue

        if kind == "ol":
            num, txt = val
            if in_refs:
                ref_n += 1
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                pf = p.paragraph_format
                pf.left_indent = Cm(0.75); pf.first_line_indent = Cm(-0.75)
                pf.space_after = Pt(2)
                add_rich(p, f"{num}. {txt}", SZ_SMALL)
            else:
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                pf = p.paragraph_format
                pf.left_indent = Cm(0.75); pf.first_line_indent = Cm(-0.75)
                pf.space_after = Pt(3)
                add_rich(p, f"{num}. {txt}", SZ_BODY)
            i += 1
            continue

        if kind == "ul":
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            pf = p.paragraph_format
            pf.left_indent = Cm(0.75); pf.first_line_indent = Cm(-0.4)
            pf.space_after = Pt(3)
            add_rich(p, "• " + val, SZ_BODY)
            i += 1
            continue

        # paragraph
        txt = val
        if txt.startswith("**Keywords:**"):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.space_after = Pt(10)
            style_run(p.add_run("Keywords: "), SZ_SMALL, bold=True)
            add_rich(p, txt.replace("**Keywords:**", "").strip(), SZ_SMALL)
            i += 1
            continue

        cap = re.match(r"^\*\*(Table \d)\.\*\*\s*(.*)$", txt)
        if cap:
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(3)
            style_run(p.add_run(f"{cap.group(1)}. "), SZ_SMALL, bold=True)
            add_rich(p, cap.group(2), SZ_SMALL)
            i += 1
            continue

        body_par(doc, txt)
        # place any figure first referenced in this paragraph
        for n in sorted(set(int(x) for x in re.findall(r"Figure (\d)", txt))):
            if n not in placed_figs and n in FIGURES:
                add_figure(doc, n)
                placed_figs.add(n)
        i += 1

    for n in FIGURES:
        if n not in placed_figs:
            print(f"  !! figure {n} never placed")

    doc.save(OUT)
    print(f"written: {OUT}")
    print(f"  figures placed: {sorted(placed_figs)}")
    print(f"  reference entries: {ref_n}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        raise SystemExit("The historical default source is superseded. Run python scripts/build_revision_docx.py for the current manuscript.")
    main()
