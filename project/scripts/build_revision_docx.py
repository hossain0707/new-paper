"""Use the existing document layout with the new source and evidence-derived figures."""
from pathlib import Path
import sys
import build_mdpi_docx as builder
ROOT=Path(__file__).resolve().parents[1]
FIGURES={
    1:('fig1_protocol.png','Evaluated GlucoSense workflow: observed CGM/HR processing, patient-held-out forecasting, RF attribution and locally checked explanation.'),
    2:('fig6_grounding.png','Generation contract acceptance and synthetic detector rejection. Fallbacks are excluded from LLM acceptance; source-corruption cases expose the source-truth boundary.'),
    3:('fig2_forecast_mae.png','Subject-macro MAE on identical test origins. Learned models use seed-averaged predictions; input availability differs between cohorts.'),
    4:('fig4_heterogeneity.png','RF 30-minute participant heterogeneity. Participant 907 is highlighted; the variability association is descriptive, not causal.'),
    5:('fig5_clarke.png','RF 30-minute forecasts with complete Clarke boundaries and logarithmic density. Display axes span 0–450 mg/dL; numerical metrics use all pairs. Clarke zones do not certify clinical safety.'),
    6:('fig3_paired_effects.png','Paired participant-MAE differences and unadjusted patient-bootstrap 95% intervals. Negative values favor the first method; Table 4 reports Holm-adjusted decisions.'),
    7:('../../revision_v3/figures/fig7_additional_evidence.png','Additional descriptive evidence. (A) Frozen DDATSHR-to-AZT1D point estimates; all methods and intervals are in Table 7. (B) Chronos-2 minus primary CGM RF on identical origins; RF additionally uses time-of-day. (C) RF CGM+HR minus CGM with three inner validation participants. Bars in B/C are 95% paired participant-bootstrap intervals; negative differences favor the first method. Shared training folds limit independence; these are not additional confirmatory tests.'),
}

if __name__=='__main__':
    builder.SRC=ROOT/'manuscript/manuscript_REVISED_EVIDENCE.md'
    builder.OUT=ROOT.parent/'manuscript_REVISED_EVIDENCE.docx'
    builder.FIGDIR=ROOT/'results/revision_v2/figures';builder.FIGURES=FIGURES
    builder.main()
    # The rewrite is extensive: highlight the entire revised body, while stating this
    # is a review aid and not Word's tracked-deletion history.
    from docx import Document
    from docx.enum.text import WD_COLOR_INDEX
    doc=Document(builder.OUT)
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    import re
    from copy import deepcopy
    from docx.text.run import Run
    from docx.shared import Cm
    last=doc.inline_shapes[-1]
    ratio=16.0/16.8
    last.width=int(last.width*ratio);last.height=int(last.height*ratio)
    for paragraph in doc.paragraphs:
        text=paragraph.text
        if re.search(r'\([1-6]\)$',text.strip()) and '=' in text:
            # Preserve the source equation while rendering indices as actual subscripts.
            paragraph.clear()
            for token in re.split(r'(_[A-Za-z]+)',text.replace('sqrt[','√[')):
                run=paragraph.add_run(token[1:] if token.startswith('_') else token)
                if token.startswith('_'):run.font.subscript=True
        # Match scientific indices only, preserving existing rich-text formatting.
        for run in list(paragraph.runs):
            parts=re.split(r'((?:w|f|φ|n|G|E|Σ|ŷ)_(?:RF|last|delta|p|i|j))',run.text)
            if len(parts)==1:continue
            anchor=run._r
            for part in parts:
                pieces=part.split('_',1) if re.fullmatch(r'(?:w|f|φ|n|G|E|Σ|ŷ)_(?:RF|last|delta|p|i|j)',part) else [part]
                for index,piece in enumerate(pieces):
                    element=deepcopy(run._r);anchor.addnext(element);anchor=element
                    added=Run(element,paragraph);added.text=piece
                    if index:added.font.subscript=True
            run._r.getparent().remove(run._r)
        if text.startswith(('Figure ','Table ')):paragraph.paragraph_format.keep_together=True
        if paragraph._p.xpath('.//w:drawing') or text.startswith('Table ') or re.match(r'^\d+(?:\.\d+)?\.? ',text) and len(text)<100:
            paragraph.paragraph_format.keep_with_next=True
        if text in ['Supplementary Materials','Acknowledgments','References']:
            paragraph.paragraph_format.keep_with_next=True
    for table in doc.tables:
        for index,row in enumerate(table.rows):
            properties=row._tr.get_or_add_trPr()
            properties.append(OxmlElement('w:cantSplit'))
            if index==0:properties.append(OxmlElement('w:tblHeader'))
            if index<2:
                for cell in row.cells:
                    for paragraph in cell.paragraphs:paragraph.paragraph_format.keep_with_next=True
    footer=doc.sections[0].footer.paragraphs[0]
    footer.alignment=2
    field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');footer._p.append(field)
    doc.save(builder.OUT)
    baseline=ROOT/'professor feedback/BASELINE_PROFESSOR_REVIEWED.txt'
    original=baseline.read_text() if baseline.exists() else ''
    original=re.sub(r'-\s*\n\s*(?=[a-z])','',original)
    original_normalized=re.sub(r'\s+',' ',original).strip()
    review_note=('REVIEW COPY — original title, goal and section sequence restored. Yellow highlights identify text differing from the professor-reviewed source; deletions and moves are explained in REVISION_MAP_FOR_PROFESSOR.md. This is not Word Track Changes.' if baseline.exists() else 'REVIEW COPY — the private baseline PDF text is unavailable in this extraction, so all current text is highlighted; no comparison to the original PDF is asserted.')
    note=doc.paragraphs[0].insert_paragraph_before(review_note)
    for paragraph in doc.paragraphs[1:]:
        if paragraph.style.name=='Normal' and re.sub(r'\s+',' ',paragraph.text).strip() not in original_normalized:
            for run in paragraph.runs:
                if run.text:run.font.highlight_color=WD_COLOR_INDEX.YELLOW
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    if re.sub(r'\s+',' ',paragraph.text).strip() not in original_normalized:
                        for run in paragraph.runs:run.font.highlight_color=WD_COLOR_INDEX.YELLOW
    private=ROOT/'professor feedback';private.mkdir(exist_ok=True)
    doc.save(private/'manuscript_REVISED_HIGHLIGHTED.docx')
