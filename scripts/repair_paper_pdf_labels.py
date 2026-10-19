"""Correct metadata in archived vector figures without recomputing their errors.

The archived per-job density errors cannot be recovered from figure_data.json.
Keep every path and numeric label from the original PDFs. Remove only the
incorrect beam-energy title, and add simulation provenance and the notation key.
Run with the bundled document Python (pypdf and reportlab).
"""
from io import BytesIO
from pathlib import Path
import hashlib
import json

from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO.parent / 'HGNDPaper'


def multiply(a, b):
    return [a[0]*b[0]+a[2]*b[1], a[1]*b[0]+a[3]*b[1],
            a[0]*b[2]+a[2]*b[3], a[1]*b[2]+a[3]*b[3],
            a[0]*b[4]+a[2]*b[5]+a[4], a[1]*b[4]+a[3]*b[5]+a[5]]


def remove_top_title(page, reader):
    """The original title alone has its text origin above H-20 points."""
    stream = ContentStream(page.get_contents(), reader)
    matrix = [1, 0, 0, 1, 0, 0]
    stack, result = [], []
    keep_text = True
    removed = 0
    for args, operator in stream.operations:
        if operator == b'q':
            stack.append(matrix[:])
        elif operator == b'Q':
            matrix = stack.pop()
        elif operator == b'cm':
            matrix = multiply(matrix, list(map(float, args)))
        if operator == b'BT':
            keep_text = matrix[5] <= float(page.mediabox.height)-20
            removed += not keep_text
        if keep_text:
            result.append((args, operator))
        if operator == b'ET':
            keep_text = True
    assert removed == 1, f'Unexpected archived title structure: {removed}'
    stream.operations = result
    page.replace_contents(stream)
    assert 'Xe+CsI' not in page.extract_text()


def main():
    # Embed the annotation font; Base-14 Helvetica otherwise needs a reader
    # substitution and does not satisfy the manuscript's embedded-font rule.
    pdfmetrics.registerFont(TTFont('FigureSans', '/System/Library/Fonts/Supplemental/Arial.ttf'))
    manifest = {'operation': 'metadata correction; original curves and uncertainties preserved',
                'figures': []}
    files = [('b_full_analysis/fig1_b_distributions.pdf', 'b_distributions', True),
             ('b_full_analysis/fig2_job_structure.pdf', 'job_structure', False),
             ('b_full_analysis/fig7_spot_response.pdf', 'spot_response', False),
             ('retrain_eval/fig1_energy_performance.pdf', 'energy_performance', False)]
    for source_name, stem, wrong_title in files:
        source = REPO/'results'/source_name
        reader = PdfReader(source)
        original = PdfWriter()
        page = original.add_page(reader.pages[0])
        if wrong_title:
            remove_top_title(page, original)
        w, h = float(page.mediabox.width), float(page.mediabox.height)
        extra = 9 if wrong_title else 22
        overlay = BytesIO()
        canvas = Canvas(overlay, pagesize=(w, h+extra))
        canvas.setFont('FigureSans', 7)
        # This corrects the energy variable, not the numerical simulation data.
        heading = ('MC truth + reconstructed: SMASH Xe+Cs, beam Ekin/A = 2.87 GeV'
                   if stem == 'energy_performance' else
                   'MC truth: SMASH Xe+Cs, beam Ekin/A = 2.87 GeV; minimum bias')
        canvas.drawCentredString(w/2, h+extra-8, heading)
        canvas.setFont('FigureSans', 6.5)
        detail = ('Archived four-epoch comparison; single-seed point estimates'
                  if stem == 'energy_performance' else
                  'Archived S_pot labels denote U_sym (MeV); statistical uncertainties')
        canvas.drawCentredString(w/2, h+extra-17, detail)
        canvas.save()
        writer = PdfWriter()
        target = writer.add_blank_page(width=w, height=h+extra)
        target.merge_page(page)
        target.merge_page(PdfReader(overlay).pages[0])
        dest = PAPER/'figs'/f'{stem}.pdf'
        with dest.open('wb') as f:
            writer.write(f)
        text = PdfReader(dest).pages[0].extract_text()
        assert 'beam Ekin/A = 2.87 GeV' in text
        assert 'Xe+CsI' not in text
        manifest['figures'].append({'source':str(source.relative_to(REPO)),
            'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'output':str(dest.relative_to(PAPER)),
            'output_sha256':hashlib.sha256(dest.read_bytes()).hexdigest()})
    (PAPER/'data/figure_label_corrections.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
