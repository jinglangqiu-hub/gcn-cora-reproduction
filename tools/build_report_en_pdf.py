"""Typeset docs/report_en.md as a linked PDF, retaining all text, tables, and figures."""

import argparse
import hashlib
import html
import json
from pathlib import Path
import re

import reportlab
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, KeepTogether, PageBreak, Paragraph, Preformatted, SimpleDocTemplate,
    Spacer, Table, TableStyle,
)
from PIL import Image as PILImage

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'docs/report_en.md'
ORIGIN = 'https://github.com/jinglangqiu-hub/gcn-cora-reproduction'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plain_dashes(text):
    return text.translate(str.maketrans({'–': '-', '—': '-', '‑': '-', '−': '-'})).replace('→', '->')


def link_url(target):
    if '://' in target:
        return target
    relative, _, anchor = target.partition('#')
    path = (SOURCE.parent/relative).resolve()
    repo_path = path.relative_to(ROOT).as_posix()
    kind = 'tree' if path.is_dir() else 'blob'
    return f'{ORIGIN}/{kind}/main/{repo_path}' + (f'#{anchor}' if anchor else '')


def markup(text):
    text = html.escape(plain_dashes(text), quote=False)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)',
                  lambda m: f'<link href="{html.escape(link_url(html.unescape(m[2])), quote=True)}" color="#245C91">{m[1]}</link>', text)
    text = re.sub(r'`([^`]+)`', r'<font name="Vera" color="#243447">\1</font>', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<i>\1</i>', text)
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fonts = Path(reportlab.__file__).parent/'fonts'
    for name, filename in [('Vera', 'Vera.ttf'), ('Vera-Bold', 'VeraBd.ttf'),
                           ('Vera-Italic', 'VeraIt.ttf'), ('Vera-BoldItalic', 'VeraBI.ttf')]:
        pdfmetrics.registerFont(TTFont(name, str(fonts/filename)))
    pdfmetrics.registerFontFamily('Vera', normal='Vera', bold='Vera-Bold',
                                 italic='Vera-Italic', boldItalic='Vera-BoldItalic')
    styles = getSampleStyleSheet()
    ink, blue, gray = colors.HexColor('#243447'), colors.HexColor('#245C91'), colors.HexColor('#66717E')
    body = ParagraphStyle('ReportBody', fontName='Vera', fontSize=9.1, leading=13.0,
                          textColor=ink, spaceAfter=7, splitLongWords=True)
    title = ParagraphStyle('ReportTitle', parent=body, fontName='Vera-Bold',
                           fontSize=20, leading=26, spaceAfter=12)
    h2 = ParagraphStyle('ReportH2', parent=body, fontName='Vera-Bold',
                        fontSize=14, leading=19, spaceAfter=12, keepWithNext=True)
    h3 = ParagraphStyle('ReportH3', parent=body, fontName='Vera-Bold',
                        fontSize=11, leading=15, spaceBefore=8, keepWithNext=True)
    caption = ParagraphStyle('ReportCaption', parent=body, fontSize=8.0, leading=11,
                             textColor=gray, spaceAfter=10)
    cell = ParagraphStyle('ReportCell', parent=body, fontSize=7.8, leading=10.5, spaceAfter=0)
    code_style = ParagraphStyle('ReportCode', parent=body, fontName='Courier',
                                fontSize=8.1, leading=11.5, textColor=ink,
                                spaceBefore=3, spaceAfter=10)
    bullet = ParagraphStyle('ReportBullet', parent=body, leftIndent=10, firstLineIndent=-8)
    doc = SimpleDocTemplate(str(output), pagesize=A4, rightMargin=43, leftMargin=43,
                            topMargin=44, bottomMargin=43, title='Reproducing the Two-Layer GCN: Core Experiments on Cora',
                            author='JinglangQiu', subject='English report v1.1; Cora fixed-split GCN/MLP reproduction')
    story, images, table_count = [], [], 0
    lines = SOURCE.read_text(encoding='utf-8').splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        if line.startswith('```'):
            index += 1
            block = []
            while index < len(lines) and not lines[index].startswith('```'):
                block.append(plain_dashes(lines[index]))
                index += 1
            story.append(Preformatted('\n'.join(block), code_style))
        elif line.startswith('|'):
            raw = []
            while index < len(lines) and lines[index].startswith('|'):
                row = lines[index]
                if not re.fullmatch(r'[| :\-]+', row):
                    raw.append([s.strip() for s in row.strip('|').split('|')])
                index += 1
            columns = len(raw[0])
            if columns == 6:
                ratios = [.105, .09, .235, .24, .17, .16] if table_count == 0 else [.09, .18, .18, .19, .18, .18]
            elif columns == 3:
                ratios = [.24, .22, .54] if table_count == 1 else [.20, .35, .45]
            else:
                ratios = [.29, .71]
            widths = [doc.width*r/sum(ratios) for r in ratios]
            data = [[Paragraph(markup(value), cell) for value in row] for row in raw]
            table = Table(data, colWidths=widths, repeatRows=1, hAlign='LEFT')
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E7EFF7')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F6F8FA')]),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LINEBELOW', (0, 0), (-1, 0), .6, blue),
                ('LEFTPADDING', (0, 0), (-1, -1), 6), ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6), ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.extend([table, Spacer(1, 9)])
            table_count += 1
            continue
        elif line.startswith('!['):
            target = re.search(r'\]\(([^)]+)\)', line)[1]
            path = SOURCE.parent/target
            with PILImage.open(path) as image:
                width, height = image.size
            image = Image(str(path), width=doc.width, height=doc.width*height/width)
            image.hAlign = 'CENTER'
            block = [image, Spacer(1, 5)]
            if index+2 < len(lines) and lines[index+2].startswith('*Figure '):
                block.append(Paragraph(markup(lines[index+2]), caption))
                index += 2
            story.append(KeepTogether(block))
            images.append(target)
        elif line.startswith('# '):
            story.append(Paragraph(markup(line[2:]), title))
        elif line.startswith('## '):
            heading = line[3:]
            if re.match(r'\d+\.', heading) or heading == 'References and evidence notes':
                story.append(PageBreak())
            story.append(Paragraph(markup(heading), h2))
        elif line.startswith('### '):
            story.append(Paragraph(markup(line[4:]), h3))
        elif line.startswith('- '):
            story.append(Paragraph('- '+markup(line[2:]), bullet))
        else:
            story.append(Paragraph(markup(line), body))
        index += 1

    def page_chrome(canvas, document):
        canvas.saveState()
        canvas.setFont('Vera', 7)
        canvas.setFillColor(gray)
        canvas.drawString(43, A4[1]-26, 'GCN on Cora | Reproduction report v1.1 | English edition')
        canvas.setStrokeColor(colors.HexColor('#DCE2E8'))
        canvas.line(43, 32, A4[0]-43, 32)
        canvas.drawString(43, 20, 'JinglangQiu | Experiments: 2026-10-29 | English edition: 2026-11-02')
        canvas.drawRightString(A4[0]-43, 20, str(document.page))
        canvas.restoreState()

    doc.build(story, onFirstPage=page_chrome, onLaterPages=page_chrome)
    manifest = {'source': 'docs/report_en.md', 'source_sha256': sha256(SOURCE),
                'builder': 'tools/build_report_en_pdf.py', 'builder_sha256': sha256(Path(__file__)),
                'images_sha256': {p: sha256(SOURCE.parent/p) for p in images},
                'tables': table_count, 'figures': len(images),
                'pdf_name': output.name, 'pdf_sha256': sha256(output),
                'rendering': 'ReportLab; embedded Vera fonts; Unicode dashes and arrows rendered as ASCII',
                'training_performed': False, 'test_predictions_recomputed': False}
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2)+'\n',
                                                   encoding='utf-8', newline='\n')
    print(f'Created {output.name}; {table_count} tables, {len(images)} figures.')


if __name__ == '__main__':
    main()
