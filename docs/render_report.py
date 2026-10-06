from pathlib import Path
import json, html, shutil, subprocess
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import reportlab

CSS='''body{max-width:900px;margin:50px auto;padding:0 28px;font:17px/1.6 Georgia,serif;color:#18212c}h1,h2,h3{font-family:Arial,sans-serif;line-height:1.2}h1{font-size:34px}h2{font-size:23px;margin-top:36px}a{color:#185d80;overflow-wrap:anywhere}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;border-bottom:1px solid #cad4df;text-align:left;vertical-align:top}th{background:#edf2f6}.note{background:#f1f5f8;border-left:4px solid #367291;padding:16px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f4f6;padding:16px;font-size:13px}.subtitle{font-size:21px;color:#45586b}.small{font-size:14px;color:#526171}section{margin-bottom:25px}@media print{body{max-width:none;margin:0}.page{break-after:page}a{color:inherit}}'''
def page(title,body):return '<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>'+html.escape(title)+'</title><style>'+CSS+'</style></head><body>'+body+'</body></html>'
def note(r):return f'Originally released publicly in March {r["year"]}. This archival technical-report edition was prepared on 5 October 2026 from the public project record. The original release date is distinct from the intended October 2026 archival deposit. No Zenodo deposit or DOI registration has been completed for this edition; the actual deposit date will be recorded when publication occurs.'

font_dir=Path('/System/Library/Fonts/Supplemental')
fallback_dir=Path(reportlab.__file__).parent/'fonts'
for font_name,filename,fallback in [('Helvetica','Arial.ttf','Vera.ttf'),('Helvetica-Bold','Arial Bold.ttf','VeraBd.ttf'),('Times-Roman','Times New Roman.ttf','Vera.ttf')]:
 font_path=font_dir/filename
 if not font_path.exists(): font_path=fallback_dir/fallback
 pdfmetrics.registerFont(TTFont(font_name,str(font_path)))
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleA',fontName='Helvetica-Bold',fontSize=24,leading=28,spaceAfter=10,textColor=colors.HexColor('#142b3d')))
styles.add(ParagraphStyle(name='SubA',fontName='Helvetica',fontSize=12,leading=16,spaceAfter=12,textColor=colors.HexColor('#385568')))
styles.add(ParagraphStyle(name='BodyA',fontName='Times-Roman',fontSize=10.5,leading=14,spaceAfter=9))
styles.add(ParagraphStyle(name='HeadA',fontName='Helvetica-Bold',fontSize=12,leading=15,spaceBefore=12,spaceAfter=8,textColor=colors.HexColor('#142b3d'),keepWithNext=True))
styles.add(ParagraphStyle(name='SmallA',fontName='Helvetica',fontSize=8,leading=11,spaceAfter=8,splitLongWords=True))
styles.add(ParagraphStyle(name='CellA',fontName='Helvetica',fontSize=8.3,leading=11,spaceAfter=0))
styles.add(ParagraphStyle(name='NoteA',fontName='Helvetica',fontSize=8.1,leading=11,spaceBefore=6,spaceAfter=12,backColor=colors.HexColor('#eff4f7'),borderPadding=8))
def p(s,sty='BodyA'):return Paragraph(html.escape(s),styles[sty])
def pdf(r,d):
 story=[p(r['title'],'TitleA'),p(r['subtitle'],'SubA'),p('Michael Hla','SubA'),p(f'Original release: March {r["year"]}  |  Archival edition: October 2026','SmallA'),p(note(r),'NoteA'),p('Abstract','HeadA'),p(r['abstract'])]
 for i,blocks in enumerate(r['pages']):
  if i:story.append(PageBreak())
  for typ,val in blocks:
   if typ in ['p','h','formula','caption']:story.append(p(val,{'p':'BodyA','h':'HeadA','formula':'SmallA','caption':'SmallA'}[typ]))
   elif typ=='table':
    heads,rows=val; data=[[p(v,'CellA') for v in row] for row in [heads]+rows]
    widths=[118,180,177] if len(heads)==3 else [94,118,128,135]
    t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8f0f5')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,-1),.4,colors.HexColor('#cad5dd')),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]));story.extend([t,Spacer(1,10)])
   elif typ=='refs':
    for n,(label,url) in enumerate(val,1):
     story.append(Paragraph(f'[{n}] {html.escape(label)} <a href="{html.escape(url)}" color="#185d80">Source</a>.',styles['SmallA']))
 def foot(c,doc):
  c.setStrokeColor(colors.HexColor('#c3cdd5'));c.line(60,45,A4[0]-60,45);c.setFont('Helvetica',8);c.setFillColor(colors.HexColor('#536675'))
  c.drawString(60,32,r['title']+' | October 2026 archival edition');c.drawRightString(A4[0]-60,32,str(doc.page))
 doc=SimpleDocTemplate(str(d/(r['slug']+'.pdf')),pagesize=A4,rightMargin=60,leftMargin=60,topMargin=48,bottomMargin=60,title=r['title'],author='Michael Hla',subject=r['subtitle'])
 doc.build(story,onFirstPage=foot,onLaterPages=foot)

def report_html(r):
 body=f'<h1>{html.escape(r["title"])}</h1><p class="subtitle">{html.escape(r["subtitle"])}</p><p><strong>Michael Hla</strong></p><p>Original release: March {r["year"]} | Archival edition: October 2026</p><p class="note">{html.escape(note(r))}</p><h2>Abstract</h2><p>{html.escape(r["abstract"])}</p>'
 for blocks in r['pages']:
  for typ,v in blocks:
   if typ=='h':body+='<h2>'+html.escape(v)+'</h2>'
   elif typ=='p':body+='<p>'+html.escape(v)+'</p>'
   elif typ=='formula':body+='<pre>'+html.escape(v)+'</pre>'
   elif typ=='caption':body+='<p class="small">'+html.escape(v)+'</p>'
   elif typ=='table':
    heads,rows=v;body+='<table><thead><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in heads)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'
   elif typ=='refs':body+='<ol>'+''.join(f'<li>{html.escape(label)} <a href="{html.escape(url)}">{html.escape(url)}</a></li>' for label,url in v)+'</ol>'
 return page(r['title'],body)


if __name__ == '__main__':
 import argparse
 parser=argparse.ArgumentParser(description='Render an archival report from its editable content JSON; requires reportlab.')
 parser.add_argument('source',type=Path)
 parser.add_argument('--output-dir',type=Path)
 args=parser.parse_args()
 report=json.loads(args.source.read_text())
 dest=args.output_dir or args.source.parent
 dest.mkdir(parents=True,exist_ok=True)
 pdf(report,dest)
 (dest/(report['slug']+'.html')).write_text(report_html(report))
 print('Rendered',report['slug'])
