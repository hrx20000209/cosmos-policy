#!/usr/bin/env python3
"""Render REPORT_ZH.md to reproducible HTML and PDF with a system CJK font."""
from __future__ import annotations
import argparse
from pathlib import Path
import markdown
from weasyprint import HTML
def main():
 p=argparse.ArgumentParser(); p.add_argument('--output-dir',type=Path,required=True); a=p.parse_args(); src=a.output_dir/'REPORT_ZH.md'
 body=markdown.markdown(src.read_text(),extensions=['tables','fenced_code'])
 css="""@page{size:A4;margin:18mm}body{font-family:'Noto Sans CJK SC','WenQuanYi Zen Hei','DejaVu Sans',sans-serif;line-height:1.55;color:#222}h1,h2{color:#15385f}code{font-family:monospace;background:#f2f2f2;padding:1px 3px}table{border-collapse:collapse;width:100%;font-size:9pt}td,th{border:1px solid #aaa;padding:4px}blockquote{border-left:4px solid #c44;padding-left:10px;color:#555}img{max-width:100%}"""
 html=f'<!doctype html><html lang="zh"><meta charset="utf-8"><style>{css}</style><body>{body}</body></html>'
 hp=a.output_dir/'REPORT_ZH.html'; hp.write_text(html); HTML(string=html,base_url=str(a.output_dir)).write_pdf(a.output_dir/'REPORT_ZH.pdf'); print(a.output_dir/'REPORT_ZH.pdf')
if __name__=='__main__': main()
