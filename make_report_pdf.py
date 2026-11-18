"""Render report.html to report.pdf.

Needs playwright (pip install playwright, then `playwright install chromium`).
If the installed browser build does not match the playwright version, point
CHROME_PATH at any Chromium/Chrome binary instead:

    CHROME_PATH="/path/to/Chrome for Testing" python make_report_pdf.py

A4, 16-18 mm margins, page numbers in the footer, print stylesheet from the
@media print block in report.html.
"""
import os
from pathlib import Path

import fitz
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "report.html"
OUT = ROOT / "report.pdf"

FOOTER = (
    '<div style="width:100%;text-align:center;font-size:8px;color:#777;'
    'font-family:-apple-system,Arial,sans-serif;">'
    '<span class="pageNumber"></span> / <span class="totalPages"></span></div>'
)

if __name__ == "__main__":
    exe = os.environ.get("CHROME_PATH")
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=exe) if exe else p.chromium.launch()
        page = browser.new_page()
        page.goto(SRC.as_uri(), wait_until="load")
        page.emulate_media(media="print")
        page.wait_for_timeout(400)
        page.pdf(
            path=str(OUT),
            format="A4",
            print_background=True,
            display_header_footer=True,
            header_template="<div></div>",
            footer_template=FOOTER,
            margin={"top": "18mm", "bottom": "16mm", "left": "16mm", "right": "16mm"},
            prefer_css_page_size=False,
        )
        browser.close()

    doc = fitz.open(str(OUT))
    doc.set_metadata({
        "title": "House Prices — Method Report",
        "author": "we are nai looong (Kaggle group we-are-nai-looong)",
        "subject": "Kaggle House Prices: public LB 0.11569, rank 236/4957 (top 4.76%)",
        "keywords": "kaggle, house prices, ensemble, lightgbm, xgboost, catboost, outlier analysis",
        "creator": "report.html -> Chromium print to PDF",
        "producer": "PyMuPDF",
    })
    doc.save(str(OUT), incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP)
    print(f"{OUT.name}: {doc.page_count} pages, {OUT.stat().st_size} bytes")
    doc.close()
