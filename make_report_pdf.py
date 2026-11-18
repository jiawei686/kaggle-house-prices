"""Render report.html to report.pdf.

Chromium is driven over CDP so that the two accessibility options which
playwright's page.pdf() does not expose can be switched on:

    generateTaggedPDF        structure tree, so screen readers and text
                             extractors get headings, lists and tables
    generateDocumentOutline  PDF bookmarks from the h1/h2/h3 hierarchy

If the installed browser build does not match the playwright version, point
CHROME_PATH at any Chromium/Chrome binary:

    CHROME_PATH="/path/to/Chrome for Testing" python make_report_pdf.py

A4, 16-18 mm margins, page numbers in the footer, print stylesheet from the
@media print block in report.html.
"""
import base64
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

MM = 25.4  # millimetres per inch

PDF_OPTIONS = {
    "printBackground": True,
    "paperWidth": 210 / MM,
    "paperHeight": 297 / MM,
    "marginTop": 18 / MM,
    "marginBottom": 16 / MM,
    "marginLeft": 16 / MM,
    "marginRight": 16 / MM,
    "displayHeaderFooter": True,
    "headerTemplate": "<div></div>",
    "footerTemplate": FOOTER,
    "preferCSSPageSize": False,
    "generateTaggedPDF": True,
    "generateDocumentOutline": True,
}

META = {
    "title": "House Prices - Method Report",
    "author": "we are nai looong (Kaggle group we-are-nai-looong)",
    "subject": "Kaggle House Prices: public LB 0.11569, rank 236/4957 (top 4.76%)",
    "keywords": (
        "kaggle, house prices, ensemble, non-negative least squares, lightgbm, "
        "xgboost, catboost, outlier analysis, error analysis"
    ),
    "creator": "report.html -> Chromium print to PDF",
    "producer": "PyMuPDF",
}


def main() -> None:
    exe = os.environ.get("CHROME_PATH")
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=exe) if exe else p.chromium.launch()
        page = browser.new_page()
        page.goto(SRC.as_uri(), wait_until="load")
        page.emulate_media(media="print")
        page.wait_for_timeout(400)

        client = page.context.new_cdp_session(page)
        result = client.send("Page.printToPDF", PDF_OPTIONS)
        OUT.write_bytes(base64.b64decode(result["data"]))
        browser.close()

    doc = fitz.open(str(OUT))
    doc.set_metadata(META)
    doc.save(str(OUT), incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP)
    toc = doc.get_toc()
    print(f"{OUT.name}: {doc.page_count} pages, {OUT.stat().st_size} bytes, "
          f"{len(toc)} bookmarks")
    doc.close()


if __name__ == "__main__":
    main()
