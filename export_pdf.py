# export_pdf.py — Playwright/Chromium HTML -> PDF (Windows)
import argparse
import os
import platform
import subprocess
from pathlib import Path
from playwright.sync_api import sync_playwright

def parse_args():
    p = argparse.ArgumentParser(description="Export an HTML file to PDF via Playwright/Chromium.")
    p.add_argument("--html", required=True, help="Path to input HTML file.")
    p.add_argument("--pdf",  required=True, help="Path to output PDF file.")
    p.add_argument("--format", default="Letter", help="PDF page format (e.g., Letter, A4).")
    p.add_argument("--margin", default="0.5in", help='Uniform margins (e.g., 0.5in, 10mm).')
    p.add_argument("--no-bg", action="store_true", help="Disable printing backgrounds.")
    p.add_argument("--no-css-page-size", action="store_true", help="Ignore @page size and use --format instead.")
    return p.parse_args()

def main():
    args = parse_args()

    html_path = Path(args.html).resolve()
    if not html_path.exists():
        print(f"[ERROR] HTML not found: {html_path}")
        raise SystemExit(1)

    pdf_path = Path(args.pdf).resolve()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    html_uri = html_path.as_uri()  # file:///C:/.../reports/report_YYYY-MM.html

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        # Load the actual file so relative assets (assets/...) resolve correctly
        page.goto(html_uri, wait_until="networkidle")

        # Use print CSS
        try:
            page.emulate_media(media="print")
        except Exception:
            pass

        # Best-effort waits (don’t fail if something’s missing)
        try:
            page.wait_for_function("Array.from(document.images).every(i => i.complete)", timeout=2000)
        except Exception:
            pass
        try:
            page.wait_for_function("document.fonts && document.fonts.status === 'loaded'", timeout=1500)
        except Exception:
            pass

        pdf_options = {
            "path": str(pdf_path),
            "print_background": (not args.no_bg),
            "margin": {"top": args.margin, "right": args.margin, "bottom": args.margin, "left": args.margin},
            "prefer_css_page_size": (not args.no_css_page_size),
        }
        if args.no_css_page_size and args.format:
            pdf_options["format"] = args.format

        page.pdf(**pdf_options)
        browser.close()

    # Auto-open PDF on Windows
    opened = False
    try:
        os.startfile(str(pdf_path))  # type: ignore[attr-defined]
        opened = True
    except Exception:
        pass
    if not opened and platform.system() == "Windows":
        try:
            subprocess.Popen(["cmd", "/c", "start", "", str(pdf_path)], shell=True)
            opened = True
        except Exception:
            pass

    print(f"[OK] PDF written: {pdf_path}")
    if opened:
        print("[OK] PDF opened in your default viewer.")
    else:
        print(f"[INFO] Could not auto-open. You can open it with:\n  start \"\" \"{pdf_path}\"")

if __name__ == "__main__":
    main()



