# run_report.py — One-command orchestration for rolling 30d + 12m report
# Usage:
#   python run_report.py
#   python run_report.py --client test-aws --asof 2025-10-19
#
# Requires:
#   - get_costs.py (updated for --asof)
#   - make_report.py (outputs reports/report_<label>.html)
#   - export_pdf.py (accepts: --html, --pdf, --format, --margin)

import argparse
import os
import sys
import subprocess
from pathlib import Path
from datetime import date, timedelta

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR     = PROJECT_ROOT / "data"
REPORTS_DIR  = PROJECT_ROOT / "reports"

def run(cmd, env=None):
    print(f"\n$ {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr)
        raise SystemExit(f"[ERROR] Command failed: {' '.join(cmd)}")
    if res.stdout.strip():
        print(res.stdout)
    if res.stderr.strip():
        print(res.stderr)
    return res

def latest_html() -> Path | None:
    REPORTS_DIR.mkdir(exist_ok=True, parents=True)
    htmls = sorted(REPORTS_DIR.glob("report_*.html"), key=lambda p: p.stat().st_mtime, reverse=True)
    return htmls[0] if htmls else None

def derive_pdf_path(html_path: Path) -> Path:
    # report_<label>.html -> report_<label>.pdf
    return html_path.with_suffix(".pdf")

def main():
    ap = argparse.ArgumentParser(description="Generate rolling 30d + 12m AWS cost report (HTML + PDF) in one command.")
    ap.add_argument("--client", type=str, default=None, help="AWS profile name to use (sets AWS_PROFILE).")
    ap.add_argument("--asof", type=str, default=None, help="Anchor date YYYY-MM-DD (default = yesterday).")
    ap.add_argument("--format", type=str, default="Letter", help="PDF page size for export_pdf.py (e.g. Letter, A4).")
    ap.add_argument("--margin", type=str, default="0.5in", help="PDF margin for export_pdf.py (CSS-style unit).")
    args = ap.parse_args()

    env = os.environ.copy()
    if args.client:
        env["AWS_PROFILE"] = args.client
        print(f"[info] Using AWS profile: {args.client}")

    # Default as-of = yesterday (to avoid partial last-day data)
    asof = args.asof
    if not asof:
        asof = (date.today() - timedelta(days=1)).isoformat()
        print(f"[info] Using default as-of: {asof}")

    # 1) Pull data (writes two CSVs into data/)
    run([sys.executable, "get_costs.py", "--asof", asof], env=env)

    # 2) Build HTML (writes reports/report_<label>.html)
    run([sys.executable, "make_report.py"], env=env)

    # 3) Find newest HTML and export to PDF
    html_path = latest_html()
    if not html_path:
        raise SystemExit("[ERROR] No report_*.html found in reports/. Did make_report.py run?")
    pdf_path = derive_pdf_path(html_path)

    # 4) Export PDF via your existing exporter
    run([
        sys.executable, "export_pdf.py",
        "--html", str(html_path),
        "--pdf",  str(pdf_path),
        "--format", args.format,
        "--margin", args.margin,
    ], env=env)

    print("\n[OK] End-to-end complete.")
    print(f"HTML: {html_path}")
    print(f"PDF : {pdf_path}")

if __name__ == "__main__":
    main()


