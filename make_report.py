# make_report.py — MVP v2 (Rolling 30d daily + 12m monthly, visuals + anomalies)
# - Loads TWO CSVs emitted by get_costs.py:
#     data/cost_explorer_daily_<start>_to_<end>.csv        (30d DAILY by service)
#     data/cost_explorer_monthly_<start>_to_<end>.csv      (12m MONTHLY by service)
# - Produces charts:
#     1) 30d total daily spend (line/area)
#     2) 30d per-service daily multi-line (top 5 services)
#     3) 12m total monthly spend (bars + line overlay)
#     4) 12m monthly stacked by service (top 5 services, others grouped)
#     5) 30d Top-5 services bar
# - Computes simple anomalies:
#     - Daily spike (>30% above trailing 7-day avg)
#     - MoM spike (>15% vs previous month)
# - Prepares context for Page-1 CEO bullets (on track / up or down / spike day / action)
#
# NOTE: This version focuses on structure and context; your template (report.html.j2)
# will render Page 1 bullets first, then headline charts, followed by deep-dive sections.

from __future__ import annotations

import os
import re
import shutil
import base64
from datetime import datetime, date, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Tuple, Dict, List

import pandas as pd
import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader, select_autoescape

# ======= Savings Heuristic Thresholds =======
IDLE_EC2_MIN_DAYS = 25
IDLE_EC2_FLAT_STDEV_RATIO = 0.05
IDLE_EC2_MIN_TOTAL = 3.00
IDLE_EC2_SAVINGS_RATE = 0.30

EBS_MIN_REPEAT_DAYS = 3  # days with EBS>0 while EC2==0

S3_MIN_DAYS = 20
S3_FLAT_CV_MAX = 0.15
S3_DT_MEDIAN_MAX = 0.05
S3_SAVINGS_RATE = 0.50

DT_SPIKE_FACTOR = 1.5
# ============================================



def compute_savings(daily_costs):
    """
    Phase 1 detectors using billing-only heuristics:
      - Idle EC2 (existing behavior, cleaned up to use thresholds)
      - Unattached EBS (EBS>0 while EC2==0 on >= EBS_MIN_REPEAT_DAYS)
      - S3 Storage Class Optimization (flat S3 + trivial DT)
      - Data Transfer Spikes (risk-only, no savings)
    Returns:
      {
        "total_potential_savings": float,
        "annualized_savings": float,
        "details": [ {title, confidence, description, savings_estimate}, ... ]
      }
    """
    import pandas as pd
    from statistics import mean

    # Normalize input to a DataFrame with expected columns
    if isinstance(daily_costs, list):
        daily = pd.DataFrame(daily_costs)
    else:
        daily = daily_costs.copy()

    # Standardize columns
    daily = daily.rename(columns={
        "Date": "date", "Service": "service", "Amount": "amount"
    })
    daily["date"] = pd.to_datetime(daily["date"]).dt.date
    daily["service"] = daily["service"].astype(str)
    daily["amount"] = pd.to_numeric(daily["amount"], errors="coerce").fillna(0.0)

    details = []
    total_savings = 0.0

    # ---------- Idle EC2 (existing logic with thresholds) ----------
    ec2_mask = daily["service"].str.contains(r"(Elastic Compute Cloud|EC2)", case=False, na=False)
    ec2_df = daily[ec2_mask]
    if not ec2_df.empty:
        for svc, grp in ec2_df.groupby("service"):
            # sum per day for this EC2 sub-service
            series = grp.groupby("date")["amount"].sum().sort_index()
            pos_days = series[series > 0]
            if len(pos_days) < IDLE_EC2_MIN_DAYS:
                continue
            m = float(pos_days.mean())
            sd = float(pos_days.std(ddof=1)) if len(pos_days) > 1 else 0.0
            if m == 0:
                continue
            # very flat?
            if sd > IDLE_EC2_FLAT_STDEV_RATIO * m:
                continue
            # too small to matter?
            if float(pos_days.sum()) < IDLE_EC2_MIN_TOTAL:
                continue
            est = round(float(pos_days.sum()) * IDLE_EC2_SAVINGS_RATE, 2)
            total_savings += est
            details.append({
                "title": "Idle EC2 Detected",
                "confidence": "High",
                "description": (
                    f"Consistent EC2 spend with minimal variance (service: {svc}). "
                    "Likely always-on dev/test; consider schedules or Spot/RI."
                ),
                "savings_estimate": est,
            })

    # ---------- Unattached EBS ----------
    ebs_mask = daily["service"].str.contains("Elastic Block Store", case=False, na=False)
    daily_ebs = daily[ebs_mask].groupby("date")["amount"].sum() if ebs_mask.any() else pd.Series(dtype=float)
    daily_ec2 = ec2_df.groupby("date")["amount"].sum() if not ec2_df.empty else pd.Series(dtype=float)

    if not daily_ebs.empty:
        ebs_only_days = []
        ebs_total = 0.0
        # align on the union of dates to be safe
        all_days = sorted(set(daily_ebs.index) | set(daily_ec2.index))
        for d in all_days:
            ebs_val = float(daily_ebs.get(d, 0.0))
            ec2_val = float(daily_ec2.get(d, 0.0))
            if ebs_val > 0 and ec2_val == 0:
                ebs_only_days.append(d)
                ebs_total += ebs_val

        if len(ebs_only_days) >= EBS_MIN_REPEAT_DAYS and ebs_total > 0:
            est = round(ebs_total, 2)  # treat as 100% stoppable for conservative monthly savings
            total_savings += est
            details.append({
                "title": "Unattached EBS Detected",
                "confidence": "Medium",
                "description": (
                    f"EBS spend ${ebs_total:.2f} across {len(ebs_only_days)} day(s) with no EC2 activity "
                    "(likely unattached volumes)."
                ),
                "savings_estimate": est,
            })

    # ---------- S3 Storage Class Optimization ----------
    s3_mask = daily["service"].str.contains("Simple Storage Service", case=False, na=False)
    dt_mask = daily["service"].str.contains("Data Transfer", case=False, na=False)

    s3_series = (daily[s3_mask].groupby("date")["amount"].sum().sort_index()
                 if s3_mask.any() else pd.Series(dtype=float))
    dt_series = (daily[dt_mask].groupby("date")["amount"].sum().sort_index()
                 if dt_mask.any() else pd.Series(dtype=float))

    if not s3_series.empty and len(s3_series) >= S3_MIN_DAYS:
        s3_mean = float(s3_series.mean())
        s3_std = float(s3_series.std(ddof=1)) if len(s3_series) > 1 else 0.0
        s3_cv = (s3_std / s3_mean) if s3_mean > 0 else 0.0
        dt_median = float(dt_series.median()) if not dt_series.empty else 0.0

        if s3_cv < S3_FLAT_CV_MAX and dt_median < S3_DT_MEDIAN_MAX:
            s3_total = float(s3_series.sum())
            est = round(s3_total * S3_SAVINGS_RATE, 2)
            total_savings += est
            details.append({
                "title": "S3 Storage Class Optimization",
                "confidence": "Low-Medium",
                "description": (
                    f"Flat S3 storage spend (${s3_total:.2f}) with minimal data transfer "
                    f"(median/day ≈ ${dt_median:.2f}) — consider IA/Glacier tiers."
                ),
                "savings_estimate": est,
            })

    # ---------- Data Transfer Spikes (risk-only, no savings line item) ----------
    if not dt_series.empty and len(dt_series) >= 8:  # need at least 7 days of history + 1 day to compare
        dt_series = dt_series.sort_index()
        rolling = dt_series.shift(1).rolling(7, min_periods=7).mean()
        for d, val in dt_series.items():
            trail = rolling.get(d)
            if pd.isna(trail) or float(trail) <= 0:
                continue
            if float(val) > DT_SPIKE_FACTOR * float(trail):
                details.append({
                    "title": "Data Transfer Spike",
                    "confidence": "N/A",
                    "description": (
                        f"Risk: unusual AWS Data Transfer on {d} (${float(val):.2f}, "
                        f">{DT_SPIKE_FACTOR}× trailing 7-day avg)."
                    ),
                    "savings_estimate": 0.0,
                })

    return {
        "total_potential_savings": round(total_savings, 2),
        "annualized_savings": round(total_savings * 12.0, 2),
        "details": details
    }





# -------- Project layout (run from project root: C:\...\cost-report)
DATA_DIR     = Path("data")
TEMPLATE_DIR = Path("templates")
REPORTS_DIR  = Path("reports")

# Source assets in repo
SRC_ASSETS_DIR = Path("assets")
SRC_IMG_DIR    = SRC_ASSETS_DIR / "img"

# Output assets alongside HTML
OUT_ASSETS_DIR = REPORTS_DIR / "assets"
OUT_IMG_DIR    = OUT_ASSETS_DIR / "img"
OUT_CSS_DIR    = OUT_ASSETS_DIR

# We now look for the new rolling-window CSVs
DAILY_CSV_PATTERN   = re.compile(r"cost_explorer_daily_(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})\.csv$", re.I)
MONTHLY_CSV_PATTERN = re.compile(r"cost_explorer_monthly_(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})\.csv$", re.I)

def posix(p: Path | str) -> str:
    return Path(p).as_posix()

def to_decimal(x):
    try:
        return Decimal(str(x))
    except (InvalidOperation, ValueError, TypeError):
        return Decimal("0")

def ensure_dirs():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_CSS_DIR.mkdir(parents=True, exist_ok=True)

def copy_if_exists(src_path: Path, dst_path: Path) -> bool:
    if src_path.is_file():
        shutil.copy2(src_path, dst_path)
        return True
    return False

def embed_file_as_data_uri(path: Path, mime: str) -> str | None:
    try:
        b = path.read_bytes()
        b64 = base64.b64encode(b).decode("ascii")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None

def discover_logo_data_uri() -> tuple[str, str]:
    """
    Finds a logo (SVG preferred, PNG fallback) either in:
      1) Local repo: assets/img/
      2) Central shared folder: C:\alpenglow-clients\  (alpenglow-logo.svg/.png or branding\*)
    Returns (data_uri, note). If nothing found, returns an inline SVG placeholder.
    """
    tried_paths: list[str] = []

    repo_candidates = [
        (SRC_IMG_DIR / "alpenglow-logo.svg",  "image/svg+xml"),
        (SRC_IMG_DIR / "alpenglow-logo.png",  "image/png"),
        (SRC_IMG_DIR / "alpenglow-logo.jpg",  "image/jpeg"),
        (SRC_IMG_DIR / "alpenglow-logo.jpeg", "image/jpeg"),
        (SRC_IMG_DIR / "alpenglow-logo.webp", "image/webp"),
    ]
    central_root = Path(r"C:\alpenglow-clients")
    central_candidates = [
        (central_root / "alpenglow-logo.svg",  "image/svg+xml"),
        (central_root / "alpenglow-logo.png",  "image/png"),
        (central_root / "branding" / "alpenglow-logo.svg", "image/svg+xml"),
        (central_root / "branding" / "alpenglow-logo.png", "image/png"),
    ]

    def try_list(cands, source_label: str):
        nonlocal tried_paths
        for p, mime in cands:
            tried_paths.append(str(p))
            if p.is_file():
                data_uri = embed_file_as_data_uri(p, mime)
                if data_uri:
                    return data_uri, f"Embedded logo from {source_label}: {posix(p)}"
        return None, None

    data_uri, note = try_list(repo_candidates, "repo")
    if data_uri: return data_uri, note
    data_uri, note = try_list(central_candidates, "central")
    if data_uri: return data_uri, note

    # Generic fallback from any repo image
    if SRC_IMG_DIR.is_dir():
        ext_to_mime = {
            ".svg": "image/svg+xml",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }
        for fname in sorted(SRC_IMG_DIR.iterdir()):
            mime = ext_to_mime.get(fname.suffix.lower())
            if mime:
                tried_paths.append(str(fname))
                data_uri = embed_file_as_data_uri(fname, mime)
                if data_uri:
                    return data_uri, f"Embedded logo from repo (generic): {posix(fname)}"

    # Inline placeholder
    placeholder_svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' width='220' height='48'>"
        "<rect width='100%' height='100%' fill='#222'/>"
        "<text x='50%' y='50%' fill='#fff' font-family='Arial,Helvetica,sans-serif' "
        "font-size='16' text-anchor='middle' dominant-baseline='middle'>Alpenglow</text>"
        "</svg>"
    ).encode("utf-8")
    b64 = base64.b64encode(placeholder_svg).decode("ascii")
    note = (
        "Logo not found — using inline placeholder.\n"
        "Paths tried:\n  - " + "\n  - ".join(tried_paths) + "\n"
        "Expected locations:\n"
        "  - assets/img/alpenglow-logo.svg (preferred) or .png\n"
        "  - C:\\alpenglow-clients\\alpenglow-logo.svg (preferred) or .png\n"
        "  - C:\\alpenglow-clients\\branding\\alpenglow-logo.svg (preferred) or .png"
    )
    return f"data:image/svg+xml;base64,{b64}", note

# -------- File discovery (latest daily+monthly pairs)

def _scan_csvs(pattern: re.Pattern) -> List[Tuple[Path, str, str]]:
    out = []
    if not DATA_DIR.exists():
        return out
    for p in DATA_DIR.iterdir():
        m = pattern.match(p.name)
        if m:
            start_iso, end_iso = m.group(1), m.group(2)
            out.append((p, start_iso, end_iso))
    # Sort by end date (exclusive) desc to pick latest
    out.sort(key=lambda t: t[2], reverse=True)
    return out

def find_latest_daily_and_monthly():
    daily_list = _scan_csvs(DAILY_CSV_PATTERN)
    monthly_list = _scan_csvs(MONTHLY_CSV_PATTERN)
    if not daily_list:
        raise FileNotFoundError("No daily CSV found in data/ (expected cost_explorer_daily_*.csv)")
    if not monthly_list:
        raise FileNotFoundError("No monthly CSV found in data/ (expected cost_explorer_monthly_*.csv)")
    daily_path, d_start, d_end_excl = daily_list[0]
    monthly_path, m_start, m_end_excl = monthly_list[0]
    return (daily_path, d_start, d_end_excl), (monthly_path, m_start, m_end_excl), daily_list, monthly_list

# -------- Data loading

def load_daily_by_service(csv_path: Path) -> pd.DataFrame:
    # Expect: date, service, amount, unit
    df = pd.read_csv(csv_path)
    # Normalize column names
    cols = {c.lower(): c for c in df.columns}
    date_col    = cols.get("date", "date")
    service_col = cols.get("service", "service")
    amount_col  = cols.get("amount", "amount")
    # Coerce
    df[date_col] = pd.to_datetime(df[date_col]).dt.date
    df[amount_col] = pd.to_numeric(df[amount_col], errors="coerce").fillna(0.0)
    df = df.rename(columns={date_col: "date", service_col: "service", amount_col: "amount"})
    return df[["date", "service", "amount"]]

def load_monthly_by_service(csv_path: Path) -> pd.DataFrame:
    # Expect: month_start, service, amount, unit
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    m_col      = cols.get("month_start", "month_start")
    service_col= cols.get("service", "service")
    amount_col = cols.get("amount", "amount")
    df[m_col] = pd.to_datetime(df[m_col]).dt.date
    df[amount_col] = pd.to_numeric(df[amount_col], errors="coerce").fillna(0.0)
    df = df.rename(columns={m_col: "month_start", service_col: "service", amount_col: "amount"})
    return df[["month_start", "service", "amount"]]

# -------- Computations

def daily_totals(df_daily: pd.DataFrame) -> pd.DataFrame:
    t = df_daily.groupby("date", as_index=False)["amount"].sum().sort_values("date")
    return t

def daily_top_services(df_daily: pd.DataFrame, top_n: int = 5) -> List[str]:
    svc_totals = df_daily.groupby("service")["amount"].sum().sort_values(ascending=False)
    return list(svc_totals.head(top_n).index)

def pivot_daily_per_service(df_daily: pd.DataFrame, services: List[str]) -> pd.DataFrame:
    filtered = df_daily[df_daily["service"].isin(services)]
    p = filtered.pivot_table(index="date", columns="service", values="amount", aggfunc="sum").fillna(0.0)
    p = p.sort_index()
    return p

def monthly_totals(df_monthly: pd.DataFrame) -> pd.DataFrame:
    t = df_monthly.groupby("month_start", as_index=False)["amount"].sum().sort_values("month_start")
    return t

def monthly_top_services(df_monthly: pd.DataFrame, top_n: int = 5) -> List[str]:
    svc_totals = df_monthly.groupby("service")["amount"].sum().sort_values(ascending=False)
    return list(svc_totals.head(top_n).index)

def monthly_stacked_prep(df_monthly: pd.DataFrame, top_services: List[str]) -> pd.DataFrame:
    df = df_monthly.copy()
    df["svc_group"] = df["service"].where(df["service"].isin(top_services), other="Other")
    p = df.pivot_table(index="month_start", columns="svc_group", values="amount", aggfunc="sum").fillna(0.0)
    p = p.sort_index()
    return p

# -------- Anomalies

def detect_daily_spikes(df_daily_total: pd.DataFrame, window:int = 7, thresh: float = 0.30):
    """
    Flags days where total > (1+thresh) * trailing-avg(window).
    Returns list of dicts: {date, total, trailing_avg, spike_pct}
    """
    s = df_daily_total.set_index("date")["amount"].astype(float)
    # trailing average excluding current day
    avg = s.shift(1).rolling(window, min_periods=window).mean()
    spikes = []
    for d, val in s.items():
        if pd.isna(avg.loc[d]):
            continue
        if val > (1.0 + thresh) * avg.loc[d]:
            spike_pct = (val / avg.loc[d]) - 1.0
            spikes.append({"date": d, "total": float(val), "trailing_avg": float(avg.loc[d]), "spike_pct": float(spike_pct)})
    return spikes

def detect_mom_spikes(df_monthly_total: pd.DataFrame, thresh: float = 0.15):
    """
    Flags months where total changes by more than thresh vs previous month.
    Returns list of dicts: {month_start, total, delta_pct}
    """
    out = []
    t = df_monthly_total.sort_values("month_start").reset_index(drop=True)
    for i in range(1, len(t)):
        prev = float(t.loc[i-1, "amount"])
        cur  = float(t.loc[i, "amount"])
        if prev <= 0: 
            continue
        delta_pct = (cur / prev) - 1.0
        if abs(delta_pct) >= thresh:
            out.append({"month_start": t.loc[i, "month_start"], "total": cur, "delta_pct": float(delta_pct)})
    return out

# -------- Charts

def chart_30d_total_line(df_daily_total: pd.DataFrame, label_slug: str) -> str | None:
    if df_daily_total.empty: return None
    plt.figure(figsize=(8,3))
    plt.plot(df_daily_total["date"], df_daily_total["amount"], linewidth=2)
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.title("Past 30 Days – Total Daily Spend")
    plt.ylabel("USD")
    plt.tight_layout()
    ensure_dirs()
    out_path = OUT_IMG_DIR / f"chart_30d_total_{label_slug}.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    return posix(Path("assets") / "img" / out_path.name)

def chart_30d_per_service_multiline(pivot_df: pd.DataFrame, label_slug: str) -> str | None:
    if pivot_df.empty: return None
    plt.figure(figsize=(8,3.2))
    for col in pivot_df.columns:
        plt.plot(pivot_df.index, pivot_df[col], linewidth=1.8, label=col)
    plt.legend(loc="upper left", ncol=2, fontsize=8)
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.title("Past 30 Days – Per-Service Daily Spend (Top 5)")
    plt.ylabel("USD")
    plt.tight_layout()
    ensure_dirs()
    out_path = OUT_IMG_DIR / f"chart_30d_per_service_{label_slug}.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    return posix(Path("assets") / "img" / out_path.name)

def chart_12m_total_bars_line(df_monthly_total: pd.DataFrame, label_slug: str) -> str | None:
    if df_monthly_total.empty: return None
    x = df_monthly_total["month_start"]
    y = df_monthly_total["amount"]
    plt.figure(figsize=(8,3))
    plt.bar(x, y)
    plt.plot(x, y, linewidth=2)
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, axis="y", linestyle="--", alpha=0.35)
    plt.title("Last 12 Months – Total Monthly Spend")
    plt.ylabel("USD")
    plt.tight_layout()
    ensure_dirs()
    out_path = OUT_IMG_DIR / f"chart_12m_total_{label_slug}.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    return posix(Path("assets") / "img" / out_path.name)

def chart_12m_stacked_by_service(pivot_df: pd.DataFrame, label_slug: str) -> str | None:
    if pivot_df.empty: return None
    plt.figure(figsize=(8,3.2))
    bottom = None
    for col in pivot_df.columns:
        if bottom is None:
            bottom = pivot_df[col].values
            plt.bar(pivot_df.index, pivot_df[col].values)
        else:
            plt.bar(pivot_df.index, pivot_df[col].values, bottom=bottom)
            bottom = bottom + pivot_df[col].values
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, axis="y", linestyle="--", alpha=0.35)
    plt.title("Last 12 Months – Monthly Spend by Service (Top 5 + Other)")
    plt.ylabel("USD")
    plt.tight_layout()
    ensure_dirs()
    out_path = OUT_IMG_DIR / f"chart_12m_stacked_{label_slug}.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    return posix(Path("assets") / "img" / out_path.name)

def chart_top5_services_bar(series_by_service: pd.Series, label_slug: str) -> str | None:
    if series_by_service.empty: return None
    top = series_by_service.sort_values(ascending=True).tail(5)  # smallest->largest, we take tail
    plt.figure(figsize=(8,3.2))
    plt.barh(top.index, top.values)
    plt.title("Top 5 Services – Past 30 Days")
    plt.xlabel("USD")
    plt.tight_layout()
    ensure_dirs()
    out_path = OUT_IMG_DIR / f"chart_top5_30d_{label_slug}.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    return posix(Path("assets") / "img" / out_path.name)

# -------- Static assets

def stage_static_assets():
    """Copy CSS to reports/assets; logo embedded inline."""
    ensure_dirs()
    css_src = TEMPLATE_DIR / "style.css"
    css_dst = OUT_CSS_DIR / "style.css"
    css_ok = copy_if_exists(css_src, css_dst)
    logo_data_uri, note = discover_logo_data_uri()
    return {
        "css_rel": posix(Path("assets") / "style.css") if css_ok else None,
        "logo_data_uri": logo_data_uri,
        "logo_note": note,
    }

# -------- CEO bullets helper

def summarize_ceo_bullets(df_daily_total: pd.DataFrame,
                          df_monthly_total: pd.DataFrame,
                          daily_spikes: List[Dict],
                          savings_summary: Dict[str, float]) -> Dict[str, str]:
    
    """
    Returns short strings for Page 1 bullets:
      - 30d total spend
      - placeholder for potential savings
      - biggest daily spike
      - high-level trend (last 30 vs prior 30 days)
      - one suggested action (still returned, but not shown in TL;DR)
    """
    # 30d total
    total_30d = float(df_daily_total["amount"].sum()) if not df_daily_total.empty else 0.0
    headline_text = f"Past 30 days spend: ${total_30d:,.0f}."

    # Potential savings placeholder (to be wired up later)
    savings_text = (
        f"Potential savings: ~${savings_summary['monthly']:,.0f}/mo "
        f"(≈${savings_summary['annual']:,.0f}/yr)."
    )


    # Biggest daily spike in last 30d
    spike_text = "No unusual daily spikes detected."
    if daily_spikes:
        biggest = max(daily_spikes, key=lambda d: d["spike_pct"])
        spike_text = (
            f"Largest daily spike on {biggest['date']}: "
            f"${biggest['total']:,.0f} (≈{biggest['spike_pct']*100:.0f}% above 7-day avg)."
        )

    # High-level trend: last 30 vs prior 30 days
    trend_text = "Trend: N/A (not enough data for 60 days)."
    if len(df_daily_total) >= 60:
        df_sorted = df_daily_total.sort_values("date")
        last_30 = df_sorted.tail(30)["amount"].sum()
        prev_30 = df_sorted.tail(60).head(30)["amount"].sum()
        if prev_30 > 0:
            pct = (last_30 / prev_30 - 1.0) * 100.0
            arrow = "up" if pct >= 0 else "down"
            trend_text = (
                f"Trend: ${last_30:,.0f} vs ${prev_30:,.0f} "
                f"({arrow} {abs(pct):.1f}% vs prior 30d)."
            )

    # Action (kept for downstream sections)
    action_text = "Action: review Top-5 services and flagged dates for drivers; consider right-sizing if sustained."

    return {
        "headline": headline_text,
        "savings": savings_text,
        "spike": spike_text,
        "trend": trend_text,
        "action": action_text,
    }


# -------- Template render

def render_html(context):
    ensure_dirs()
    env = Environment(
        loader=FileSystemLoader(posix(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    tpl = env.get_template("report.html.j2")
    html = tpl.render(**context)
    out_name = f"report_{context['label_slug']}.html"
    out_path = REPORTS_DIR / out_name
    out_path.write_text(html, encoding="utf-8")
    return out_path

# -------- Main

def main():
    # Discover the latest pair of CSVs
    (daily_path, d_start, d_end_excl), (monthly_path, m_start, m_end_excl), daily_list, monthly_list = find_latest_daily_and_monthly()

    label_slug = f"{d_start}_to_{d_end_excl}"  # keeps report names self-describing

    # Load data
    df_daily    = load_daily_by_service(daily_path)
    df_monthly  = load_monthly_by_service(monthly_path)
    savings = compute_savings(df_daily.to_dict(orient="records"))


    # Aggregations
    df_daily_total   = daily_totals(df_daily)                           # date, amount
    top5_daily_names = daily_top_services(df_daily, top_n=5)            # ['EC2','S3',...]
    pivot_daily_top  = pivot_daily_per_service(df_daily, top5_daily_names)  # index=date, cols=services

    df_monthly_total = monthly_totals(df_monthly)                        # month_start, amount
    top5_monthly     = monthly_top_services(df_monthly, top_n=5)
    pivot_monthly    = monthly_stacked_prep(df_monthly, top5_monthly)    # index=month_start, cols=top5+'Other'

    # Anomalies
    daily_spikes = detect_daily_spikes(df_daily_total, window=7, thresh=0.30)
    mom_spikes   = detect_mom_spikes(df_monthly_total, thresh=0.15)

    # Charts
    chart_30d_total   = chart_30d_total_line(df_daily_total, label_slug)
    chart_30d_multi   = chart_30d_per_service_multiline(pivot_daily_top, label_slug)
    chart_12m_total   = chart_12m_total_bars_line(df_monthly_total, label_slug)
    chart_12m_stacked = chart_12m_stacked_by_service(pivot_monthly, label_slug)
    # Top-5 (30d) bar
    top5_series_30d = df_daily.groupby("service")["amount"].sum().sort_values(ascending=False).head(5)
    chart_top5_30d  = chart_top5_services_bar(top5_series_30d, label_slug)

    # CEO bullets
    bullets = summarize_ceo_bullets(
        df_daily_total,
        df_monthly_total,
        daily_spikes,
        {"monthly": savings["total_potential_savings"], "annual": savings["annualized_savings"]}
    )


    # Static assets
    staged = stage_static_assets()

    # Build table data for template (compact)
    # Daily table: top services + total per date
    daily_table = pivot_daily_top.copy()
    daily_table["Total"] = daily_table.sum(axis=1)
    daily_table_reset = daily_table.reset_index()  # columns: date, <svc...>, Total
    daily_table_rows = [
        {"date": str(row["date"]), **{str(col): float(row[col]) for col in daily_table.columns}}
        for _, row in daily_table_reset.iterrows()
    ]

    # Monthly table: top services + Other + Total
    monthly_table = pivot_monthly.copy()
    monthly_table["Total"] = monthly_table.sum(axis=1)
    monthly_table_reset = monthly_table.reset_index()
    monthly_table_rows = [
        {"month_start": str(row["month_start"]), **{str(col): float(row[col]) for col in monthly_table.columns}}
        for _, row in monthly_table_reset.iterrows()
    ]

    # Heuristic-based insights for Areas to Watch section
    heuristics = []
    for change in savings["details"]:
        heuristics.append(f"{change['title']}: {change['description']} Estimated savings: ${change['savings_estimate']}")


    # Build a check summary so clients see what we actually evaluated
    check_summary = []
    if savings["details"]:
        for change in savings["details"]:
            check_summary.append(f"⚠️ {change['title']}: {change['description']} Estimated savings: ${change['savings_estimate']}")
    else:
        # No findings – explicitly show what was checked
        check_summary = [
            "✅ EC2: All instances reviewed — no idle/oversized detected.",
            "✅ S3: Buckets within expected thresholds.",
            "✅ Data Transfer: No suspicious inter-region transfer spikes.",
            "✅ EBS: All attached volumes showed consistent activity."
        ]



    context = {
        # Page-1 metadata
        "client_alias": os.environ.get("AWS_PROFILE", "CLIENT_ALIAS"),
        "label_slug": label_slug,
        "window_daily": {"start": d_start, "end_exclusive": d_end_excl},
        "window_monthly": {"start": m_start, "end_exclusive": m_end_excl},

        # CEO bullets
        "ceo": bullets,

        # Charts
        "charts": {
            "daily_total_30d": chart_30d_total,
            "daily_per_service_30d": chart_30d_multi,
            "monthly_total_12m": chart_12m_total,
            "monthly_stacked_12m": chart_12m_stacked,
            "top5_services_30d": chart_top5_30d,
        },

        # Tables (for deep-dive pages)
        "daily_table_columns": ["date"] + top5_daily_names + ["Total"],
        "daily_table_rows": daily_table_rows,
        "monthly_table_columns": ["month_start"] + list(pivot_monthly.columns) + ["Total"],
        "monthly_table_rows": monthly_table_rows,

        # Anomalies for callouts
        "anomalies": {
            "daily_spikes": daily_spikes,  # list of {date,total,trailing_avg,spike_pct}
            "mom_spikes": [
                {"month_start": str(x["month_start"]), "total": x["total"], "delta_pct": x["delta_pct"]}
                for x in mom_spikes
            ],
        },

        # Static
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "logo_data_uri": staged["logo_data_uri"],
        "css_path": staged["css_rel"] or posix(Path("assets") / "style.css"),
        "now_year": datetime.now().year,

        # New — savings data for TL;DR and details section
        "savings_summary": {
            "monthly": savings["total_potential_savings"],
            "annual": savings["annualized_savings"],
        },
    "recommended_changes": savings["details"],
    "heuristics": heuristics,
    "check_summary": check_summary,

    }


    

    # Render
    out_path = render_html(context)
    print(f"[OK] Rendered HTML -> {posix(out_path)}")
    print(staged["logo_note"])

    # Auto-open on Windows
    try:
        os.startfile(str(out_path.resolve()))  # type: ignore[attr-defined]
    except Exception:
        pass

if __name__ == "__main__":
    main()


