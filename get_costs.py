# get_costs.py — Rolling windows (30d daily + 12m monthly), by service, write CSVs
#
# Usage examples:
#   python get_costs.py
#   python get_costs.py --asof 2025-10-19
#   python get_costs.py --profile CLIENT_ALIAS --asof 2025-10-19
#
# What it does:
# - Determines an "as-of" date (default = yesterday, local time).
# - Pulls DAILY costs (by service) for the last 30 COMPLETE days ending at as-of.
# - Pulls MONTHLY costs (by service) for the last 12 COMPLETE months (ending at the
#   last complete month relative to as-of).
# - Writes two CSVs into ./data:
#     data/cost_explorer_daily_<start>_to_<end>.csv
#     data/cost_explorer_monthly_<start>_to_<end>.csv
#
# Notes:
# - End dates passed to Cost Explorer are EXCLUSIVE per AWS API.
# - We also print brief summaries to the console for quick sanity checks.

import os
import sys
import csv
import json
import argparse
from datetime import date, datetime, timedelta

import boto3


# ----------------------------
# Date helpers (no extra deps)
# ----------------------------

def _first_of_month(d: date) -> date:
    return d.replace(day=1)

def _add_months(d: date, months: int) -> date:
    # Simple month arithmetic without dateutil
    year = d.year + (d.month - 1 + months) // 12
    month = (d.month - 1 + months) % 12 + 1
    day = 1
    return date(year, month, day)

def _iso(d: date) -> str:
    return d.isoformat()

def compute_windows(asof: date):
    """
    Compute:
      - 30d DAILY window: last 30 complete days up to 'asof'
      - 12m MONTHLY window: last 12 complete months ending at month BEFORE 'asof'’s month
    Returns:
      daily_start, daily_end_excl, monthly_start, monthly_end_excl  (all ISO strings)
    """
    # DAILY (30d): include the 'asof' day; CE end is exclusive so add +1 day
    daily_end_excl = asof + timedelta(days=1)
    daily_start = asof - timedelta(days=29)  # 30 consecutive days

    # MONTHLY (12m): last complete month is the first of the current month (exclusive end)
    this_month_first = _first_of_month(asof)
    monthly_end_excl = this_month_first  # exclusive end at first day of current month
    monthly_start = _add_months(this_month_first, -12)  # first day 12 months prior

    return _iso(daily_start), _iso(daily_end_excl), _iso(monthly_start), _iso(monthly_end_excl)


# ----------------------------
# Cost Explorer paging helper
# ----------------------------

def ce_get_all(ce, **kwargs):
    """Fetch all pages for get_cost_and_usage."""
    results = []
    token = None
    while True:
        resp = ce.get_cost_and_usage(NextPageToken=token, **kwargs) if token else ce.get_cost_and_usage(**kwargs)
        results.append(resp)
        token = resp.get("NextPageToken")
        if not token:
            break
    return results


# ----------------------------
# Extraction helpers
# ----------------------------

def write_grouped_csv_from_ce_responses(responses, out_csv_path, granularity: str):
    """
    Write a normalized CSV from Cost Explorer grouped-by-SERVICE responses.
    Columns:
      For DAILY:   date, service, amount, unit
      For MONTHLY: month_start, service, amount, unit
    """
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)

    field_date = "date" if granularity == "DAILY" else "month_start"
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([field_date, "service", "amount", "unit"])

        total_amount = 0.0
        unit_seen = "USD"

        for page in responses:
            for t in page.get("ResultsByTime", []):
                period_start = t.get("TimePeriod", {}).get("Start")
                groups = t.get("Groups", [])
                unit = "USD"
                for g in groups:
                    keys = g.get("Keys", [])
                    service = keys[0] if keys else "Other"
                    metrics = g.get("Metrics", {}).get("UnblendedCost", {})
                    amount = float(metrics.get("Amount", "0") or "0")
                    unit = metrics.get("Unit", unit)
                    writer.writerow([period_start, service, f"{amount:.6f}", unit])
                    total_amount += amount
                unit_seen = unit or unit_seen

        # Small console hint
        print(f"[OK] Wrote {out_csv_path}")
        print(f"     Approx total ({granularity} sum of groups): {total_amount:.2f} {unit_seen}")


# ----------------------------
# Main
# ----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Pull AWS Cost Explorer data into CSVs for 30d daily + 12m monthly (by service).")
    p.add_argument("--asof", type=str, default=None, help="Anchor date (YYYY-MM-DD). Default = yesterday (local).")
    p.add_argument("--profile", type=str, default=None, help="AWS profile to use (sets AWS_PROFILE).")
    return p.parse_args()

def main():
    args = parse_args()

    # Optional profile
    if args.profile:
        os.environ["AWS_PROFILE"] = args.profile
        print(f"Using AWS profile: {args.profile}")

    # Determine as-of date (default = yesterday)
    if args.asof:
        try:
            asof = datetime.strptime(args.asof, "%Y-%m-%d").date()
        except ValueError:
            print("ERROR: --asof must be YYYY-MM-DD")
            sys.exit(2)
    else:
        asof = date.today() - timedelta(days=1)
    print(f"As-of date: {asof.isoformat()} (defaulting to yesterday if not provided)")

    daily_start, daily_end_excl, monthly_start, monthly_end_excl = compute_windows(asof)
    print(f"30d DAILY window     : {daily_start} to {daily_end_excl} (End exclusive)")
    print(f"12m MONTHLY window   : {monthly_start} to {monthly_end_excl} (End exclusive)")

    ce = boto3.client("ce")

    # --- DAILY by SERVICE (last 30 days) ---
    daily_kwargs = dict(
        TimePeriod={"Start": daily_start, "End": daily_end_excl},
        Granularity="DAILY",
        Metrics=["UnblendedCost"],
        GroupBy=[{"Type": "DIMENSION", "Key": "SERVICE"}],
    )
    daily_pages = ce_get_all(ce, **daily_kwargs)

    # --- MONTHLY by SERVICE (last 12 complete months) ---
    monthly_kwargs = dict(
        TimePeriod={"Start": monthly_start, "End": monthly_end_excl},
        Granularity="MONTHLY",
        Metrics=["UnblendedCost"],
        GroupBy=[{"Type": "DIMENSION", "Key": "SERVICE"}],
    )
    monthly_pages = ce_get_all(ce, **monthly_kwargs)

    # Output filenames (encode ranges so they’re self-describing)
    daily_csv = f"data/cost_explorer_daily_{daily_start}_to_{daily_end_excl}.csv"
    monthly_csv = f"data/cost_explorer_monthly_{monthly_start}_to_{monthly_end_excl}.csv"

    # Write CSVs
    write_grouped_csv_from_ce_responses(daily_pages, daily_csv, granularity="DAILY")
    write_grouped_csv_from_ce_responses(monthly_pages, monthly_csv, granularity="MONTHLY")

    # Optional: also save raw JSON (first page) for quick debugging
    debug_json_path = f"data/debug_daily_firstpage_{daily_start}_to_{daily_end_excl}.json"
    with open(debug_json_path, "w", encoding="utf-8") as jf:
        json.dump(daily_pages[0], jf, indent=2)
    print(f"[info] Saved first DAILY page JSON to: {debug_json_path}")

    debug_json_path2 = f"data/debug_monthly_firstpage_{monthly_start}_to_{monthly_end_excl}.json"
    with open(debug_json_path2, "w", encoding="utf-8") as jf:
        json.dump(monthly_pages[0], jf, indent=2)
    print(f"[info] Saved first MONTHLY page JSON to: {debug_json_path2}")

    print("\nDone. Next step: generate HTML/PDF from these CSVs in make_report.py.")

if __name__ == "__main__":
    main()


