# convert_costs_to_csv.py
# Read raw CE JSON and produce a CSV with basic validations.

import os
import json
import glob
from decimal import Decimal, InvalidOperation
import pandas as pd

DATA_DIR = "data"

def find_latest_json():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "cost_explorer_*_to_*.json")))
    if not files:
        raise FileNotFoundError("No JSON files found in ./data. Run get_costs.py first.")
    return files[-1]

def to_decimal(x):
    try:
        return Decimal(str(x))
    except (InvalidOperation, ValueError, TypeError):
        return Decimal("0")


def main():
    infile = find_latest_json()
    with open(infile, "r", encoding="utf-8") as f:
        payload = json.load(f)

    rbt = payload.get("ResultsByTime", [])
    if not rbt:
        raise RuntimeError("ResultsByTime missing/empty in JSON.")

    period = rbt[0]
    start = period.get("TimePeriod", {}).get("Start")
    end = period.get("TimePeriod", {}).get("End")

    total_obj = period.get("Total", {}).get("UnblendedCost", {})
    total_amount = to_decimal(total_obj.get("Amount"))
    total_unit = total_obj.get("Unit", "USD")

    groups = period.get("Groups", [])

    rows = []
    if groups:
        for g in groups:
            keys = g.get("Keys", [])
            service = keys[0] if keys else "UNKNOWN"
            amt = to_decimal(g.get("Metrics", {}).get("UnblendedCost", {}).get("Amount"))
            rows.append(
                {
                    "month_start": start,
                    "month_end_exclusive": end,
                    "service": service,
                    "amount": str(amt),  # keep as string for CSV readability
                    "unit": total_unit,
                }
            )
    else:
        # No groups: still emit a single row with service = "ALL"
        rows.append(
            {
                "month_start": start,
                "month_end_exclusive": end,
                "service": "ALL",
                "amount": str(total_amount),
                "unit": total_unit,
            }
        )

    df = pd.DataFrame(rows, columns=["month_start", "month_end_exclusive", "service", "amount", "unit"])


    # Validation: sum of grouped amounts should equal the Total for the month (within 1 cent)
    grouped_sum = sum((to_decimal(a) for a in df["amount"]), Decimal("0"))
    epsilon = Decimal("0.01")  # 1 cent tolerance
    matches_total = abs(grouped_sum - total_amount) <= epsilon

    # Save CSV next to the JSON
    outfile = infile.replace(".json", ".csv")
    df.to_csv(outfile, index=False, encoding="utf-8")

    print(f"Read JSON: {infile}")
    print(f"Wrote CSV: {outfile}")
    print(f"Groups: {len(groups)}")
    print(f"Grouped sum: {grouped_sum} {total_unit}")
    print(f"Monthly total: {total_amount} {total_unit}")
    print("VALIDATION: OK" if matches_total else "VALIDATION: MISMATCH (check data/rounding)")

if __name__ == "__main__":
    main()


