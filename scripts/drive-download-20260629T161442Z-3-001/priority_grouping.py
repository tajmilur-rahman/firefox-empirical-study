#!/usr/bin/env python3
"""
Priority Grouping Script
- Groups all labels by priority from Claude Word Match output
- Removes duplicates
- Outputs priority_groups.xlsx with one column per priority

Usage:
    python priority_grouping.py
"""

import sys

def install(pkg):
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

try:
    import pandas as pd
except ImportError:
    install("pandas openpyxl"); import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_FILE  = r"C:\Users\avish\Downloads\priority_bugs_labeled_claude.csv"
OUTPUT_FILE = r"C:\Users\avish\Downloads\priority_groups.xlsx"
PRIORITIES  = ["P1", "P2", "P3", "P4", "P5"]
# ─────────────────────────────────────────────────────────────────────────────

def run():
    print("Loading Claude Word Match output...")
    df = pd.read_csv(INPUT_FILE)
    print(f"  Loaded {len(df)} bugs")
    print()

    result = {}

    for pri in PRIORITIES:
        labels = []
        seen = set()
        pri_df = df[df['priority'] == pri]

        for row in pri_df['matched_labels']:
            for label in str(row).split(';'):
                label = label.strip()
                if label and label not in seen:
                    seen.add(label)
                    labels.append(label)

        result[pri] = pd.Series(labels)
        print(f"  {pri}: {len(labels)} unique labels")

    print()
    print(f"Saving to {OUTPUT_FILE}...")
    out = pd.DataFrame(result)
    out.to_excel(OUTPUT_FILE, index=False)
    print(f"Done!")


if __name__ == "__main__":
    run()
