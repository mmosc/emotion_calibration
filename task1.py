""""
Aggregate (user, song) interactions and binarize with a >=2 threshold.
Simply run:
    python task1.py
Expected input file:
    listening_history.csv
Output files:
    counts.csv                — all user-song counts
    interactions_binarized.csv — only pairs with count >=2 (label=5)
"""

import pandas as pd
from pathlib import Path

# === File paths ===
INPUT_FILE = "listening_history.csv"
COUNTS_FILE = "counts.csv"
OUTPUT_FILE = "interactions_binarized.csv"

def main():
    in_path = Path(INPUT_FILE)
    if not in_path.exists():
        print(f"Input file not found: {in_path}")
        return

    # Read CSV — auto-detect separator (comma, tab, semicolon)
    df = pd.read_csv(in_path, sep=None, engine="python", dtype=str)
    if df.empty:
        print("Input file is empty — nothing to process.")
        return

    # Identify columns
    cols = {c.lower().strip(): c for c in df.columns}
    user_col = cols.get("user") or list(df.columns)[0]
    item_col = cols.get("song") or cols.get("item") or list(df.columns)[1]

    # Keep only user and song columns
    work = df[[user_col, item_col]].rename(columns={user_col: "user", item_col: "song"}).dropna()

    # Count interactions
    counts = (
        work.groupby(["user", "song"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
    )

    # Save all counts
    counts.to_csv(COUNTS_FILE, index=False)
    print(f"Saved counts to {COUNTS_FILE} ({len(counts)} rows)")

    # Filter where count >= 2
    filtered = counts[counts["count"] >= 2].copy()
    if filtered.empty:
        print("No (user, song) pairs with count >= 2 found.")
        filtered[["user", "song"]] = []
    else:
        filtered = filtered.drop(columns=["count"])
        filtered["label"] = 5  # assign label 5

    # Save binarized output
    filtered.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved binarized interactions to {OUTPUT_FILE} ({len(filtered)} rows)")

if __name__ == "__main__":
    main()
