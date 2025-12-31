import pandas as pd
from pathlib import Path

# === CONFIG ===
INPUT_FILE = "data/listening_history.csv"
COUNTS_FILE = "outputs/01_preprocessing/counts.csv"
OUTPUT_FILE = "outputs/01_preprocessing/interactions_binarized.csv"

IMPLICIT_FEEDBACK_SCORE = 5  # Score assigned to filtered interactions

def main():
    in_path = Path(INPUT_FILE)
    if not in_path.exists():
        print(f"Input file not found: {in_path}")
        return

    # Read CSV
    df = pd.read_csv(in_path, sep=None, engine="python", dtype=str)
    if df.empty:
        print("Input file is empty — nothing to process.")
        return

    # Identify columns
    cols = {c.lower().strip(): c for c in df.columns}
    user_col = cols.get("user") or list(df.columns)[0]
    item_col = cols.get("song") or cols.get("item") or list(df.columns)[1]

    # Keeping only user and song columns
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
        filtered["label"] = IMPLICIT_FEEDBACK_SCORE

    # Save binarized output
    filtered.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved binarized interactions to {OUTPUT_FILE} ({len(filtered)} rows)")

if __name__ == "__main__":
    main()
