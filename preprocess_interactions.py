import pandas as pd
from pathlib import Path

# === CONFIG ===
INPUT_FILE = "data/listening_history.csv"
COUNTS_FILE = "outputs/01_preprocessing/counts.csv"
OUTPUT_FILE = "outputs/01_preprocessing/interactions_binarized.csv"

IMPLICIT_FEEDBACK_SCORE = 5  # Score assigned to filtered interactions

def filter_k_core(df, k=5):
    """
    Iteratively filters users and items with fewer than k interactions.
    """
    print(f"Applying {k}-core filtering...")
    while True:
        prev_size = len(df)
        
        # Filter items
        item_counts = df.groupby("song").size()
        df = df[df["song"].isin(item_counts[item_counts >= k].index)]
        
        # Filter users
        user_counts = df.groupby("user").size()
        df = df[df["user"].isin(user_counts[user_counts >= k].index)]
        
        if len(df) == prev_size:
            break
            
    return df

def main():
    Path(COUNTS_FILE).parent.mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)

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

    # Count interactions (to handle noise reduction first)
    counts = (
        work.groupby(["user", "song"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
    )

    # Save all raw counts for reference
    counts.to_csv(COUNTS_FILE, index=False)
    print(f"Saved raw counts to {COUNTS_FILE} ({len(counts)} rows)")

    # 1. Noise reduction: Keep only (user, song) with >= 2 total listens
    filtered = counts[counts["count"] >= 2].copy()
    
    # 2. Apply 5-core filtering on the resulting unique interactions
    filtered = filter_k_core(filtered, k=5)

    if filtered.empty:
        print("No interactions left after 5-core filtering.")
    else:
        filtered = filtered.drop(columns=["count"])
        filtered["label"] = IMPLICIT_FEEDBACK_SCORE

    # Save binarized output
    filtered.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved 5-core binarized interactions to {OUTPUT_FILE} ({len(filtered)} rows)")

if __name__ == "__main__":
    main()
