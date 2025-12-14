import pandas as pd
import numpy as np

# ============== CONFIG =================
INTERACTIONS = "interactions_binarized.csv"
GEMS         = "id_highest_gems.tsv"
BPR_RECS     = "user_top100_BPR.tsv"

OUT_RECS_CAL = "user_top100_BPR_calibrated.tsv"  # output file (Top-10)
TOP_K        = 10
LAMBDA       = 0.3   # 0 = only BPR ranking, 1 = only emotions
# =======================================

print("Loading data...")

# 1) Load interactions
inter = pd.read_csv(INTERACTIONS)
inter.columns = ["user", "song", "label"]

# 2) Load emotion labels and renamed columns
gems = pd.read_csv(GEMS, sep="\t")
gems = gems.rename(columns={"id": "song", "highest_gem": "emotion"})

# Map song -> emotion
song2emotion = dict(zip(gems["song"], gems["emotion"]))

# 3) Build user emotion distribution P_u
merged = inter.merge(gems, on="song", how="left")

# Count per (user, emotion)
counts = merged.groupby(["user", "emotion"]).size().unstack(fill_value=0)

# Normalize rows to probabilities P_u(e)
P = counts.div(counts.sum(axis=1), axis=0).fillna(0)

print(f"Built emotion distribution for {len(P)} users.")
EMOTIONS = list(P.columns)  # emotion ids


# 4) Load BPR recommendations
recs = pd.read_csv(BPR_RECS, sep="\t")
recs["list"] = recs["recommended_items"].apply(lambda x: x.split(","))

print(f"Loaded BPR recs for {len(recs)} users.")


# 5) Calibration-aware re-ranking
rows = []

for idx, row in recs.iterrows():
    user = row["user_id"]
    items = row["list"]

    # If user has no emotion history, keep original top-k
    if user not in P.index:
        topk_items = items[:TOP_K]
        rows.append({
            "user_id": user,
            "recommended_items": ",".join(topk_items)
        })
        continue

    P_u = P.loc[user]   # Series indexed by emotion id

    scored_items = []
    n = len(items)

    for rank, item in enumerate(items):
        # Base BPR score from rank (higher score for higher rank)
        base_score = (n - rank) / n  # in (0,1]

        # Emotion of this item
        emo = song2emotion.get(item, None)

        # Calibration term: how much user likes this emotion
        if emo in P_u.index:
            calib = P_u.loc[emo]
        else:
            calib = 0.0

        # New score: mix of BPR ranking and emotional alignment
        new_score = (1 - LAMBDA) * base_score + LAMBDA * calib

        scored_items.append((item, new_score))

    # Sort by new score
    scored_items.sort(key=lambda x: x[1], reverse=True)

    # Take calibrated Top-K
    topk_items = [it for it, s in scored_items[:TOP_K]]

    rows.append({
        "user_id": user,
        "recommended_items": ",".join(topk_items)
    })

# Save calibrated recommendations
out_df = pd.DataFrame(rows)
out_df.to_csv(OUT_RECS_CAL, sep="\t", index=False)

print("Saved calibrated BPR recommendations to:", OUT_RECS_CAL)
