import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon

# ===  Base directory: folder where THIS script lives ===
base = os.path.dirname(__file__)

# ===  Build portable paths ===
inter_path = os.path.join(base, "interactions_binarized.csv")
recs_path  = os.path.join(base, "user_top100.tsv")
gems_path  = os.path.join(base, "id_highest_gems.tsv")

# === Load the files ===
interactions = pd.read_csv(inter_path)
recs = pd.read_csv(recs_path, sep="\t")
gems = pd.read_csv(gems_path, sep="\t")


# Make sure column names match
interactions.columns = ["user", "item", "label"]
gems.columns = ["item", "emotion"]

# === Merge emotions into past interactions ===
interactions = interactions.merge(gems, on="item", how="left")

# === Expand the recommendations (user_top100.tsv) ===
rows = []
for _, row in recs.iterrows():
    user = row["user_id"]
    items = row["recommended_items"].split(",")
    for item in items:
        rows.append({"user": user, "item": item})

rec_expanded = pd.DataFrame(rows)
rec_expanded = rec_expanded.merge(gems, on="item", how="left")

# === Get list of all emotion categories ===
emotions = gems["emotion"].unique()

# === Define helper for normalized emotion distribution ===
def emotion_distribution(df, user_col="user"):
    counts = df.groupby([user_col, "emotion"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=emotions, fill_value=0)
    probs = counts.div(counts.sum(axis=1), axis=0)
    return probs

# Compute per-user distributions
P = emotion_distribution(interactions)
Q = emotion_distribution(rec_expanded)

# === Compute KL and JSD per user ===
kl_values, jsd_values = [], []

for user in P.index:
    if user not in Q.index:
        continue
    p = P.loc[user].values + 1e-12
    q = Q.loc[user].values + 1e-12

    kl = np.sum(p * np.log(p / q))
    jsd = jensenshannon(p, q, base=2) ** 2

    kl_values.append(kl)
    jsd_values.append(jsd)

# === Aggregate ===
results = pd.DataFrame({"user": P.index[:len(kl_values)], "KL": kl_values, "JSD": jsd_values})
print("Average KL:", np.mean(kl_values))
print("Average JSD:", np.mean(jsd_values))

results.to_csv(r"calibration_metrics_per_user.tsv", sep="\t", index=False)
print("Saved per-user metrics to calibration_metrics_per_user.tsv")
