import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon

# ============== CONFIG =================
INTERACTIONS = "outputs/01_preprocessing/interactions_binarized.csv"
GEMS         = "data/id_highest_gems.tsv"
RANDOM_RECS  = "outputs/02_base_recs/user_top100_random.tsv"
TOP_K = 10
# =======================================


# ---------- nDCG ----------
def ndcg_at_k(items, relevant, k=10):
    dcg = 0.0
    for i, item in enumerate(items[:k]):
        if item in relevant:
            dcg += 1 / np.log2(i + 2)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0


print("Loading data...")

# ---------- Load interactions ----------
inter = pd.read_csv(INTERACTIONS)
inter.columns = ["user", "song", "label"]
user_history = inter.groupby("user")["song"].apply(set).to_dict()

# ---------- Load emotion labels ----------
gems = pd.read_csv(GEMS, sep="\t")
gems = gems.rename(columns={"id": "song", "highest_gem": "emotion"})
song2emotion = dict(zip(gems["song"], gems["emotion"]))

EMOTIONS = sorted(gems["emotion"].unique())
emo2idx = {e: i for i, e in enumerate(EMOTIONS)}
num_emotions = len(EMOTIONS)

# ---------- Build P_u once ----------
merged = inter.merge(gems, on="song", how="left")
counts = merged.groupby(["user", "emotion"]).size().unstack(fill_value=0)
P = counts.reindex(columns=EMOTIONS, fill_value=0)
P = P.div(P.sum(axis=1), axis=0).fillna(0)

# ---------- Load Random recommendations ----------
try:
    recs = pd.read_csv(RANDOM_RECS, sep="\t")
    recs["list"] = recs["recommended_items"].apply(lambda x: str(x).split(","))
except FileNotFoundError:
    print(f"Error: {RANDOM_RECS} not found. Run generate_random_recommendations.py first.")
    exit(1)

ndcg_vals = []
kl_vals = []
jsd_vals = []

# ---------- Evaluation ----------
for _, row in recs.iterrows():
    user = row["user_id"]
    items = row["list"][:TOP_K]

    # --- nDCG ---
    relevant = user_history.get(user, set())
    ndcg_vals.append(ndcg_at_k(items, relevant, TOP_K))

    if user not in P.index:
        continue

    # --- Fast Q_u ---
    q_counts = np.zeros(num_emotions)
    for item in items:
        emo = song2emotion.get(item)
        if emo in emo2idx:
            q_counts[emo2idx[emo]] += 1

    q = q_counts / TOP_K
    p = P.loc[user].values + 1e-12
    q = q + 1e-12

    kl_vals.append(np.sum(p * np.log(p / q)))
    jsd_vals.append(jensenshannon(p, q, base=2) ** 2)

# ---------- Summary ----------
results = {
    "Model": "Random",
    "nDCG@10_mean": np.mean(ndcg_vals),
    "nDCG@10_std": np.std(ndcg_vals),
    "KL@10_mean": np.mean(kl_vals),
    "KL@10_std": np.std(kl_vals),
    "JSD@10_mean": np.mean(jsd_vals),
    "JSD@10_std": np.std(jsd_vals),
}

df = pd.DataFrame([results])
# Output to evaluation folder
df.to_csv("outputs/04_evaluation/random_evaluation_summary.csv", index=False)

print("Saved Random evaluation summary to outputs/04_evaluation/random_evaluation_summary.csv")
print(df)
