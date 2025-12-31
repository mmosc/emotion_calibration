import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
from pathlib import Path

# ============= CONFIG ==============
DATA = "outputs/01_preprocessing/interactions_binarized.csv"
GEMS = "data/id_highest_gems.tsv"

# Recommendation outputs for each model
BPR_RECS      = "outputs/02_base_recs/user_top100_BPR.tsv"
ITEMKNN_RECS  = "outputs/02_base_recs/user_top100_itemknn.tsv"
MOSTPOP_RECS  = "outputs/02_base_recs/user_top100_mostpop.tsv"
RANDOM_RECS   = "outputs/02_base_recs/user_top100_random.tsv"

OUTPUT_TABLE = "outputs/04_evaluation/evaluation_summary.csv"
OUTPUT_KLJSD = "outputs/04_evaluation/calibration_all_models.csv"

# ===================================

def main():
    # ====== Step 0: Path Handling ======
    # Ensure we can find the files if running from project root
    files_to_check = [DATA, GEMS]
    missing = [f for f in files_to_check if not os.path.exists(f)]
    if missing:
        print(f"Error: Missing input files: {missing}")
        return

    # ====== Step 1: Load data ==========
    inter = pd.read_csv(DATA)
    inter.columns = ["user", "song", "label"]

    gems = pd.read_csv(GEMS, sep="\t")
    gems.columns = ["song", "emotion"]

    EMOTIONS = sorted(gems["emotion"].dropna().unique())

    # Build user history for NDCG
    user_history = (
        inter.groupby("user")["song"]
        .apply(set)
        .to_dict()
    )

    # ====== Load recommendation files ===========
    def load_recs(path, name):
        if not os.path.exists(path):
            print(f"Warning: {name} file not found at {path}. Skipping.")
            return None
        df = pd.read_csv(path, sep="\t")
        df["list"] = df["recommended_items"].apply(lambda x: str(x).split(","))
        return df[["user_id", "list"]]

    models = {}
    for name, path in [("BPR", BPR_RECS), ("ItemKNN", ITEMKNN_RECS), 
                       ("MostPop", MOSTPOP_RECS), ("Random", RANDOM_RECS),
                       ("Random_Calib", "outputs/03_calibration/user_top10_random_calitune.tsv")]:
        recs_df = load_recs(path, name)
        if recs_df is not None:
            models[name] = recs_df

    if not models:
        print("No recommendation files found to evaluate.")
        return

    # ====== Step 2: NDCG@10 per user ===========
    def ndcg_at_k(recommended_items, relevant_items, k=10):
        rec_k = recommended_items[:k]
        dcg = sum([1/np.log2(i+2) for i, item in enumerate(rec_k) if item in relevant_items])
        idcg = sum([1/np.log2(i+2) for i in range(min(len(relevant_items), k))])
        return dcg / idcg if idcg > 0 else 0

    def compute_ndcg(model_df):
        rows = []
        for idx, row in model_df.iterrows():
            user = row["user_id"]
            recs = row["list"]
            relevant = user_history.get(user, set())
            score = ndcg_at_k(recs, relevant, k=10)
            rows.append(score)
        return np.array(rows)

    ndcg_scores = {name: compute_ndcg(df) for name, df in models.items()}

    # ====== Step 3: KL@10 + JSD@10 per user ==========
    def emotion_dist(df, user_col="user", item_col="song"):
        merged = df.merge(gems, left_on=item_col, right_on="song", how="left")
        counts = merged.groupby([user_col, "emotion"]).size().unstack(fill_value=0)
        counts = counts.reindex(columns=EMOTIONS, fill_value=0)
        probs = counts.div(counts.sum(axis=1), axis=0)
        return probs

    P = emotion_dist(inter, user_col="user", item_col="song")   

    def rec_emotion_dist(rec_df, k=10):
        rows = []
        for idx, row in rec_df.iterrows():
            user = row["user_id"]
            for s in row["list"][:k]:
                rows.append({"user": user, "song": s})
        df = pd.DataFrame(rows)
        return emotion_dist(df, user_col="user", item_col="song")

    KL = {}
    JSD = {}

    for model_name, rec_df in models.items():
        Q = rec_emotion_dist(rec_df, k=10)
        shared_users = set(P.index).intersection(Q.index)
        
        kl_vals, jsd_vals = [], []
        
        for u in shared_users:
            p = P.loc[u, EMOTIONS].values + 1e-12
            q = Q.loc[u, EMOTIONS].values + 1e-12
            
            kl = np.sum(p * np.log(p / q))
            jsd = jensenshannon(p, q, base=2) ** 2
            
            kl_vals.append(kl)
            jsd_vals.append(jsd)

        KL[model_name] = np.array(kl_vals)
        JSD[model_name] = np.array(jsd_vals)

    # ====== Step 4: Create Summary Table ==========
    rows = []
    for model in models.keys():
        rows.append({
            "Model": model,
            "nDCG@10_mean": ndcg_scores[model].mean(),
            "nDCG@10_std": ndcg_scores[model].std(),
            "KL@10_mean": KL[model].mean(),
            "KL@10_std": KL[model].std(),
            "JSD@10_mean": JSD[model].mean(),
            "JSD@10_std": JSD[model].std(),
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_TABLE, index=False)
    print("Saved summary table to:", OUTPUT_TABLE)

    # ====== Step 5: Save per-user KL/JSD for all models ==========
    all_rows = []
    for model in models.keys():
        for kl, jsd in zip(KL[model], JSD[model]):
            all_rows.append({"model": model, "KL": kl, "JSD": jsd})

    pd.DataFrame(all_rows).to_csv(OUTPUT_KLJSD, index=False)
    print("Saved KL/JSD per-user file to:", OUTPUT_KLJSD)

    # ====== Step 6: Boxplots ============
    def make_boxplot(data_dict, title):
        plt.figure(figsize=(8,6))
        plt.boxplot(data_dict.values(), tick_labels=data_dict.keys())
        plt.title(title)
        plt.ylabel(title)
        plt.grid(True)
        
        # Save to evaluation folder
        filename = f"outputs/04_evaluation/{title.replace('/', '_').replace(' ', '_')}.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {filename}")

    make_boxplot(ndcg_scores, "nDCG@10")
    make_boxplot(KL, "KL@10 Divergence")
    make_boxplot(JSD, "Jensen-Shannon Divergence (JSD@10)")

if __name__ == "__main__":
    main()
