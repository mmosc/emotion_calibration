import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt

# CONFIG
DATA = "outputs/01_preprocessing/interactions_binarized.csv"
GEMS = "data/id_highest_gems.tsv"

BPR_RECS      = "outputs/02_base_recs/user_top100_BPR.tsv"
ITEMKNN_RECS  = "outputs/02_base_recs/user_top100_itemknn.tsv"
MOSTPOP_RECS  = "outputs/02_base_recs/user_top100_mostpop.tsv"
RANDOM_RECS   = "outputs/02_base_recs/user_top100_random.tsv"

OUTPUT_TABLE = "outputs/04_evaluation/evaluation_summary.csv"
OUTPUT_KLJSD = "outputs/04_evaluation/calibration_all_models.csv"

def ndcg_at_k(recommended_items, relevant_items, k=10):
    rec_k = recommended_items[:k]
    dcg = 0.0
    for i, item in enumerate(rec_k):
        if item in relevant_items:
            dcg += 1 / np.log2(i + 2)
    
    idcg = 0.0
    for i in range(min(len(relevant_items), k)):
        idcg += 1 / np.log2(i + 2)
    
    return dcg / idcg if idcg > 0 else 0

def main():
    os.makedirs(os.path.dirname(OUTPUT_TABLE), exist_ok=True)

    # Load data
    inter = pd.read_csv(DATA)
    inter.columns = ["user", "song", "label"]

    gems = pd.read_csv(GEMS, sep="\t")
    gems.columns = ["song", "emotion"]

    EMOTIONS = sorted(gems["emotion"].dropna().unique())

    # Build user history
    user_history = {}
    for user, group in inter.groupby("user"):
        user_history[user] = set(group["song"])

    # Load recommendations
    models = {}
    paths = [
        ("BPR", BPR_RECS), 
        ("ItemKNN", ITEMKNN_RECS), 
        ("MostPop", MOSTPOP_RECS), 
        ("Random", RANDOM_RECS)
    ]

    for name, path in paths:
        if os.path.exists(path):
            df = pd.read_csv(path, sep="\t")
            df["list"] = df["recommended_items"].apply(lambda x: str(x).split(","))
            models[name] = df[["user_id", "list"]]

    # Evaluation
    ndcg_results = {}
    kl_results = {}
    jsd_results = {}

    # Build P_u (user history distribution)
    merged = inter.merge(gems, on="song", how="left")
    counts = merged.groupby(["user", "emotion"]).size().unstack(fill_value=0)
    P = counts.reindex(columns=EMOTIONS, fill_value=0)
    P = P.div(P.sum(axis=1), axis=0)

    for name, model_df in models.items():
        print(f"Evaluating {name}...")
        ndcg_vals = []
        kl_vals = []
        jsd_vals = []

        for idx, row in model_df.iterrows():
            user = row["user_id"]
            recs = row["list"]
            
            # NDCG
            relevant = user_history.get(user, set())
            ndcg_vals.append(ndcg_at_k(recs, relevant, k=10))

            # Emotion Dist for Recs
            if user in P.index:
                q_counts = np.zeros(len(EMOTIONS))
                emo_map = dict(zip(gems["song"], gems["emotion"]))
                emo2idx = {e: i for i, e in enumerate(EMOTIONS)}
                
                for s in recs[:10]:
                    e = emo_map.get(s)
                    if e in emo2idx:
                        q_counts[emo2idx[e]] += 1
                
                q = (q_counts / 10) + 1e-12
                p = P.loc[user].values + 1e-12
                
                kl = np.sum(p * np.log(p / q))
                jsd = jensenshannon(p, q, base=2) ** 2
                
                kl_vals.append(kl)
                jsd_vals.append(jsd)

        ndcg_results[name] = np.array(ndcg_vals)
        kl_results[name] = np.array(kl_vals)
        jsd_results[name] = np.array(jsd_vals)

    # Summary Table
    summary_rows = []
    for name in ndcg_results.keys():
        summary_rows.append({
            "Model": name,
            "nDCG@10_mean": ndcg_results[name].mean(),
            "nDCG@10_std": ndcg_results[name].std(),
            "KL@10_mean": kl_results[name].mean(),
            "KL@10_std": kl_results[name].std(),
            "JSD@10_mean": jsd_results[name].mean(),
            "JSD@10_std": jsd_results[name].std(),
        })

    pd.DataFrame(summary_rows).to_csv(OUTPUT_TABLE, index=False)
    
    # Per-user results
    user_rows = []
    for name in kl_results.keys():
        for kl, jsd in zip(kl_results[name], jsd_results[name]):
            user_rows.append({"model": name, "KL": kl, "JSD": jsd})
    pd.DataFrame(user_rows).to_csv(OUTPUT_KLJSD, index=False)

    #  Plotting
    for metric_name, data in [("nDCG@10", ndcg_results), ("KL@10", kl_results), ("JSD@10", jsd_results)]:
        plt.figure(figsize=(8,6))
        plt.boxplot(data.values(), labels=data.keys())
        plt.title(metric_name)
        plt.ylabel("Score")
        plt.grid(True)
        plt.savefig(f"outputs/04_evaluation/{metric_name.replace('@', '_')}_boxplot.png")
        plt.close()

if __name__ == "__main__":
    main()
