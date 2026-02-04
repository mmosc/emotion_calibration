import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
import calibration_utils as utils

# ============== CONFIG =================
INTERACTIONS = "outputs/01_preprocessing/interactions_binarized.csv"
GEMS         = "data/id_highest_gems.tsv"
SCORE_TYPE = 'model'  # options: 'rank', 'model'

# Model File Patterns
MODELS = {
    "BPR":      f"outputs/03_calibration/user_top10_bpr_calibrated_{SCORE_TYPE}_lambda_{{}}.tsv",
    "ItemKNN":  f"outputs/03_calibration/user_top10_itemknn_calibrated_{SCORE_TYPE}_lambda_{{}}.tsv",
    "MostPop":  f"outputs/03_calibration/user_top10_mostpop_calibrated_{SCORE_TYPE}_lambda_{{}}.tsv",
}

# The Random baseline 
RANDOM_REC = "outputs/02_base_recs/user_top100_random.tsv"

LAMBDAS = [0.5]
TOP_K = 10
# =======================================

def ndcg_at_k(items, relevant, k=10):
    dcg = 0.0
    for i, item in enumerate(items[:k]):
        if item in relevant:
            dcg += 1 / np.log2(i + 2)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0

def format_metric(mean_val, std_val):
    if mean_val is None: return ""
    return f"{mean_val:.4f}_{{{std_val:.4f}}}"

def evaluate_recs(recs_df, user_history, P, song2emotion, emo2idx, num_emotions):
    ndcg_vals = []
    kl_vals = []
    jsd_vals = []

    for _, row in recs_df.iterrows():
        user = row["user_id"]
        # Ensure list
        if isinstance(row["recommended_items"], str):
             items = row["recommended_items"].split(",")
        else:
             items = []

        # --- nDCG ---
        relevant = user_history.get(user, set())
        ndcg_vals.append(ndcg_at_k(items, relevant, TOP_K))

        if user not in P.index:
            continue

        # --- Fast Q_u ---
        q_counts = np.zeros(num_emotions)
        for item in items[:TOP_K]:
            emo = song2emotion.get(item)
            if emo in emo2idx:
                q_counts[emo2idx[emo]] += 1

        q = q_counts / TOP_K
        p = P.loc[user].values + 1e-12
        q = q + 1e-12

        kl_vals.append(np.sum(p * np.log(p / q)))
        jsd_vals.append(jensenshannon(p, q, base=2) ** 2)

    return (
        np.mean(ndcg_vals), np.std(ndcg_vals),
        np.mean(kl_vals), np.std(kl_vals),
        np.mean(jsd_vals), np.std(jsd_vals)
    )

def main():
    # 1) Load data
    inter, gems, song2emotion, emotions = utils.load_interactions_and_gems(INTERACTIONS, GEMS)
    emo2idx = {e: i for i, e in enumerate(emotions)}
    num_emotions = len(emotions)

    user_history = inter.groupby("user")["song"].apply(set).to_dict()

    # 2) Build P_u
    P = utils.build_emotion_distribution(inter, gems)
    P = P.reindex(columns=emotions, fill_value=0)

    rows = []

    # --- Evaluate Random First ---
    print("Evaluating Random...")
    try:
        rand_recs = pd.read_csv(RANDOM_REC, sep="\t")
        n_m, n_s, k_m, k_s, j_m, j_s = evaluate_recs(rand_recs, user_history, P, song2emotion, emo2idx, num_emotions)
        rows.append({
            "Model": "Random",
            "nDCG@10(mean)_{std}": format_metric(n_m, n_s),
            "KL@10(mean)_{std}": format_metric(k_m, k_s),
            "JSD@10(mean)_{std}": format_metric(j_m, j_s),
        })
    except FileNotFoundError:
        print(f"Warning: Random file {RANDOM_REC} not found.")

    # --- Evaluate Calibrated Models ---
    for model_name, file_pattern in MODELS.items():
        for LAMBDA in LAMBDAS:
            filename = file_pattern.format(LAMBDA)
            
            try:
                recs = pd.read_csv(filename, sep="\t")
            except FileNotFoundError:
                print(f"Skipping {model_name} lambda={LAMBDA} (File not found)")
                continue

            print(f"Evaluating {model_name} | lambda={LAMBDA}")
            n_m, n_s, k_m, k_s, j_m, j_s = evaluate_recs(recs, user_history, P, song2emotion, emo2idx, num_emotions)
            
            rows.append({
                "Model": f"{model_name}_lambda={LAMBDA}",
                "nDCG@10(mean)_{std}": format_metric(n_m, n_s),
                "KL@10(mean)_{std}": format_metric(k_m, k_s),
                "JSD@10(mean)_{std}": format_metric(j_m, j_s),
            })

    # Save
    if rows:
        df = pd.DataFrame(rows)
       
        cols = ["Model", "nDCG@10(mean)_{std}", "KL@10(mean)_{std}", "JSD@10(mean)_{std}"]
        df = df[cols]
        
        out_path = "outputs/04_evaluation/lambda_table_formatted.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved formatted table to {out_path}")
    else:
        print("No results computed.")

if __name__ == "__main__":
    main()
