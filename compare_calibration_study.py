import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
import calibration_utils as utils
import os

# ============== CONFIG =================
INTERACTIONS = "outputs/01_preprocessing/interactions_binarized.csv"
GEMS         = "data/id_highest_gems.tsv"
TOP_K = 10

# Methods to compare
METHODS = {
    "Greedy (CaliTune)": {
        "pattern": "outputs/03_calibration/user_top10_{}_calitune_model_lambda_{}.tsv",
        "color": "blue",
        "marker": "o"
    },
    "Linear heuristic": {
        "pattern": "outputs/03_calibration/user_top10_{}_linear_lambda_{}.tsv",
        "color": "red",
        "marker": "s"
    }
}

MODELS = ["BPR", "ItemKNN", "MostPop"]
LAMBDAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# =======================================

def ndcg_at_k(items, relevant, k=10):
    dcg = 0.0
    for i, item in enumerate(items[:k]):
        if item in relevant:
            dcg += 1 / np.log2(i + 2)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0

def evaluate_file(filename, user_history, P, song2emotion, emo2idx, num_emotions):
    try:
        recs_df = pd.read_csv(filename, sep="\t")
    except FileNotFoundError:
        return None
        
    ndcg_vals, jsd_vals, kl_vals = [], [], []

    for _, row in recs_df.iterrows():
        user = row["user_id"]
        if isinstance(row["recommended_items"], str):
            items = row["recommended_items"].split(",")
        else:
            items = []

        # nDCG
        relevant = user_history.get(user, set())
        ndcg_vals.append(ndcg_at_k(items, relevant, TOP_K))

        # JSD and KL
        if user in P.index:
            q_counts = np.zeros(num_emotions)
            for item in items[:TOP_K]:
                emo = song2emotion.get(item)
                if emo in emo2idx:
                    q_counts[emo2idx[emo]] += 1
            
            q = (q_counts / TOP_K) + 1e-12
            p = P.loc[user].values + 1e-12
            
            # JSD
            jsd_vals.append(jensenshannon(p, q, base=2) ** 2)
            
            # KL Divergence
            kl = np.sum(p * np.log(p / q))
            kl_vals.append(kl)

    return np.mean(ndcg_vals), np.mean(jsd_vals), np.mean(kl_vals)

def main():
    # 1) Load data
    inter, gems, song2emotion, emotions = utils.load_interactions_and_gems(INTERACTIONS, GEMS)
    emo2idx = {e: i for i, e in enumerate(emotions)}
    num_emotions = len(emotions)
    user_history = inter.groupby("user")["song"].apply(set).to_dict()
    P = utils.build_emotion_distribution(inter, gems)
    P = P.reindex(columns=emotions, fill_value=0)

    # 2) Gather Results
    results = []

    for model in MODELS:
        plt.figure(figsize=(10, 6))
        
        for method_name, config in METHODS.items():
            x_vals = [] # nDCG
            y_vals = [] # JSD
            valid_lambdas = []

            for lam in LAMBDAS:
                # Handle formatting for float lambdas
                lam_str = str(float(lam))
                if lam_str.endswith(".0"): lam_str = lam_str[:-2]
                
                # Check for both formats just in case
                fname = config["pattern"].format(model, lam)
                if not os.path.exists(fname):
                    fname = config["pattern"].format(model, float(lam))
                
                metrics = evaluate_file(fname, user_history, P, song2emotion, emo2idx, num_emotions)
                
                if metrics:
                    ndcg, jsd, kl = metrics
                    x_vals.append(ndcg)
                    y_vals.append(jsd)
                    valid_lambdas.append(lam)
                    results.append({
                        "Model": model,
                        "Method": method_name,
                        "Lambda": lam,
                        "nDCG": ndcg,
                        "JSD": jsd,
                        "KL": kl
                    })

            if x_vals:
                # Sort by lambda for the line plot
                sorted_idx = np.argsort(valid_lambdas)
                plt.plot(np.array(x_vals)[sorted_idx], np.array(y_vals)[sorted_idx], 
                         label=method_name, color=config["color"], marker=config["marker"])
                
                # Annotate lambda values
                for i, lam in enumerate(valid_lambdas):
                    plt.annotate(f"λ={lam}", (x_vals[i], y_vals[i]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)

        plt.title(f"Calibration Trade-off: Greedy vs Linear ({model})")
        plt.xlabel("Accuracy (nDCG@10) → Higher is Better")
        plt.ylabel("Calibration Error (JSD@10) ↓ Lower is Better")
        # Invert X axis so "more accurate" is on the right, but often people like it this way.
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        
        # Save plots
        plot_path = f"outputs/04_evaluation/comparison_{model}_tradeoff.png"
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        print(f"Saved comparison plot for {model} to {plot_path}")

    # 3) Save Results CSV
    if results:
        res_df = pd.DataFrame(results)
        res_path = "outputs/04_evaluation/comparison_study_results.csv"
        res_df.to_csv(res_path, index=False)
        print(f"Saved detailed comparison CSV to {res_path}")

if __name__ == "__main__":
    main()
