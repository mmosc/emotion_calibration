import pandas as pd
import calibration_utils as utils
import os

# ================= CONFIG =================
INTERACTIONS = "outputs/01_preprocessing/interactions_binarized.csv"
GEMS         = "data/id_highest_gems.tsv"

MODELS = {
    "BPR": "outputs/02_base_recs/user_top100_BPR.tsv",
    "ItemKNN": "outputs/02_base_recs/user_top100_itemknn.tsv",
    "MostPop": "outputs/02_base_recs/user_top100_mostpop.tsv"
}

TOP_K = 10
LAMBDAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 1.0]
# Note: Linear re-ranking as implemented in utils uses rank-based scoring by default.
# =========================================

def main():
    # 1) Load data
    inter, gems, song2emotion, emotions = utils.load_interactions_and_gems(INTERACTIONS, GEMS)
    
    # 2) Build user emotion distribution
    P = utils.build_emotion_distribution(inter, gems)
    P = P.reindex(columns=emotions, fill_value=0)

    # 3) Process each model
    for model_name, path in MODELS.items():
        if not os.path.exists(path):
            print(f"Skipping {model_name}: file not found at {path}")
            continue
            
        print(f"\nProcessing {model_name} with Linear Re-ranking...")
        recs = pd.read_csv(path, sep="\t")
        recs["list"] = recs["recommended_items"].apply(lambda x: str(x).split(","))

        for lam in LAMBDAS:
            print(f"  Calibrating with lambda = {lam}")
            rows = []

            for _, row in recs.iterrows():
                user = row["user_id"]
                items = row["list"][:50] # Top-50 candidates as in greedy

                if user not in P.index:
                    top10 = items[:TOP_K]
                else:
                    # Call the linear re-ranking method
                    top10 = utils.rerank_linear(
                        user_id=user,
                        items=items,
                        P_user=P.loc[user],
                        song2emotion=song2emotion,
                        lam=lam,
                        top_k=TOP_K
                    )

                rows.append({
                    "user_id": user,
                    "recommended_items": ",".join(top10)
                })

            out_df = pd.DataFrame(rows)
            out_filename = f"outputs/03_calibration/user_top10_{model_name}_linear_lambda_{lam}.tsv"
            out_df.to_csv(out_filename, sep="\t", index=False)
            print(f"  Saved: {out_filename}")

    print("\nDone. Batch linear calibration completed.")

if __name__ == "__main__":
    main()
