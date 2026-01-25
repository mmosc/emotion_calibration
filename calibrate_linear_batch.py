import pandas as pd
import calibration_utils as utils
import os

# Settings
INTERACTIONS = "outputs/01_preprocessing/interactions_binarized.csv"
GEMS         = "data/id_highest_gems.tsv"

MODELS = {
    "BPR": "outputs/02_base_recs/user_top100_BPR.tsv",
    "ItemKNN": "outputs/02_base_recs/user_top100_itemknn.tsv",
    "MostPop": "outputs/02_base_recs/user_top100_mostpop.tsv"
}

TOP_K = 10
LAMBDAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

def main():
    # Load data
    print("Loading data...")
    inter, gems, song2emotion, emotions = utils.load_interactions_and_gems(INTERACTIONS, GEMS)
    emo2idx = {e: i for i, e in enumerate(emotions)}
    num_emotions = len(emotions)
    
    # User profiles
    P = utils.build_emotion_distribution(inter, gems)
    P = P.reindex(columns=emotions, fill_value=0)

    # Process models for rank-based (linear) re-ranking
    for model_name, path in MODELS.items():
        if not os.path.exists(path):
            print(f"Skipping {model_name} (path not found).")
            continue
            
        print(f"Ranking-based calibration for {model_name}...")
        recs = pd.read_csv(path, sep="\t")
        recs["list"] = recs["recommended_items"].apply(lambda x: str(x).split(","))

        for lam in LAMBDAS:
            print(f"  Processing Lambda {lam}...")
            final_results = []

            for _, row in recs.iterrows():
                user = row["user_id"]
                items = row["list"][:100]

                if user not in P.index:
                    top10 = items[:TOP_K]
                else:
                    top10 = utils.rerank_greedy_jsd(
                        user_id=user,
                        items=items,
                        P_user_dist=P.loc[user].values,
                        song2emotion=song2emotion,
                        emo2idx=emo2idx,
                        num_emotions=num_emotions,
                        lam=lam,
                        top_k=TOP_K,
                        scores=None,
                        score_type='rank'
                    )

                final_results.append({
                    "user_id": user,
                    "recommended_items": ",".join(top10)
                })

            # Save the results
            out_df = pd.DataFrame(final_results)
            save_name = f"outputs/03_calibration/user_top10_{model_name.lower()}_linear_lambda_{lam}.tsv"
            out_df.to_csv(save_name, sep="\t", index=False)

    print("Linear batch calibration finished.")

if __name__ == "__main__":
    main()
