import pandas as pd
import calibration_utils as utils
import os

INTERACTIONS = "outputs/01_preprocessing/interactions_binarized.csv"
GEMS = "data/id_highest_gems.tsv"

# Models 
MODELS = {
    "ItemKNN": "outputs/02_base_recs/user_top100_itemknn.tsv",
    "MostPop": "outputs/02_base_recs/user_top100_mostpop.tsv",
}

TOP_K = 10
LAMBDAS = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
SCORE_TYPE = 'model' # options: 'rank', 'model'

def main():
    # Load all base data
    print("Loading data...")
    inter, gems, song2emotion, emotions = utils.load_interactions_and_gems(INTERACTIONS, GEMS)
    emo2idx = {e: i for i, e in enumerate(emotions)}
    num_emotions = len(emotions)

    # User history profiles
    P = utils.build_emotion_distribution(inter, gems)
    P = P.reindex(columns=emotions, fill_value=0)

    # Process each model
    for model_name, path in MODELS.items():
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
            
        print(f"Current model: {model_name}")
        recs = pd.read_csv(path, sep="\t")
        recs["list"] = recs["recommended_items"].apply(lambda x: str(x).split(","))
        
        # Check if model scores are available
        if SCORE_TYPE == 'model' and "scores" in recs.columns:
            recs["scores_list"] = recs["scores"].apply(lambda x: [float(s) for s in str(x).split(",")])
            print("Using model scores for re-ranking.")
        else:
            recs["scores_list"] = None
            print(f"Using {SCORE_TYPE} relevance.")

        # Loop through lambda selection
        for lam in LAMBDAS:
            print(f"  Calibrating for Lambda {lam}...")
            rows = []

            for _, row in recs.iterrows():
                user = row["user_id"]
                items = row["list"]
                scores = row["scores_list"] if row["scores_list"] is not None else None

                if user not in P.index:
                    top10 = items[:TOP_K]
                else:
                    # Run greedy JSD algorithm
                    top10 = utils.rerank_greedy_jsd(
                        user_id=user,
                        items=items,
                        P_user_dist=P.loc[user].values,
                        song2emotion=song2emotion,
                        emo2idx=emo2idx,
                        num_emotions=num_emotions,
                        lam=lam,
                        top_k=TOP_K,
                        scores=scores,
                        score_type=SCORE_TYPE
                    )

                rows.append({
                    "user_id": user,
                    "recommended_items": ",".join(top10)
                })

            # Save results
            out_df = pd.DataFrame(rows)
            save_dest = f"outputs/03_calibration/user_top10_{model_name.lower()}_calitune_{SCORE_TYPE}_lambda_{lam}.tsv"
            out_df.to_csv(save_dest, sep="\t", index=False)

    print("Main model calibration loop finished.")

if __name__ == "__main__":
    main()
