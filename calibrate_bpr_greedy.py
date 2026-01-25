import pandas as pd
import calibration_utils as utils

# Settings
INTERACTIONS = "outputs/01_preprocessing/interactions_binarized.csv"
GEMS         = "data/id_highest_gems.tsv"
BPR_RECS     = "outputs/02_base_recs/user_top100_BPR.tsv"

TOP_K = 10
LAMBDAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SCORE_TYPE = 'model' # options: 'rank', 'model'

def main():
    # Load data
    print("Loading data...")
    inter, gems, song2emotion, emotions = utils.load_interactions_and_gems(INTERACTIONS, GEMS)
    emo2idx = {e: i for i, e in enumerate(emotions)}
    num_emotions = len(emotions)

    # Build user profiles
    P = utils.build_emotion_distribution(inter, gems)
    P = P.reindex(columns=emotions, fill_value=0)

    # Load BPR Recommendations
    print("Loading BPR recs...")
    recs = pd.read_csv(BPR_RECS, sep="\t")
    recs["list"] = recs["recommended_items"].apply(lambda x: str(x).split(","))
    
    if SCORE_TYPE == 'model' and "scores" in recs.columns:
        recs["scores_list"] = recs["scores"].apply(lambda x: [float(s) for s in str(x).split(",")])
        print("Using model scores.")
    else:
        recs["scores_list"] = None
        print("Using rank-based relevance.")

    # Loop through each lambda value
    for lam in LAMBDAS:
        print(f"Calibrating BPR for lambda {lam}...")
        results = []

        for idx, row in recs.iterrows():
            user = row["user_id"]
            candidates = row["list"][:100]
            scores = row["scores_list"][:100] if row["scores_list"] is not None else None

            if user not in P.index:
                # Fallback to original top-K if no profile
                top10 = candidates[:TOP_K]
            else:
                # Greedy re-ranking
                top10 = utils.rerank_greedy_jsd(
                    user_id=user,
                    items=candidates,
                    P_user_dist=P.loc[user].values,
                    song2emotion=song2emotion,
                    emo2idx=emo2idx,
                    num_emotions=num_emotions,
                    lam=lam,
                    top_k=TOP_K,
                    scores=scores,
                    score_type=SCORE_TYPE
                )

            results.append({
                "user_id": user,
                "recommended_items": ",".join(top10)
            })

        # Save to file
        out_df = pd.DataFrame(results)
        save_path = f"outputs/03_calibration/user_top10_bpr_calitune_{SCORE_TYPE}_lambda_{lam}.tsv"
        out_df.to_csv(save_path, sep="\t", index=False)
        print(f"Saved: {save_path}")

    print("BPR calibration finished.")

if __name__ == "__main__":
    main()
