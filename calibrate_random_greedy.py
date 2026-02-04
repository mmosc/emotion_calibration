import pandas as pd
import calibration_utils as utils

# Settings
INTERACTIONS = "outputs/01_preprocessing/interactions_binarized.csv"
GEMS         = "data/id_highest_gems.tsv"
INPUT_RECS   = "outputs/02_base_recs/user_top100_random.tsv"
SAVE_FILE    = "outputs/03_calibration/user_top10_random_calibrated.tsv"

TOP_K = 10
LAMBDA_VAL = 1.0 

def main():
    # Load interactions and metadata
    print("Loading data...")
    inter, gems, song2emotion, emotions = utils.load_interactions_and_gems(INTERACTIONS, GEMS)
    emo2idx = {e: i for i, e in enumerate(emotions)}
    num_emotions = len(emotions)

    # Build emotion profiles for users
    P = utils.build_emotion_distribution(inter, gems)
    P = P.reindex(columns=emotions, fill_value=0)

    # Load Random recommendations
    print("Loading Random base recs...")
    recs = pd.read_csv(INPUT_RECS, sep="\t")
    recs["list"] = recs["recommended_items"].apply(lambda x: str(x).split(","))

    print(f"Applying Greedy JSD Calibration (Lambda={LAMBDA_VAL})...")
    final_rows = []

    for _, row in recs.iterrows():
        user = row["user_id"]
        # Use top 50 as candidates for random calibration
        items = row["list"][:50] 

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
                lam=LAMBDA_VAL,
                top_k=TOP_K
            )

        final_rows.append({
            "user_id": user,
            "recommended_items": ",".join(top10)
        })

    # Save to file
    out_df = pd.DataFrame(final_rows)
    out_df.to_csv(SAVE_FILE, sep="\t", index=False)
    print(f"Calibration results saved to {SAVE_FILE}")

if __name__ == "__main__":
    main()
