import pandas as pd
import calibration_utils as utils

# ================= CONFIG =================
INTERACTIONS = "outputs/01_preprocessing/interactions_binarized.csv"
GEMS         = "data/id_highest_gems.tsv"
INPUT_RECS   = "outputs/02_base_recs/user_top100_itemknn.tsv"

TOP_K = 10
LAMBDAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 1.0]
SCORE_TYPE = 'model' # options: 'rank', 'model'
# =========================================

def main():
    # 1) Load data
    inter, gems, song2emotion, emotions = utils.load_interactions_and_gems(INTERACTIONS, GEMS)
    emo2idx = {e: i for i, e in enumerate(emotions)}
    num_emotions = len(emotions)

    # 2) Build user emotion distribution
    P = utils.build_emotion_distribution(inter, gems)
    # Reindex to ensure consistent order with 'emotions' list for vector ops
    P = P.reindex(columns=emotions, fill_value=0)

    # 3) Load ItemKNN Recs
    print(f"Loading ItemKNN recs from {INPUT_RECS}...")
    recs = pd.read_csv(INPUT_RECS, sep="\t")
    recs["list"] = recs["recommended_items"].apply(lambda x: str(x).split(","))

    if SCORE_TYPE == 'model' and "scores" in recs.columns:
        recs["scores_list"] = recs["scores"].apply(lambda x: [float(s) for s in str(x).split(",")])
        print("Using real model scores for re-ranking.")
    else:
        recs["scores_list"] = None
        print(f"Using {SCORE_TYPE} relevance for re-ranking.")

    # 4) Loop over lambdas
    for lam in LAMBDAS:
        print(f"Calibrating ItemKNN (Greedy JSD) with lambda = {lam}")
        rows = []

        for _, row in recs.iterrows():
            user = row["user_id"]
            items = row["list"][:50] # candidates
            scores = row["scores_list"][:50] if row["scores_list"] is not None else None

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
                    scores=scores,
                    score_type=SCORE_TYPE
                )

            rows.append({
                "user_id": user,
                "recommended_items": ",".join(top10)
            })

        out_df = pd.DataFrame(rows)
        out_filename = f"outputs/03_calibration/user_top10_itemknn_calitune_{SCORE_TYPE}_lambda_{lam}.tsv"
        out_df.to_csv(out_filename, sep="\t", index=False)
        print(f"Saved: {out_filename}")

    print("\nDone. ItemKNN CaliTune (Greedy) re-ranking completed.")

if __name__ == "__main__":
    main()
