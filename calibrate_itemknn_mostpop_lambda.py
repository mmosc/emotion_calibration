import pandas as pd
import calibration_utils as utils

# ================= CONFIG =================
INTERACTIONS = "outputs/01_preprocessing/interactions_binarized.csv"
GEMS = "data/id_highest_gems.tsv"

MODELS = {
    "ItemKNN": "outputs/02_base_recs/user_top100_itemknn.tsv",
    "MostPop": "outputs/02_base_recs/user_top100_mostpop.tsv",
}

TOP_K = 10
LAMBDAS = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
SCORE_TYPE = 'model' # options: 'rank', 'model'
# ========================================

def main():
    # 1) Load data
    inter, gems, song2emotion, emotions = utils.load_interactions_and_gems(INTERACTIONS, GEMS)
    emo2idx = {e: i for i, e in enumerate(emotions)}
    num_emotions = len(emotions)

    # 2) Build user emotion distribution P_u
    P = utils.build_emotion_distribution(inter, gems)
    P = P.reindex(columns=emotions, fill_value=0)

    # 3) Main loop
    for model_name, rec_file in MODELS.items():
        print(f"\nProcessing {model_name}")

        try:
            recs = pd.read_csv(rec_file, sep="\t")
            recs["list"] = recs["recommended_items"].apply(lambda x: str(x).split(","))
            
            if SCORE_TYPE == 'model' and "scores" in recs.columns:
                recs["scores_list"] = recs["scores"].apply(lambda x: [float(s) for s in str(x).split(",")])
                print(f"  Using real {model_name} scores/popularity for re-ranking.")
            else:
                recs["scores_list"] = None
                print(f"  Using {SCORE_TYPE} relevance for {model_name} re-ranking.")

        except FileNotFoundError:
            print(f"  Warning: {rec_file} not found. Skipping.")
            continue

        for LAMBDA in LAMBDAS:
            print(f"  Lambda = {LAMBDA}")
            rows = []

            for _, row in recs.iterrows():
                user = row["user_id"]
                items = row["list"]
                scores = row["scores_list"] if row["scores_list"] is not None else None

                if user not in P.index:
                    top10 = items[:TOP_K]
                else:
                    # Using greedy JSD re-ranking 
                    top10 = utils.rerank_greedy_jsd(
                        user_id=user,
                        items=items,
                        P_user_dist=P.loc[user].values,
                        song2emotion=song2emotion,
                        emo2idx=emo2idx,
                        num_emotions=num_emotions,
                        lam=LAMBDA,
                        top_k=TOP_K,
                        scores=scores,
                        score_type=SCORE_TYPE
                    )

                rows.append({
                    "user_id": user,
                    "recommended_items": ",".join(top10)
                })

            out = pd.DataFrame(rows)
            out_filename = f"outputs/03_calibration/user_top10_{model_name}_calitune_{SCORE_TYPE}_lambda_{LAMBDA}.tsv"
            out.to_csv(out_filename, sep="\t", index=False)
            print(f"  Saved: {out_filename}")

    print("\nDone. CaliTune-calibrated Top-10 files created.")

if __name__ == "__main__":
    main()
