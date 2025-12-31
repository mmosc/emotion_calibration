import pandas as pd
import calibration_utils as utils

# ================= CONFIG =================
INTERACTIONS = "outputs/01_preprocessing/interactions_binarized.csv"
GEMS         = "data/id_highest_gems.tsv"
INPUT_RECS   = "outputs/02_base_recs/user_top100_random.tsv"
OUTPUT_FILE  = "outputs/03_calibration/user_top10_random_calitune.tsv"

TOP_K = 10
LAMBDA_VAL = 1.0 
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

    # 3) Load Random Recs
    print(f"Loading Random recs from {INPUT_RECS}...")
    try:
        recs = pd.read_csv(INPUT_RECS, sep="\t")
        recs["list"] = recs["recommended_items"].apply(lambda x: str(x).split(","))
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_RECS}. Make sure to generate random recs first.")
        return

    print(f"Calibrating Random (Greedy JSD) with lambda = {LAMBDA_VAL} (Fixed)...")
    rows = []

    for _, row in recs.iterrows():
        user = row["user_id"]
        # top 50 as candidates
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
                top_k=T
                OP_K
            )

        rows.append({
            "user_id": user,
            "recommended_items": ",".join(top10)
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_FILE, sep="\t", index=False)
    print(f"Saved: {OUTPUT_FILE}")
    print("\nDone. Random CaliTune (Greedy) re-ranking completed.")

if __name__ == "__main__":
    main()
