import os
import pandas as pd
import numpy as np

def main():

    # Load interactions (correct path)
    interactions = pd.read_csv(r"C:\Users\Emra\Desktop\PR\interactions_binarized.csv")

    # Correct item column is: "song"
    all_items = interactions["song"].unique()

    # Load users from BPR recommendations
    recs_bpr = pd.read_csv(r"C:\Users\Emra\Desktop\PR\user_top100_BPR.tsv", sep="\t")
    user_ids = recs_bpr["user_id"].tolist()

    rows = []

    for u in user_ids:
        # sample 100 random *unique* items
        sampled_items = np.random.choice(all_items, size=100, replace=False)

        rows.append({
            "user_id": u,
            "recommended_items": ",".join(map(str, sampled_items))
        })

    out_path = r"C:\Users\Emra\Desktop\PR\user_top100_random.tsv"
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)

    print(f"Saved Random recommendations to {out_path}")

if __name__ == "__main__":
    main()
