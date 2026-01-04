import os
import pandas as pd
import numpy as np

def main():

    os.chdir(r"C:\Users\Emra\Desktop\PR")

    # Load interactions (correct path)
    interactions = pd.read_csv("outputs/01_preprocessing/interactions_binarized.csv")
    popularity = interactions.groupby("song").size().sort_values(ascending=False)

    # Top 100 most popular items
    top100_items = popularity.head(100).index.tolist()

    # Load users from BPR recommendations (to get user list)
    recs_bpr = pd.read_csv("outputs/02_base_recs/user_top100_BPR.tsv", sep="\t")
    user_ids = recs_bpr["user_id"].tolist()

    rows = []
    for u in user_ids:
        rows.append({
            "user_id": u,
            "recommended_items": ",".join(map(str, top100_items))
        })

    out_path = "outputs/02_base_recs/user_top100_mostpop.tsv"
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)

    print(f"Saved Most Popular recommendations to {out_path}")


if __name__ == "__main__":
    main()