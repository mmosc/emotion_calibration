import pandas as pd
import os

# CONFIG
GEMS_PATH = "data/id_highest_gems.tsv"
INTERACTIONS_PATH = "outputs/01_preprocessing/interactions_binarized.csv"
OUTPUT_DIR = "outputs/04_evaluation"
SCORE_TYPE = 'model' # options: 'rank', 'model'
MODELS = ["BPR", "ItemKNN", "MostPop"]

FILE_PATTERNS = {
    "BPR": f"outputs/03_calibration/user_top10_BPR_calitune_{SCORE_TYPE}_lambda_{{}}.tsv",
    "ItemKNN": f"outputs/03_calibration/user_top10_itemknn_calitune_{SCORE_TYPE}_lambda_{{}}.tsv",
    "MostPop": f"outputs/03_calibration/user_top10_mostpop_calitune_{SCORE_TYPE}_lambda_{{}}.tsv"
}

def load_gems():
    gems = pd.read_csv(GEMS_PATH, sep="\t")
    if "id" in gems.columns and "highest_gem" in gems.columns:
        gems = gems.rename(columns={"id": "song", "highest_gem": "emotion"})
    return gems, dict(zip(gems["song"], gems["emotion"]))

def get_recommendation_distribution(lam, song2emotion):
    dist_dict = {}
    for model in MODELS:
        file_path = FILE_PATTERNS[model].format(float(lam))
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found.")
            continue
        df = pd.read_csv(file_path, sep="\t")
        all_items = []
        for items_str in df["recommended_items"]:
            if isinstance(items_str, str):
                all_items.extend(items_str.split(","))
        emotions = [song2emotion.get(item) for item in all_items if song2emotion.get(item)]
        if emotions:
            dist_dict[model] = pd.Series(emotions).value_counts(normalize=True)
    return pd.DataFrame(dist_dict).fillna(0).sort_index()

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    print("Loading GEM data...")
    gems, song2emotion = load_gems()
    
    # 1. Catalog Distribution
    print("Exporting Catalog Distribution...")
    cat_dist = gems["emotion"].value_counts(normalize=True).sort_index()
    cat_dist.to_csv(f"{OUTPUT_DIR}/distribution_catalog.csv", header=["proportion"])
    
    # 2. User Profile Distribution
    print("Exporting User Profile Distribution...")
    if os.path.exists(INTERACTIONS_PATH):
        inter = pd.read_csv(INTERACTIONS_PATH)
        inter.columns = ["user", "song", "label"]
        inter["emotion"] = inter["song"].map(song2emotion)
        user_dist = inter["emotion"].value_counts(normalize=True).sort_index()
        user_dist.to_csv(f"{OUTPUT_DIR}/distribution_user_profile.csv", header=["proportion"])
    else:
        print(f"Warning: {INTERACTIONS_PATH} not found.")
    
    # 3. Lambda = 0 Distribution
    print(f"Exporting Lambda 0 (Accuracy) Distribution ({SCORE_TYPE})...")
    df_lam0 = get_recommendation_distribution(0.0, song2emotion)
    if not df_lam0.empty:
        df_lam0.to_csv(f"{OUTPUT_DIR}/distribution_{SCORE_TYPE}_lambda_0.csv")
    
    # 4. Lambda = 1 Distribution
    print(f"Exporting Lambda 1 (Calibration) Distribution ({SCORE_TYPE})...")
    df_lam1 = get_recommendation_distribution(1.0, song2emotion)
    if not df_lam1.empty:
        df_lam1.to_csv(f"{OUTPUT_DIR}/distribution_{SCORE_TYPE}_lambda_1.csv")

    print(f"\nSuccess! Files created in: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()
