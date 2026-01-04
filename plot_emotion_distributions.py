import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

# ================= CONFIG =================
GEMS_PATH = "data/id_highest_gems.tsv"
OUTPUT_DIR = "outputs/04_evaluation"

# Models 
MODELS = ["BPR", "ItemKNN", "MostPop"]

# Path 
FILE_PATTERNS = {
    "BPR": "outputs/03_calibration/user_top10_BPR_calitune_lambda_{}.tsv",
    "ItemKNN": "outputs/03_calibration/user_top10_itemknn_calitune_lambda_{}.tsv",
    "MostPop": "outputs/03_calibration/user_top10_mostpop_calitune_lambda_{}.tsv"
}
# =========================================

def load_gems():
    print(f"Loading gems from {GEMS_PATH}...")
    gems = pd.read_csv(GEMS_PATH, sep="\t")
    # Columns usually: id, highest_gem. Rename for consistency
    if "id" in gems.columns and "highest_gem" in gems.columns:
        gems = gems.rename(columns={"id": "song", "highest_gem": "emotion"})
    
    song2emotion = dict(zip(gems["song"], gems["emotion"]))
    return gems, song2emotion

def get_catalog_distribution(gems_df):
    """
    1. Percentage of tracks of each emotion label in the whole catalog
    """
    counts = gems_df["emotion"].value_counts(normalize=True).sort_index()
    return counts

def get_recommendation_distribution(lam, song2emotion):
    
    
    
    dist_dict = {}

    for model in MODELS:
        file_path = FILE_PATTERNS[model].format(lam)
        if not os.path.exists(file_path):
            print(f"Warning: File not found {file_path}. Skipping {model}.")
            continue
        
        print(f"Processing {model} (Lambda={lam})...")
        df = pd.read_csv(file_path, sep="\t")
        
        # Flatten all recommendations
        # "item1,item2,..." -> ["item1", "item2", ...]
        all_items = []
        for items_str in df["recommended_items"]:
            if isinstance(items_str, str):
                all_items.extend(items_str.split(","))
        
        # Map to emotions
        emotions = []
        for item in all_items:
            emo = song2emotion.get(item)
            if emo:
                emotions.append(emo)
        
        # Count and normalize
        
        if emotions:
            counts = pd.Series(emotions).value_counts(normalize=True)
            dist_dict[model] = counts
        else:
            dist_dict[model] = pd.Series(dtype=float)

    # Combine into one DF
    if dist_dict:
        combined_df = pd.DataFrame(dist_dict).fillna(0).sort_index()
        return combined_df
    return pd.DataFrame()

def plot_catalog_dist(counts, save_path):
    plt.figure(figsize=(10, 6))
    
    # Simple bar plot
    # x = emotions, y = percentage
    emotions = counts.index
    values = counts.values * 100 # Convert to %
    
    sns.barplot(x=emotions, y=values, color="skyblue")
    
    plt.title("Emotion Distribution in Catalog (Percentage of Tracks)", fontsize=14)
    plt.ylabel("Percentage (%)", fontsize=12)
    plt.xlabel("Emotion", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig(save_path)
    plt.close()
    print(f"Saved catalog plot to {save_path}")

def plot_model_dist(df, title, save_path):
    if df.empty:
        print(f"No data to plot for {title}")
        return

    # Prepare data 
    # Melt: Emotion | Model | Percentage
    df_reset = df.reset_index().rename(columns={"index": "Emotion"})
    melted = df_reset.melt(id_vars="Emotion", var_name="Model", value_name="Percentage")
    melted["Percentage"] = melted["Percentage"] * 100 # Convert to %

    plt.figure(figsize=(12, 6))
    sns.barplot(data=melted, x="Emotion", y="Percentage", hue="Model")
    
    plt.title(title, fontsize=14)
    plt.ylabel("Percentage (%)", fontsize=12)
    plt.xlabel("Emotion", fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    gems, song2emotion = load_gems()

    # 1. Catalog Distribution
    print("\n--- 1. Catalog Distribution ---")
    cat_counts = get_catalog_distribution(gems)
    plot_catalog_dist(cat_counts, f"{OUTPUT_DIR}/plot_catalog_dist.png")

    # 2. Top-10 (Lambda=0)
    print("\n--- 2. Lambda=0 Distribution ---")
    df_lam0 = get_recommendation_distribution(0.0, song2emotion)
    plot_model_dist(df_lam0, "Emotion Distribution in Top-10 Recs (Lambda=0)", f"{OUTPUT_DIR}/plot_lambda_0_dist.png")

    # 3. Top-10 (Lambda=1)
    print("\n--- 3. Lambda=1 Distribution ---")
    df_lam1 = get_recommendation_distribution(1.0, song2emotion)
    plot_model_dist(df_lam1, "Emotion Distribution in Top-10 Recs (Lambda=1)", f"{OUTPUT_DIR}/plot_lambda_1_dist.png")

    print("\nDone.")

if __name__ == "__main__":
    main()
