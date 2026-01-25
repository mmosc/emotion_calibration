import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# CONFIG
INTERACTIONS_PATH = "outputs/01_preprocessing/interactions_binarized.csv"
GEMS_PATH = "data/id_highest_gems.tsv"
OUTPUT_DIR = "outputs/04_evaluation"

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Load data
    print("Loading interactions and gems...")
    inter = pd.read_csv(INTERACTIONS_PATH)
    inter.columns = ["user", "song", "label"]

    gems = pd.read_csv(GEMS_PATH, sep="\t")
    if "id" in gems.columns:
        gems = gems.rename(columns={"id": "song", "highest_gem": "emotion"})
    
    song2emotion = dict(zip(gems["song"], gems["emotion"]))

    # 2. Map history interactions to emotions
    print("Calculating user profile distribution...")
    inter["emotion"] = inter["song"].map(song2emotion)
    valid_inter = inter.dropna(subset=["emotion"])
    
    # Get distribution counts as percentages
    counts = valid_inter["emotion"].value_counts(normalize=True).sort_index()
    emotions = counts.index
    values = counts.values * 100 
    
    # 3. Plotting using Seaborn (restoring original style)
    plt.figure(figsize=(10, 6))
    
    sns.barplot(x=emotions, y=values, color="lightgreen")
    
    plt.title("User Profile Emotion Distribution (User History)", fontsize=14)
    plt.ylabel("Percentage (%)")
    plt.xlabel("Emotion")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    save_path = f"{OUTPUT_DIR}/plot_user_profile_dist.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    main()
