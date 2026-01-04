import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================= CONFIG =================
INTERACTIONS_PATH = "outputs/01_preprocessing/interactions_binarized.csv"
GEMS_PATH = "data/id_highest_gems.tsv"
OUTPUT_DIR = "outputs/04_evaluation"
OUTPUT_FILE = f"{OUTPUT_DIR}/plot_user_profile_dist.png"
# =========================================

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Loading interactions from {INTERACTIONS_PATH}...")
    inter = pd.read_csv(INTERACTIONS_PATH)
    # Typical columns: user_id, item_id, label (or similar)
    # Adjust header names if necessary based on your file structure
    inter.columns = ["user", "song", "label"]

    print(f"Loading gems from {GEMS_PATH}...")
    gems = pd.read_csv(GEMS_PATH, sep="\t")
    if "id" in gems.columns and "highest_gem" in gems.columns:
        gems = gems.rename(columns={"id": "song", "highest_gem": "emotion"})
    
    song2emotion = dict(zip(gems["song"], gems["emotion"]))

    # Map interactions to emotions
    print("Mapping user history to emotions...")
    # Filter interactions to only those with known emotions
    inter["emotion"] = inter["song"].map(song2emotion)
    
    # Drop items without emotion mapping (if any)
    valid_inter = inter.dropna(subset=["emotion"])
    
    if valid_inter.empty:
        print("Error: No valid interactions found with emotion mappings.")
        return

    # Count distribution
    counts = valid_inter["emotion"].value_counts(normalize=True).sort_index()
    
    # Plot
    print("Plotting...")
    plt.figure(figsize=(10, 6))
    
    emotions = counts.index
    values = counts.values * 100 # Convert to %
    
    sns.barplot(x=emotions, y=values, color="lightgreen")
    
    plt.title("User Profile Emotion Distribution (User History)", fontsize=14)
    plt.ylabel("Percentage (%)", fontsize=12)
    plt.xlabel("Emotion", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig(OUTPUT_FILE)
    plt.close()
    print(f"Saved plot to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
