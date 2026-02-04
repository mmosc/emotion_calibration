import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

# CONFIG
GEMS_PATH = "data/id_highest_gems.tsv"
INTERACTIONS_PATH = "outputs/01_preprocessing/interactions_binarized.csv"
OUTPUT_DIR = "outputs/04_evaluation"
SCORE_TYPE = 'model' 

MODELS = ["BPR", "ItemKNN", "MostPop"]

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Load basic data
    print("Loading data...")
    gems = pd.read_csv(GEMS_PATH, sep="\t")
    if "id" in gems.columns:
        gems = gems.rename(columns={"id": "song", "highest_gem": "emotion"})
    
    song2emotion = dict(zip(gems["song"], gems["emotion"]))
    all_emotions = sorted(gems["emotion"].dropna().unique())

    # Catalog distribution
    cat_counts = gems["emotion"].value_counts(normalize=True).sort_index()

    # 2. Catalog vs History Plot
    print("Processing History Distribution...")
    inter = pd.read_csv(INTERACTIONS_PATH)
    inter.columns = ["user", "song", "label"]
    
    # Map history to emotions
    hist_emos = inter["song"].map(song2emotion).dropna()
    hist_counts = hist_emos.value_counts(normalize=True).sort_index()

    # Create comparison dataframe for Hist vs Cat
    df_hist = pd.DataFrame({
        "Emotion": all_emotions,
        "Catalog": cat_counts.reindex(all_emotions, fill_value=0).values * 100,
        "Listening History": hist_counts.reindex(all_emotions, fill_value=0).values * 100
    })
    
    melt_hist = df_hist.melt(id_vars="Emotion", var_name="Type", value_name="Percentage")
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=melt_hist, x="Emotion", y="Percentage", hue="Type")
    plt.title("Catalog vs Listening History Emotion Distribution", fontsize=14)
    plt.ylabel("Percentage (%)")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plot_catalog_vs_history.png")
    plt.close()

    # 3. Model Distributions (Lambda 0 and 1)
    for lam in [0.0, 1.0]:
        print(f"Processing Models for Lambda {lam}...")
        
        plot_data = {"Emotion": all_emotions}
        # Add catalog as baseline
        plot_data["Catalog"] = cat_counts.reindex(all_emotions, fill_value=0).values * 100
        
        for model in MODELS:
            fname = f"outputs/03_calibration/user_top10_{model.lower()}_calibrated_{SCORE_TYPE}_lambda_{lam}.tsv"
            if os.path.exists(fname):
                df = pd.read_csv(fname, sep="\t")
                recs = []
                for s in df["recommended_items"]:
                    if isinstance(s, str):
                        recs.extend(s.split(","))
                
                m_emos = [song2emotion[r] for r in recs if r in song2emotion]
                m_counts = pd.Series(m_emos).value_counts(normalize=True).reindex(all_emotions, fill_value=0)
                plot_data[model] = m_counts.values * 100
        
        # Melt and plot
        df_models = pd.DataFrame(plot_data)
        melt_models = df_models.melt(id_vars="Emotion", var_name="Source", value_name="Percentage")
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=melt_models, x="Emotion", y="Percentage", hue="Source")
        plt.title(f"Catalog vs Top-10 Recs (Lambda={lam})", fontsize=14)
        plt.ylabel("Percentage (%)")
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/plot_lambda_{lam}_dist.png")
        plt.close()

    print("Done.")

if __name__ == "__main__":
    main()
