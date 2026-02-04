import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
import os
import glob

# ================= CONFIG =================
INTERACTIONS_PATH = "outputs/01_preprocessing/interactions_binarized.csv"
GEMS_PATH         = "data/id_highest_gems.tsv"
OUTPUT_DIR        = "outputs/04_evaluation/methodology_analysis"
SCORE_TYPE        = 'model' 

MODELS = ["BPR", "ItemKNN", "MostPop"]
LAMBDAS = [0.0, 1.0] 



FILE_PATTERNS = {
    "BPR": f"outputs/03_calibration/user_top10_BPR_calibrated_{SCORE_TYPE}_lambda_{{}}.tsv",
    "ItemKNN": f"outputs/03_calibration/user_top10_itemknn_calibrated_{SCORE_TYPE}_lambda_{{}}.tsv",
    "MostPop": f"outputs/03_calibration/user_top10_mostpop_calibrated_{SCORE_TYPE}_lambda_{{}}.tsv"
}
# =========================================

def load_data():
    print(f"Loading Gems from {GEMS_PATH}...")
    gems = pd.read_csv(GEMS_PATH, sep="\t")
    if "id" in gems.columns and "highest_gem" in gems.columns:
        gems = gems.rename(columns={"id": "song", "highest_gem": "emotion"})
    
    song2emotion = dict(zip(gems["song"], gems["emotion"]))
    emotions = sorted(gems["emotion"].dropna().unique())
    emo2idx = {e: i for i, e in enumerate(emotions)}
    
    print(f"Loading Interactions from {INTERACTIONS_PATH}...")
    if os.path.exists(INTERACTIONS_PATH):
        inter = pd.read_csv(INTERACTIONS_PATH)
        inter.columns = ["user", "song", "label"]
    else:
        # Fallback/Mock for testing if file missing in active workspace context (unlikely)
        raise FileNotFoundError(f"{INTERACTIONS_PATH} not found")
        
    return gems, inter, song2emotion, emotions, emo2idx

def compute_dist(item_list, song2emotion, num_emotions, emo2idx):
    counts = np.zeros(num_emotions)
    total = 0
    for item in item_list:
        if item in song2emotion:
            e = song2emotion[item]
            if e in emo2idx:
                counts[emo2idx[e]] += 1
                total += 1
    
    if total == 0:
        return np.zeros(num_emotions)
    return counts / total

def main():
    if not os.path.exists(OUTPUT_DIR):
        print(f"Creating output directory: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR)

    gems, inter, song2emotion, emotions, emo2idx = load_data()
    num_emotions = len(emotions)
    print(f"Detected {num_emotions} emotions: {emotions}")
    
    # --- B. Emotion Distributions ---
    
    # 1. Catalog Distribution (P_cat)
    print("Computing Catalog Distribution...")
    cat_dist = compute_dist(gems['song'].tolist(), song2emotion, num_emotions, emo2idx)
    
    # 2. Users' Listening History Aggregated (P_hist_aggr)
    print("Computing Aggregated History Distribution...")
   
    # This implies we just iterate over the interactions column 'song'
    hist_dist_aggr = compute_dist(inter['song'].tolist(), song2emotion, num_emotions, emo2idx)
    
    # Saved these for plotting later if needed
    pd.DataFrame({
        "emotion": emotions,
        "catalog": cat_dist,
        "history_aggr": hist_dist_aggr
    }).to_csv(f"{OUTPUT_DIR}/dist_catalog_vs_history.csv", index=False)

    # --- C. JSD Analysis ---
    
    results = []
    
    # C.1 Aggregated Level
    # JSD(Catalog || History)
    jsd_cat_hist = jensenshannon(cat_dist, hist_dist_aggr, base=2) ** 2
    print(f"JSD(Catalog || Aggr History) = {jsd_cat_hist:.6f}")
    
    results.append({
        "Analysis": "Aggr: Catalog vs History",
        "Model": "N/A",
        "Lambda": "N/A",
        "JSD": jsd_cat_hist
    })
    
    # JSD(Catalog || Aggr Recs) -> computed per model loop below
    
    # C.2 User Level
    # JSD(P^u || Catalog)
    # First, build P^u for all users
    print("Building P^u for all users...")
    user_groups = inter.groupby("user")["song"]
    
    user_jsd_cat_vals = []
    user_profiles = {}
    
    for user, tracks in user_groups:
        p_u = compute_dist(tracks, song2emotion, num_emotions, emo2idx)
        user_profiles[user] = p_u
        # JSD(P^u || Catalog)
        val = jensenshannon(p_u, cat_dist, base=2) ** 2
        user_jsd_cat_vals.append(val)
        
    mean_user_jsd_cat = np.mean(user_jsd_cat_vals)
    print(f"Mean User JSD(P^u || Catalog) = {mean_user_jsd_cat:.6f}")
    
    results.append({
        "Analysis": "User Level Mean: P^u vs Catalog",
        "Model": "N/A",
        "Lambda": "N/A",
        "JSD": mean_user_jsd_cat
    })
    
    # Loop over models for Recs-related JSDs
    for model in MODELS:
        # Check for files
        pattern = FILE_PATTERNS[model]
        # Find all available lambdas for this model
        # Just iterating a fixed set for simplicity and to match the likely available files
        available_lambdas = [0.0, 1.0] # We can check others if they exist, but these are critical
        
        for lam in available_lambdas:
            fname = pattern.format(lam)
            if not os.path.exists(fname):
                continue
                
            print(f"Processing {model} Lambda={lam}...")
            recs_df = pd.read_csv(fname, sep="\t")
            
            # --- Aggregated Recs Distribution ---
            all_recs = []
            for item_str in recs_df["recommended_items"]:
                if isinstance(item_str, str):
                    all_recs.extend(item_str.split(","))
            
            recs_dist_aggr = compute_dist(all_recs, song2emotion, num_emotions, emo2idx)
            
            # JSD(Catalog || Aggr Recs)
            jsd_cat_recs = jensenshannon(cat_dist, recs_dist_aggr, base=2) ** 2
            
            results.append({
                "Analysis": "Aggr: Catalog vs Recs",
                "Model": model,
                "Lambda": lam,
                "JSD": jsd_cat_recs
            })
            
            # Note: User Level JSD(P^u || R^u) is already in evaluate_lambda_table.py
             
            # I'll calculate it for completeness to have it in one file.
            
            user_jsd_pu_ru = []
            for _, row in recs_df.iterrows():
                u = row["user_id"]
                if u not in user_profiles: 
                    continue
                
                if isinstance(row["recommended_items"], str):
                    items = row["recommended_items"].split(",")
                else:
                    items = []
                
                r_u = compute_dist(items, song2emotion, num_emotions, emo2idx)
                # JSD(P^u || R^u)
                val = jensenshannon(user_profiles[u], r_u, base=2) ** 2
                user_jsd_pu_ru.append(val)
            
            if user_jsd_pu_ru:
                mean_jsd_pu_ru = np.mean(user_jsd_pu_ru)
                results.append({
                    "Analysis": "User Level Mean: P^u vs R^u",
                    "Model": model,
                    "Lambda": lam,
                    "JSD": mean_jsd_pu_ru
                })

    # Save Results
    res_df = pd.DataFrame(results)
    print("\n--- Final Results ---")
    print(res_df)
    res_df.to_csv(f"{OUTPUT_DIR}/jsd_analysis_results.csv", index=False)
    print(f"\nSaved analysis to {OUTPUT_DIR}/jsd_analysis_results.csv")

if __name__ == "__main__":
    main()
