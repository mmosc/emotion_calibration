import os
import pandas as pd
import glob
import numpy as np
from pathlib import Path
from recbole.quick_start import load_data_and_model

def main():
    # Setup paths
    project_root = Path.cwd()
    calitune_path = project_root / "CaliTune"
    
    if not calitune_path.exists() and Path("CaliTune").exists():
        calitune_path = Path("CaliTune").absolute()
    
    os.chdir(calitune_path)
    
    # Load dataset using any available model
    model_files = glob.glob(str(calitune_path / "saved" / "BPR-*.pth"))
    if not model_files:
        print("Error: No BPR model found. Need it to load the dataset split.")
        return
    
    model_path = max(model_files, key=os.path.getmtime)
    rel_model_path = os.path.relpath(model_path, start=calitune_path)
    
    print("Loading dataset split...")
    _, _, dataset, _, _, test_data = load_data_and_model(rel_model_path)
    
    # Calculate relative popularity from test set
    print("Calculating relative popularity in test set...")
    test_inter = test_data.dataset.inter_matrix(form='csr')
    item_counts = np.array(test_inter.sum(axis=0)).flatten()
    total_test_interactions = item_counts.sum()
    
    if total_test_interactions == 0:
        print("Error: Test set is empty!")
        return
    
    item_rel_pop = item_counts / total_test_interactions
    
    # Get top 100 most popular items (excluding padding ID 0)
    top100_iids = np.argsort(item_rel_pop)[-101:]
    top100_iids = top100_iids[top100_iids != 0][-100:][::-1]
    top100_probs = item_rel_pop[top100_iids]
    
    # Convert to external IDs
    valid_uids = list(range(1, dataset.user_num))
    user_tokens = dataset.id2token(dataset.uid_field, valid_uids)
    item_tokens = dataset.id2token(dataset.iid_field, top100_iids)
    
    # Create recommendations (same for all users)
    item_tokens_str = ",".join(map(str, item_tokens))
    scores_str = ",".join(map(str, top100_probs))
    
    print(f"Generating MostPop recommendations for {len(user_tokens)} users...")
    rows = [{"user_id": u_tok, "recommended_items": item_tokens_str, "scores": scores_str} 
            for u_tok in user_tokens]
    
    # Save to TSV
    out_path = project_root / "outputs" / "02_base_recs" / "user_top100_mostpop.tsv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    
    print(f"✅ Saved MostPop recommendations to {out_path}")

if __name__ == "__main__":
    main()