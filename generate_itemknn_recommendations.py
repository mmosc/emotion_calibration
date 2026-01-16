import os
import torch
import pandas as pd
import glob
from pathlib import Path
from recbole.quick_start import load_data_and_model
from recbole.data.interaction import Interaction

def main():
    # Setup paths
    project_root = Path.cwd()
    calitune_path = project_root / "CaliTune"
    
    if not calitune_path.exists() and Path("CaliTune").exists():
        calitune_path = Path("CaliTune").absolute()
    
    os.chdir(calitune_path)
    
    # Load latest ItemKNN model
    model_files = glob.glob(str(calitune_path / "saved" / "ItemKNN-*.pth"))
    if not model_files:
        print("Error: No ItemKNN model found in saved/ directory.")
        return
    
    model_path = max(model_files, key=os.path.getmtime)
    rel_model_path = os.path.relpath(model_path, start=calitune_path)
    print(f"Loading model: {rel_model_path}")
    
    config, model, dataset, _, _, _ = load_data_and_model(rel_model_path)
    
    device = torch.device(config["device"])
    model.to(device)
    model.eval()
    
    # Generate recommendations
    user_field = dataset.uid_field
    item_field = dataset.iid_field
    valid_uids = list(range(1, dataset.user_num))  # Skip padding ID 0
    k = 100
    
    all_user_items = []
    all_user_scores = []
    
    print(f"Generating ItemKNN recommendations for {len(valid_uids)} users...")
    
    for uid in valid_uids:
        interaction = Interaction({user_field: torch.tensor([uid], device=device)}).to(device)
        scores = model.full_sort_predict(interaction)
        
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)
        
        topk_score, topk_idx = torch.topk(scores, k=k, dim=1)
        all_user_items.append(topk_idx.cpu())
        all_user_scores.append(topk_score.cpu())
        
        if uid % 500 == 0:
            print(f"Processed {uid}/{len(valid_uids)} users")
    
    # Convert to external IDs
    topk_indices = torch.cat(all_user_items, dim=0)
    topk_scores = torch.cat(all_user_scores, dim=0)
    
    user_tokens = dataset.id2token(user_field, valid_uids)
    item_tokens = dataset.id2token(item_field, topk_indices)
    
    # Save to TSV
    rows = []
    for u_tok, items, scores_list in zip(user_tokens, item_tokens, topk_scores):
        rows.append({
            "user_id": u_tok,
            "recommended_items": ",".join(map(str, items.tolist() if hasattr(items, "tolist") else items)),
            "scores": ",".join(map(str, scores_list.tolist() if hasattr(scores_list, "tolist") else scores_list))
        })
    
    out_path = project_root / "outputs" / "02_base_recs" / "user_top100_itemknn.tsv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    
    print(f"✅ Saved ItemKNN top-100 per user to {out_path}")

if __name__ == "__main__":
    main()