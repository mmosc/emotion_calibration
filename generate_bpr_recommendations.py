import os
import torch
import pandas as pd
import glob
from pathlib import Path
from recbole.quick_start import load_data_and_model
from recbole.data.interaction import Interaction

def main():
    # Detect project root 
    project_root = Path.cwd()
    calitune_path = project_root / "CaliTune"

    if not calitune_path.exists():
        print(f"Warning: CaliTune directory not found at {calitune_path}")
        # fallback to relative if running from PR
        if Path("CaliTune").exists():
            calitune_path = Path("CaliTune").absolute()

    # Move into CaliTune project folder so relative paths work 
    os.chdir(calitune_path)
    print(f"Changed working directory to: {os.getcwd()}")

    # ----------------------
    # 1. Load BPR model
    # ----------------------
    # Auto-discover latest BPR model
    model_files = glob.glob(str(calitune_path / "saved" / "BPR-*.pth"))
    if not model_files:
        print("Error: No BPR model found in saved/ directory.")
        return
    
    # Sort by modification time to get the latest
    model_path = max(model_files, key=os.path.getmtime)
    
    rel_model_path = os.path.relpath(model_path, start=calitune_path)
    print(f"Loading model: {rel_model_path}")

    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(rel_model_path)

    device = torch.device(config["device"])
    model.to(device)
    model.eval()

    user_field = dataset.uid_field
    item_field = dataset.iid_field
    user_num = dataset.user_num
    k = 100

    all_user_items = []

    # ----------------------
    # 2. Generate top-100 recommendations
    # ----------------------
    print(f"Generating recommendations for {user_num} users...")
    
    for uid in range(user_num):
        u_tensor = torch.tensor([uid], device=device)
        interaction = Interaction({user_field: u_tensor}).to(device)
        scores = model.full_sort_predict(interaction)

        if scores.dim() == 1:
            scores = scores.unsqueeze(0)

        _, topk_idx = torch.topk(scores, k=k, dim=1)
        all_user_items.append(topk_idx.cpu())

        if (uid + 1) % 500 == 0:
            print(f"Processed {uid + 1}/{user_num} users")

    topk_indices = torch.cat(all_user_items, dim=0)

    # ----------------------
    # 3. Convert internal → external IDs
    # ----------------------
    user_tokens = dataset.id2token(user_field, list(range(user_num)))
    item_tokens = dataset.id2token(item_field, topk_indices)

    rows = []
    for user_tok, items in zip(user_tokens, item_tokens.tolist()):
        rows.append({
            "user_id": user_tok,
            "recommended_items": ",".join(items)
        })

    # ----------------------
    # 4. Save file
    # ----------------------
    # Save back to project root
    out_path = project_root / "outputs" / "02_base_recs" / "user_top100_BPR.tsv"
    # Ensure directory exists 
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)

    print(f"\n✅ Saved BPR top-100 per user to {out_path}")

if __name__ == "__main__":
    main()
