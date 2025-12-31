import os
import torch
import pandas as pd
import glob
from pathlib import Path
from recbole.quick_start import load_data_and_model
from recbole.data.interaction import Interaction

def main():
    project_root = Path.cwd()
    calitune_path = project_root / "CaliTune"
    
    if not calitune_path.exists():
         if Path("CaliTune").exists():
            calitune_path = Path("CaliTune").absolute()

    os.chdir(calitune_path)
    print(f"Changed working directory to: {os.getcwd()}")

    # Auto-discover ItemKNN model
    model_files = glob.glob(str(calitune_path / "saved" / "ItemKNN-*.pth"))
    if not model_files:
        print("Error: No ItemKNN model found in saved/ directory.")
        return

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

    print(f"Generating ItemKNN recommendations for {user_num} users...")

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

    user_tokens = dataset.id2token(user_field, list(range(user_num)))
    item_tokens = dataset.id2token(item_field, topk_indices)

    rows = []
    for u_tok, items in zip(user_tokens, item_tokens.tolist()):
        rows.append({
            "user_id": u_tok,
            "recommended_items": ",".join(items),
        })

    out_path = project_root / "outputs" / "02_base_recs" / "user_top100_itemknn.tsv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    print(f"Saved top-100 per user to {out_path}")

if __name__ == "__main__":
    main()