import os
import torch
import pandas as pd
from recbole.quick_start import load_data_and_model
from recbole.data.interaction import Interaction

def main():

    # Move into CaliTune project folder so relative paths work
    os.chdir(r"C:\Users\Emra\Desktop\PR\CaliTune")

    # ----------------------
    # 1. Load BPR model
    # ----------------------
    model_path = r"saved\BPR-Oct-22-2025_11-00-51.pth"  # 👉 your actual file here
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(model_path)

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
    out_path = r"C:\Users\Emra\Desktop\PR\user_top100_BPR.tsv"
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)

    print(f"\n✅ Saved BPR top-100 per user to {out_path}")

if __name__ == "__main__":
    main()
