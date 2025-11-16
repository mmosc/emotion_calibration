import os
import torch
import pandas as pd
from recbole.quick_start import load_data_and_model
from recbole.data.interaction import Interaction


def main():

    # Changing to the project directory where the model and data are stored
    os.chdir(r"C:\Users\Emra\Desktop\PR\CaliTune")


    # Load the pre-trained BPR model and associated data
    model_path = r"saved\ItemKNN-Nov-16-2025_20-42-36.pth"
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(model_path)

    # Set up device (GPU if available, else CPU) and put model in evaluation mode
    device = torch.device(config["device"])
    model.to(device)
    model.eval()

    # Get field names and user count from dataset
    user_field = dataset.uid_field  # "user" - field name for user IDs
    item_field = dataset.iid_field  #  "item" - field name for item IDs
    user_num = dataset.user_num  # total number of users in the dataset
    k = 100  # number of recommendations per user

    # List to store top-k items for each user
    all_user_items = []

    # Generate recommendations for each user
    for uid in range(user_num):
        # Create interaction object for current user
        # We need to format the input properly for the model
        u_tensor = torch.tensor([uid], device=device)
        interaction = Interaction({user_field: u_tensor}).to(device)

        # Get prediction scores for ALL items for this user
        # full_sort_predict returns scores for every item in the catalog
        scores = model.full_sort_predict(interaction)  # returns 1D version

        # Ensure scores have the right shape [1, n_items]
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)  # make it [1, n_items]

        # Select top-k items with highest scores
        # torch.topk returns both values and indices - we only need indices
        _, topk_idx = torch.topk(scores, k=k, dim=1)  # shape: [1, k]
        all_user_items.append(topk_idx.cpu())  # Move to CPU and store

        # Progress tracking - print every 500 users
        if (uid + 1) % 500 == 0:
            print(f"processed {uid + 1}/{user_num} users")

    # Combine all user recommendations into single tensor
    # Result shape: [user_num, k]
    topk_indices = torch.cat(all_user_items, dim=0)

    # Convert internal IDs back to original tokens/IDs
    # Map user indices to their original user IDs
    user_tokens = dataset.id2token(user_field, list(range(user_num)))
    # Map item indices to their original item IDs for all recommendations
    item_tokens = dataset.id2token(item_field, topk_indices)

    # Build DataFrame for output
    rows = []
    for u_tok, items in zip(user_tokens, item_tokens.tolist()):
        rows.append({
            "user_id": u_tok,  # Original user ID
            "recommended_items": ",".join(items),  # Comma-separated list of recommended items
            # Items are ordered from best (highest score) to worst (lowest score)
        })

    # Save results to TSV file
    out_path = r"C:\Users\Emra\Desktop\PR\user_top100.tsv"
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    print(f"Saved top-100 per user to {out_path}")


if __name__ == "__main__":
    main()