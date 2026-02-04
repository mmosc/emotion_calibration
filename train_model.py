import sys
import torch

# Patch torch.load for PyTorch >= 2.6 compatibility with RecBole
_orig_load = torch.load
torch.load = lambda *a, **kw: _orig_load(*a, **dict(kw, weights_only=False))

from recbole.quick_start import run_recbole

CONFIGS = {
    "BPR": "recbole_configs/Recbole_BPR_config_my_dataset.yaml",
    "ItemKNN": "recbole_configs/Recbole_ItemKNN_config_my_dataset.yaml",
}

if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else None
    if model not in CONFIGS:
        print(f"Usage: python train_model.py <{'|'.join(CONFIGS.keys())}>")
        sys.exit(1)

    run_recbole(model=model, dataset="my_dataset", config_file_list=[CONFIGS[model]])
