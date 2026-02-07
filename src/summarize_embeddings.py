import torch
from pathlib import Path

ROOT = Path("data/embeddings")

def summarize(path: Path):
    """Print a concise summary of a tensor file."""
    try:
        x = torch.load(path, map_location="cpu")
        if not torch.is_tensor(x):
            print(f"{path.name}: not a tensor")
            return
        shape = tuple(x.shape)
        mean = x.float().mean().item()
        std = x.float().std().item()
        dtype = x.dtype
        print(f"{path.name:35} | shape={shape} | mean={mean:.6f} | std={std:.6f} | dtype={dtype}")
    except Exception as e:
        print(f"{path.name}: ERROR ({e})")

if __name__ == "__main__":
    print("=== Embedding Summary ===")
    # Group files by dataset
    embeddings = sorted(ROOT.rglob("*.pt"))
    groups = {
        "CIFAR_50k": [],
        "CIFAR_5k": [],
        "CIFAR_5k_test": [],
        "adv_probes": [],
        "other": []
    }
    
    for pt in embeddings:
        path_str = str(pt)
        if "CIFAR_5k_test" in path_str:
            groups["CIFAR_5k_test"].append(pt)
        elif "CIFAR_5k" in path_str:
            groups["CIFAR_5k"].append(pt)
        elif "CIFAR_50k" in path_str:
            groups["CIFAR_50k"].append(pt)
        elif "adv_probes" in path_str:
            groups["adv_probes"].append(pt)
        else:
            groups["other"].append(pt)
    
    for group_name in ["CIFAR_50k", "CIFAR_5k", "CIFAR_5k_test", "adv_probes", "other"]:
        files = groups.get(group_name, [])
        if not files:
            continue
        print(f"\n--- {group_name} ---")
        for pt in sorted(files):
            summarize(pt)
