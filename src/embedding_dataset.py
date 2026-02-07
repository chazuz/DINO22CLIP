import os
import torch
from torch.utils.data import Dataset

class EmbeddingDataset(Dataset):
    """
    Dataset for paired embeddings (CLIP <-> DINO).

    Args:
        data_dir (str): folder containing `z_clip.pt` and `z_dino.pt`
        direction (str): "dino2clip" or "clip2dino"
    """
    def __init__(self, data_dir, direction="dino2clip"):
        self.data_dir = os.path.abspath(data_dir)  # absolute path for portability
        self.direction = direction
        self.load_embeddings()

    def load_embeddings(self):
        """Load embeddings and set X, Y depending on direction"""
        # Safe CPU loading to ensure portability
        self.z_dino = torch.load(os.path.join(self.data_dir, "dino.pt"), map_location="cpu")
        self.z_clip = torch.load(os.path.join(self.data_dir, "clip.pt"), map_location="cpu")

        if self.direction == "dino2clip":
            self.X = self.z_dino
            self.Y = self.z_clip
        elif self.direction == "clip2dino":
            self.X = self.z_clip
            self.Y = self.z_dino
        else:
            raise ValueError(
                f"Invalid direction '{self.direction}'. Must be 'dino2clip' or 'clip2dino'."
            )

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ================================
# Optional CLI test for sanity check
# ================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test EmbeddingDataset loading.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data/embeddings",
        help="Path to folder containing dino.pt and clip.pt",
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="dino2clip",
        choices=["dino2clip", "clip2dino"],
        help="Mapping direction: 'dino2clip' or 'clip2dino'",
    )
    args = parser.parse_args()

    dataset = EmbeddingDataset(args.data_dir, args.direction)
    print(f"Loaded dataset with {len(dataset)} samples.")
    print(f"X shape: {dataset.X.shape}, Y shape: {dataset.Y.shape}")
    print(f"First sample X[0]: {dataset.X[0][:5]}")  # show first 5 elements for sanity
