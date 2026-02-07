"""
mlp_baseline/src/train.py

Train MLP models to map between CLIP and DINO embeddings.

USAGE:
    # Train both directions (default)
    python mlp_baseline/src/train.py \
        --data_dir data/embeddings/CIFAR_50k \
        --model_dir mlp_baseline/models

    # Custom hyperparameters
    python mlp_baseline/src/train.py \
        --data_dir data/embeddings/CIFAR_50k \
        --model_dir mlp_baseline/models \
        --hidden_dim 256 \
        --n_hidden 2 \
        --epochs 50 \
        --batch_size 128 \
        --lr 1e-3 \
        --device cuda
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from pathlib import Path

# Assuming your EmbeddingDataset is in src/
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.embedding_dataset import EmbeddingDataset


# ----------------------------
# MLP Model
# ----------------------------
class MLP(nn.Module):
    """Simple feed-forward network for mapping embeddings."""
    
    def __init__(self, input_dim, output_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


# ----------------------------
# Training function
# ----------------------------
def train_mlp(
    dataset,
    hidden_dim=256,
    n_hidden=2,
    epochs=50,
    batch_size=128,
    lr=1e-3,
    device='cpu'
):
    """Train MLP on the given dataset."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    input_dim = dataset.X.shape[1]
    output_dim = dataset.Y.shape[1]
    
    model = MLP(input_dim, output_dim, hidden_dim, n_hidden).to(device).float()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for x, y in loader:
            x = x.to(device).float()
            y = y.to(device).float()
            
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
        
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.6f}")
    
    return model


# ----------------------------
# Main function
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train MLP to map embeddings (DINO ⇄ CLIP)"
    )
    
    # Paths
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/embeddings/CIFAR_50k",
        help="Directory containing z_clip.pt and z_dino.pt"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="mlp_baseline/models",
        help="Where to save trained models"
    )
    
    # Hyperparameters
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--n_hidden", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    
    device = args.device
    print(f"Using device: {device}")
    
    # Train both directions
    directions = [
        ("dino2clip", "z_dino.pt", "z_clip.pt"),
        ("clip2dino", "z_clip.pt", "z_dino.pt")
    ]
    
    for name, src_file, tgt_file in directions:
        print(f"\n{'='*60}")
        print(f"Training MLP for {name}")
        print(f"{'='*60}")
        
        # Load dataset
        dataset = EmbeddingDataset(
            data_dir=args.data_dir,
            direction=name
        )
        
        # Train model
        model = train_mlp(
            dataset,
            hidden_dim=args.hidden_dim,
            n_hidden=args.n_hidden,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device
        )
        
        # Save trained model
        model_path = Path(args.model_dir) / f"{name}_mlp.pt"
        torch.save(model.state_dict(), model_path)
        print(f"[✓] Saved trained MLP to: {model_path}")
        
        # Generate and save embeddings for validation
        with torch.no_grad():
            transformed = model(dataset.X.to(device).float()).cpu()
        
        output_name = (
            "z_clip_from_dino_mlp.pt" if name == "dino2clip"
            else "z_dino_from_clip_mlp.pt"
        )
        output_path = Path(args.data_dir) / output_name
        torch.save(transformed, output_path)
        print(f"[✓] Saved transformed embeddings to: {output_path}")
    
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
