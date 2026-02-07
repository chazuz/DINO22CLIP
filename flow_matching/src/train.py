"""
flow_matching/src/train.py

Train flow matching models to map between CLIP and DINO embeddings.

Flow matching learns a continuous velocity field that interpolates between
source and target embedding spaces.

USAGE:
    # Train both directions (default)
    python flow_matching/src/train.py \
        --data_dir data/embeddings/CIFAR_50k \
        --model_dir flow_matching/models

    # Custom hyperparameters
    python flow_matching/src/train.py \
        --data_dir data/embeddings/CIFAR_50k \
        --model_dir flow_matching/models \
        --epochs 50 \
        --batch_size 256 \
        --lr 1e-4 \
        --device cuda
"""

import torch
import argparse
from torch.utils.data import DataLoader
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from flow_matching.src.model import ConditionalFlowModel
from flow_matching.src.utils import FlowMatching
from src.embedding_dataset import EmbeddingDataset


def train_direction(
    direction,
    data_dir,
    model_dir,
    epochs=50,
    batch_size=256,
    lr=1e-4,
    device='cpu',
    seed=42
):
    """
    Train one direction of flow matching.
    
    Args:
        direction: 'clip2dino' or 'dino2clip'
        data_dir: Directory containing z_clip.pt and z_dino.pt
        model_dir: Where to save trained model
        epochs: Number of training epochs
        batch_size: Training batch size
        lr: Learning rate
        device: 'cuda' or 'cpu'
        seed: Random seed for reproducibility
    """
    torch.manual_seed(seed)
    
    print("="*60)
    print(f"Training: {direction}")
    print("="*60)
    
    # Load dataset
    dataset = EmbeddingDataset(data_dir, direction)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Source dim: {dataset.X.shape[1]}, Target dim: {dataset.Y.shape[1]}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")
    print(f"Device: {device}\n")
    
    # Initialize model and flow matching
    model = ConditionalFlowModel(
        target_dim=dataset.Y.shape[1],
        cond_dim=dataset.X.shape[1]
    ).to(device)
    
    flow = FlowMatching()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)  # conditioning (source embeddings)
            y_batch = y_batch.to(device)  # target embeddings
            
            # Sample trajectory point and true velocity
            x_t, t, u_t = flow.sample_trajectory(y_batch)
            
            # Predict velocity conditioned on source embeddings
            v_pred = model(x_t, t, x_batch)
            
            # Flow matching loss (MSE between predicted and true velocity)
            loss = torch.nn.functional.mse_loss(v_pred, u_t)
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * y_batch.size(0)
        
        # Print progress
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.6f}")
    
    # Save final model
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{direction}_flow.pt"
    torch.save(model.state_dict(), model_path)
    print(f"[✓] Saved: {model_path}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train flow matching models (DINO ⇄ CLIP)"
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
        default="flow_matching/models",
        help="Where to save trained models"
    )
    
    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("FLOW MATCHING TRAINING - BOTH DIRECTIONS")
    print("="*60 + "\n")
    
    # Train both directions
    for direction in ["dino2clip", "clip2dino"]:
        train_direction(
            direction=direction,
            data_dir=args.data_dir,
            model_dir=args.model_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            seed=args.seed
        )
    
    print("="*60)
    print("✓ Training complete for both directions!")
    print("="*60)


if __name__ == "__main__":
    main()
