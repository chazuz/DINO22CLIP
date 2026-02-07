"""
mlp_baseline/src/run.py

Run trained MLP model on embeddings (inference only, no training).

This script handles both one-way and round-trip translations:
- One-way: Just specify input, model, output
- Round-trip: Chain two calls (use output of first as input to second)

USAGE:
    # One-way CLIP → DINO
    python mlp_baseline/src/run.py \
        --input data/embeddings/CIFAR_50k/clip.pt \
        --model mlp_baseline/models/clip2dino_mlp.pt \
        --output data/embeddings/CIFAR_50k/dino_by_clip_mlp.pt

    # Round-trip: CLIP → DINO → CLIP
    # Step 1:
    python mlp_baseline/src/run.py \
        --input data/embeddings/CIFAR_50k/clip.pt \
        --model mlp_baseline/models/clip2dino_mlp.pt \
        --output data/embeddings/CIFAR_50k/dino_intermediate.pt
    
    # Step 2:
    python mlp_baseline/src/run.py \
        --input data/embeddings/CIFAR_50k/dino_intermediate.pt \
        --model mlp_baseline/models/dino2clip_mlp.pt \
        --output data/embeddings/CIFAR_50k/clip_roundtrip.pt
"""

import os
import argparse
import torch
import torch.nn as nn
from pathlib import Path


def build_mlp_from_state_dict(state_dict):
    """
    Reconstruct MLP architecture exactly from a saved state_dict.
    
    This allows us to load models without knowing the architecture beforehand.
    Works by extracting layer dimensions from the weight matrices.
    """
    # Extract Linear layers in order
    weights = [(k, v) for k, v in state_dict.items() if k.endswith("weight")]
    
    layers = []
    for i, (name, W) in enumerate(weights):
        out_dim, in_dim = W.shape
        layers.append(nn.Linear(in_dim, out_dim))
        
        # Add ReLU except after last layer
        if i < len(weights) - 1:
            layers.append(nn.ReLU())
    
    model = nn.Sequential(*layers)
    model.load_state_dict(state_dict)
    return model


def load_embeddings(path):
    """Load embeddings from .pt file."""
    embeddings = torch.load(path, map_location='cpu')
    print(f"[✓] Loaded {path} | shape={embeddings.shape}")
    return embeddings


@torch.no_grad()
def generate_embeddings(
    input_path,
    model_path,
    output_path,
    device='cuda' if torch.cuda.is_available() else 'cpu',
):
    """
    Generate target embeddings using trained MLP model.
    
    Args:
        input_path: Path to source embeddings (.pt file)
        model_path: Path to trained model checkpoint (.pt file)
        output_path: Path to save generated embeddings (.pt file)
        device: 'cuda' or 'cpu'
    """
    # Load source embeddings
    X = load_embeddings(input_path).float().to(device)
    
    # Load model (architecture auto-reconstructed from state_dict)
    state = torch.load(model_path, map_location=device)
    model = build_mlp_from_state_dict(state).to(device)
    model.eval()
    print(f"[✓] Loaded model from {model_path}")
    
    # Generate embeddings
    print(f"[→] Generating embeddings...")
    Y = model(X).cpu()
    
    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(Y, output_path)
    
    print(f"[✓] Saved {output_path}")
    print(f"    Input shape:  {X.shape}")
    print(f"    Output shape: {Y.shape}")


def main():
    parser = argparse.ArgumentParser(
        description="Run trained MLP (no training)"
    )
    
    # Required arguments
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input embeddings (.pt file)"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pt file)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save output embeddings (.pt file)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--device",
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help="Device: 'cuda' or 'cpu' (default: auto-detect)"
    )
    
    args = parser.parse_args()
    
    generate_embeddings(
        input_path=args.input,
        model_path=args.model,
        output_path=args.output,
        device=args.device,
    )


if __name__ == "__main__":
    main()
