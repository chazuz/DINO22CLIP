"""
flow_matching/src/run.py

Run trained flow matching model on embeddings (inference only, no training).

This script handles both one-way and round-trip translations:
- One-way: Just specify input, model, output, target-dim
- Round-trip: Chain two calls (use output of first as input to second)

USAGE:
    # One-way CLIP → DINO
    python flow_matching/src/run.py \
        --input data/embeddings/CIFAR_50k/clip.pt \
        --model flow_matching/models/clip2dino_flow.pt \
        --output data/embeddings/CIFAR_50k/dino_by_clip_flow.pt \
        --target-dim 768

    # Round-trip: CLIP → DINO → CLIP
    # Step 1:
    python flow_matching/src/run.py \
        --input data/embeddings/CIFAR_50k/clip.pt \
        --model flow_matching/models/clip2dino_flow.pt \
        --output data/embeddings/CIFAR_50k/dino_intermediate.pt \
        --target-dim 768
    
    # Step 2:
    python flow_matching/src/run.py \
        --input data/embeddings/CIFAR_50k/dino_intermediate.pt \
        --model flow_matching/models/dino2clip_flow.pt \
        --output data/embeddings/CIFAR_50k/clip_roundtrip.pt \
        --target-dim 512
"""

import torch
import argparse
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from flow_matching.src.model import ConditionalFlowModel
from flow_matching.src.utils import FlowMatching


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
    target_dim,
    num_steps=50,
    device='cuda' if torch.cuda.is_available() else 'cpu',
):
    """
    Generate target embeddings using trained flow matching model.
    
    Args:
        input_path: Path to source embeddings (.pt file)
        model_path: Path to trained model checkpoint (.pt file)
        output_path: Path to save generated embeddings (.pt file)
        target_dim: Dimension of target embedding space
        num_steps: Number of ODE integration steps (default: 50)
        device: 'cuda' or 'cpu'
    """
    # Load source embeddings
    z_cond = load_embeddings(input_path).to(device)
    cond_dim = z_cond.shape[1]
    
    # Load model
    model = ConditionalFlowModel(
        target_dim=target_dim,
        cond_dim=cond_dim,
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"[✓] Loaded model from {model_path}")
    
    # Generate embeddings via ODE integration
    print(f"[→] Generating embeddings ({num_steps} integration steps)...")
    flow_matching = FlowMatching(num_steps=num_steps)
    samples = flow_matching.sample_loop(
        model,
        z_cond,
        shape=(z_cond.shape[0], target_dim),
    )
    
    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(samples.cpu(), output_path)
    
    print(f"[✓] Saved {output_path}")
    print(f"    Input shape:  {z_cond.shape}")
    print(f"    Output shape: {samples.shape}")


def main():
    parser = argparse.ArgumentParser(
        description="Run trained flow matching model (no training)"
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
    parser.add_argument(
        "--target-dim",
        type=int,
        required=True,
        help="Dimension of target embedding space (CLIP=512, DINO=768)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--num-steps",
        type=int,
        default=50,
        help="Number of ODE integration steps (default: 50)"
    )
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
        target_dim=args.target_dim,
        num_steps=args.num_steps,
        device=args.device,
    )


if __name__ == "__main__":
    main()
