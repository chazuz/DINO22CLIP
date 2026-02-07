"""
================================================================================
UNIFIED DIFFUSION GENERATION SCRIPT
================================================================================

This script replaces ALL separate diffusion generation scripts:
  ❌ diffusion_forward_oneway.py
  ❌ diffusion_forward_roundtrip.py  
  ❌ ddim_forward_oneway.py
  ❌ ddim_forward_roundtrip.py

One script handles everything because:
  1. DDPM vs DDIM = just different sampling methods (--sampler flag)
  2. One-way vs Round-trip = same process, just different input embeddings

================================================================================
BASIC USAGE
================================================================================

ONE-WAY TRANSLATION (CLIP → DINO):
    python diffusion/generate.py \
        --input data/embeddings/CIFAR_50k/clip.pt \
        --model diffusion/models/clip2dino.pt \
        --output data/embeddings/CIFAR_50k/dino_by_clip_diffusion.pt \
        --target-dim 768

ROUND-TRIP TRANSLATION (CLIP → DINO → CLIP):
    Step 1: CLIP → DINO
    python diffusion/generate.py \
        --input data/embeddings/CIFAR_50k/clip.pt \
        --model diffusion/models/clip2dino.pt \
        --output data/embeddings/CIFAR_50k/dino_intermediate.pt \
        --target-dim 768
    
    Step 2: DINO → CLIP (using intermediate embeddings from step 1)
    python diffusion/generate.py \
        --input data/embeddings/CIFAR_50k/dino_intermediate.pt \
        --model diffusion/models/dino2clip.pt \
        --output data/embeddings/CIFAR_50k/clip_roundtrip.pt \
        --target-dim 512

SWITCH BETWEEN DDPM AND DDIM:
    # DDPM (default, stochastic, uses all 1000 timesteps)
    --sampler ddpm
    
    # DDIM (deterministic, faster, typically 50 steps)
    --sampler ddim --ddim-steps 50

================================================================================
REQUIRED ARGUMENTS
================================================================================
  --input       Path to source embeddings (.pt file)
  --model       Path to trained diffusion model checkpoint (.pt file)  
  --output      Where to save generated embeddings (.pt file)
  --target-dim  Dimension of target embedding space (CLIP=512, DINO=768)

================================================================================
OPTIONAL ARGUMENTS
================================================================================
  --sampler       'ddpm' or 'ddim' (default: ddpm)
  --ddim-steps    Number of steps for DDIM (default: 50, ignored for ddpm)
  --ddim-eta      Stochasticity for DDIM: 0.0=deterministic, 1.0=like DDPM
  --batch-size    Batch size for generation (default: 256)

================================================================================
"""

import torch
import argparse
from pathlib import Path
from diffusion.src.model import ConditionalDenoiser
from diffusion.src.utils import Diffusion
from diffusion.configs.diffusion_config import DEVICE


def load_embeddings(path):
    """Load embeddings from .pt file."""
    embeddings = torch.load(path, map_location='cpu')
    print(f"[✓] Loaded {path} | shape={embeddings.shape}")
    return embeddings


def generate_embeddings(
    input_path,
    model_path,
    output_path,
    target_dim,
    sampler='ddpm',
    ddim_steps=50,
    ddim_eta=0.0,
    batch_size=256,
):
    """
    Generate target embeddings using trained diffusion model.
    
    Args:
        input_path: Path to source embeddings (.pt file)
        model_path: Path to trained model checkpoint (.pt file)
        output_path: Path to save generated embeddings (.pt file)
        target_dim: Dimension of target embeddings
        sampler: 'ddpm' or 'ddim'
        ddim_steps: Number of steps for DDIM sampling (ignored for DDPM)
        ddim_eta: Stochasticity parameter for DDIM (0.0 = deterministic)
        batch_size: Batch size for generation (for memory management)
    """
    # Load source embeddings
    z_cond = load_embeddings(input_path).to(DEVICE)
    cond_dim = z_cond.shape[1]
    
    # Initialize model
    model = ConditionalDenoiser(
        target_dim=target_dim,
        cond_dim=cond_dim,
    ).to(DEVICE)
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print(f"[✓] Loaded model from {model_path}")
    
    # Initialize diffusion
    diffusion = Diffusion()
    
    # Generate embeddings
    print(f"[→] Generating with {sampler.upper()} sampler...")
    with torch.no_grad():
        if sampler == 'ddpm':
            samples = diffusion.sample_loop(
                model,
                z_cond,
                shape=(z_cond.shape[0], target_dim),
            )
        elif sampler == 'ddim':
            samples = diffusion.ddim_sample_loop(
                model,
                z_cond,
                shape=(z_cond.shape[0], target_dim),
                ddim_steps=ddim_steps,
                eta=ddim_eta,
            )
        else:
            raise ValueError(f"Unknown sampler: {sampler}. Use 'ddpm' or 'ddim'.")
    
    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(samples.cpu(), output_path)
    print(f"[✓] Saved {output_path} | shape={samples.shape}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings using trained diffusion model"
    )
    
    # Required arguments
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input (source) embeddings (.pt file)"
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
        help="Path to save generated embeddings (.pt file)"
    )
    parser.add_argument(
        "--target-dim",
        type=int,
        required=True,
        help="Dimension of target embedding space"
    )
    
    # Optional arguments
    parser.add_argument(
        "--sampler",
        type=str,
        default="ddpm",
        choices=["ddpm", "ddim"],
        help="Sampling method: 'ddpm' (stochastic) or 'ddim' (deterministic)"
    )
    parser.add_argument(
        "--ddim-steps",
        type=int,
        default=50,
        help="Number of denoising steps for DDIM (default: 50, ignored for DDPM)"
    )
    parser.add_argument(
        "--ddim-eta",
        type=float,
        default=0.0,
        help="Stochasticity parameter for DDIM (0.0 = deterministic, 1.0 = DDPM)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for generation (for memory management)"
    )
    
    args = parser.parse_args()
    
    generate_embeddings(
        input_path=args.input,
        model_path=args.model,
        output_path=args.output,
        target_dim=args.target_dim,
        sampler=args.sampler,
        ddim_steps=args.ddim_steps,
        ddim_eta=args.ddim_eta,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
