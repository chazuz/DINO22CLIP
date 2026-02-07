#!/usr/bin/env python3
"""
PCA comparison of round-trip embeddings for CIFAR subsets.

Layout:
  Row 1 (CLIP space):
    - clip (original)
    - clip_roundtrip_mlp
    - clip_roundtrip_flow
    - clip_roundtrip_diffusion
    - clip_roundtrip_diffusion_norm

  Row 2 (DINO space):
    - dino (original)
    - dino_roundtrip_mlp
    - dino_roundtrip_flow
    - dino_roundtrip_diffusion
    - dino_roundtrip_diffusion_norm

Usage:
    python src/eval/eval_pca_roundtrip.py --size 50k
    python src/eval/eval_pca_roundtrip.py --size 5k
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from matplotlib.patches import Patch
from scipy.spatial import procrustes
from sklearn.decomposition import PCA

# ----------------------------
# Configuration
# ----------------------------
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

COLORS = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
    '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe'
]

PCA_PARAMS = {
    "n_components": 2,
    "random_state": 42,
}

OUTPUT_DIR = Path("logs/eval")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Helper functions
# ----------------------------
def load_embedding(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Embedding not found: {path}")
    emb = torch.load(path, map_location="cpu")
    if isinstance(emb, dict):
        emb = emb.get("embeddings", emb.get("z", emb))
    return emb.numpy() if torch.is_tensor(emb) else emb

def load_labels(emb_dir: Path):
    path = emb_dir / "labels.pt"
    if not path.exists():
        raise FileNotFoundError(f"Labels not found: {path}")
    labels = torch.load(path, map_location="cpu")
    return labels.numpy() if torch.is_tensor(labels) else labels

def run_pca(embedding: np.ndarray):
    reducer = PCA(**PCA_PARAMS)
    return reducer.fit_transform(embedding)

def align_procrustes(reference: np.ndarray, target: np.ndarray):
    _, aligned, disparity = procrustes(reference, target)
    return aligned, disparity

def plot_pca_subplot(ax, coords, labels, title):
    for i, color in enumerate(COLORS):
        mask = labels == i
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=color,
            label=CIFAR10_CLASSES[i],
            s=3,
            alpha=0.6,
            edgecolors="none",
            rasterized=True,
        )
    ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

def create_legend(fig):
    legend_elements = [
        Patch(facecolor=COLORS[i], label=CIFAR10_CLASSES[i], alpha=0.6)
        for i in range(10)
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=5,
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, 0.01),
        columnspacing=1.5,
        handlelength=2.0,
    )

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--size",
        type=str,
        choices=["5k", "50k"],
        default="50k",
        help="Which CIFAR subset to visualize",
    )
    args = parser.parse_args()

    emb_dir = Path(f"data/embeddings/CIFAR_{args.size}")
    output_file_pdf = OUTPUT_DIR / f"pca_roundtrip_{args.size}.pdf"
    output_file_png = OUTPUT_DIR / f"pca_roundtrip_{args.size}.png"

    print(f"Loading labels from {emb_dir}...")
    labels = load_labels(emb_dir)
    print(f"✓ Loaded {len(labels)} labels")

    # ------------------------
    # Embedding files: originals + round-trips
    # ------------------------
    embedding_files = {
        # CLIP row
        "clip": "clip.pt",
        "clip_rt_mlp": "clip_roundtrip_mlp.pt",
        "clip_rt_flow": "clip_roundtrip_flow.pt",
        "clip_rt_diff": "clip_roundtrip_diffusion.pt",
        "clip_rt_diff_norm": "clip_roundtrip_diffusion_norm.pt",
        # DINO row
        "dino": "dino.pt",
        "dino_rt_mlp": "dino_roundtrip_mlp.pt",
        "dino_rt_flow": "dino_roundtrip_flow.pt",
        "dino_rt_diff": "dino_roundtrip_diffusion.pt",
        "dino_rt_diff_norm": "dino_roundtrip_diffusion_norm.pt",
    }

    embeddings = {}
    for key, filename in embedding_files.items():
        try:
            embeddings[key] = load_embedding(emb_dir / filename)
            print(
                f"✓ Loaded {key} ({embeddings[key].shape}, "
                f"std={embeddings[key].std():.2f})"
            )
        except FileNotFoundError:
            print(f"⚠ {key} not found")
            embeddings[key] = None

    # PCA
    pca_coords = {}
    print("\nRunning PCA...")
    for key, emb in embeddings.items():
        if emb is not None:
            pca_coords[key] = run_pca(emb)
            print(f"✓ {key} → 2D")
        else:
            pca_coords[key] = None

    # Align CLIP row to original CLIP
    row1_keys = [
        "clip",
        "clip_rt_mlp",
        "clip_rt_flow",
        "clip_rt_diff",
        "clip_rt_diff_norm",
    ]
    ref = pca_coords["clip"]
    if ref is not None:
        for key in row1_keys[1:]:
            if pca_coords[key] is not None:
                pca_coords[key], disp = align_procrustes(ref, pca_coords[key])
                print(f"Aligned {key} to CLIP (disparity: {disp:.4f})")

    # Align DINO row to original DINO
    row2_keys = [
        "dino",
        "dino_rt_mlp",
        "dino_rt_flow",
        "dino_rt_diff",
        "dino_rt_diff_norm",
    ]
    ref = pca_coords["dino"]
    if ref is not None:
        for key in row2_keys[1:]:
            if pca_coords[key] is not None:
                pca_coords[key], disp = align_procrustes(ref, pca_coords[key])
                print(f"Aligned {key} to DINO (disparity: {disp:.4f})")

    # Plot 2 x 5 grid
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    layout = [
        ("clip", "CLIP (Original)", 0, 0),
        ("clip_rt_mlp", "CLIP RT (MLP)", 0, 1),
        ("clip_rt_flow", "CLIP RT (Flow)", 0, 2),
        ("clip_rt_diff", "CLIP RT (Diffusion)", 0, 3),
        ("clip_rt_diff_norm", "CLIP RT (Diffusion, norm)", 0, 4),
        ("dino", "DINO (Original)", 1, 0),
        ("dino_rt_mlp", "DINO RT (MLP)", 1, 1),
        ("dino_rt_flow", "DINO RT (Flow)", 1, 2),
        ("dino_rt_diff", "DINO RT (Diffusion)", 1, 3),
        ("dino_rt_diff_norm", "DINO RT (Diffusion, norm)", 1, 4),
    ]

    for key, title, row, col in layout:
        ax = axes[row, col]
        if pca_coords[key] is not None:
            plot_pca_subplot(ax, pca_coords[key], labels, title)
        else:
            ax.text(
                0.5,
                0.5,
                f"{title}\n(not available)",
                ha="center",
                va="center",
                fontsize=9,
                color="gray",
            )
            ax.axis("off")

    create_legend(fig)
    plt.tight_layout(rect=[0, 0.06, 1, 1.0])
    plt.savefig(output_file_pdf, dpi=300, bbox_inches="tight")
    plt.savefig(output_file_png, dpi=150, bbox_inches="tight")

    print(f"\n✓ Saved PDF: {output_file_pdf}")
    print(f"✓ Saved PNG: {output_file_png}")


if __name__ == "__main__":
    main()

