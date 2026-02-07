#!/usr/bin/env python3
"""
PCA comparison of Diffusion / DDIM / Flow / MLP embeddings for CIFAR subsets.

Usage:
    python src/eval/eval_pca.py --size 5k
    python src/eval/eval_pca.py --size 50k
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from matplotlib.patches import Patch
from scipy.spatial import procrustes
from sklearn.decomposition import PCA  # new

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
    # Centering is handled by PCA internally
    reducer = PCA(**PCA_PARAMS)
    return reducer.fit_transform(embedding)

def align_procrustes(reference: np.ndarray, target: np.ndarray):
    _, aligned, disparity = procrustes(reference, target)
    return aligned, disparity

def plot_pca_subplot(ax, coords, labels, title, show_ylabel=False):
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
    ax.set_ylabel("" if not show_ylabel else "")
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
# Main function
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--size",
        type=str,
        choices=["5k", "50k"],
        default="5k",
        help="Which CIFAR subset to visualize",
    )
    args = parser.parse_args()

    emb_dir = Path(f"data/embeddings/CIFAR_{args.size}")
    output_file_pdf = OUTPUT_DIR / f"pca_all_{args.size}.pdf"
    output_file_png = OUTPUT_DIR / f"pca_all_{args.size}.png"

    print(f"Loading labels from {emb_dir}...")
    labels = load_labels(emb_dir)
    print(f"✓ Loaded {len(labels)} labels")

    # ------------------------
    # Embedding files to compare
    # ------------------------
    embedding_files = {
        "clip": "clip.pt",
        "dino": "dino.pt",
        "clip_by_dino_diffusion": "clip_by_dino_diffusion.pt",
        "clip_by_dino_ddim": "clip_by_dino_ddim.pt",
        "dino_by_clip_diffusion": "dino_by_clip_diffusion.pt",
        "dino_by_clip_ddim": "dino_by_clip_ddim.pt",
        "clip_by_dino_flow": "clip_by_dino_flow.pt",
        "clip_by_dino_mlp": "clip_by_dino_mlp.pt",
        "dino_by_clip_flow": "dino_by_clip_flow.pt",
        "dino_by_clip_mlp": "dino_by_clip_mlp.pt",
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

    # Run PCA
    pca_coords = {}
    print("\nRunning PCA...")
    for key, emb in embeddings.items():
        if emb is not None:
            pca_coords[key] = run_pca(emb)
            print(f"✓ {key} → 2D")
        else:
            pca_coords[key] = None

    # ------------------------
    # Align embeddings to their respective references
    # Row 1: CLIP reference, align DINO→CLIP embeddings
    # Row 2: DINO reference, align CLIP→DINO embeddings
    # ------------------------
    row1_keys = ["clip", "dino_by_clip_mlp", "dino_by_clip_diffusion",
                 "dino_by_clip_ddim", "dino_by_clip_flow"]
    row2_keys = ["dino", "clip_by_dino_mlp", "clip_by_dino_diffusion",
                 "clip_by_dino_ddim", "clip_by_dino_flow"]

    # Align row 1
    ref = pca_coords["clip"]
    if ref is not None:
        for key in row1_keys[1:]:
            if pca_coords[key] is not None:
                pca_coords[key], disp = align_procrustes(ref, pca_coords[key])
                print(f"Aligned {key} (disparity: {disp:.4f})")

    # Align row 2
    ref = pca_coords["dino"]
    if ref is not None:
        for key in row2_keys[1:]:
            if pca_coords[key] is not None:
                pca_coords[key], disp = align_procrustes(ref, pca_coords[key])
                print(f"Aligned {key} (disparity: {disp:.4f})")

    # ------------------------
    # Plot: 2 rows x 5 columns
    # ------------------------
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    layout = [
        ("clip", "CLIP (Original)", 0, 0),
        ("dino_by_clip_mlp", "DINO→CLIP (MLP)", 0, 1),
        ("dino_by_clip_diffusion", "DINO→CLIP (Diffusion)", 0, 2),
        ("dino_by_clip_ddim", "DINO→CLIP (DDIM)", 0, 3),
        ("dino_by_clip_flow", "DINO→CLIP (Flow)", 0, 4),
        ("dino", "DINO (Original)", 1, 0),
        ("clip_by_dino_mlp", "CLIP→DINO (MLP)", 1, 1),
        ("clip_by_dino_diffusion", "CLIP→DINO (Diffusion)", 1, 2),
        ("clip_by_dino_ddim", "CLIP→DINO (DDIM)", 1, 3),
        ("clip_by_dino_flow", "CLIP→DINO (Flow)", 1, 4),
    ]

    for key, title, row, col in layout:
        if pca_coords[key] is not None:
            plot_pca_subplot(
                axes[row, col],
                pca_coords[key],
                labels,
                title,
                show_ylabel=(col == 0),
            )
        else:
            axes[row, col].text(
                0.5,
                0.5,
                f"{title}\n(not available)",
                ha="center",
                va="center",
                fontsize=9,
                color="gray",
            )
            axes[row, col].axis("off")

    create_legend(fig)
    plt.tight_layout(rect=[0, 0.06, 1, 1.0])
    plt.savefig(output_file_pdf, dpi=300, bbox_inches="tight")
    plt.savefig(output_file_png, dpi=150, bbox_inches="tight")

    print(f"\n✓ Saved PDF: {output_file_pdf}")
    print(f"✓ Saved PNG: {output_file_png}")


if __name__ == "__main__":
    main()

