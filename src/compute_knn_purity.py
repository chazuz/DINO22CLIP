#!/usr/bin/env python3
"""
compute_knn_purity.py

Compute kNN purity for CIFAR embeddings (DINO2CLIP project).
Handles all models: base, mlp, diffusion (DDPM), DDIM, flow.
Separates forward vs roundtrip embeddings and skips labels/invalid files.
Generates:
 - LaTeX table (knn_purity_table.tex)
 - PDF summary plot (knn_purity_summary.pdf)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
BASE_PATH = Path("data/embeddings/CIFAR_50k")
KS = [5, 10, 50, 100, 500]  # k values for kNN purity
OUTPUT_TEX = "knn_purity_table.tex"
OUTPUT_PDF = "knn_purity_summary.pdf"

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def compute_knn_purity(embeddings, labels, ks):
    """
    embeddings: numpy array (N_samples, dim)
    labels: numpy array (N_samples,)
    ks: list of k values
    Returns: dict {k: purity}
    """
    nbrs = NearestNeighbors(n_neighbors=max(ks)+1, algorithm='auto').fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)

    # Remove self from neighbors (first column)
    indices = indices[:, 1:]

    purities = {}
    for k in ks:
        topk = indices[:, :k]
        match = (labels[topk] == labels[:, None])
        purities[k] = match.mean()
    return purities

# -----------------------------
# LOAD EMBEDDINGS
# -----------------------------
embedding_files = sorted(BASE_PATH.glob("*.pt"))
embeddings_dict = {}

for f in embedding_files:
    name = f.stem
    try:
        emb = torch.load(f).numpy()
        if emb.ndim != 2:
            print(f"Skipping {name}: not 2D (shape={emb.shape})")
            continue
        embeddings_dict[name] = emb
    except Exception as e:
        print(f"Error loading {name}: {e}, skipping.")

# Load labels (assumes labels.pt exists)
labels_path = BASE_PATH / "labels.pt"
if not labels_path.exists():
    raise FileNotFoundError(f"Cannot find labels.pt in {BASE_PATH}")
labels = torch.load(labels_path).numpy()

# -----------------------------
# COMPUTE kNN PURITY
# -----------------------------
results = {}
for name, emb in embeddings_dict.items():
    purities = compute_knn_purity(emb, labels, KS)
    results[name] = purities
    print(f"{name}: {purities}")

# -----------------------------
# PLOT SUMMARY
# -----------------------------
plt.figure(figsize=(10, 6))
for name, purities in results.items():
    plt.plot(KS, [purities[k] for k in KS], marker='o', label=name)

plt.xlabel("k")
plt.ylabel("kNN Purity")
plt.title("kNN Purity across embeddings")
plt.xticks(KS)
plt.grid(True)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_PDF)
plt.close()
print(f"Saved summary plot to {OUTPUT_PDF}")

# -----------------------------
# SAVE LATEX TABLE
# -----------------------------
with open(OUTPUT_TEX, "w") as f:
    f.write("\\begin{tabular}{l" + "c"*len(KS) + "}\n")
    f.write("Embedding & " + " & ".join([str(k) for k in KS]) + " \\\\\n")
    f.write("\\hline\n")
    for name, purities in results.items():
        row = [f"{purities[k]:.3f}" for k in KS]
        f.write(f"{name} & " + " & ".join(row) + " \\\\\n")
    f.write("\\end{tabular}\n")
print(f"Saved LaTeX table to {OUTPUT_TEX}")

