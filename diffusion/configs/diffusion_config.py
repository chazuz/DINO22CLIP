import torch
from pathlib import Path

# -------------------------------
# Device and seed
# -------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# -------------------------------
# Project paths
# -------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]  # ~/big/DINO22CLIP
DATA_DIR = BASE_DIR / "data" / "embeddings"
MODEL_DIR = BASE_DIR / "diffusion/models"
LOG_DIR = BASE_DIR / "diffusion/logs"

# -------------------------------
# Diffusion hyperparameters
# -------------------------------
TIMESTEPS = 1000
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
EPOCHS = 50

# -------------------------------
# Optional: beta schedule
# -------------------------------
BETA_START = 1e-4
BETA_END = 0.02
