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
BASE_DIR = Path(__file__).resolve().parents[2]  # Project root
DATA_DIR = BASE_DIR / "data" / "embeddings"
MODEL_DIR = BASE_DIR / "flow_matching" / "models"
LOG_DIR = BASE_DIR / "flow_matching" / "logs"

# -------------------------------
# Flow matching hyperparameters
# -------------------------------
BATCH_SIZE = 256              # Same as diffusion
LEARNING_RATE = 1e-4          # Same as diffusion
EPOCHS = 50                   # Same as diffusion for fair comparison

# -------------------------------
# Flow matching specific
# -------------------------------
NUM_INTEGRATION_STEPS = 50    # For ODE solver during inference
                              # Flow needs fewer steps than diffusion (50 vs 1000)
                              # due to straight-line interpolation paths

SIGMA_MIN = 1e-5              # Small noise for numerical stability in ODE solver

