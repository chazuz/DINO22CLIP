import torch
from torch.utils.data import DataLoader
from diffusion.src.model import ConditionalDenoiser
from diffusion.src.utils import Diffusion
from diffusion.configs.diffusion_config import EPOCHS, BATCH_SIZE, LEARNING_RATE, SEED
from src.embedding_dataset import EmbeddingDataset
from pathlib import Path

# -----------------------------
# Set device and seed
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
print("Using device:", DEVICE)

# -----------------------------
# Paths for data and models
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[2]  # project root
DATA_DIR = BASE_DIR / "data" / "embeddings"
MODEL_DIR = BASE_DIR / "diffusion" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Training function
# -----------------------------
def train(direction="dino2clip", dataset_name="CIFAR_50k"):
    data_path = DATA_DIR / dataset_name
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    dataset = EmbeddingDataset(data_path, direction)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = ConditionalDenoiser(target_dim=dataset.Y.shape[1], cond_dim=dataset.X.shape[1]).to(DEVICE)
    diffusion = Diffusion()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        total_loss = 0.0
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            t = torch.randint(0, diffusion.T, (y_batch.shape[0],), device=DEVICE)
            noise = torch.randn_like(y_batch)
            z_t = diffusion.q_sample(y_batch, t, noise)
            noise_pred = model(z_t, t, x_batch)
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y_batch.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{EPOCHS} | Avg loss: {avg_loss:.6f}")

        # Save checkpoint every 10 epochs and at last epoch
        if (epoch + 1) % 10 == 0 or (epoch + 1) == EPOCHS:
            ckpt_path = MODEL_DIR / f"{direction}_epoch{epoch+1}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

# -----------------------------
# Command line interface
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train diffusion model on embeddings")
    parser.add_argument("--direction", type=str, default="dino2clip",
                        choices=["dino2clip", "clip2dino"], help="Direction of mapping")
    parser.add_argument("--dataset", type=str, default="CIFAR_50k",
                        choices=["CIFAR_50k", "CIFAR_5k"], help="Dataset to use")
    args = parser.parse_args()

    train(args.direction, args.dataset)
