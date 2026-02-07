#!/bin/bash
#SBATCH --job-name=$1                # Job name (first argument)
#SBATCH --output=logs/%x_%j.out      # Standard output (%x = job-name, %j = job-id)
#SBATCH --error=logs/%x_%j.err       # Standard error
#SBATCH --partition=NvidiaAll        # Change to GPU partition if needed
#SBATCH --time=01:00:00              # Max time
#SBATCH --chdir=/home/w/webere/big/DINO22CLIP  # Base project dir

# ========================================
# Activate virtual environment
# ========================================
echo "Activating virtual environment..."
source ./venv/bin/activate

export PYTHONPATH=$PWD:$PYTHONPATH 


# ========================================
# Ensure Python prints are line-buffered
# ========================================
python -u - <<EOF
import sys
sys.stdout.reconfigure(line_buffering=True)
EOF

# ========================================
# Run the Python script
# ========================================
# $2 = path to Python script relative to project root
# Additional args can be passed after the script path
echo "Running script: $2"
python -u $2 "${@:3}"

echo "Job finished at $(date)"

# ========================================
# USAGE EXAMPLES
# ========================================
# From base folder, run:
#
# Extract embeddings
# sbatch run_task.slurm extract_embeddings src/extract_embeddings.py
#
# Train MLP (CPU)
# sbatch run_task.slurm train_mlp src/mlp_baseline/src/train_mlp.py
#
# Train MLP (GPU, if available)
# sbatch run_task.slurm train_mlp src/mlp_baseline/src/train_mlp.py --device cuda
#
# Train diffusion model
# sbatch run_task.slurm train_diffusion src/diffusion/src/train_diffusion.py \
#     --source_path data/embeddings/z_dino.pt \
#     --target_path data/embeddings/z_clip.pt \
#     --model_name dino2clip
#
# Run full-dataset diffusion
# sbatch run_task.slurm diffuse_full src/diffusion/src/diffuse_full_dataset.py
#
# Train flow model
# sbatch run_task.slurm train_flow src/flow_matching/src/train_cond_flow.py --direction dino2clip --device cuda
#
# Note:
# - "${@:3}" forwards any additional arguments to your Python script
# - Use --device cuda if your script supports GPU
# - All paths are relative to the base folder
