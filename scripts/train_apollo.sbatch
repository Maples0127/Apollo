#!/bin/bash
#SBATCH --job-name=apollo
#SBATCH --partition=RTX5880Ada
#SBATCH --gres=gpu:1
#SBATCH --nodelist=ailab02
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/data/cse12212308/exp/logs/%x_%j.out
#SBATCH --error=/data/cse12212308/exp/logs/%x_%j.err
#
# Usage: sbatch scripts/train_apollo.sbatch [config]
#        APOLLO_RESUME=1 sbatch --export=ALL,... scripts/train_apollo.sbatch config  # continue
#   default config: configs/apollo_smoke.yaml
#
# NOTE: pinned to ailab02 because datasets / HDF5 / exp dirs live on its
#       node-local /data disk (NFS /home is too slow for training I/O).

set -euo pipefail

CONF="${1:-configs/apollo_smoke.yaml}"
RESUME_FLAG=""
if [ "${APOLLO_RESUME:-0}" = "1" ]; then
  RESUME_FLAG="--resume"
fi
DATA_ROOT="${DATA_ROOT:-/data/cse12212308}"
EXP_ROOT="${EXP_ROOT:-$DATA_ROOT/exp}"

mkdir -p "$EXP_ROOT/logs" "$EXP_ROOT/wandb"

source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate look2hear

export WANDB_MODE=offline
export WANDB_DIR="$EXP_ROOT/wandb"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cd "$HOME/code/Apollo"

echo "== job $SLURM_JOB_ID on $(hostname) =="
echo "== config: $CONF =="
date
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count())"

python -u train.py --conf_dir="$CONF" $RESUME_FLAG
