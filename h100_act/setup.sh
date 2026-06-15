#!/usr/bin/env bash
# One-time environment setup for training LeRobot on a fresh H100 box.
# Tested target: Ubuntu 22.04 + CUDA, Python 3.12.
set -euo pipefail

# ---- 0. system deps (ffmpeg is required for video-backed datasets) ----
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo apt-get install -y git git-lfs ffmpeg build-essential
fi
git lfs install || true

# ---- 1. get the code (skip if you already have the repo) ----
if [ ! -d lerobot ]; then
  git clone https://github.com/huggingface/lerobot.git
fi
cd lerobot

# ---- 2. python env ----
# LeRobot needs Python 3.12+. If your box has it, a venv is enough:
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# ---- 3. install lerobot (base install includes ACT + PyTorch) ----
# This pulls a CUDA-enabled torch on a CUDA box. ACT needs no extra extras.
pip install -e .

# ---- 4. sanity: confirm CUDA + entry point ----
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
which lerobot-train

# ---- 5. auth (needed to PULL the dataset if private + to PUSH the policy) ----
# export HF_TOKEN=hf_xxx   then:
huggingface-cli login --token "${HF_TOKEN:-}" --add-to-git-credential || \
  echo ">> Set HF_TOKEN and re-run 'huggingface-cli login' if dataset is private or you want to push."
# optional experiment tracking:
# wandb login   # or set WANDB_API_KEY

echo "Setup complete. Next: bash ../train.sh"
