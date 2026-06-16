#!/usr/bin/env bash
#
# multi-gpu-dinodt-train.sh
# ─────────────────────────────────────────────────────────────────────────────
# One-shot runbook to train the DINOv3 diffusion-transformer policy (`dino_dt`)
# on MULTIPLE GPUs using LeRobot's native accelerate integration.
#
# This branch (dinov3-diffusion-model) vendors the custom `dino_dt` policy into
# upstream LeRobot 0.5.2, which already supports multi-GPU via HuggingFace
# accelerate. So you get near-linear speedup on a 4x/8x box for the (compute-
# bound) diffusion training.
#
# ── HOW TO GET THIS ONTO A FRESH GPU BOX ─────────────────────────────────────
#   # Option A: clone the branch, the script is in the repo root
#   git clone -b dinov3-diffusion-model https://github.com/Msrikrishna/lerobot.git
#   cd lerobot
#   bash multi-gpu-dinodt-train.sh
#
#   # Option B: grab just this script, it clones the repo for you
#   curl -fsSL https://raw.githubusercontent.com/Msrikrishna/lerobot/dinov3-diffusion-model/multi-gpu-dinodt-train.sh -o train.sh
#   bash train.sh
#
# Prereqs on the box: an NVIDIA GPU (run `nvidia-smi`), internet, and a HF token
# with the DINOv3 license accepted:
#   https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m  (click "Agree")
#
# COST WARNING: a GPU box bills by the minute. Delete it when done (`brev delete`).
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CONFIG — edit these, then run.                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# --- Repo / branch (where the dino_dt port lives) ---------------------------
REPO_URL="https://github.com/Msrikrishna/lerobot.git"
BRANCH="dinov3-diffusion-model"
WORKDIR="$HOME/lerobot"            # where the repo is cloned / found

# --- Data + output ----------------------------------------------------------
DATASET_REPO_ID="srik410/pushT_book"          # HF dataset to train on
POLICY_REPO_ID="srik410/dino_dt_pushT_book"   # HF repo to push the trained policy to (YOUR account)
OUTPUT_DIR="outputs/train/dino_dt_pushT_book" # local checkpoint dir

# --- Multi-GPU ---------------------------------------------------------------
NUM_GPUS=4                         # number of GPUs to shard across (e.g. 4 or 8)
MIXED_PRECISION="bf16"             # bf16 on H100/A100; use "fp16" on older cards

# --- Training hyperparameters ----------------------------------------------
BATCH_SIZE=32                      # PER-GPU batch. Effective batch = BATCH_SIZE * NUM_GPUS
STEPS=100000                       # total gradient steps
SAVE_FREQ=10000                    # checkpoint every N steps
NUM_WORKERS=8                      # dataloader workers PER process
IMAGE_SIZE=256                     # dino_dt image size, multiple of 16 (256 -> 16x16 tokens)
OPTIMIZER_LR="1e-4"                # base LR. With NUM_GPUS the effective batch grows —
                                   # consider scaling up (e.g. ~2e-4 for 4 GPUs). See note below.

# --- Smoke test -------------------------------------------------------------
SMOKE_TEST="false"                 # "true" = quick 20-step run to verify the box works
                                   # (overrides STEPS=20, no checkpointing, no push, no wandb).
                                   # Run this FIRST on a fresh box, then set back to "false".

# --- Misc -------------------------------------------------------------------
WANDB_ENABLE="true"                # "false" to skip Weights & Biases
JOB_NAME="dino_dt_pushT_book_${NUM_GPUS}gpu"
CONDA_ENV="lerobot"
HF_TOKEN=""                        # optional: paste an HF token for non-interactive login.
                                   # Leave empty to run `hf auth login` interactively instead.
PUSH_TO_HUB="true"                 # push the final policy to POLICY_REPO_ID

# ╚═══════════════════════════════════════════════════════════════════════════╝
#   You normally don't need to edit below this line.
# ─────────────────────────────────────────────────────────────────────────────

# Smoke-test mode: override to a fast, throwaway run that just proves the box,
# the data pipeline, and DDP all work end to end.
if [ "$SMOKE_TEST" = "true" ]; then
  STEPS=20
  SAVE_FREQ=1000000          # effectively never checkpoint
  PUSH_TO_HUB="false"
  WANDB_ENABLE="false"
  JOB_NAME="${JOB_NAME}_smoke"
  echo "==> SMOKE TEST mode: STEPS=$STEPS, no checkpoint/push/wandb"
fi

echo "==> dino_dt multi-GPU training"
echo "    dataset   : $DATASET_REPO_ID"
echo "    push to   : $POLICY_REPO_ID"
echo "    GPUs      : $NUM_GPUS x  (per-GPU batch $BATCH_SIZE -> effective $((BATCH_SIZE * NUM_GPUS)))"
echo "    steps     : $STEPS"
echo

# 1) Sanity: GPUs visible
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found — is this a GPU box?" >&2; exit 1
fi
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# 2) conda (install Miniforge if absent — avoids Anaconda ToS prompt)
if ! command -v conda >/dev/null 2>&1; then
  echo "==> Installing Miniforge"
  wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh
  bash /tmp/miniforge.sh -b -p "$HOME/miniforge3"
fi
source "$HOME/miniforge3/etc/profile.d/conda.sh"

# 3) Clone or update the repo+branch
if [ -d "$WORKDIR/.git" ]; then
  echo "==> Repo exists at $WORKDIR — fetching $BRANCH"
  git -C "$WORKDIR" fetch origin "$BRANCH"
  git -C "$WORKDIR" checkout "$BRANCH"
  git -C "$WORKDIR" pull --ff-only origin "$BRANCH"
else
  echo "==> Cloning $REPO_URL ($BRANCH) -> $WORKDIR"
  git clone -b "$BRANCH" "$REPO_URL" "$WORKDIR"
fi
cd "$WORKDIR"

# 4) Python env + deps
if ! conda env list | grep -qE "^\s*${CONDA_ENV}\s"; then
  echo "==> Creating conda env '$CONDA_ENV'"
  conda create -n "$CONDA_ENV" python=3.10 -y
fi
conda activate "$CONDA_ENV"

# torchcodec needs ffmpeg < 8 to decode the dataset's mp4 videos
conda install -y -c conda-forge "ffmpeg<8"
# install upstream lerobot WITH the training extra (pulls in accelerate + wandb)
pip install -e '.[training]'

# 5) HF auth (gated DINOv3 backbone + dataset access)
if [ -n "$HF_TOKEN" ]; then
  hf auth login --token "$HF_TOKEN"
else
  echo "==> Run 'hf auth login' if you haven't (needs DINOv3 license accepted)."
  hf auth whoami >/dev/null 2>&1 || hf auth login
fi

# 6) Verify the dino_dt policy is registered in this checkout
python -c "from lerobot.policies.factory import get_policy_class; print('dino_dt ->', get_policy_class('dino_dt').__name__)"

# 7) Launch multi-GPU training via accelerate
#    --multi_gpu + --num_processes=N shards the batch across N GPUs, syncs grads,
#    and logs/checkpoints only on the main process. The frozen DINOv3 backbone is
#    handled by LeRobot's find_unused_parameters=True DDP setting.
echo "==> Launching training on $NUM_GPUS GPUs"
accelerate launch \
  --multi_gpu \
  --num_processes="$NUM_GPUS" \
  --mixed_precision="$MIXED_PRECISION" \
  "$(which lerobot-train)" \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --policy.type=dino_dt \
  --policy.device=cuda \
  --policy.image_size="$IMAGE_SIZE" \
  --policy.optimizer_lr="$OPTIMIZER_LR" \
  --batch_size="$BATCH_SIZE" \
  --steps="$STEPS" \
  --save_freq="$SAVE_FREQ" \
  --num_workers="$NUM_WORKERS" \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$JOB_NAME" \
  --wandb.enable="$WANDB_ENABLE" \
  --policy.push_to_hub="$PUSH_TO_HUB" \
  --policy.repo_id="$POLICY_REPO_ID"

echo
echo "==> Done. Checkpoints in $WORKDIR/$OUTPUT_DIR/checkpoints/"
echo "    Policy pushed to: https://huggingface.co/$POLICY_REPO_ID"
echo "    REMEMBER to delete the GPU box to stop billing."

# ── Notes ────────────────────────────────────────────────────────────────────
# • Run inside tmux so it survives SSH drops:  tmux new -s train  (Ctrl-b d to detach)
# • LR scaling: LeRobot does NOT auto-scale LR for multi-GPU. Effective batch is
#   BATCH_SIZE*NUM_GPUS, so bump OPTIMIZER_LR up (linear or sqrt rule) if you see
#   slower convergence — e.g. 1e-4 -> 2e-4 for 4 GPUs.
# • First do a smoke test:  set SMOKE_TEST="true"  and run — it does a throwaway
#   20-step run (no checkpoint/push/wandb). Confirm loss prints on all ranks, then
#   set SMOKE_TEST="false" for the real run.
# • Resume an interrupted run: re-run with the same OUTPUT_DIR and add --resume=true.
