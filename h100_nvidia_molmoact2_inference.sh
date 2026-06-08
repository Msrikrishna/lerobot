#!/usr/bin/env bash
# =============================================================================
# h100_nvidia_molmoact2_inference.sh
# -----------------------------------------------------------------------------
# Stand up the LeRobot async inference *policy server* from scratch on an
# NVIDIA Brev H100 box, using uv.
#
# The policy server is an empty container: which policy/checkpoint to run and
# all its parameters are sent by the RobotClient during the first handshake.
# So this script only installs deps and starts the server on --host/--port.
#
# Run ON the Brev H100 box (Ubuntu 22.04 + NVIDIA driver/CUDA preinstalled):
#   bash h100_nvidia_molmoact2_inference.sh
#
# Then expose the port from Brev and start the RobotClient on your robot host
# (see the command printed at the end).
# =============================================================================

set -euo pipefail

# ----------------------------- Configuration ---------------------------------
REPO_URL="${REPO_URL:-https://github.com/huggingface/lerobot.git}"
REPO_DIR="${REPO_DIR:-$HOME/lerobot}"
HOST="${HOST:-0.0.0.0}"     # bind all interfaces so a remote client can reach it
PORT="${PORT:-8080}"
# MolmoAct2 weights are downloaded by the server at handshake. Set HF_TOKEN if gated.
HF_TOKEN="${HF_TOKEN:-}"

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# ----------------------------- 0. GPU sanity ---------------------------------
log "Checking for NVIDIA GPU"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found — MolmoAct2 needs an NVIDIA GPU."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# ----------------------------- 1. System deps --------------------------------
log "Installing system packages (git, git-lfs, ffmpeg, curl)"
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends git git-lfs curl ca-certificates ffmpeg
git lfs install

# ----------------------------- 2. uv -----------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  log "Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

# ----------------------------- 3. Repo + env ---------------------------------
if [ ! -d "$REPO_DIR/.git" ]; then
  log "Cloning LeRobot into $REPO_DIR"
  git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

# `uv sync` builds the locked env. async = gRPC transport; molmoact2 = model deps
# (the server needs them installed to instantiate the policy at handshake).
log "Syncing environment (uv sync --extra molmoact2 --extra async)"
uv sync --locked --extra molmoact2 --extra async

log "Verifying CUDA is visible to PyTorch"
uv run python -c "import torch; assert torch.cuda.is_available(); print('CUDA OK:', torch.cuda.get_device_name(0))"

# ----------------------------- 4. Hugging Face auth (optional) ---------------
if [ -n "$HF_TOKEN" ]; then
  log "Logging in to Hugging Face Hub"
  uv run huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential || \
    echo "WARNING: HF login failed; gated/private repos may not download."
fi

# ----------------------------- 5. Start the policy server --------------------
# Set SETUP_ONLY=1 to stop here (env ready) without starting the blocking server,
# e.g. to run the HF->LeRobot checkpoint conversion first.
if [ "${SETUP_ONLY:-0}" = "1" ]; then
  log "SETUP_ONLY=1 — environment ready, not starting the server."
  exit 0
fi

log "Starting async policy server on ${HOST}:${PORT}"
exec uv run python -m lerobot.async_inference.policy_server \
  --host="$HOST" \
  --port="$PORT"
