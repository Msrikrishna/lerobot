#!/usr/bin/env bash
# Train an ACT policy on srik410/makermods_pick_book_parallel_grasp_merged.
# Single H100 (80GB). Run from inside the lerobot repo with the venv active:
#   source .venv/bin/activate && bash /path/to/train.sh
set -euo pipefail

# ----------------------- config -----------------------
DATASET="srik410/makermods_pick_book_parallel_grasp_merged"
HF_USER="srik410"                      # where the trained policy is pushed
RUN="act_book_parallel_grasp"
OUT="outputs/train/${RUN}"

STEPS=100000                           # ACT default; ~a few hrs on one H100
BATCH=64                               # H100 80GB handles this for ACT easily
WORKERS=8                              # bump if video decode is the bottleneck
SAVE_FREQ=10000
LOG_FREQ=200
# ------------------------------------------------------

# Resume automatically if a checkpoint already exists in OUT.
RESUME_ARGS=()
if [ -d "${OUT}/checkpoints/last" ]; then
  echo ">> Found existing checkpoint -> resuming ${OUT}"
  RESUME_ARGS=(--config_path="${OUT}/checkpoints/last/pretrained_model/train_config.json" --resume=true)
fi

lerobot-train \
  --dataset.repo_id="${DATASET}" \
  --policy.type=act \
  --policy.device=cuda \
  --policy.use_amp=true \
  --batch_size="${BATCH}" \
  --num_workers="${WORKERS}" \
  --steps="${STEPS}" \
  --save_freq="${SAVE_FREQ}" \
  --log_freq="${LOG_FREQ}" \
  --output_dir="${OUT}" \
  --job_name="${RUN}" \
  --policy.push_to_hub=true \
  --policy.repo_id="${HF_USER}/${RUN}" \
  --wandb.enable=true \
  "${RESUME_ARGS[@]}"

echo "Done. Checkpoints in ${OUT}/checkpoints/  (last/ is the most recent)."
