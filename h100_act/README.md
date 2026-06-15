# Train ACT on H100 — makermods_pick_book_parallel_grasp

Trains an [ACT](https://github.com/huggingface/lerobot) policy on
`srik410/makermods_pick_book_parallel_grasp_merged` on a single H100.

## Run it
```bash
# on the H100 box
export HF_TOKEN=hf_xxx            # needed to push the policy (and to pull if private)
bash setup.sh                    # one-time: deps + venv + install + login
cd lerobot && source .venv/bin/activate
bash ../h100_act/train.sh        # trains; auto-resumes if a checkpoint exists
```

## What the command does
`lerobot-train` with:
- `--policy.type=act` — ACT (auto-adapts to your dataset's cameras + motor dims).
- `--policy.device=cuda --policy.use_amp=true` — GPU + mixed precision (H100 loves bf16).
- `--batch_size=64 --num_workers=8` — uses the H100; ACT is small so 64 fits in 80GB.
- `--steps=100000` — ACT default (~a few hours on one H100).
- `--save_freq=10000` — checkpoint every 10k steps to `outputs/train/<run>/checkpoints/`.
- `--policy.push_to_hub=true --policy.repo_id=srik410/act_book_parallel_grasp` — pushes the
  trained policy to the Hub.
- `--wandb.enable=true` — live loss/metrics (run `wandb login` first, or set `--wandb.enable=false`).

## Knobs (edit the top of `train.sh`)
- **Out of memory?** lower `BATCH` (32, 16, 8).
- **GPU underused / slow?** the bottleneck is usually video decoding — raise `WORKERS`
  (12–16) before raising batch.
- **Longer/shorter training:** change `STEPS`.
- **Exactly reproduce ACT paper defaults:** set `BATCH=8` (original ACT batch size).

## After training — use the policy
Best checkpoint is `outputs/train/<run>/checkpoints/last/pretrained_model`. Evaluate / run on the
robot by pointing at it:
```bash
lerobot-record ... --policy.path=outputs/train/act_book_parallel_grasp/checkpoints/last/pretrained_model
# or from the Hub:
... --policy.path=srik410/act_book_parallel_grasp
```

## Multi-GPU (if your box has 2+ H100s)
```bash
accelerate launch --multi_gpu --num_processes=<N_GPUS> --mixed_precision=bf16 \
  $(which lerobot-train) \
  --dataset.repo_id=srik410/makermods_pick_book_parallel_grasp_merged \
  --policy.type=act --policy.device=cuda \
  --batch_size=64 --num_workers=8 --steps=100000 \
  --output_dir=outputs/train/act_book_parallel_grasp \
  --job_name=act_book_parallel_grasp \
  --policy.repo_id=srik410/act_book_parallel_grasp --wandb.enable=true
```
(With `--multi_gpu`, `batch_size` is **per-GPU**.)
