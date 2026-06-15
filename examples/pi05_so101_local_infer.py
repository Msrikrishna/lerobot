"""Local pi0.5 inference smoke-test on a Mac (no robot, no simulator).

Loads a pi0.5 checkpoint, pulls a real frame from a LeRobot dataset, runs the
full preprocess -> select_action -> postprocess pipeline, and prints the
predicted action vector. This verifies the checkpoint loads and produces a
correctly-shaped action on Apple Silicon (MPS) or CPU.

IMPORTANT about `MODEL_ID`:
  - Point it at YOUR finetuned checkpoint (e.g. "srik410/pi05_pick_book").
    Its expected camera/state keys match the dataset it was trained on, so the
    preprocess step lines up and this script runs end-to-end.
  - The raw "lerobot/pi05_base" checkpoint expects its own training-time camera
    keys (base_0_rgb, left/right_wrist_0_rgb), which will NOT match a single
    `observation.images.hand_cam` SO-101 dataset. Use base only after finetuning.

Run:  python examples/pi05_so101_local_infer.py
"""

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.pi05 import PI05Policy

# --- Config -----------------------------------------------------------------
MODEL_ID = "srik410/pi05_pick_book"  # your finetuned pi0.5 checkpoint
DATASET_ID = "srik410/makermods_pick_book_parallel_grasp_merged"
TASK = "pick up the book"  # natural-language instruction the policy is conditioned on
FRAME_INDEX = 0


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    device = pick_device()
    print(f"Using device: {device}")

    # 1. Load the policy.
    model = PI05Policy.from_pretrained(MODEL_ID)
    model.to(device)
    model.eval()

    # 2. Build the matching pre/post processor pipeline, forced onto our device.
    #    (Defaults to CUDA if available; the override lets it run on MPS/CPU.)
    preprocess, postprocess = make_pre_post_processors(
        model.config,
        MODEL_ID,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    # 3. Grab one real frame from the dataset to use as the observation.
    dataset = LeRobotDataset(DATASET_ID)
    sample = dataset[FRAME_INDEX]

    # A dataset sample already has observation.* keys; the policy also needs a
    # language task. Keep only what the policy consumes + the task string.
    frame = {k: v for k, v in sample.items() if k.startswith("observation")}
    frame["task"] = TASK

    # 4. Run inference. reset() clears the action-chunk queue between episodes.
    model.reset()
    with torch.no_grad():
        obs = preprocess(frame)
        action = model.select_action(obs)
        action = postprocess(action)

    action = action.squeeze().cpu()
    print(f"Predicted action ({action.numel()} dims): {action.tolist()}")


if __name__ == "__main__":
    main()
