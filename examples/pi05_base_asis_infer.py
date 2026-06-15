"""Run pi0.5 BASE 'as is' on a single SO-101 frame (Mac, no robot / no sim).

This loads `lerobot/pi05_base` with NO finetuning and forces a frame from your
single-camera SO-101 dataset through it. To do that it must bridge the gap
between your data and what the base checkpoint expects:

  * Image keys: the base checkpoint was trained with its own camera keys
    (e.g. base_0_rgb / left_wrist_0_rgb / right_wrist_0_rgb). We discover those
    from `model.config.image_features` and copy your single `hand_cam` image
    into every one of them, resized to the model's expected resolution.
  * State width: pi0.5 pads state/action to max_state_dim/max_action_dim (32),
    so your 6-dim SO-101 state fits after padding. We pad/truncate to the
    checkpoint's expected state dimension.

⚠️  The predicted actions are NOT meaningful. The base model has never seen the
SO-101 embodiment, the joint ordering, or this camera. This script only proves
the checkpoint loads and the full pipeline runs end-to-end on your machine.
For useful actions you must finetune (see pi05 training command).

Run:  python examples/pi05_base_asis_infer.py
"""

import torch
import torch.nn.functional as F

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.pi05 import PI05Policy

MODEL_ID = "lerobot/pi05_base"
DATASET_ID = "srik410/makermods_pick_book_parallel_grasp_merged"
SRC_IMAGE_KEY = "observation.images.hand_cam"
STATE_KEY = "observation.state"
TASK = "pick up the book"
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

    model = PI05Policy.from_pretrained(MODEL_ID)
    model.to(device)
    model.eval()

    preprocess, postprocess = make_pre_post_processors(
        model.config,
        MODEL_ID,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    dataset = LeRobotDataset(DATASET_ID)
    sample = dataset[FRAME_INDEX]

    # --- Source image from your dataset: shape (C, H, W), float in [0, 1] ----
    src_img = sample[SRC_IMAGE_KEY]
    if src_img.ndim == 3:
        src_img = src_img.unsqueeze(0)  # -> (1, C, H, W) for interpolate

    # --- Build the frame the BASE checkpoint expects -------------------------
    frame: dict[str, object] = {}

    # Copy the single camera into every image key the model was trained on,
    # resized to the model's expected resolution.
    target_h, target_w = model.config.image_resolution
    resized = F.interpolate(src_img, size=(target_h, target_w), mode="bilinear", align_corners=False)
    expected_image_keys = list(model.config.image_features.keys())
    print(f"Checkpoint image keys: {expected_image_keys}")
    for key in expected_image_keys:
        frame[key] = resized.squeeze(0).clone()

    # State: pad/truncate your 6-dim SO-101 state to the checkpoint's width.
    state_ft = model.config.robot_state_feature
    if state_ft is not None:
        expected_state_dim = state_ft.shape[0]
        src_state = sample[STATE_KEY].flatten().float()
        if src_state.numel() < expected_state_dim:
            src_state = F.pad(src_state, (0, expected_state_dim - src_state.numel()))
        else:
            src_state = src_state[:expected_state_dim]
        # Use the checkpoint's own state key name.
        state_key = next(iter(k for k, ft in model.config.input_features.items() if ft is state_ft))
        frame[state_key] = src_state
        print(f"Checkpoint state key: {state_key} (dim {expected_state_dim})")

    frame["task"] = TASK

    model.reset()
    with torch.no_grad():
        obs = preprocess(frame)
        action = model.select_action(obs)
        action = postprocess(action)

    action = action.squeeze().cpu()
    print(f"Predicted action ({action.numel()} dims): {action.tolist()}")
    print("NOTE: values are not meaningful without finetuning on SO-101 data.")


if __name__ == "__main__":
    main()
