"""Convert the original-HF MolmoAct2-SO100_101 checkpoint into a LeRobot-format
checkpoint that the async policy server can load via `from_pretrained`.

Run on the H100 (inside the lerobot repo, with the molmoact2 env):
    uv run python convert_molmoact2_to_lerobot.py
"""

import os

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy

OUT = os.environ.get("OUT", os.path.expanduser("~/molmoact2_so101_lerobot"))
HF_CHECKPOINT = os.environ.get("HF_CHECKPOINT", "allenai/MolmoAct2-SO100_101")
NORM_TAG = os.environ.get("NORM_TAG", "so100_so101_molmoact2")
# Single camera (you have one). Key must match what the robot_client streams.
IMAGE_KEYS = [k.strip() for k in os.environ.get("IMAGE_KEYS", "observation.images.top").split(",") if k.strip()]

# SO-101: 6-DoF joint state + action, RGB camera(s) at 480x640.
input_features = {k: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)) for k in IMAGE_KEYS}
input_features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(6,))
output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))}

cfg = MolmoAct2Config(
    checkpoint_path=HF_CHECKPOINT,
    norm_tag=NORM_TAG,
    inference_action_mode="continuous",
    image_keys=IMAGE_KEYS,
    input_features=input_features,
    output_features=output_features,
    device="cuda",
    model_dtype="bfloat16",
)

print(f"Loading {HF_CHECKPOINT} (norm_tag={NORM_TAG}, image_keys={IMAGE_KEYS}) ...")
policy = MolmoAct2Policy(cfg)

print(f"Saving LeRobot checkpoint -> {OUT}")
policy.save_pretrained(OUT)

pre, post = make_pre_post_processors(policy_cfg=cfg)
pre.save_pretrained(OUT)
post.save_pretrained(OUT)

print("DONE:", OUT)
print("Files:", sorted(os.listdir(OUT)))
