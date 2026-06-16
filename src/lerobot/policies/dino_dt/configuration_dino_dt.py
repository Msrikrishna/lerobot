# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig


@PreTrainedConfig.register_subclass("dino_dt")
@dataclass
class DinoDTConfig(PreTrainedConfig):
    """Configuration for the DINO Diffusion Transformer policy.

    A Diffusion Policy whose vision backbone is a **frozen DINOv3 ViT** and whose
    denoiser is a **transformer that cross-attends to the DINOv3 patch tokens**
    (instead of the stock ResNet + conditional U-Net).

    I/O contract mirrors the stock `diffusion` policy:
        - "observation.state" is required as an input key.
        - At least one "observation.image*" key is required (cross-attention needs tokens).
        - "action" is required as an output key.

    Notes
    -----
    * VISUAL normalization is IDENTITY: images stay in [0, 1] and the DINOv3 encoder
      applies its own ImageNet normalization internally (same choice pi0 makes).
    * The DINOv3 backbone is gated on the HF Hub — run `huggingface-cli login` and
      accept the license once before first use.
    """

    # Input / output structure.
    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # The original implementation doesn't sample frames for the last few steps,
    # which avoids excessive padding and leads to improved training results.
    drop_n_last_frames: int = 7  # horizon - n_action_steps - n_obs_steps + 1

    # --- Vision backbone (frozen DINOv3 ViT) ---
    dino_model: str = "facebook/dinov3-vits16-pretrain-lvd1689m"
    image_size: int = 256  # multiple of 16 -> 16x16 = 256 patch tokens for /16 models
    freeze_backbone: bool = True

    # --- Diffusion Transformer denoiser ---
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 6
    dim_feedforward_scale: int = 4
    dropout: float = 0.1

    # --- Noise scheduler ---
    noise_scheduler_type: str = "DDIM"
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0

    # --- Inference ---
    num_inference_steps: int | None = 16

    # --- Loss computation ---
    do_mask_loss_for_padding: bool = False

    # --- Training presets ---
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    def __post_init__(self):
        super().__post_init__()

        supported_prediction_types = ["epsilon", "sample"]
        if self.prediction_type not in supported_prediction_types:
            raise ValueError(
                f"`prediction_type` must be one of {supported_prediction_types}. Got {self.prediction_type}."
            )
        supported_noise_schedulers = ["DDPM", "DDIM"]
        if self.noise_scheduler_type not in supported_noise_schedulers:
            raise ValueError(
                f"`noise_scheduler_type` must be one of {supported_noise_schedulers}. "
                f"Got {self.noise_scheduler_type}."
            )
        if self.n_action_steps > self.horizon:
            raise ValueError(
                f"`n_action_steps` ({self.n_action_steps}) cannot exceed `horizon` ({self.horizon})."
            )
        if self.n_action_steps > self.horizon - self.n_obs_steps + 1:
            raise ValueError(
                "The receding-horizon control scheme requires "
                "`n_action_steps <= horizon - n_obs_steps + 1`. "
                f"Got n_action_steps={self.n_action_steps}, horizon={self.horizon}, "
                f"n_obs_steps={self.n_obs_steps}."
            )
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"`d_model` ({self.d_model}) must be divisible by `n_heads` ({self.n_heads})."
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    def validate_features(self) -> None:
        if len(self.image_features) == 0:
            raise ValueError(
                "DinoDT requires at least one image input (the denoiser cross-attends to DINOv3 "
                "patch tokens). No `observation.image*` feature was found."
            )

        # Check that all input images have the same shape.
        first_image_key, first_image_ft = next(iter(self.image_features.items()))
        for key, image_ft in self.image_features.items():
            if image_ft.shape != first_image_ft.shape:
                raise ValueError(
                    f"`{key}` does not match `{first_image_key}`, but we expect all image shapes to match."
                )

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
