#!/usr/bin/env python

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
"""DINO Diffusion Transformer policy.

A Diffusion Policy whose perception is a **frozen DINOv3 ViT** and whose denoiser is a
**transformer decoder that cross-attends to the DINOv3 patch tokens**. Architecture ported
from the standalone `so101_dino_dp` project; the framework integration (queues, processors,
checkpointing) mirrors the stock `DiffusionPolicy`.
"""

import math
from collections import deque

import torch
import torch.nn.functional as F  # noqa: N812
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import Tensor, nn
from transformers import AutoModel

from lerobot.policies.dino_dt.configuration_dino_dt import DinoDTConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    populate_queues,
)
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DinoDTPolicy(PreTrainedPolicy):
    """DINO Diffusion Transformer policy (frozen DINOv3 + cross-attention diffusion denoiser)."""

    config_class = DinoDTConfig
    name = "dino_dt"

    def __init__(self, config: DinoDTConfig, **kwargs):
        # 0.5.2 make_policy passes dataset_stats (and possibly other kwargs) to the
        # constructor; normalization is handled by the processor pipeline, so we
        # accept and ignore them here, matching the stock policies' **kwargs.
        super().__init__(config)
        config.validate_features()
        self.config = config

        # queues hold the n latest observations and the planned action chunk during rollout
        self._queues = None

        self.model = DinoDiffusionTransformerModel(config)

        self.reset()

    def get_optim_params(self) -> dict:
        # Frozen backbone params have requires_grad=False; only the projection +
        # transformer + embeddings are returned to the optimizer.
        return [p for p in self.model.parameters() if p.requires_grad]

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`."""
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        # stack n latest observations from the queue -> (B, n_obs_steps, ...)
        batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        actions = self.model.generate_actions(batch)
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Select a single action given environment observations.

        Caches a history of observations and a generated action trajectory; executes
        `n_action_steps` actions before re-planning (receding horizon).
        """
        if ACTION in batch:
            batch.pop(ACTION)

        if self.config.image_features:
            batch = dict(batch)  # shallow copy so adding a key doesn't mutate the caller's dict
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        # NOTE: must happen after stacking the images into a single key.
        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        action = self._queues[ACTION].popleft()
        return action

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        """Run the batch through the model and compute the training loss."""
        if self.config.image_features:
            batch = dict(batch)  # shallow copy
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        loss = self.model.compute_loss(batch)
        # no auxiliary outputs to log
        return loss, None


def _make_noise_scheduler(name: str, **kwargs) -> DDPMScheduler | DDIMScheduler:
    if name == "DDPM":
        return DDPMScheduler(**kwargs)
    elif name == "DDIM":
        return DDIMScheduler(**kwargs)
    else:
        raise ValueError(f"Unsupported noise scheduler type {name}")


def _sinusoidal_embedding(timesteps: Tensor, dim: int) -> Tensor:
    """Standard diffusion timestep embedding. timesteps: [B] -> [B, dim]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=timesteps.device) / max(half - 1, 1)
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class DINOv3Encoder(nn.Module):
    """Frozen DINOv3 backbone -> small trainable linear projection to d_model.

    Returns the full token sequence (CLS + register + patch tokens) so the diffusion
    head can cross-attend over dense spatial features.
    """

    def __init__(self, model_id: str, d_model: int, image_size: int, freeze: bool = True):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_id)
        self.freeze = freeze
        if freeze:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad = False

        hidden = self.backbone.config.hidden_size
        self.proj = nn.Linear(hidden, d_model)  # trainable
        self.image_size = image_size
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        # keep the backbone in eval (frozen) even when the policy is in train mode
        super().train(mode)
        if self.freeze:
            self.backbone.eval()
        return self

    def _backbone_tokens(self, x: Tensor) -> Tensor:
        # x: [B, 3, H, W] floats in [0, 1]
        x = F.interpolate(
            x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
        )
        x = (x - self.mean) / self.std
        ctx = torch.no_grad() if self.freeze else torch.enable_grad()
        with ctx:
            out = self.backbone(pixel_values=x)
        return out.last_hidden_state  # [B, seq, hidden]

    def forward(self, x: Tensor) -> Tensor:
        tok = self._backbone_tokens(x)
        return self.proj(tok)  # [B, seq, d_model]  (grad flows through proj)


class DinoDiffusionTransformerModel(nn.Module):
    """Encodes observations into a memory token sequence and denoises an action chunk by
    cross-attending to that memory."""

    def __init__(self, config: DinoDTConfig):
        super().__init__()
        self.config = config

        d_model = config.d_model
        self.state_dim = config.robot_state_feature.shape[0]
        self.action_dim = config.action_feature.shape[0]
        self.num_cameras = len(config.image_features)

        # Vision: frozen DINOv3 -> trainable projection to d_model
        self.vision = DINOv3Encoder(config.dino_model, d_model, config.image_size, config.freeze_backbone)

        # Robot state -> token
        self.state_proj = nn.Sequential(
            nn.Linear(self.state_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )

        # Tag memory tokens by camera / obs step
        self.cam_embed = nn.Parameter(torch.randn(self.num_cameras, d_model) * 0.02)
        self.obs_step_embed = nn.Parameter(torch.randn(config.n_obs_steps, d_model) * 0.02)

        # Action chunk tokens
        self.action_in = nn.Linear(self.action_dim, d_model)
        self.action_pos = nn.Parameter(torch.randn(config.horizon, d_model) * 0.02)

        # Diffusion timestep conditioning
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )

        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward_scale * d_model,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=config.n_layers)
        self.action_out = nn.Linear(d_model, self.action_dim)

        self.noise_scheduler = _make_noise_scheduler(
            config.noise_scheduler_type,
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
        )

        if config.num_inference_steps is None:
            self.num_inference_steps = config.num_train_timesteps
        else:
            self.num_inference_steps = config.num_inference_steps

    def _encode_memory(self, batch: dict[str, Tensor]) -> Tensor:
        """Build the cross-attention memory: [B, M, d_model].

        Expects batch[OBS_STATE]  : (B, n_obs_steps, state_dim)
                batch[OBS_IMAGES] : (B, n_obs_steps, num_cameras, C, H, W)
        """
        state = batch[OBS_STATE]
        n_obs = state.shape[1]
        mem = []

        if self.config.image_features:
            imgs = batch[OBS_IMAGES]  # (B, n_obs, num_cam, C, H, W)
            for ci in range(self.num_cameras):
                for t in range(n_obs):
                    tok = self.vision(imgs[:, t, ci])  # (B, seq, d_model)
                    tok = tok + self.cam_embed[ci][None, None] + self.obs_step_embed[t][None, None]
                    mem.append(tok)

        for t in range(n_obs):
            s = self.state_proj(state[:, t])[:, None]  # (B, 1, d_model)
            s = s + self.obs_step_embed[t][None, None]
            mem.append(s)

        return torch.cat(mem, dim=1)  # (B, M, d_model)

    def _denoise(self, noisy_actions: Tensor, timesteps: Tensor, memory: Tensor) -> Tensor:
        """Predict noise. noisy_actions: [B, horizon, action_dim] -> same shape."""
        x = self.action_in(noisy_actions) + self.action_pos[None]
        temb = self.time_mlp(_sinusoidal_embedding(timesteps, self.config.d_model))  # [B, d]
        x = x + temb[:, None]
        out = self.decoder(tgt=x, memory=memory)
        return self.action_out(out)

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        memory = self._encode_memory(batch)

        trajectory = batch[ACTION]  # (B, horizon, action_dim)
        eps = torch.randn(trajectory.shape, device=trajectory.device)
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        noisy = self.noise_scheduler.add_noise(trajectory, eps, timesteps)

        pred = self._denoise(noisy, timesteps, memory)

        if self.config.prediction_type == "epsilon":
            target = eps
        elif self.config.prediction_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {self.config.prediction_type}")

        loss = F.mse_loss(pred, target, reduction="none")

        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(
                    "You need to provide 'action_is_pad' in the batch when "
                    f"{self.config.do_mask_loss_for_padding=}."
                )
            in_episode_bound = ~batch["action_is_pad"]
            loss = loss * in_episode_bound.unsqueeze(-1)

        return loss.mean()

    def generate_actions(self, batch: dict[str, Tensor]) -> Tensor:
        n_obs = batch[OBS_STATE].shape[1]
        assert n_obs == self.config.n_obs_steps
        batch_size = batch[OBS_STATE].shape[0]
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        memory = self._encode_memory(batch)

        sample = torch.randn(
            size=(batch_size, self.config.horizon, self.action_dim),
            dtype=dtype,
            device=device,
        )

        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            model_output = self._denoise(
                sample,
                torch.full(sample.shape[:1], t, dtype=torch.long, device=device),
                memory,
            )
            sample = self.noise_scheduler.step(model_output, t, sample).prev_sample

        # Extract `n_action_steps` worth of actions, starting from the current observation.
        start = n_obs - 1
        end = start + self.config.n_action_steps
        return sample[:, start:end]
