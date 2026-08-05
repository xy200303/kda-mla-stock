from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from kda_mla_stock.core.config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.normalized_shape = (hidden_size,)
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(hidden_states, self.normalized_shape, self.weight, self.eps)


class CausalDepthwiseConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            groups=channels,
            bias=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        sequence = hidden_states.transpose(1, 2)
        sequence = F.pad(sequence, (self.kernel_size - 1, 0))
        return self.conv(sequence).transpose(1, 2)


def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    first, second = hidden_states.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, rotary_dim: int, theta: float) -> None:
        super().__init__()
        frequencies = 1.0 / (
            theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
        )
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.register_buffer("cosine_cache", torch.empty(0), persistent=False)
        self.register_buffer("sine_cache", torch.empty(0), persistent=False)

    def _update_cache(self, sequence_length: int, reference: torch.Tensor) -> None:
        cache_matches = (
            self.cosine_cache.shape[0] >= sequence_length
            and self.cosine_cache.device == reference.device
            and self.cosine_cache.dtype == reference.dtype
        )
        if cache_matches:
            return
        positions = torch.arange(sequence_length, device=reference.device, dtype=torch.float32)
        angles = torch.outer(positions, self.frequencies.float())
        embeddings = torch.cat((angles, angles), dim=-1)
        self.cosine_cache = embeddings.cos().to(reference.dtype)
        self.sine_cache = embeddings.sin().to(reference.dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        sequence_length = hidden_states.shape[-2]
        self._update_cache(sequence_length, hidden_states)
        cosine = self.cosine_cache[:sequence_length].view(1, 1, sequence_length, -1)
        sine = self.sine_cache[:sequence_length].view(1, 1, sequence_length, -1)
        return hidden_states * cosine + _rotate_half(hidden_states) * sine


def upgrade_concatenated_parameter(
    state_dict: dict[str, torch.Tensor],
    prefix: str,
    new_name: str,
    legacy_names: tuple[str, ...],
) -> None:
    new_key = prefix + new_name
    legacy_keys = tuple(prefix + name for name in legacy_names)
    if new_key in state_dict or not all(key in state_dict for key in legacy_keys):
        return
    state_dict[new_key] = torch.cat([state_dict.pop(key) for key in legacy_keys], dim=0)


class GatedFeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_up_proj = nn.Linear(
            config.hidden_size,
            2 * config.intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        upgrade_concatenated_parameter(
            state_dict,
            prefix,
            "gate_up_proj.weight",
            ("gate_proj.weight", "up_proj.weight"),
        )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(hidden_states).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)
