from __future__ import annotations

import torch


def validate_features(
    features: torch.Tensor,
    attention_mask: torch.Tensor | None,
    num_features: int,
) -> None:
    if features.ndim != 3 or features.shape[-1] != num_features:
        raise ValueError(f"features must have shape [batch, sequence, {num_features}]")
    if attention_mask is not None and attention_mask.shape != features.shape[:2]:
        raise ValueError("attention_mask must match the first two feature dimensions")


def last_valid_state(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    if attention_mask is None:
        return hidden_states[:, -1]
    last_indices = attention_mask.long().sum(dim=1).sub(1).clamp_min(0)
    batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[batch_indices, last_indices]


def sinusoidal_positions(hidden_states: torch.Tensor) -> torch.Tensor:
    sequence_length, hidden_size = hidden_states.shape[1:]
    positions = torch.arange(sequence_length, device=hidden_states.device).float().unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, hidden_size, 2, device=hidden_states.device).float()
        * (-torch.log(torch.tensor(10_000.0, device=hidden_states.device)) / hidden_size)
    )
    encoding = hidden_states.new_zeros(sequence_length, hidden_size)
    encoding[:, 0::2] = torch.sin(positions * frequencies).to(hidden_states.dtype)
    encoding[:, 1::2] = torch.cos(positions * frequencies[: hidden_size // 2]).to(
        hidden_states.dtype
    )
    return encoding.unsqueeze(0)
