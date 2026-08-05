from __future__ import annotations

import torch
from torch import nn

from kda_mla_stock.core.config import ModelConfig
from kda_mla_stock.models.kda_mla.common import GatedFeedForward, RMSNorm
from kda_mla_stock.models.kda_mla.kda import KimiDeltaAttention
from kda_mla_stock.models.kda_mla.mla import MultiHeadLatentAttention


class EncoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_index: int) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        if layer_index in config.kda_layers:
            self.attention: nn.Module = KimiDeltaAttention(config)
        else:
            self.attention = MultiHeadLatentAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.feed_forward = GatedFeedForward(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.dropout(
            self.attention(self.attention_norm(hidden_states), attention_mask)
        )
        hidden_states = hidden_states + self.dropout(
            self.feed_forward(self.ffn_norm(hidden_states))
        )
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        return hidden_states


class StockForecaster(nn.Module):
    """Causal KDA/MLA encoder for future-return prediction."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.input_projection = nn.Linear(config.num_features, config.hidden_size)
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.layers = nn.ModuleList(
            EncoderLayer(config, layer_index) for layer_index in range(config.num_hidden_layers)
        )
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, config.num_targets),
        )
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_uniform_(module.weight, nonlinearity="linear")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != self.config.num_features:
            raise ValueError(
                f"features must have shape [batch, sequence, {self.config.num_features}]"
            )
        if attention_mask is not None and attention_mask.shape != features.shape[:2]:
            raise ValueError("attention_mask must match the first two feature dimensions")
        hidden_states = self.input_norm(self.input_projection(features))
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        hidden_states = self.final_norm(hidden_states)

        if attention_mask is None:
            pooled = hidden_states[:, -1]
        else:
            last_indices = attention_mask.long().sum(dim=1).sub(1).clamp_min(0)
            batch_indices = torch.arange(features.shape[0], device=features.device)
            pooled = hidden_states[batch_indices, last_indices]
        return self.head(pooled)
