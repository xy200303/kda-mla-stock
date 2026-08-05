from __future__ import annotations

import torch
from torch import nn

from kda_mla_stock.core.config import ModelConfig
from kda_mla_stock.models.common.sequence import last_valid_state, validate_features


class LSTMForecaster(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        if config.architecture != "lstm":
            raise ValueError("LSTMForecaster requires architecture=lstm")
        self.config = config
        self.recurrent = nn.LSTM(
            input_size=config.num_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_hidden_layers,
            batch_first=True,
            dropout=config.dropout if config.num_hidden_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, config.num_targets),
        )

    def forward(
        self, features: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        validate_features(features, attention_mask, self.config.num_features)
        hidden_states, _ = self.recurrent(features)
        return self.head(last_valid_state(hidden_states, attention_mask))
