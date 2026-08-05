from __future__ import annotations

import torch
from torch import nn

from kda_mla_stock.core.config import ModelConfig
from kda_mla_stock.models.common.sequence import (
    last_valid_state,
    sinusoidal_positions,
    validate_features,
)


class TransformerForecaster(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        if config.architecture != "transformer":
            raise ValueError("TransformerForecaster requires architecture=transformer")
        self.config = config
        self.input_projection = nn.Linear(config.num_features, config.hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.num_hidden_layers,
            norm=nn.LayerNorm(config.hidden_size),
            enable_nested_tensor=False,
        )
        self.head = nn.Linear(config.hidden_size, config.num_targets)

    def forward(
        self, features: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        validate_features(features, attention_mask, self.config.num_features)
        hidden_states = self.input_projection(features)
        hidden_states = hidden_states + sinusoidal_positions(hidden_states)
        sequence_length = features.shape[1]
        causal_mask = torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=features.device,
        ).triu(1)
        padding_mask = None if attention_mask is None else ~attention_mask.bool()
        hidden_states = self.encoder(
            hidden_states,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )
        return self.head(last_valid_state(hidden_states, attention_mask))
