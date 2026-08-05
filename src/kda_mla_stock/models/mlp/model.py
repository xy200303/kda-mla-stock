from __future__ import annotations

import torch
from torch import nn

from kda_mla_stock.core.config import ModelConfig
from kda_mla_stock.models.common.sequence import last_valid_state, validate_features


class TemporalMLPForecaster(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        if config.architecture != "mlp":
            raise ValueError("TemporalMLPForecaster requires architecture=mlp")
        self.config = config
        layers: list[nn.Module] = []
        input_size = config.num_features * 3
        for layer_index in range(config.num_hidden_layers):
            layers.extend(
                [
                    nn.Linear(
                        input_size if layer_index == 0 else config.hidden_size,
                        config.hidden_size,
                    ),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                ]
            )
        layers.append(nn.Linear(config.hidden_size, config.num_targets))
        self.network = nn.Sequential(*layers)

    def forward(
        self, features: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        validate_features(features, attention_mask, self.config.num_features)
        last = last_valid_state(features, attention_mask)
        if attention_mask is None:
            mean = features.mean(dim=1)
            std = features.std(dim=1, unbiased=False)
        else:
            weights = attention_mask.unsqueeze(-1).to(features.dtype)
            count = weights.sum(dim=1).clamp_min(1.0)
            mean = (features * weights).sum(dim=1) / count
            variance = ((features - mean.unsqueeze(1)).square() * weights).sum(dim=1) / count
            std = variance.sqrt()
        return self.network(torch.cat((last, mean, std), dim=-1))
