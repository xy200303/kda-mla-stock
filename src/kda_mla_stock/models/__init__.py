from __future__ import annotations

import json
from pathlib import Path

from kda_mla_stock.configuration import ModelConfig
from kda_mla_stock.models.neural import StockForecaster, build_model, count_parameters
from kda_mla_stock.models.traditional import (
    SUPPORTED_TRADITIONAL_MODELS,
    TraditionalModelConfig,
    build_estimator,
    count_estimator_parameters,
    fit_estimator,
    load_estimator,
    save_estimator,
)

ModelConfiguration = ModelConfig | TraditionalModelConfig


def load_model_config(path: str | Path) -> ModelConfiguration:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("architecture") in SUPPORTED_TRADITIONAL_MODELS:
        return TraditionalModelConfig.from_dict(payload)
    return ModelConfig.from_dict(payload)


__all__ = [
    "ModelConfiguration",
    "StockForecaster",
    "TraditionalModelConfig",
    "build_estimator",
    "build_model",
    "count_estimator_parameters",
    "count_parameters",
    "fit_estimator",
    "load_estimator",
    "load_model_config",
    "save_estimator",
]
