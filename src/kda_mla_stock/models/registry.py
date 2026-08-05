from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from torch import nn

from kda_mla_stock.core.config import EstimatorConfig, ModelConfig
from kda_mla_stock.models.gru.model import GRUForecaster
from kda_mla_stock.models.hist_gbdt.model import build_estimator as build_hist_gbdt
from kda_mla_stock.models.kda_mla.model import StockForecaster
from kda_mla_stock.models.lightgbm.model import (
    build_estimator as build_lightgbm,
)
from kda_mla_stock.models.lightgbm.model import (
    fit_estimator as fit_lightgbm,
)
from kda_mla_stock.models.lstm.model import LSTMForecaster
from kda_mla_stock.models.mlp.model import TemporalMLPForecaster
from kda_mla_stock.models.random_forest.model import build_estimator as build_random_forest
from kda_mla_stock.models.ridge.model import build_estimator as build_ridge
from kda_mla_stock.models.transformer.model import TransformerForecaster

TorchBuilder = Callable[[ModelConfig], nn.Module]
EstimatorBuilder = Callable[[dict[str, Any], int], Any]
EstimatorFit = Callable[[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray], None]


@dataclass(frozen=True)
class ModelRegistration:
    backend: Literal["torch", "estimator"]
    artifact_name: str
    torch_builder: TorchBuilder | None = None
    estimator_builder: EstimatorBuilder | None = None
    estimator_fit: EstimatorFit | None = None


MODEL_REGISTRY: dict[str, ModelRegistration] = {
    "kda_mla": ModelRegistration("torch", "best.safetensors", torch_builder=StockForecaster),
    "lstm": ModelRegistration("torch", "best.safetensors", torch_builder=LSTMForecaster),
    "gru": ModelRegistration("torch", "best.safetensors", torch_builder=GRUForecaster),
    "transformer": ModelRegistration(
        "torch", "best.safetensors", torch_builder=TransformerForecaster
    ),
    "mlp": ModelRegistration("torch", "best.safetensors", torch_builder=TemporalMLPForecaster),
    "ridge": ModelRegistration("estimator", "model.joblib", estimator_builder=build_ridge),
    "random_forest": ModelRegistration(
        "estimator", "model.joblib", estimator_builder=build_random_forest
    ),
    "hist_gbdt": ModelRegistration(
        "estimator", "model.joblib", estimator_builder=build_hist_gbdt
    ),
    "lightgbm": ModelRegistration(
        "estimator",
        "model.joblib",
        estimator_builder=build_lightgbm,
        estimator_fit=fit_lightgbm,
    ),
}


def get_registration(architecture: str) -> ModelRegistration:
    try:
        return MODEL_REGISTRY[architecture]
    except KeyError as error:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"unsupported architecture {architecture!r}; expected one of {supported}"
        ) from error


def build_model(config: ModelConfig) -> nn.Module:
    registration = get_registration(config.architecture)
    if registration.backend != "torch" or registration.torch_builder is None:
        raise TypeError(f"{config.architecture} is not a torch model")
    return registration.torch_builder(config)


def build_estimator(config: EstimatorConfig, seed: int) -> Any:
    registration = get_registration(config.architecture)
    if registration.backend != "estimator" or registration.estimator_builder is None:
        raise TypeError(f"{config.architecture} is not an estimator model")
    return registration.estimator_builder(dict(config.params), seed)


def fit_estimator(
    estimator: Any,
    config: EstimatorConfig,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    valid_features: np.ndarray,
    valid_targets: np.ndarray,
) -> None:
    registration = get_registration(config.architecture)
    if registration.estimator_fit is not None:
        registration.estimator_fit(
            estimator,
            train_features,
            train_targets,
            valid_features,
            valid_targets,
        )
        return
    estimator.fit(train_features, train_targets)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
