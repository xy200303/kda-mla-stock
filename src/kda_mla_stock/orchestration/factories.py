from __future__ import annotations

from pathlib import Path
from typing import Any

from kda_mla_stock.core.artifacts import load_estimator, load_torch_model
from kda_mla_stock.core.config import (
    EstimatorConfig,
    ModelConfiguration,
    TrainingConfig,
)
from kda_mla_stock.core.runtime import configure_torch_runtime, resolve_device, set_seed
from kda_mla_stock.data.loader import create_data_loader
from kda_mla_stock.data.market import FEATURE_COLUMNS
from kda_mla_stock.data.module import MarketDataModule
from kda_mla_stock.models.registry import build_estimator, build_model
from kda_mla_stock.training.estimator_trainer import EstimatorTrainer
from kda_mla_stock.training.torch_trainer import TorchTrainer
from kda_mla_stock.validation.estimator_validator import EstimatorValidator
from kda_mla_stock.validation.torch_validator import TorchValidator


def create_trainer(
    model_config: ModelConfiguration,
    training_config: TrainingConfig,
    data_module: MarketDataModule,
    *,
    resume_from: str | Path | None = None,
    requested_device: str | None = None,
):
    if isinstance(model_config, EstimatorConfig):
        if resume_from is not None:
            raise ValueError("estimator models do not support checkpoint resume")
        train_data = data_module.tabular("train", model_config.aggregation_windows)
        valid_data = data_module.tabular("valid", model_config.aggregation_windows)
        estimator = build_estimator(model_config, training_config.seed)
        validator = EstimatorValidator(valid_data)
        return EstimatorTrainer(
            estimator,
            train_data,
            valid_data,
            training_config,
            model_config,
            data_module.normalization,
            validator,
        )

    if model_config.num_features != len(FEATURE_COLUMNS):
        raise ValueError(
            f"model expects {model_config.num_features} features, pipeline produces "
            f"{len(FEATURE_COLUMNS)}"
        )
    device = resolve_device(requested_device)
    configure_torch_runtime(training_config, device)
    set_seed(training_config.seed)
    model = build_model(model_config).to(device)
    valid_loader = create_data_loader(
        data_module.sequence("valid"), training_config, shuffle=False
    )
    validator = TorchValidator(valid_loader, device, training_config.mixed_precision)
    return TorchTrainer(
        model,
        data_module.splits,
        training_config,
        model_config,
        data_module.normalization,
        validator,
        device,
        resume_from=resume_from,
    )


def create_checkpoint_validator(
    model_config: ModelConfiguration,
    training_config: TrainingConfig,
    data_module: MarketDataModule,
    checkpoint_dir: str | Path,
    split: str,
    *,
    requested_device: str | None = None,
) -> tuple[Any, Any]:
    checkpoint = Path(checkpoint_dir)
    if isinstance(model_config, EstimatorConfig):
        dataset = data_module.tabular(split, model_config.aggregation_windows)
        return EstimatorValidator(dataset), load_estimator(checkpoint / "model.joblib")

    device = resolve_device(requested_device)
    configure_torch_runtime(training_config, device)
    model = build_model(model_config).to(device)
    load_torch_model(model, checkpoint / "best.safetensors", device)
    loader = create_data_loader(data_module.sequence(split), training_config, shuffle=False)
    validator = TorchValidator(
        loader,
        device,
        training_config.mixed_precision,
        description=f"Evaluate {split}",
    )
    return validator, model
