from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from kda_mla_stock.configuration import ModelConfig, TrainingConfig
from kda_mla_stock.data import FEATURE_COLUMNS, NormalizationStats
from kda_mla_stock.datasets import MarketDatasetBundle, load_dataset_bundle
from kda_mla_stock.evaluation import evaluate_and_write_predictions
from kda_mla_stock.models import (
    ModelConfiguration,
    TraditionalModelConfig,
    build_estimator,
    build_model,
    count_estimator_parameters,
    count_parameters,
    fit_estimator,
    load_estimator,
    load_model_config,
    save_estimator,
)
from kda_mla_stock.qlib_evaluation import QlibBacktestConfig
from kda_mla_stock.training import (
    configure_torch_runtime,
    create_data_loader,
    load_model,
    predict_loader,
    resolve_device,
    set_seed,
    train_model,
    write_json,
)


class Trainer:
    def __init__(
        self,
        model_config: ModelConfiguration,
        training_config: TrainingConfig,
        *,
        resume_from: str | Path | None = None,
        requested_device: str | None = None,
    ) -> None:
        self.model_config = model_config
        self.training_config = training_config
        self.resume_from = resume_from
        self.requested_device = requested_device

    def run(self, bundle: MarketDatasetBundle | None = None) -> dict[str, Any]:
        self.training_config.validate()
        print(f"loading and engineering features from {self.training_config.data_path}")
        bundle = bundle or load_dataset_bundle(self.training_config)
        print(
            f"dataset samples: {bundle.split_sizes}, "
            f"training stride={self.training_config.train_stride}"
        )
        if isinstance(self.model_config, TraditionalModelConfig):
            if self.resume_from is not None:
                raise ValueError("traditional estimators do not support checkpoint resume")
            return self._train_traditional(bundle)
        return self._train_neural(bundle)

    def _train_neural(self, bundle: MarketDatasetBundle) -> dict[str, Any]:
        config = self.model_config
        if config.num_features != len(FEATURE_COLUMNS):
            raise ValueError(
                f"model expects {config.num_features} features, pipeline produces "
                f"{len(FEATURE_COLUMNS)}"
            )
        model = build_model(config)
        print(
            f"model: architecture={config.architecture}, "
            f"trainable parameters={count_parameters(model):,}"
        )
        return train_model(
            model,
            bundle.splits,
            self.training_config,
            config,
            bundle.normalization,
            resume_from=self.resume_from,
            requested_device=self.requested_device,
        )

    def _train_traditional(self, bundle: MarketDatasetBundle) -> dict[str, Any]:
        config = self.model_config
        set_seed(self.training_config.seed)
        output_dir = Path(self.training_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        config.save_json(output_dir / "model_config.json")
        self.training_config.save_json(output_dir / "training_config.json")
        bundle.normalization.save_json(output_dir / "normalization.json")

        train = bundle.tabular("train", config.aggregation_windows)
        valid = bundle.tabular("valid", config.aggregation_windows)
        print(
            f"traditional features: train={len(train.targets):,}, "
            f"valid={len(valid.targets):,}, dimensions={train.features.shape[1]}"
        )
        estimator = build_estimator(config, self.training_config.seed)
        started_at = time.perf_counter()
        fit_estimator(
            estimator,
            config,
            train.features,
            train.targets,
            valid.features,
            valid.targets,
        )
        fit_seconds = time.perf_counter() - started_at
        save_estimator(estimator, output_dir / "model.joblib")
        best_iteration = getattr(estimator, "best_iteration_", None)
        summary = {
            "architecture": config.architecture,
            "seed": self.training_config.seed,
            "trainable_parameters": count_estimator_parameters(estimator),
            "feature_count": train.features.shape[1],
            "train_samples": len(train.targets),
            "validation_samples": len(valid.targets),
            "fit_seconds": fit_seconds,
            "best_epoch": int(best_iteration) if best_iteration is not None else 0,
        }
        write_json(summary, output_dir / "train_summary.json")
        predictions = valid.predictions_frame(estimator.predict(valid.features))
        evaluate_and_write_predictions(
            predictions,
            self.training_config,
            output_dir / "evaluation_valid",
            "valid",
            metadata={"architecture": config.architecture, "fit_seconds": fit_seconds},
        )
        return summary


class Valer:
    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        requested_device: str | None = None,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_config = load_model_config(self.checkpoint_dir / "model_config.json")
        self.training_config = TrainingConfig.from_json(
            self.checkpoint_dir / "training_config.json"
        )
        self.normalization = NormalizationStats.from_json(
            self.checkpoint_dir / "normalization.json"
        )
        self.requested_device = requested_device

    def run(
        self,
        split: str = "test",
        *,
        data_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        qlib_config: QlibBacktestConfig | None = None,
    ) -> dict[str, Any]:
        if split not in {"valid", "test"}:
            raise ValueError("split must be valid or test")
        bundle = load_dataset_bundle(
            self.training_config,
            normalization=self.normalization,
            data_path=data_path,
        )
        if isinstance(self.model_config, TraditionalModelConfig):
            predictions, loss, metadata = self._predict_traditional(bundle, split)
            history_path = None
        else:
            predictions, loss, metadata = self._predict_neural(bundle, split)
            history_path = self.checkpoint_dir / "history.json"
        destination = Path(output_dir or self.checkpoint_dir / f"evaluation_{split}")
        return evaluate_and_write_predictions(
            predictions,
            self.training_config,
            destination,
            split,
            loss=loss,
            qlib_config=qlib_config,
            history_path=history_path,
            metadata=metadata,
        )

    def _predict_neural(
        self,
        bundle: MarketDatasetBundle,
        split: str,
    ) -> tuple[Any, float, dict[str, Any]]:
        config = self.model_config
        if not isinstance(config, ModelConfig):
            raise TypeError("neural prediction requires ModelConfig")
        device = resolve_device(self.requested_device)
        configure_torch_runtime(self.training_config, device)
        model = build_model(config).to(device)
        load_model(model, self.checkpoint_dir / "best.safetensors", device)
        loader = create_data_loader(bundle.splits[split], self.training_config, shuffle=False)
        started_at = time.perf_counter()
        loss, predictions = predict_loader(
            model,
            loader,
            device,
            self.training_config.mixed_precision,
            progress_description=f"Evaluate {split}",
        )
        prediction_seconds = time.perf_counter() - started_at
        return predictions, loss, {
            "device": str(device),
            "prediction_seconds": prediction_seconds,
        }

    def _predict_traditional(
        self,
        bundle: MarketDatasetBundle,
        split: str,
    ) -> tuple[Any, None, dict[str, Any]]:
        config = self.model_config
        if not isinstance(config, TraditionalModelConfig):
            raise TypeError("traditional prediction requires TraditionalModelConfig")
        tabular = bundle.tabular(split, config.aggregation_windows)
        estimator = load_estimator(self.checkpoint_dir / "model.joblib")
        started_at = time.perf_counter()
        predictions = tabular.predictions_frame(estimator.predict(tabular.features))
        prediction_seconds = time.perf_counter() - started_at
        return predictions, None, {
            "architecture": config.architecture,
            "prediction_seconds": prediction_seconds,
        }


Evaluator = Valer

__all__ = ["Evaluator", "Trainer", "Valer"]
