from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from kda_mla_stock.core.artifacts import save_estimator, write_json
from kda_mla_stock.core.config import EstimatorConfig, TrainingConfig
from kda_mla_stock.core.runtime import set_seed
from kda_mla_stock.data.market import NormalizationStats
from kda_mla_stock.data.tabular import TabularDataset
from kda_mla_stock.models.common.estimator import count_estimator_parameters
from kda_mla_stock.models.registry import fit_estimator
from kda_mla_stock.validation.estimator_validator import EstimatorValidator


class EstimatorTrainer:
    def __init__(
        self,
        estimator: Any,
        train_data: TabularDataset,
        valid_data: TabularDataset,
        training_config: TrainingConfig,
        model_config: EstimatorConfig,
        normalization_stats: NormalizationStats,
        validator: EstimatorValidator,
    ) -> None:
        self.estimator = estimator
        self.train_data = train_data
        self.valid_data = valid_data
        self.training_config = training_config
        self.model_config = model_config
        self.normalization_stats = normalization_stats
        self.validator = validator

    def train(self) -> dict[str, Any]:
        set_seed(self.training_config.seed)
        output_dir = Path(self.training_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model_config.save_json(output_dir / "model_config.json")
        self.training_config.save_json(output_dir / "training_config.json")
        self.normalization_stats.save_json(output_dir / "normalization.json")

        print(
            f"estimator features: train={len(self.train_data.targets):,}, "
            f"valid={len(self.valid_data.targets):,}, "
            f"dimensions={self.train_data.features.shape[1]}"
        )
        started_at = time.perf_counter()
        fit_estimator(
            self.estimator,
            self.model_config,
            self.train_data.features,
            self.train_data.targets,
            self.valid_data.features,
            self.valid_data.targets,
        )
        fit_seconds = time.perf_counter() - started_at
        validation = self.validator.validate(self.estimator)
        save_estimator(self.estimator, output_dir / "model.joblib")
        validation.predictions.to_csv(
            output_dir / "best_validation_predictions.csv", index=False
        )
        best_iteration = getattr(self.estimator, "best_iteration_", None)
        summary = {
            "architecture": self.model_config.architecture,
            "seed": self.training_config.seed,
            "trainable_parameters": count_estimator_parameters(self.estimator),
            "feature_count": self.train_data.features.shape[1],
            "train_samples": len(self.train_data.targets),
            "validation_samples": len(self.valid_data.targets),
            "fit_seconds": fit_seconds,
            "best_epoch": int(best_iteration) if best_iteration is not None else 0,
            "best_validation_loss": validation.loss,
            "best_validation_rank_ic": validation.metrics["rank_ic_mean"],
        }
        write_json(summary, output_dir / "train_summary.json")
        return summary
