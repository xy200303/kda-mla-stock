from __future__ import annotations

from pathlib import Path
from typing import Any

from kda_mla_stock.core.config import (
    EstimatorConfig,
    ModelConfiguration,
    TrainingConfig,
    load_model_config,
)
from kda_mla_stock.data.market import NormalizationStats
from kda_mla_stock.data.module import MarketDataModule, load_data_module
from kda_mla_stock.evaluation.backtests.qlib import QlibBacktestConfig
from kda_mla_stock.evaluation.evaluator import evaluate_and_write_predictions
from kda_mla_stock.orchestration.factories import (
    create_checkpoint_validator,
    create_trainer,
)


class TrainRunner:
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

    def run(self, data_module: MarketDataModule | None = None) -> dict[str, Any]:
        self.training_config.validate()
        print(f"loading and engineering features from {self.training_config.data_path}")
        data_module = data_module or load_data_module(self.training_config)
        print(
            f"dataset samples: {data_module.split_sizes}, "
            f"training stride={self.training_config.train_stride}"
        )
        trainer = create_trainer(
            self.model_config,
            self.training_config,
            data_module,
            resume_from=self.resume_from,
            requested_device=self.requested_device,
        )
        model = getattr(trainer, "model", None)
        if model is not None:
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            print(
                f"model: architecture={self.model_config.architecture}, "
                f"trainable parameters={parameter_count:,}"
            )
        return trainer.train()


class EvaluationRunner:
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
        data_module = load_data_module(
            self.training_config,
            normalization=self.normalization,
            data_path=data_path,
        )
        validator, model = create_checkpoint_validator(
            self.model_config,
            self.training_config,
            data_module,
            self.checkpoint_dir,
            split,
            requested_device=self.requested_device,
        )
        validation = validator.validate(model)
        destination = Path(output_dir or self.checkpoint_dir / f"evaluation_{split}")
        history_path = (
            None
            if isinstance(self.model_config, EstimatorConfig)
            else self.checkpoint_dir / "history.json"
        )
        return evaluate_and_write_predictions(
            validation.predictions,
            self.training_config,
            destination,
            split,
            loss=validation.loss,
            qlib_config=qlib_config,
            history_path=history_path,
            metadata={
                "architecture": self.model_config.architecture,
                **validation.metadata,
            },
        )


Trainer = TrainRunner
Valer = EvaluationRunner
Evaluator = EvaluationRunner

__all__ = ["EvaluationRunner", "Evaluator", "TrainRunner", "Trainer", "Valer"]
