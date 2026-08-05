from __future__ import annotations

import time
from typing import Any

import numpy as np

from kda_mla_stock.core.contracts import ValidationResult
from kda_mla_stock.data.tabular import TabularDataset
from kda_mla_stock.evaluation.metrics import evaluate_predictions


class EstimatorValidator:
    def __init__(self, dataset: TabularDataset) -> None:
        self.dataset = dataset

    def validate(self, model: Any, epoch: int | None = None) -> ValidationResult:
        del epoch
        started_at = time.perf_counter()
        predictions = self.dataset.predictions_frame(model.predict(self.dataset.features))
        prediction_seconds = time.perf_counter() - started_at
        loss = float(np.mean(np.square(predictions["prediction"] - predictions["target"])))
        return ValidationResult(
            loss=loss,
            predictions=predictions,
            metrics=evaluate_predictions(predictions),
            metadata={"prediction_seconds": prediction_seconds},
        )
