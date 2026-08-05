from __future__ import annotations

import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from kda_mla_stock.core.contracts import ValidationResult
from kda_mla_stock.evaluation.metrics import evaluate_predictions
from kda_mla_stock.evaluation.predictor import predict_loader


class TorchValidator:
    def __init__(
        self,
        loader: DataLoader,
        device: torch.device,
        mixed_precision: str,
        *,
        description: str = "Valid",
    ) -> None:
        self.loader = loader
        self.device = device
        self.mixed_precision = mixed_precision
        self.description = description

    def validate(self, model: nn.Module, epoch: int | None = None) -> ValidationResult:
        description = self.description if epoch is None else f"{self.description} {epoch + 1}"
        started_at = time.perf_counter()
        loss, predictions = predict_loader(
            model,
            self.loader,
            self.device,
            self.mixed_precision,
            progress_description=description,
        )
        return ValidationResult(
            loss=loss,
            predictions=predictions,
            metrics=evaluate_predictions(predictions),
            metadata={
                "device": str(self.device),
                "prediction_seconds": time.perf_counter() - started_at,
            },
        )
