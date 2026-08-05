from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import pandas as pd


@dataclass
class ValidationResult:
    loss: float | None
    predictions: pd.DataFrame
    metrics: dict[str, float | int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class Validator(Protocol):
    def validate(self, model: Any, epoch: int | None = None) -> ValidationResult: ...


class ModelTrainer(Protocol):
    def train(self) -> dict[str, Any]: ...
