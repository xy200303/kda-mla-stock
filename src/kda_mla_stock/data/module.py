from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from kda_mla_stock.core.config import TrainingConfig
from kda_mla_stock.data.market import (
    NormalizationStats,
    fit_normalization_stats,
    load_and_engineer_market_data,
)
from kda_mla_stock.data.tabular import TabularDataset, build_tabular_dataset
from kda_mla_stock.data.window import StockWindowDataset, build_window_datasets


@dataclass
class MarketDataModule:
    frame: pd.DataFrame
    normalization: NormalizationStats
    splits: dict[str, StockWindowDataset]
    data_path: Path

    @property
    def split_sizes(self) -> dict[str, int]:
        return {name: len(dataset) for name, dataset in self.splits.items()}

    def sequence(self, split: str) -> StockWindowDataset:
        if split not in self.splits:
            raise ValueError(f"unknown dataset split: {split}")
        return self.splits[split]

    def tabular(self, split: str, aggregation_windows: list[int]) -> TabularDataset:
        if split not in self.splits:
            raise ValueError(f"unknown dataset split: {split}")
        return build_tabular_dataset(self.splits[split], aggregation_windows)


def load_data_module(
    config: TrainingConfig,
    *,
    normalization: NormalizationStats | None = None,
    data_path: str | Path | None = None,
) -> MarketDataModule:
    resolved_path = Path(data_path or config.data_path).expanduser()
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"data file not found: {resolved_path}. Run a data preparation script first."
        )
    frame = load_and_engineer_market_data(resolved_path, config.horizon)
    stats = normalization or fit_normalization_stats(frame, config.train_end)
    splits = build_window_datasets(
        frame,
        stats,
        config.sequence_length,
        config.train_end,
        config.valid_end,
        config.train_stride,
    )
    empty_splits = [name for name, dataset in splits.items() if len(dataset) == 0]
    if empty_splits:
        raise ValueError(f"dataset splits have no samples: {', '.join(empty_splits)}")
    return MarketDataModule(frame, stats, splits, resolved_path)


# Compatibility aliases for callers created before the data-module refactor.
MarketDatasetBundle = MarketDataModule
load_dataset_bundle = load_data_module
