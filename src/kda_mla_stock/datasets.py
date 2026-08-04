from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from kda_mla_stock.configuration import TrainingConfig
from kda_mla_stock.data import (
    NormalizationStats,
    StockWindowDataset,
    build_window_datasets,
    fit_normalization_stats,
    load_and_engineer_market_data,
)


@dataclass
class MarketDatasetBundle:
    frame: pd.DataFrame
    normalization: NormalizationStats
    splits: dict[str, StockWindowDataset]
    data_path: Path

    @property
    def split_sizes(self) -> dict[str, int]:
        return {name: len(dataset) for name, dataset in self.splits.items()}

    def tabular(self, split: str, aggregation_windows: list[int]) -> TabularDataset:
        if split not in self.splits:
            raise ValueError(f"unknown dataset split: {split}")
        return build_tabular_dataset(self.splits[split], aggregation_windows)


@dataclass
class TabularDataset:
    features: np.ndarray
    targets: np.ndarray
    dates: np.ndarray
    symbols: np.ndarray
    feature_names: list[str]

    def predictions_frame(self, predictions: np.ndarray) -> pd.DataFrame:
        values = np.asarray(predictions, dtype=np.float64).reshape(-1)
        if len(values) != len(self.targets):
            raise ValueError("prediction count does not match tabular dataset")
        return pd.DataFrame(
            {
                "date": pd.to_datetime(self.dates),
                "symbol": self.symbols,
                "prediction": values,
                "target": self.targets,
            }
        )


def load_dataset_bundle(
    config: TrainingConfig,
    *,
    normalization: NormalizationStats | None = None,
    data_path: str | Path | None = None,
) -> MarketDatasetBundle:
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
    return MarketDatasetBundle(frame, stats, splits, resolved_path)


def _feature_names(base_names: list[str], windows: list[int]) -> list[str]:
    names = [f"{name}_last" for name in base_names]
    for window in windows:
        names.extend(f"{name}_mean_{window}" for name in base_names)
        names.extend(f"{name}_std_{window}" for name in base_names)
    return names


def build_tabular_dataset(
    dataset: StockWindowDataset,
    aggregation_windows: list[int],
) -> TabularDataset:
    windows = sorted(window for window in aggregation_windows if window <= dataset.sequence_length)
    if not windows:
        raise ValueError("no aggregation window fits within sequence_length")
    feature_names = _feature_names(dataset.store.feature_names, windows)
    row_count = len(dataset.references)
    feature_count = len(feature_names)
    features = np.empty((row_count, feature_count), dtype=np.float32)
    targets = np.empty(row_count, dtype=np.float32)
    dates = np.empty(row_count, dtype="datetime64[ns]")
    symbols = np.empty(row_count, dtype=object)

    positions_by_symbol: dict[int, list[tuple[int, int]]] = {}
    for row_index, reference in enumerate(dataset.references):
        positions_by_symbol.setdefault(reference.symbol_index, []).append(
            (row_index, reference.anchor_index)
        )

    base_feature_count = len(dataset.store.feature_names)
    for symbol_index, positions in positions_by_symbol.items():
        row_indices = np.fromiter((row for row, _anchor in positions), dtype=np.int64)
        anchors = np.fromiter((anchor for _row, anchor in positions), dtype=np.int64)
        values = dataset.store.features[symbol_index]
        clean_values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        prefix = np.vstack(
            (
                np.zeros((1, base_feature_count), dtype=np.float64),
                np.cumsum(clean_values, axis=0, dtype=np.float64),
            )
        )
        squared_prefix = np.vstack(
            (
                np.zeros((1, base_feature_count), dtype=np.float64),
                np.cumsum(np.square(clean_values), axis=0, dtype=np.float64),
            )
        )
        features[row_indices, :base_feature_count] = values[anchors]
        offset = base_feature_count
        for window in windows:
            ends = anchors + 1
            starts = ends - window
            means = (prefix[ends] - prefix[starts]) / window
            second_moments = (squared_prefix[ends] - squared_prefix[starts]) / window
            standard_deviations = np.sqrt(np.maximum(second_moments - np.square(means), 0.0))
            features[row_indices, offset : offset + base_feature_count] = means
            offset += base_feature_count
            features[row_indices, offset : offset + base_feature_count] = standard_deviations
            offset += base_feature_count
        targets[row_indices] = dataset.store.targets[symbol_index][anchors]
        dates[row_indices] = dataset.store.dates[symbol_index][anchors]
        symbols[row_indices] = dataset.store.symbols[symbol_index]

    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ValueError("tabular feature extraction produced non-finite values")
    return TabularDataset(features, targets, dates, symbols, feature_names)
