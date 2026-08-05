from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from kda_mla_stock.data.market import NormalizationStats, apply_normalization


@dataclass(frozen=True)
class WindowReference:
    symbol_index: int
    anchor_index: int


class MarketWindowStore:
    def __init__(self, frame: pd.DataFrame, feature_names: list[str]) -> None:
        self.feature_names = feature_names
        self.symbols: list[str] = []
        self.features: list[np.ndarray] = []
        self.targets: list[np.ndarray] = []
        self.dates: list[np.ndarray] = []
        self.target_dates: list[np.ndarray] = []
        self._valid_anchor_cache: dict[tuple[int, int], np.ndarray] = {}

        for symbol, symbol_frame in frame.groupby("symbol", sort=True, observed=True):
            self.symbols.append(str(symbol))
            self.features.append(
                symbol_frame.loc[:, feature_names].to_numpy(dtype=np.float32, copy=True)
            )
            self.targets.append(symbol_frame["target"].to_numpy(dtype=np.float32, copy=True))
            self.dates.append(symbol_frame["date"].to_numpy(dtype="datetime64[ns]", copy=True))
            self.target_dates.append(
                symbol_frame["target_date"].to_numpy(dtype="datetime64[ns]", copy=True)
            )

    def _valid_window_anchors(self, symbol_index: int, sequence_length: int) -> np.ndarray:
        cache_key = (symbol_index, sequence_length)
        if cache_key in self._valid_anchor_cache:
            return self._valid_anchor_cache[cache_key]
        finite_rows = np.isfinite(self.features[symbol_index]).all(axis=1)
        invalid_count = np.concatenate(
            (np.array([0], dtype=np.int64), np.cumsum(~finite_rows, dtype=np.int64))
        )
        anchors = np.arange(len(finite_rows))
        starts = anchors - sequence_length + 1
        valid = starts >= 0
        eligible_anchors = anchors[valid]
        eligible_starts = starts[valid]
        valid[valid] = (
            invalid_count[eligible_anchors + 1] - invalid_count[eligible_starts]
        ) == 0
        self._valid_anchor_cache[cache_key] = valid
        return valid

    def references_for_split(
        self,
        split: Literal["train", "valid", "test"],
        sequence_length: int,
        train_end: str | pd.Timestamp,
        valid_end: str | pd.Timestamp,
        stride: int = 1,
    ) -> list[WindowReference]:
        if stride <= 0:
            raise ValueError("stride must be positive")
        train_cutoff = np.datetime64(pd.Timestamp(train_end), "ns")
        valid_cutoff = np.datetime64(pd.Timestamp(valid_end), "ns")
        references: list[WindowReference] = []
        for symbol_index in range(len(self.features)):
            dates = self.dates[symbol_index]
            target_dates = self.target_dates[symbol_index]
            targets = self.targets[symbol_index]
            eligible = self._valid_window_anchors(symbol_index, sequence_length)
            eligible = eligible & np.isfinite(targets) & ~np.isnat(target_dates)
            if split == "train":
                eligible = eligible & (dates <= train_cutoff) & (target_dates <= train_cutoff)
            elif split == "valid":
                eligible = eligible & (dates > train_cutoff) & (target_dates <= valid_cutoff)
            else:
                eligible = eligible & (dates > valid_cutoff)
            anchor_indices = np.flatnonzero(eligible)
            if split == "train" and stride > 1:
                anchor_indices = anchor_indices[::stride]
            references.extend(
                WindowReference(symbol_index, int(anchor_index))
                for anchor_index in anchor_indices
            )
        return references


class StockWindowDataset(Dataset[dict[str, torch.Tensor | str]]):
    def __init__(
        self,
        store: MarketWindowStore,
        references: list[WindowReference],
        sequence_length: int,
    ) -> None:
        self.store = store
        self.references = references
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return len(self.references)

    def _features_and_target(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        reference = self.references[index]
        anchor = reference.anchor_index
        start = anchor - self.sequence_length + 1
        features = self.store.features[reference.symbol_index][start : anchor + 1]
        target = self.store.targets[reference.symbol_index][anchor]
        return torch.from_numpy(features), torch.tensor([target], dtype=torch.float32)

    def training_item(self, index: int) -> dict[str, torch.Tensor]:
        features, target = self._features_and_target(index)
        return {"features": features, "target": target}

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        reference = self.references[index]
        features, target = self._features_and_target(index)
        anchor = reference.anchor_index
        date = self.store.dates[reference.symbol_index][anchor].astype(np.int64)
        return {
            "features": features,
            "target": target,
            "date": torch.tensor(date, dtype=torch.int64),
            "symbol": self.store.symbols[reference.symbol_index],
        }


def build_window_datasets(
    frame: pd.DataFrame,
    stats: NormalizationStats,
    sequence_length: int,
    train_end: str | pd.Timestamp,
    valid_end: str | pd.Timestamp,
    train_stride: int = 1,
) -> dict[str, StockWindowDataset]:
    normalized = apply_normalization(frame, stats)
    store = MarketWindowStore(normalized, stats.feature_names)
    datasets = {}
    for split in ("train", "valid", "test"):
        references = store.references_for_split(
            split,
            sequence_length,
            train_end,
            valid_end,
            train_stride if split == "train" else 1,
        )
        datasets[split] = StockWindowDataset(store, references, sequence_length)
    return datasets
