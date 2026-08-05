from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from kda_mla_stock.core.config import TrainingConfig


class TrainingDatasetView(Dataset):
    """Skip date and symbol metadata for optimizer-only batches."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        training_item = getattr(self.dataset, "training_item", None)
        if training_item is not None:
            return training_item(index)
        item = self.dataset[index]
        return {"features": item["features"], "target": item["target"]}


def create_data_loader(
    dataset: Dataset,
    config: TrainingConfig,
    *,
    shuffle: bool,
    training: bool = False,
) -> DataLoader:
    loader_dataset = TrainingDatasetView(dataset) if training else dataset
    loader_kwargs: dict[str, Any] = {}
    if config.num_workers > 0:
        loader_kwargs["prefetch_factor"] = config.prefetch_factor
        loader_kwargs["persistent_workers"] = True
    return DataLoader(
        loader_dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and torch.cuda.is_available(),
        **loader_kwargs,
    )
