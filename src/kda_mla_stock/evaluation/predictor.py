from __future__ import annotations

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from kda_mla_stock.core.runtime import autocast_context


@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mixed_precision: str = "no",
    progress_description: str | None = None,
) -> tuple[float, pd.DataFrame]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    rows: list[pd.DataFrame] = []
    batches = (
        tqdm(
            loader,
            desc=progress_description,
            unit="batch",
            dynamic_ncols=True,
            leave=False,
        )
        if progress_description
        else loader
    )
    for batch in batches:
        features = batch["features"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        with autocast_context(device, mixed_precision):
            predictions = model(features)
            loss = F.mse_loss(predictions.float(), targets.float(), reduction="sum")
        batch_size = features.shape[0]
        total_loss += float(loss.item())
        total_samples += batch_size
        if progress_description:
            batches.set_postfix(loss=f"{total_loss / total_samples:.6f}", refresh=False)
        rows.append(
            pd.DataFrame(
                {
                    "date": pd.to_datetime(batch["date"].cpu().numpy()),
                    "symbol": list(batch["symbol"]),
                    "prediction": predictions.float().cpu().numpy().reshape(-1),
                    "target": targets.float().cpu().numpy().reshape(-1),
                }
            )
        )
    if total_samples == 0:
        raise ValueError("evaluation dataset has no samples")
    return total_loss / total_samples, pd.concat(rows, ignore_index=True)
