from __future__ import annotations

import json
import math
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from tensorboardX import SummaryWriter
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from kda_mla_stock.configuration import ModelConfig, TrainingConfig
from kda_mla_stock.data import NormalizationStats
from kda_mla_stock.metrics import evaluate_predictions


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str | None = None) -> torch.device:
    if requested:
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _autocast_context(device: torch.device, precision: str):
    if precision == "no":
        return nullcontext()
    if device.type == "cuda":
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)
    if device.type == "cpu" and precision == "bf16":
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    return nullcontext()


def create_data_loader(
    dataset: Dataset,
    config: TrainingConfig,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )


def _state_dict_for_safetensors(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model.state_dict().items()
    }


def save_model(model: nn.Module, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_file(_state_dict_for_safetensors(model), str(destination))


def load_model(model: nn.Module, path: str | Path, device: torch.device) -> None:
    state = load_file(str(path), device=str(device))
    model.load_state_dict(state)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    return value


def write_json(payload: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mixed_precision: str = "no",
) -> tuple[float, pd.DataFrame]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    rows: list[pd.DataFrame] = []
    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        with _autocast_context(device, mixed_precision):
            predictions = model(features)
            loss = F.mse_loss(predictions.float(), targets.float(), reduction="sum")
        batch_size = features.shape[0]
        total_loss += float(loss.item())
        total_samples += batch_size
        dates = pd.to_datetime(batch["date"].cpu().numpy())
        rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "symbol": list(batch["symbol"]),
                    "prediction": predictions.float().cpu().numpy().reshape(-1),
                    "target": targets.float().cpu().numpy().reshape(-1),
                }
            )
        )
    if total_samples == 0:
        raise ValueError("evaluation dataset has no samples")
    return total_loss / total_samples, pd.concat(rows, ignore_index=True)


def train_model(
    model: nn.Module,
    datasets: dict[str, Dataset],
    training_config: TrainingConfig,
    model_config: ModelConfig,
    normalization_stats: NormalizationStats,
    resume_from: str | Path | None = None,
    requested_device: str | None = None,
) -> dict[str, Any]:
    training_config.validate()
    if len(datasets["train"]) == 0:
        raise ValueError("training dataset has no samples")
    if len(datasets["valid"]) == 0:
        raise ValueError("validation dataset has no samples")
    set_seed(training_config.seed)
    device = resolve_device(requested_device)
    model.to(device)
    train_loader = create_data_loader(datasets["train"], training_config, shuffle=True)
    valid_loader = create_data_loader(datasets["valid"], training_config, shuffle=False)
    optimizer = AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, training_config.epochs))
    use_scaler = device.type == "cuda" and training_config.mixed_precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    output_dir = Path(training_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_config.save_json(output_dir / "model_config.json")
    training_config.save_json(output_dir / "training_config.json")
    normalization_stats.save_json(output_dir / "normalization.json")

    start_epoch = 0
    best_validation_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    if resume_from is not None:
        resume_dir = Path(resume_from)
        if resume_dir.is_file():
            load_model(model, resume_dir, device)
        else:
            load_model(model, resume_dir / "last.safetensors", device)
            state_path = resume_dir / "last_state.pt"
            if state_path.exists():
                state = torch.load(state_path, map_location=device, weights_only=False)
                optimizer.load_state_dict(state["optimizer"])
                scheduler.load_state_dict(state["scheduler"])
                start_epoch = int(state["epoch"]) + 1
                best_validation_loss = float(state["best_validation_loss"])
                best_epoch = int(state["best_epoch"])
                epochs_without_improvement = int(state["epochs_without_improvement"])

    writer = SummaryWriter(logdir=str(output_dir / "tensorboard"))
    history_path = output_dir / "history.json"
    history = []
    if start_epoch > 0 and history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8")).get("history", [])
    for epoch in range(start_epoch, training_config.epochs):
        model.train()
        train_loss_sum = 0.0
        sample_count = 0
        for batch in train_loader:
            features = batch["features"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, training_config.mixed_precision):
                predictions = model(features)
                loss = F.mse_loss(predictions.float(), targets.float())
            if not torch.isfinite(loss):
                raise RuntimeError("training loss became non-finite")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += float(loss.item()) * features.shape[0]
            sample_count += features.shape[0]

        scheduler.step()
        train_loss = train_loss_sum / sample_count
        validation_loss, validation_predictions = predict_loader(
            model,
            valid_loader,
            device,
            training_config.mixed_precision,
        )
        validation_metrics = evaluate_predictions(validation_predictions)
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        history.append(epoch_record)
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/validation", validation_loss, epoch)
        writer.add_scalar("metrics/validation_rank_ic", validation_metrics["rank_ic_mean"], epoch)
        writer.add_scalar("optimizer/learning_rate", optimizer.param_groups[0]["lr"], epoch)
        writer.flush()
        print(
            f"epoch={epoch + 1}/{training_config.epochs} "
            f"train_loss={train_loss:.6f} valid_loss={validation_loss:.6f} "
            f"rank_ic={validation_metrics['rank_ic_mean']:.4f}"
        )

        improved = validation_loss < best_validation_loss
        if improved:
            best_validation_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            save_model(model, output_dir / "best.safetensors")
            validation_predictions.to_csv(
                output_dir / "best_validation_predictions.csv",
                index=False,
            )
        else:
            epochs_without_improvement += 1

        save_model(model, output_dir / "last.safetensors")
        torch.save(
            {
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_validation_loss": best_validation_loss,
                "best_epoch": best_epoch,
                "epochs_without_improvement": epochs_without_improvement,
            },
            output_dir / "last_state.pt",
        )
        write_json({"history": history}, history_path)
        if epochs_without_improvement >= training_config.patience:
            print(f"early stopping after {epoch + 1} epochs")
            break

    writer.close()
    result = {
        "device": str(device),
        "train_samples": len(datasets["train"]),
        "validation_samples": len(datasets["valid"]),
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "epochs_completed": len(history),
    }
    write_json(result, output_dir / "train_summary.json")
    return result
