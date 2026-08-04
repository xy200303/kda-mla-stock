from __future__ import annotations

import json
import math
import random
import time
import warnings
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
from tqdm.auto import tqdm

from kda_mla_stock.configuration import ModelConfig, TrainingConfig
from kda_mla_stock.data import NormalizationStats
from kda_mla_stock.metrics import evaluate_predictions


class _TrainingDatasetView(Dataset):
    """Avoid constructing date and symbol metadata for batches used only by the optimizer."""

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
    training: bool = False,
) -> DataLoader:
    loader_dataset = _TrainingDatasetView(dataset) if training else dataset
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


def configure_torch_runtime(config: TrainingConfig, device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
    torch.backends.cudnn.allow_tf32 = config.allow_tf32
    torch.set_float32_matmul_precision("high" if config.allow_tf32 else "highest")


def _create_optimizer(
    model: nn.Module,
    config: TrainingConfig,
    device: torch.device,
) -> tuple[AdamW, bool]:
    use_fused = config.fused_optimizer and device.type == "cuda"
    kwargs: dict[str, Any] = {
        "lr": config.learning_rate,
        "weight_decay": config.weight_decay,
    }
    if use_fused:
        kwargs["fused"] = True
    try:
        return AdamW(model.parameters(), **kwargs), use_fused
    except (RuntimeError, TypeError):
        kwargs.pop("fused", None)
        warnings.warn(
            "fused AdamW is unavailable; falling back to the standard optimizer",
            stacklevel=2,
        )
        return AdamW(model.parameters(), **kwargs), False


def _selection_value(
    config: TrainingConfig,
    validation_loss: float,
    validation_metrics: dict[str, float | int],
) -> float:
    if config.selection_metric == "validation_loss":
        return validation_loss
    return float(validation_metrics[config.selection_metric])


def _is_improved(value: float, best: float, mode: str) -> bool:
    if not math.isfinite(value):
        return False
    return value < best if mode == "min" else value > best


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
        with _autocast_context(device, mixed_precision):
            predictions = model(features)
            loss = F.mse_loss(predictions.float(), targets.float(), reduction="sum")
        batch_size = features.shape[0]
        total_loss += float(loss.item())
        total_samples += batch_size
        if progress_description:
            batches.set_postfix(loss=f"{total_loss / total_samples:.6f}", refresh=False)
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
    configure_torch_runtime(training_config, device)
    model.to(device)
    train_loader = create_data_loader(
        datasets["train"],
        training_config,
        shuffle=True,
        training=True,
    )
    valid_loader = create_data_loader(datasets["valid"], training_config, shuffle=False)
    optimizer, fused_optimizer = _create_optimizer(model, training_config, device)
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
    best_selection_score = (
        float("inf") if training_config.selection_mode == "min" else float("-inf")
    )
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
                if state.get("selection_metric") == training_config.selection_metric:
                    best_selection_score = float(
                        state.get("best_selection_score", best_validation_loss)
                    )

    execution_model: nn.Module = model
    if training_config.compile_mode != "none":
        if not hasattr(torch, "compile"):
            warnings.warn("torch.compile is unavailable in this PyTorch build", stacklevel=2)
        else:
            execution_model = torch.compile(model, mode=training_config.compile_mode)

    device_name = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
    )
    kda_modules = [
        module for module in model.modules() if module.__class__.__name__ == "KimiDeltaAttention"
    ]
    if kda_modules:
        use_fla = (
            device.type == "cuda"
            and training_config.mixed_precision in {"fp16", "bf16"}
            and all(bool(getattr(module, "fla_available", False)) for module in kda_modules)
            and all(module.config.attention_backend != "torch" for module in kda_modules)
        )
        print(f"KDA execution backend: {'fla-core' if use_fla else 'PyTorch reference'}")
    print(
        "runtime: "
        f"device={device_name}, batch_size={training_config.batch_size}, "
        f"workers={training_config.num_workers}, fused_adamw={fused_optimizer}, "
        f"tf32={training_config.allow_tf32 and device.type == 'cuda'}, "
        f"compile={training_config.compile_mode}"
    )

    writer = SummaryWriter(logdir=str(output_dir / "tensorboard"))
    history_path = output_dir / "history.json"
    history = []
    if start_epoch > 0 and history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8")).get("history", [])
    for epoch in range(start_epoch, training_config.epochs):
        execution_model.train()
        train_loss_sum = 0.0
        sample_count = 0
        epoch_started_at = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        with tqdm(
            train_loader,
            desc=f"Train {epoch + 1}/{training_config.epochs}",
            unit="batch",
            dynamic_ncols=True,
        ) as progress:
            for batch_index, batch in enumerate(progress, start=1):
                features = batch["features"].to(device, non_blocking=True)
                targets = batch["target"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with _autocast_context(device, training_config.mixed_precision):
                    predictions = execution_model(features)
                    loss = F.mse_loss(predictions.float(), targets.float())
                if not torch.isfinite(loss):
                    raise RuntimeError("training loss became non-finite")
                if batch_index == 1:
                    progress.set_postfix(stage="backward", refresh=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                batch_loss = float(loss.item())
                train_loss_sum += batch_loss * features.shape[0]
                sample_count += features.shape[0]
                progress.set_postfix(
                    loss=f"{batch_loss:.6f}",
                    average=f"{train_loss_sum / sample_count:.6f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    refresh=False,
                )

        training_elapsed_seconds = time.perf_counter() - epoch_started_at
        scheduler.step()
        train_loss = train_loss_sum / sample_count
        validation_loss, validation_predictions = predict_loader(
            execution_model,
            valid_loader,
            device,
            training_config.mixed_precision,
            progress_description=f"Valid {epoch + 1}/{training_config.epochs}",
        )
        validation_metrics = evaluate_predictions(validation_predictions)
        elapsed_seconds = time.perf_counter() - epoch_started_at
        samples_per_second = sample_count / training_elapsed_seconds
        peak_memory_gib = (
            torch.cuda.max_memory_allocated(device) / (1024**3)
            if device.type == "cuda"
            else 0.0
        )
        selection_score = _selection_value(
            training_config,
            validation_loss,
            validation_metrics,
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": elapsed_seconds,
            "training_elapsed_seconds": training_elapsed_seconds,
            "samples_per_second": samples_per_second,
            "peak_memory_gib": peak_memory_gib,
            "selection_score": selection_score,
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        history.append(epoch_record)
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/validation", validation_loss, epoch)
        writer.add_scalar("metrics/validation_rank_ic", validation_metrics["rank_ic_mean"], epoch)
        writer.add_scalar("optimizer/learning_rate", optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("performance/samples_per_second", samples_per_second, epoch)
        if device.type == "cuda":
            writer.add_scalar("performance/peak_memory_gib", peak_memory_gib, epoch)
        writer.flush()
        print(
            f"epoch={epoch + 1}/{training_config.epochs} "
            f"train_loss={train_loss:.6f} valid_loss={validation_loss:.6f} "
            f"rank_ic={validation_metrics['rank_ic_mean']:.4f} "
            f"throughput={samples_per_second:.1f} samples/s "
            f"peak_memory={peak_memory_gib:.2f} GiB"
        )

        best_validation_loss = min(best_validation_loss, validation_loss)
        improved = _is_improved(
            selection_score,
            best_selection_score,
            training_config.selection_mode,
        )
        if improved:
            best_selection_score = selection_score
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
                "best_selection_score": best_selection_score,
                "selection_metric": training_config.selection_metric,
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
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train_samples": len(datasets["train"]),
        "validation_samples": len(datasets["valid"]),
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "selection_metric": training_config.selection_metric,
        "best_selection_score": best_selection_score,
        "epochs_completed": len(history),
    }
    write_json(result, output_dir / "train_summary.json")
    return result
