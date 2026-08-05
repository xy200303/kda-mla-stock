from __future__ import annotations

import json
import math
import time
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from kda_mla_stock.core.artifacts import (
    load_torch_model,
    save_torch_model,
    write_json,
)
from kda_mla_stock.core.config import ModelConfig, TrainingConfig
from kda_mla_stock.core.runtime import (
    autocast_context,
    configure_torch_runtime,
    resolve_device,
    set_seed,
)
from kda_mla_stock.data.loader import create_data_loader
from kda_mla_stock.data.market import NormalizationStats
from kda_mla_stock.validation.torch_validator import TorchValidator


def _assert_finite_loss(loss: torch.Tensor) -> None:
    finite = torch.isfinite(loss.detach())
    if loss.device.type == "cuda" and hasattr(torch, "_assert_async"):
        torch._assert_async(finite, "training loss became non-finite")
    elif not bool(finite):
        raise RuntimeError("training loss became non-finite")


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


class TorchTrainer:
    def __init__(
        self,
        model: nn.Module,
        datasets: dict[str, Dataset],
        training_config: TrainingConfig,
        model_config: ModelConfig,
        normalization_stats: NormalizationStats,
        validator: TorchValidator,
        device: torch.device,
        *,
        resume_from: str | Path | None = None,
    ) -> None:
        self.model = model
        self.datasets = datasets
        self.config = training_config
        self.model_config = model_config
        self.normalization_stats = normalization_stats
        self.validator = validator
        self.device = device
        self.resume_from = resume_from

    def train(self) -> dict[str, Any]:
        config = self.config
        config.validate()
        if len(self.datasets["train"]) == 0:
            raise ValueError("training dataset has no samples")
        if len(self.datasets["valid"]) == 0:
            raise ValueError("validation dataset has no samples")

        set_seed(config.seed)
        configure_torch_runtime(config, self.device)
        self.model.to(self.device)
        train_loader = create_data_loader(
            self.datasets["train"], config, shuffle=True, training=True
        )
        optimizer, fused_optimizer = _create_optimizer(self.model, config, self.device)
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, config.epochs))
        use_scaler = self.device.type == "cuda" and config.mixed_precision == "fp16"
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model_config.save_json(output_dir / "model_config.json")
        config.save_json(output_dir / "training_config.json")
        self.normalization_stats.save_json(output_dir / "normalization.json")

        state = self._restore_training_state(optimizer, scheduler)
        execution_model: nn.Module = self.model
        if config.compile_mode != "none":
            if not hasattr(torch, "compile"):
                warnings.warn("torch.compile is unavailable in this PyTorch build", stacklevel=2)
            else:
                execution_model = torch.compile(self.model, mode=config.compile_mode)

        self._print_runtime(fused_optimizer)
        writer = SummaryWriter(logdir=str(output_dir / "tensorboard"))
        history_path = output_dir / "history.json"
        history: list[dict[str, Any]] = []
        if state["start_epoch"] > 0 and history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8")).get("history", [])

        for epoch in range(state["start_epoch"], config.epochs):
            train_metrics = self._train_epoch(
                execution_model,
                train_loader,
                optimizer,
                scaler,
                epoch,
            )
            scheduler.step()
            validation = self.validator.validate(execution_model, epoch)
            if validation.loss is None:
                raise RuntimeError("TorchValidator must return a validation loss")
            elapsed_seconds = time.perf_counter() - train_metrics["epoch_started_at"]
            selection_score = _selection_value(config, validation.loss, validation.metrics)
            epoch_record = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "validation_loss": validation.loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": elapsed_seconds,
                "training_elapsed_seconds": train_metrics["elapsed_seconds"],
                "samples_per_second": train_metrics["samples_per_second"],
                "peak_memory_gib": train_metrics["peak_memory_gib"],
                "selection_score": selection_score,
                **{f"validation_{key}": value for key, value in validation.metrics.items()},
            }
            history.append(epoch_record)
            self._write_tensorboard(writer, epoch_record, epoch)
            print(
                f"epoch={epoch + 1}/{config.epochs} "
                f"train_loss={train_metrics['loss']:.6f} valid_loss={validation.loss:.6f} "
                f"rank_ic={validation.metrics['rank_ic_mean']:.4f} "
                f"throughput={train_metrics['samples_per_second']:.1f} samples/s "
                f"peak_memory={train_metrics['peak_memory_gib']:.2f} GiB"
            )

            state["best_validation_loss"] = min(state["best_validation_loss"], validation.loss)
            if _is_improved(selection_score, state["best_selection_score"], config.selection_mode):
                state["best_selection_score"] = selection_score
                state["best_epoch"] = epoch
                state["epochs_without_improvement"] = 0
                save_torch_model(self.model, output_dir / "best.safetensors")
                validation.predictions.to_csv(
                    output_dir / "best_validation_predictions.csv", index=False
                )
            else:
                state["epochs_without_improvement"] += 1

            save_torch_model(self.model, output_dir / "last.safetensors")
            torch.save(
                {
                    "epoch": epoch,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_validation_loss": state["best_validation_loss"],
                    "best_selection_score": state["best_selection_score"],
                    "selection_metric": config.selection_metric,
                    "best_epoch": state["best_epoch"],
                    "epochs_without_improvement": state["epochs_without_improvement"],
                },
                output_dir / "last_state.pt",
            )
            write_json({"history": history}, history_path)
            if state["epochs_without_improvement"] >= config.patience:
                print(f"early stopping after {epoch + 1} epochs")
                break

        writer.close()
        result = {
            "device": str(self.device),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters()),
            "train_samples": len(self.datasets["train"]),
            "validation_samples": len(self.datasets["valid"]),
            "best_epoch": state["best_epoch"],
            "best_validation_loss": state["best_validation_loss"],
            "selection_metric": config.selection_metric,
            "best_selection_score": state["best_selection_score"],
            "epochs_completed": len(history),
        }
        write_json(result, output_dir / "train_summary.json")
        return result

    def _restore_training_state(
        self,
        optimizer: AdamW,
        scheduler: CosineAnnealingLR,
    ) -> dict[str, Any]:
        state: dict[str, Any] = {
            "start_epoch": 0,
            "best_validation_loss": float("inf"),
            "best_selection_score": (
                float("inf") if self.config.selection_mode == "min" else float("-inf")
            ),
            "best_epoch": -1,
            "epochs_without_improvement": 0,
        }
        if self.resume_from is None:
            return state
        resume_dir = Path(self.resume_from)
        if resume_dir.is_file():
            load_torch_model(self.model, resume_dir, self.device)
            return state
        load_torch_model(self.model, resume_dir / "last.safetensors", self.device)
        state_path = resume_dir / "last_state.pt"
        if not state_path.exists():
            return state
        restored = torch.load(state_path, map_location=self.device, weights_only=False)
        try:
            optimizer.load_state_dict(restored["optimizer"])
        except ValueError as error:
            warnings.warn(
                "optimizer state is incompatible with model parameters; "
                f"optimizer moments will restart ({error})",
                stacklevel=2,
            )
        scheduler.load_state_dict(restored["scheduler"])
        state.update(
            start_epoch=int(restored["epoch"]) + 1,
            best_validation_loss=float(restored["best_validation_loss"]),
            best_epoch=int(restored["best_epoch"]),
            epochs_without_improvement=int(restored["epochs_without_improvement"]),
        )
        if restored.get("selection_metric") == self.config.selection_metric:
            state["best_selection_score"] = float(
                restored.get("best_selection_score", state["best_validation_loss"])
            )
        return state

    def _train_epoch(
        self,
        execution_model: nn.Module,
        train_loader,
        optimizer: AdamW,
        scaler: torch.amp.GradScaler,
        epoch: int,
    ) -> dict[str, float]:
        execution_model.train()
        loss_sum = torch.zeros((), device=self.device, dtype=torch.float32)
        sample_count = 0
        average_loss = float("nan")
        started_at = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        with tqdm(
            train_loader,
            desc=f"Train {epoch + 1}/{self.config.epochs}",
            unit="batch",
            dynamic_ncols=True,
        ) as progress:
            for batch_index, batch in enumerate(progress, start=1):
                features = batch["features"].to(self.device, non_blocking=True)
                targets = batch["target"].to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(self.device, self.config.mixed_precision):
                    predictions = execution_model(features)
                    loss = F.mse_loss(predictions.float(), targets.float())
                _assert_finite_loss(loss)
                if batch_index == 1:
                    progress.set_postfix(stage="backward", refresh=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                    foreach=True if self.device.type == "cuda" else None,
                )
                scaler.step(optimizer)
                scaler.update()
                batch_size = features.shape[0]
                loss_sum.add_(loss.detach().float(), alpha=batch_size)
                sample_count += batch_size
                if batch_index % self.config.log_interval == 0 or batch_index == len(train_loader):
                    batch_loss, average_loss = torch.stack(
                        (loss.detach().float(), loss_sum / sample_count)
                    ).tolist()
                    progress.set_postfix(
                        loss=f"{batch_loss:.6f}",
                        average=f"{average_loss:.6f}",
                        lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                        refresh=False,
                    )
        elapsed = time.perf_counter() - started_at
        peak_memory = (
            torch.cuda.max_memory_allocated(self.device) / (1024**3)
            if self.device.type == "cuda"
            else 0.0
        )
        return {
            "loss": average_loss,
            "elapsed_seconds": elapsed,
            "samples_per_second": sample_count / elapsed,
            "peak_memory_gib": peak_memory,
            "epoch_started_at": started_at,
        }

    def _print_runtime(self, fused_optimizer: bool) -> None:
        device_name = (
            torch.cuda.get_device_name(self.device)
            if self.device.type == "cuda"
            else str(self.device)
        )
        kda_modules = [
            module
            for module in self.model.modules()
            if module.__class__.__name__ == "KimiDeltaAttention"
        ]
        if kda_modules:
            use_fla = (
                self.device.type == "cuda"
                and self.config.mixed_precision in {"fp16", "bf16"}
                and all(bool(getattr(module, "fla_available", False)) for module in kda_modules)
                and all(module.config.attention_backend != "torch" for module in kda_modules)
            )
            print(f"KDA execution backend: {'fla-core' if use_fla else 'PyTorch reference'}")
        print(
            "runtime: "
            f"device={device_name}, batch_size={self.config.batch_size}, "
            f"workers={self.config.num_workers}, fused_adamw={fused_optimizer}, "
            f"tf32={self.config.allow_tf32 and self.device.type == 'cuda'}, "
            f"compile={self.config.compile_mode}, log_interval={self.config.log_interval}"
        )

    @staticmethod
    def _write_tensorboard(
        writer: SummaryWriter,
        record: dict[str, Any],
        epoch: int,
    ) -> None:
        writer.add_scalar("loss/train", record["train_loss"], epoch)
        writer.add_scalar("loss/validation", record["validation_loss"], epoch)
        writer.add_scalar("metrics/validation_rank_ic", record["validation_rank_ic_mean"], epoch)
        writer.add_scalar("optimizer/learning_rate", record["learning_rate"], epoch)
        writer.add_scalar("performance/samples_per_second", record["samples_per_second"], epoch)
        if record["peak_memory_gib"] > 0:
            writer.add_scalar("performance/peak_memory_gib", record["peak_memory_gib"], epoch)
        writer.flush()


def train_model(
    model: nn.Module,
    datasets: dict[str, Dataset],
    training_config: TrainingConfig,
    model_config: ModelConfig,
    normalization_stats: NormalizationStats,
    resume_from: str | Path | None = None,
    requested_device: str | None = None,
) -> dict[str, Any]:
    device = resolve_device(requested_device)
    valid_loader = create_data_loader(
        datasets["valid"], training_config, shuffle=False
    )
    validator = TorchValidator(valid_loader, device, training_config.mixed_precision)
    return TorchTrainer(
        model,
        datasets,
        training_config,
        model_config,
        normalization_stats,
        validator,
        device,
        resume_from=resume_from,
    ).train()


save_model = save_torch_model
load_model = load_torch_model
