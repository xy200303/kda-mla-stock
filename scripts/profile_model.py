from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
from torch import nn

from kda_mla_stock.core.config import ModelConfig, TrainingConfig
from kda_mla_stock.models import count_parameters
from kda_mla_stock.models.kda_mla import (
    EncoderLayer,
    GatedFeedForward,
    KimiDeltaAttention,
    MultiHeadLatentAttention,
    StockForecaster,
)


@dataclass
class TimingResult:
    module: str
    parameters: int
    forward_ms: float
    forward_backward_ms: float
    peak_memory_gib: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile KDA, MLA, FFN, encoder layers, and the complete model"
    )
    parser.add_argument("--model-config", default="configs/model-fast.json")
    parser.add_argument("--train-config", default="configs/train.json")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--profile-steps", type=int, default=3)
    parser.add_argument("--output-dir", default="outputs/profile")
    parser.add_argument("--skip-operator-profile", action="store_true")
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def measure_action(
    action: Callable[[], torch.Tensor],
    device: torch.device,
    warmup: int,
    steps: int,
) -> float:
    for _ in range(warmup):
        action()
    synchronize(device)

    if device.type == "cuda":
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
        for start, end in zip(starts, ends, strict=True):
            start.record()
            action()
            end.record()
        synchronize(device)
        return sum(start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)) / steps

    started = time.perf_counter()
    for _ in range(steps):
        action()
    return (time.perf_counter() - started) * 1000.0 / steps


def benchmark_module(
    name: str,
    module: nn.Module,
    inputs: torch.Tensor,
    call: Callable[[nn.Module, torch.Tensor], torch.Tensor],
    device: torch.device,
    precision: str,
    warmup: int,
    steps: int,
) -> TimingResult:
    module.train()

    def forward() -> torch.Tensor:
        with torch.no_grad(), autocast_context(device, precision):
            return call(module, inputs)

    def forward_backward() -> torch.Tensor:
        module.zero_grad(set_to_none=True)
        with autocast_context(device, precision):
            output = call(module, inputs)
            loss = output.float().square().mean()
        loss.backward()
        return loss

    forward_ms = measure_action(forward, device, warmup, steps)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    forward_backward_ms = measure_action(forward_backward, device, warmup, steps)
    peak_memory = (
        torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None
    )
    return TimingResult(
        module=name,
        parameters=count_parameters(module),
        forward_ms=forward_ms,
        forward_backward_ms=forward_backward_ms,
        peak_memory_gib=peak_memory,
    )


def print_results(results: list[TimingResult], batch_size: int) -> None:
    header = (
        f"{'module':<20} {'params':>12} {'forward ms':>12} "
        f"{'fwd+bwd ms':>12} {'samples/s':>12} {'peak GiB':>10}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        samples_per_second = batch_size * 1000.0 / result.forward_backward_ms
        peak_memory = (
            f"{result.peak_memory_gib:.3f}" if result.peak_memory_gib is not None else "n/a"
        )
        print(
            f"{result.module:<20} {result.parameters:>12,} {result.forward_ms:>12.3f} "
            f"{result.forward_backward_ms:>12.3f} {samples_per_second:>12.1f} "
            f"{peak_memory:>10}"
        )


def operator_profile(
    model: StockForecaster,
    features: torch.Tensor,
    device: torch.device,
    precision: str,
    steps: int,
    max_grad_norm: float,
    log_interval: int,
    output_dir: Path,
) -> None:
    activities = [torch.profiler.ProfilerActivity.CPU]
    sort_by = "cpu_time_total"
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        sort_by = "cuda_time_total"

    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        fused=device.type == "cuda",
    )

    def training_step(profile_regions: bool, step_index: int) -> None:
        def region(name: str):
            return torch.profiler.record_function(name) if profile_regions else nullcontext()

        with region("train.zero_grad"):
            optimizer.zero_grad(set_to_none=True)
        with region("train.forward"):
            with autocast_context(device, precision):
                predictions = model(features)
                loss = predictions.float().square().mean()
        with region("train.nonfinite_check"):
            finite = torch.isfinite(loss.detach())
            if device.type == "cuda" and hasattr(torch, "_assert_async"):
                torch._assert_async(finite, "profile loss became non-finite")
            elif not bool(finite):
                raise RuntimeError("profile loss became non-finite")
        with region("train.backward"):
            loss.backward()
        with region("train.gradient_clipping"):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_grad_norm,
                foreach=True if device.type == "cuda" else None,
            )
        with region("train.optimizer_step"):
            optimizer.step()
        if (step_index + 1) % log_interval == 0:
            with region("train.periodic_log_sync"):
                float(loss.item())

    training_step(profile_regions=False, step_index=-1)
    synchronize(device)
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
    ) as profiler:
        for step_index in range(steps):
            with torch.profiler.record_function("complete_training_step"):
                training_step(profile_regions=True, step_index=step_index)
            profiler.step()
    synchronize(device)

    table = profiler.key_averages().table(sort_by=sort_by, row_limit=40)
    (output_dir / "operator_table.txt").write_text(table + "\n", encoding="utf-8")
    profiler.export_chrome_trace(str(output_dir / "trace.json"))
    print("\nTop operators")
    print(table)


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.steps <= 0 or args.profile_steps <= 0:
        raise ValueError("warmup must be non-negative and step counts must be positive")

    model_config = ModelConfig.from_json(args.model_config)
    training_config = TrainingConfig.from_json(args.train_config)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this profile on the AutoDL training instance")
    if device.type == "cpu":
        model_config = replace(model_config, attention_backend="torch")
        print("warning: CPU uses the slow KDA fallback and is suitable only for a smoke test")

    batch_size = args.batch_size or training_config.batch_size
    sequence_length = args.sequence_length or training_config.sequence_length
    precision = args.precision or (
        training_config.mixed_precision if training_config.mixed_precision != "no" else "fp32"
    )
    if device.type == "cpu" and precision == "fp16":
        raise ValueError("CPU profiling does not support fp16")

    torch.manual_seed(training_config.seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = training_config.allow_tf32
        torch.set_float32_matmul_precision("high")

    hidden = torch.randn(
        batch_size,
        sequence_length,
        model_config.hidden_size,
        device=device,
    )
    features = torch.randn(
        batch_size,
        sequence_length,
        model_config.num_features,
        device=device,
    )
    kda_layer_index = model_config.kda_layers[0]
    mla_layer_index = model_config.mla_layers[0]
    targets: list[tuple[str, Callable[[], nn.Module], torch.Tensor, Callable]] = [
        (
            "KDA attention",
            lambda: KimiDeltaAttention(model_config),
            hidden,
            lambda module, tensor: module(tensor, None),
        ),
        (
            "MLA attention",
            lambda: MultiHeadLatentAttention(model_config),
            hidden,
            lambda module, tensor: module(tensor, None),
        ),
        (
            "FFN",
            lambda: GatedFeedForward(model_config),
            hidden,
            lambda module, tensor: module(tensor),
        ),
        (
            "KDA encoder layer",
            lambda: EncoderLayer(model_config, kda_layer_index),
            hidden,
            lambda module, tensor: module(tensor, None),
        ),
        (
            "MLA encoder layer",
            lambda: EncoderLayer(model_config, mla_layer_index),
            hidden,
            lambda module, tensor: module(tensor, None),
        ),
        (
            "complete model",
            lambda: StockForecaster(model_config),
            features,
            lambda module, tensor: module(tensor),
        ),
    ]

    print(
        f"device={device}, precision={precision}, batch={batch_size}, "
        f"sequence={sequence_length}, warmup={args.warmup}, steps={args.steps}"
    )
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")

    results: list[TimingResult] = []
    for name, module_factory, inputs, call in targets:
        module = module_factory().to(device)
        results.append(
            benchmark_module(
                name,
                module,
                inputs,
                call,
                device,
                precision,
                args.warmup,
                args.steps,
            )
        )
        del module
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print_results(results, batch_size)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "environment": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch": torch.__version__,
            "precision": precision,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "warmup": args.warmup,
            "steps": args.steps,
        },
        "timings": [asdict(result) for result in results],
    }
    (output_dir / "module_timings.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    if not args.skip_operator_profile:
        model = StockForecaster(model_config).to(device)
        operator_profile(
            model,
            features,
            device,
            precision,
            args.profile_steps,
            training_config.max_grad_norm,
            training_config.log_interval,
            output_dir,
        )
    print(f"profile artifacts saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
