from __future__ import annotations

import random
from contextlib import nullcontext

import numpy as np
import torch

from kda_mla_stock.core.config import TrainingConfig


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


def autocast_context(device: torch.device, precision: str):
    if precision == "no":
        return nullcontext()
    if device.type == "cuda":
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)
    if device.type == "cpu" and precision == "bf16":
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    return nullcontext()


def configure_torch_runtime(config: TrainingConfig, device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
    torch.backends.cudnn.allow_tf32 = config.allow_tf32
    torch.set_float32_matmul_precision("high" if config.allow_tf32 else "highest")
