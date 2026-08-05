from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import nn


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


def save_torch_model(model: nn.Module, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model.state_dict().items()
    }
    save_file(state, str(destination))


def load_torch_model(model: nn.Module, path: str | Path, device: torch.device) -> None:
    model.load_state_dict(load_file(str(path), device=str(device)))


def save_estimator(estimator: Any, path: str | Path) -> None:
    try:
        import joblib
    except ImportError as error:
        raise RuntimeError("joblib is required to save estimator models") from error
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, destination)


def load_estimator(path: str | Path) -> Any:
    try:
        import joblib
    except ImportError as error:
        raise RuntimeError("joblib is required to load estimator models") from error
    return joblib.load(path)
