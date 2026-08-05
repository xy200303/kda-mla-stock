from __future__ import annotations

from typing import Any


def build_estimator(params: dict[str, Any], seed: int):
    del seed
    try:
        from sklearn.linear_model import Ridge
    except ImportError as error:
        raise RuntimeError("install estimator dependencies: pip install -e '.[ml]'") from error
    return Ridge(**params)
