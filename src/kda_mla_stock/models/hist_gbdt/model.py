from __future__ import annotations

from typing import Any


def build_estimator(params: dict[str, Any], seed: int):
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError as error:
        raise RuntimeError("install estimator dependencies: pip install -e '.[ml]'") from error
    resolved = dict(params)
    resolved.setdefault("random_state", seed)
    return HistGradientBoostingRegressor(**resolved)
