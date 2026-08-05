from __future__ import annotations

from typing import Any

import numpy as np


def build_estimator(params: dict[str, Any], seed: int):
    try:
        from lightgbm import LGBMRegressor
    except ImportError as error:
        raise RuntimeError(
            "LightGBM is unavailable; install with pip install -e '.[ml]'"
        ) from error
    resolved = dict(params)
    resolved.setdefault("random_state", seed)
    resolved.setdefault("n_jobs", -1)
    return LGBMRegressor(**resolved)


def fit_estimator(
    estimator: Any,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    valid_features: np.ndarray,
    valid_targets: np.ndarray,
) -> None:
    import lightgbm as lgb

    estimator.fit(
        train_features,
        train_targets,
        eval_set=[(valid_features, valid_targets)],
        eval_metric="l2",
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
    )
