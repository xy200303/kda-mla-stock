from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SUPPORTED_TRADITIONAL_MODELS = {"ridge", "random_forest", "hist_gbdt", "lightgbm"}


@dataclass
class TraditionalModelConfig:
    architecture: str
    aggregation_windows: list[int] = field(default_factory=lambda: [5, 20, 60, 256])
    params: dict[str, Any] = field(default_factory=dict)
    feature_scheme: str = "last_mean_std"

    def validate(self) -> None:
        if self.architecture not in SUPPORTED_TRADITIONAL_MODELS:
            supported = ", ".join(sorted(SUPPORTED_TRADITIONAL_MODELS))
            raise ValueError(f"traditional architecture must be one of {supported}")
        if self.feature_scheme != "last_mean_std":
            raise ValueError("feature_scheme must be last_mean_std")
        if not self.aggregation_windows or any(window <= 0 for window in self.aggregation_windows):
            raise ValueError("aggregation_windows must contain positive values")
        if len(self.aggregation_windows) != len(set(self.aggregation_windows)):
            raise ValueError("aggregation_windows must not contain duplicates")

    @classmethod
    def from_json(cls, path: str | Path) -> TraditionalModelConfig:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TraditionalModelConfig:
        config = cls(**payload)
        config.validate()
        return config

    def save_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def build_estimator(config: TraditionalModelConfig, seed: int):
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
        from sklearn.linear_model import Ridge
    except ImportError as error:
        raise RuntimeError("install traditional ML dependencies: pip install -e '.[ml]'") from error

    params = dict(config.params)
    if config.architecture == "ridge":
        return Ridge(**params)
    if config.architecture == "random_forest":
        params.setdefault("random_state", seed)
        params.setdefault("n_jobs", -1)
        return RandomForestRegressor(**params)
    if config.architecture == "hist_gbdt":
        params.setdefault("random_state", seed)
        return HistGradientBoostingRegressor(**params)
    try:
        from lightgbm import LGBMRegressor
    except ImportError as error:
        raise RuntimeError(
            "LightGBM is unavailable; install with pip install -e '.[ml]'"
        ) from error
    params.setdefault("random_state", seed)
    params.setdefault("n_jobs", -1)
    return LGBMRegressor(**params)


def fit_estimator(
    estimator,
    config: TraditionalModelConfig,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    valid_features: np.ndarray,
    valid_targets: np.ndarray,
) -> None:
    if config.architecture == "lightgbm":
        import lightgbm as lgb

        estimator.fit(
            train_features,
            train_targets,
            eval_set=[(valid_features, valid_targets)],
            eval_metric="l2",
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
        )
    else:
        estimator.fit(train_features, train_targets)


def count_estimator_parameters(estimator) -> int:
    if hasattr(estimator, "coef_"):
        coefficients = int(np.asarray(estimator.coef_).size)
        intercept = int(np.asarray(estimator.intercept_).size)
        return coefficients + intercept
    if hasattr(estimator, "estimators_"):
        estimators = np.asarray(estimator.estimators_, dtype=object).reshape(-1)
        return int(
            sum(tree.tree_.node_count for tree in estimators if hasattr(tree, "tree_"))
        )
    if hasattr(estimator, "_predictors"):
        return int(
            sum(
                predictor.nodes.shape[0]
                for iteration in estimator._predictors
                for predictor in iteration
            )
        )
    if hasattr(estimator, "booster_"):
        model = estimator.booster_.dump_model()
        return int(sum(tree["num_leaves"] for tree in model["tree_info"]))
    return 0


def save_estimator(estimator, path: str | Path) -> None:
    try:
        import joblib
    except ImportError as error:
        raise RuntimeError("joblib is required to save traditional models") from error
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, destination)


def load_estimator(path: str | Path):
    try:
        import joblib
    except ImportError as error:
        raise RuntimeError("joblib is required to load traditional models") from error
    return joblib.load(path)
