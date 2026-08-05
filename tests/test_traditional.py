from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kda_mla_stock.core.config import TrainingConfig
from kda_mla_stock.data import MarketWindowStore, StockWindowDataset, WindowReference
from kda_mla_stock.data.tabular import build_tabular_dataset
from kda_mla_stock.models import (
    TraditionalModelConfig,
    build_estimator,
    fit_estimator,
    load_estimator,
    save_estimator,
)
from kda_mla_stock.orchestration import EvaluationRunner, TrainRunner


def _window_dataset() -> tuple[StockWindowDataset, np.ndarray]:
    row_count = 300
    feature_names = [f"feature_{index}" for index in range(10)]
    values = np.arange(row_count * len(feature_names), dtype=np.float32).reshape(
        row_count, len(feature_names)
    )
    dates = pd.bdate_range("2020-01-01", periods=row_count)
    frame = pd.DataFrame(values, columns=feature_names)
    frame["date"] = dates
    frame["symbol"] = "TEST"
    frame["target"] = np.linspace(-0.05, 0.05, row_count, dtype=np.float32)
    frame["target_date"] = dates + pd.offsets.BDay(5)
    store = MarketWindowStore(frame, feature_names)
    references = [WindowReference(0, anchor) for anchor in range(280, row_count)]
    return StockWindowDataset(store, references, sequence_length=256), values


def test_build_tabular_dataset_extracts_causal_90_dimensional_features() -> None:
    dataset, values = _window_dataset()
    tabular = build_tabular_dataset(dataset, [5, 20, 60, 256])

    assert tabular.features.shape == (20, 90)
    assert len(tabular.feature_names) == 90
    assert np.isfinite(tabular.features).all()
    np.testing.assert_allclose(tabular.features[0, :10], values[280])
    np.testing.assert_allclose(tabular.features[0, 10:20], values[276:281].mean(axis=0))
    np.testing.assert_allclose(tabular.features[0, 20:30], values[276:281].std(axis=0))
    reference = dataset.references[0]
    assert tabular.targets[0] == dataset.store.targets[0][reference.anchor_index]
    assert tabular.dates[0] == dataset.store.dates[0][reference.anchor_index]
    assert tabular.symbols[0] == "TEST"


def test_ridge_model_round_trip(tmp_path) -> None:
    pytest.importorskip("sklearn")
    pytest.importorskip("joblib")
    dataset, _values = _window_dataset()
    tabular = build_tabular_dataset(dataset, [5, 20, 60, 256])
    config = TraditionalModelConfig(
        architecture="ridge",
        params={"alpha": 1.0, "solver": "lsqr"},
    )
    estimator = build_estimator(config, seed=42)
    fit_estimator(
        estimator,
        config,
        tabular.features,
        tabular.targets,
        tabular.features,
        tabular.targets,
    )

    path = tmp_path / "model.joblib"
    expected = estimator.predict(tabular.features)
    save_estimator(estimator, path)
    restored = load_estimator(path)

    assert path.exists()
    np.testing.assert_allclose(restored.predict(tabular.features), expected)


@pytest.mark.parametrize(
    "config_path",
    [
        "configs/ml-ridge.json",
        "configs/ml-random-forest.json",
        "configs/ml-hist-gbdt.json",
        "configs/ml-lightgbm.json",
    ],
)
def test_official_library_model_configs_fit(config_path: str) -> None:
    pytest.importorskip("sklearn")
    if "lightgbm" in config_path:
        pytest.importorskip("lightgbm")
    rng = np.random.default_rng(42)
    features = rng.normal(size=(128, 90)).astype(np.float32)
    targets = (0.2 * features[:, 0] - 0.1 * features[:, 1]).astype(np.float32)
    train_features, valid_features = features[:96], features[96:]
    train_targets, valid_targets = targets[:96], targets[96:]
    config = TraditionalModelConfig.from_json(config_path)
    estimator = build_estimator(config, seed=42)

    fit_estimator(
        estimator,
        config,
        train_features,
        train_targets,
        valid_features,
        valid_targets,
    )

    predictions = estimator.predict(valid_features)
    assert predictions.shape == valid_targets.shape
    assert np.isfinite(predictions).all()


def test_unified_trainer_and_valer_support_sklearn(
    market_frame: pd.DataFrame,
    tmp_path,
) -> None:
    pytest.importorskip("sklearn")
    pytest.importorskip("joblib")
    data_path = tmp_path / "market.csv"
    market_frame.to_csv(data_path, index=False)
    output_dir = tmp_path / "ridge-run"
    training_config = TrainingConfig(
        data_path=str(data_path),
        output_dir=str(output_dir),
        sequence_length=8,
        horizon=3,
        train_end="2020-03-20",
        valid_end="2020-04-15",
        mixed_precision="no",
    )
    model_config = TraditionalModelConfig(
        architecture="ridge",
        aggregation_windows=[2, 4, 8],
        params={"alpha": 1.0},
    )

    train_summary = TrainRunner(model_config, training_config).run()
    evaluation = EvaluationRunner(output_dir, requested_device="cpu").run("test")

    assert train_summary["architecture"] == "ridge"
    assert (output_dir / "model.joblib").exists()
    assert (output_dir / "evaluation_test" / "predictions.csv").exists()
    assert evaluation["split"] == "test"
    assert "rank_ic_mean" in evaluation["prediction_metrics"]
