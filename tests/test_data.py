from __future__ import annotations

import numpy as np
import pandas as pd

from kda_mla_stock.data import (
    FEATURE_COLUMNS,
    apply_normalization,
    build_window_datasets,
    engineer_features,
    fit_normalization_stats,
)


def test_features_do_not_depend_on_future_rows(market_frame: pd.DataFrame) -> None:
    cutoff = pd.Timestamp("2020-03-20")
    baseline = engineer_features(market_frame, horizon=5)
    changed = market_frame.copy()
    future = changed["date"] > cutoff
    changed.loc[future, ["open", "high", "low", "close"]] *= 1.7
    changed.loc[future, "volume"] *= 3
    modified = engineer_features(changed, horizon=5)

    baseline_past = baseline.loc[baseline["date"] <= cutoff, list(FEATURE_COLUMNS)]
    modified_past = modified.loc[modified["date"] <= cutoff, list(FEATURE_COLUMNS)]
    np.testing.assert_allclose(
        baseline_past.to_numpy(),
        modified_past.to_numpy(),
        equal_nan=True,
    )


def test_time_splits_purge_labels_crossing_boundaries(market_frame: pd.DataFrame) -> None:
    engineered = engineer_features(market_frame, horizon=5)
    train_end = pd.Timestamp("2020-03-31")
    valid_end = pd.Timestamp("2020-04-30")
    stats = fit_normalization_stats(engineered, train_end)
    normalized = apply_normalization(engineered, stats)
    finite_train = normalized.loc[
        normalized["date"] <= train_end,
        list(FEATURE_COLUMNS),
    ].dropna()
    assert np.isfinite(finite_train.to_numpy()).all()

    datasets = build_window_datasets(
        engineered,
        stats,
        sequence_length=8,
        train_end=train_end,
        valid_end=valid_end,
    )
    assert all(len(dataset) > 0 for dataset in datasets.values())

    train_dataset = datasets["train"]
    for reference in train_dataset.references:
        target_date = train_dataset.store.target_dates[reference.symbol_index][
            reference.anchor_index
        ]
        assert target_date <= np.datetime64(train_end)

    valid_dataset = datasets["valid"]
    for reference in valid_dataset.references:
        anchor_date = valid_dataset.store.dates[reference.symbol_index][reference.anchor_index]
        target_date = valid_dataset.store.target_dates[reference.symbol_index][
            reference.anchor_index
        ]
        assert anchor_date > np.datetime64(train_end)
        assert target_date <= np.datetime64(valid_end)

    sample = train_dataset[0]
    assert sample["features"].shape == (8, len(FEATURE_COLUMNS))
    assert sample["target"].shape == (1,)

    strided = build_window_datasets(
        engineered,
        stats,
        sequence_length=8,
        train_end=train_end,
        valid_end=valid_end,
        train_stride=5,
    )
    assert len(strided["train"]) < len(datasets["train"])
    assert len(strided["valid"]) == len(datasets["valid"])
    assert len(strided["test"]) == len(datasets["test"])
