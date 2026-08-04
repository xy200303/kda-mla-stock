from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from kda_mla_stock.qlib_evaluation import predictions_to_qlib_signal
from kda_mla_stock.reporting import plot_prediction_diagnostics, plot_training_history


def _predictions() -> pd.DataFrame:
    rows = []
    for date in pd.bdate_range("2024-01-01", periods=4):
        for symbol_index in range(5):
            rows.append(
                {
                    "date": date,
                    "symbol": f"SH{symbol_index:06d}",
                    "prediction": symbol_index * 0.01,
                    "target": (symbol_index - 2) * 0.005,
                }
            )
    return pd.DataFrame(rows)


def test_predictions_to_qlib_signal() -> None:
    signal = predictions_to_qlib_signal(_predictions())
    assert signal.index.names == ["datetime", "instrument"]
    assert signal.columns.tolist() == ["score"]
    assert len(signal) == 20

    duplicated = pd.concat([_predictions(), _predictions().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        predictions_to_qlib_signal(duplicated)

    non_finite = _predictions()
    non_finite["prediction"] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        predictions_to_qlib_signal(non_finite)


def test_research_figures_are_generated(tmp_path: Path) -> None:
    predictions = _predictions()
    daily_ic = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=4),
            "ic": [0.2, 0.1, -0.1, 0.3],
            "rank_ic": [0.3, 0.2, 0.0, 0.4],
            "count": [5, 5, 5, 5],
        }
    )
    diagnostics_path = tmp_path / "diagnostics.png"
    plot_prediction_diagnostics(predictions, daily_ic, diagnostics_path)
    assert diagnostics_path.stat().st_size > 0

    missing_history = tmp_path / "missing.json"
    assert not plot_training_history(missing_history, tmp_path / "history.png")
