from __future__ import annotations

import pandas as pd
import pytest

from kda_mla_stock.backtest import run_long_short_backtest, summarize_backtest
from kda_mla_stock.metrics import evaluate_predictions


def test_metrics_and_long_short_backtest() -> None:
    rows = []
    dates = pd.bdate_range("2024-01-01", periods=6)
    for date_index, date in enumerate(dates):
        for symbol_index in range(10):
            target = (symbol_index - 4.5) * 0.002 + date_index * 0.0001
            rows.append(
                {
                    "date": date,
                    "symbol": f"S{symbol_index:02d}",
                    "prediction": target,
                    "target": target,
                }
            )
    predictions = pd.DataFrame(rows)
    metrics = evaluate_predictions(predictions)
    assert metrics["rank_ic_mean"] == pytest.approx(1.0)
    assert metrics["direction_accuracy"] == pytest.approx(1.0)

    backtest = run_long_short_backtest(
        predictions,
        top_fraction=0.2,
        transaction_cost_bps=5.0,
        rebalance_every=1,
    )
    summary = summarize_backtest(backtest)
    assert len(backtest) == len(dates)
    assert backtest["gross_return"].gt(0).all()
    assert summary["total_return"] > 0
