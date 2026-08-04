from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from kda_mla_stock.backtest import run_long_short_backtest, summarize_backtest
from kda_mla_stock.configuration import TrainingConfig
from kda_mla_stock.metrics import daily_information_coefficients, evaluate_predictions
from kda_mla_stock.qlib_evaluation import QlibBacktestConfig, run_qlib_backtest
from kda_mla_stock.reporting import (
    plot_portfolio_report,
    plot_prediction_diagnostics,
    plot_training_history,
)
from kda_mla_stock.training import write_json


def evaluate_and_write_predictions(
    predictions: pd.DataFrame,
    training_config: TrainingConfig,
    output_dir: str | Path,
    split: str,
    *,
    loss: float | None = None,
    qlib_config: QlibBacktestConfig | None = None,
    history_path: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    prediction_metrics = evaluate_predictions(predictions)
    if loss is not None:
        prediction_metrics["loss"] = loss
    backtest = run_long_short_backtest(
        predictions,
        top_fraction=training_config.top_fraction,
        transaction_cost_bps=training_config.transaction_cost_bps,
        rebalance_every=training_config.horizon,
    )
    backtest_metrics = summarize_backtest(
        backtest,
        periods_per_year=252.0 / training_config.horizon,
    )
    daily_ic = daily_information_coefficients(predictions)
    predictions.to_csv(destination / "predictions.csv", index=False)
    daily_ic.to_csv(destination / "daily_ic.csv", index=False)
    backtest.to_csv(destination / "backtest.csv", index=False)
    plot_prediction_diagnostics(
        predictions,
        daily_ic,
        destination / "prediction_diagnostics.png",
    )
    if history_path is not None and Path(history_path).exists():
        plot_training_history(history_path, destination / "training_curves.png")

    summary: dict[str, Any] = {
        "split": split,
        "prediction_metrics": prediction_metrics,
        "diagnostic_backtest_metrics": backtest_metrics,
    }
    if metadata:
        summary.update(metadata)
    if qlib_config is not None:
        qlib_report, risk_table, indicator_table, qlib_summary = run_qlib_backtest(
            predictions,
            qlib_config,
        )
        qlib_report.to_csv(destination / "qlib_portfolio_report.csv", index=True)
        risk_table.to_csv(destination / "qlib_risk_analysis.csv", index=True)
        indicator_table.to_csv(destination / "qlib_trade_indicators.csv", index=True)
        plot_portfolio_report(qlib_report, destination / "qlib_portfolio_performance.png")
        summary["qlib_backtest"] = qlib_summary
    write_json(summary, destination / "summary.json")
    return summary
