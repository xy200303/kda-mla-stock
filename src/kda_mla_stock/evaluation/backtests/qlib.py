from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QlibBacktestConfig:
    provider_uri: str = "~/.qlib/qlib_data/cn_data"
    region: str = "cn"
    benchmark: str = "SH000300"
    topk: int = 50
    n_drop: int = 10
    hold_thresh: int = 5
    account: float = 100_000_000.0
    deal_price: str = "open"
    limit_threshold: float = 0.095
    open_cost: float = 0.0005
    close_cost: float = 0.0015
    min_cost: float = 5.0

    def validate(self) -> None:
        if self.region not in {"cn", "us"}:
            raise ValueError("region must be cn or us")
        if self.topk <= 0 or self.n_drop <= 0 or self.n_drop > self.topk:
            raise ValueError("topk must be positive and n_drop must be in [1, topk]")
        if self.hold_thresh <= 0:
            raise ValueError("hold_thresh must be positive")
        if self.account <= 0:
            raise ValueError("account must be positive")
        if self.deal_price not in {"open", "close", "vwap"}:
            raise ValueError("deal_price must be open, close, or vwap")
        if not 0.0 <= self.limit_threshold <= 1.0:
            raise ValueError("limit_threshold must be in [0, 1]")
        if min(self.open_cost, self.close_cost, self.min_cost) < 0:
            raise ValueError("transaction costs must be non-negative")


def predictions_to_qlib_signal(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "symbol", "prediction"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions are missing columns: {', '.join(sorted(missing))}")
    signal = predictions.loc[:, ["date", "symbol", "prediction"]].dropna().copy()
    signal["datetime"] = pd.to_datetime(signal.pop("date"))
    signal["instrument"] = signal.pop("symbol").astype(str)
    signal["score"] = pd.to_numeric(signal.pop("prediction"), errors="raise")
    signal = signal.loc[np.isfinite(signal["score"])].copy()
    if signal.duplicated(["datetime", "instrument"]).any():
        raise ValueError("predictions contain duplicate date/symbol rows")
    if signal.empty:
        raise ValueError("no finite predictions are available for Qlib")
    return signal.set_index(["datetime", "instrument"]).sort_index()


def _flatten_risk_analysis(analysis: dict[str, pd.DataFrame]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for series_name, frame in analysis.items():
        for metric_name, value in frame["risk"].items():
            metrics[f"{series_name}_{metric_name}"] = float(value)
    return metrics


def run_qlib_backtest(
    predictions: pd.DataFrame,
    config: QlibBacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run Qlib's TopkDropoutStrategy, exchange simulation, and risk analysis."""
    config.validate()
    try:
        import qlib
        from qlib.backtest import backtest
        from qlib.constant import REG_CN, REG_US
        from qlib.contrib.evaluate import indicator_analysis, risk_analysis
        from qlib.data import D
    except ImportError as error:
        raise RuntimeError("install the Qlib extra first: pip install -e '.[qlib]'") from error

    region = REG_CN if config.region == "cn" else REG_US
    qlib.init(provider_uri=str(Path(config.provider_uri).expanduser()), region=region)
    signal = predictions_to_qlib_signal(predictions)
    signal_dates = pd.DatetimeIndex(
        signal.index.get_level_values("datetime").unique()
    ).sort_values()
    calendar = pd.DatetimeIndex(
        D.calendar(start_time=signal_dates.min(), end_time=None, freq="day")
    )
    later_dates = calendar[calendar > signal_dates.max()]
    backtest_end = later_dates[0] if len(later_dates) else signal_dates.max()

    strategy = {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy",
        "kwargs": {
            "signal": signal,
            "topk": config.topk,
            "n_drop": config.n_drop,
            "hold_thresh": config.hold_thresh,
            "only_tradable": True,
            "risk_degree": 0.95,
        },
    }
    executor = {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            "time_per_step": "day",
            "generate_portfolio_metrics": True,
        },
    }
    portfolio_metrics, indicator_metrics = backtest(
        start_time=signal_dates.min(),
        end_time=backtest_end,
        strategy=strategy,
        executor=executor,
        benchmark=config.benchmark,
        account=config.account,
        exchange_kwargs={
            "freq": "day",
            "limit_threshold": config.limit_threshold,
            "deal_price": config.deal_price,
            "open_cost": config.open_cost,
            "close_cost": config.close_cost,
            "min_cost": config.min_cost,
        },
    )
    report, _positions = portfolio_metrics["1day"]
    indicator_bundle = indicator_metrics.get("1day")
    indicators = indicator_bundle[0] if indicator_bundle is not None else pd.DataFrame()

    returns = report["return"]
    benchmark = report["bench"]
    costs = report["cost"]
    analysis = {
        "strategy_with_cost": risk_analysis(returns - costs, freq="day", mode="product"),
        "benchmark": risk_analysis(benchmark, freq="day", mode="product"),
        "excess_without_cost": risk_analysis(
            returns - benchmark,
            freq="day",
            mode="product",
        ),
        "excess_with_cost": risk_analysis(
            returns - benchmark - costs,
            freq="day",
            mode="product",
        ),
    }
    risk_table = pd.concat(analysis, names=["return_series", "metric"])
    if indicators.empty:
        indicator_table = pd.DataFrame(columns=["value"])
    else:
        indicator_table = indicator_analysis(indicators)
    summary: dict[str, Any] = {
        "config": asdict(config),
        "signal_lag_trading_days": 1,
        "start_time": str(pd.Timestamp(report.index.min()).date()),
        "end_time": str(pd.Timestamp(report.index.max()).date()),
        "trading_days": int(len(report)),
        "risk_metrics": _flatten_risk_analysis(analysis),
        "trade_indicators": {
            str(name): float(value) for name, value in indicator_table["value"].items()
        },
    }
    return report, risk_table, indicator_table, summary
