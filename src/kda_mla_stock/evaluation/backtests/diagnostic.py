from __future__ import annotations

import math

import numpy as np
import pandas as pd


def run_long_short_backtest(
    predictions: pd.DataFrame,
    top_fraction: float = 0.2,
    transaction_cost_bps: float = 10.0,
    rebalance_every: int = 1,
) -> pd.DataFrame:
    if not 0.0 < top_fraction <= 0.5:
        raise ValueError("top_fraction must be in (0, 0.5]")
    if transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be non-negative")
    if rebalance_every <= 0:
        raise ValueError("rebalance_every must be positive")
    required = {"date", "symbol", "prediction", "target"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions are missing columns: {', '.join(sorted(missing))}")

    clean = predictions.loc[:, ["date", "symbol", "prediction", "target"]].dropna()
    dates = sorted(clean["date"].unique())[::rebalance_every]
    previous_weights: dict[str, float] = {}
    rows = []
    cost_rate = transaction_cost_bps / 10_000.0
    for date in dates:
        group = clean.loc[clean["date"] == date].sort_values("prediction", kind="stable")
        if len(group) < 2:
            continue
        selection_size = max(1, int(math.floor(len(group) * top_fraction)))
        selection_size = min(selection_size, len(group) // 2)
        short = group.head(selection_size)
        long = group.tail(selection_size)
        current_weights = {
            **{str(symbol): -1.0 / selection_size for symbol in short["symbol"]},
            **{str(symbol): 1.0 / selection_size for symbol in long["symbol"]},
        }
        all_symbols = set(previous_weights) | set(current_weights)
        turnover = 0.5 * sum(
            abs(current_weights.get(symbol, 0.0) - previous_weights.get(symbol, 0.0))
            for symbol in all_symbols
        )
        gross_return = float(long["target"].mean() - short["target"].mean())
        transaction_cost = turnover * cost_rate
        rows.append(
            {
                "date": date,
                "gross_return": gross_return,
                "turnover": turnover,
                "transaction_cost": transaction_cost,
                "net_return": gross_return - transaction_cost,
                "long_count": selection_size,
                "short_count": selection_size,
            }
        )
        previous_weights = current_weights

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["equity"] = (1.0 + result["net_return"]).cumprod()
    result["drawdown"] = result["equity"] / result["equity"].cummax() - 1.0
    return result


def summarize_backtest(
    backtest: pd.DataFrame,
    periods_per_year: float = 252.0,
) -> dict[str, float | int]:
    if backtest.empty:
        return {
            "periods": 0,
            "total_return": float("nan"),
            "annualized_return": float("nan"),
            "annualized_sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "average_turnover": float("nan"),
        }
    returns = backtest["net_return"]
    total_return = float(backtest["equity"].iloc[-1] - 1.0)
    years = len(backtest) / periods_per_year
    annualized_return = (
        float((1.0 + total_return) ** (1.0 / years) - 1.0)
        if years > 0 and total_return > -1.0
        else float("nan")
    )
    return_std = float(returns.std(ddof=1)) if len(returns) > 1 else float("nan")
    sharpe = (
        float(returns.mean() / return_std * np.sqrt(periods_per_year))
        if np.isfinite(return_std) and return_std > 0
        else float("nan")
    )
    return {
        "periods": int(len(backtest)),
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_sharpe": sharpe,
        "max_drawdown": float(backtest["drawdown"].min()),
        "average_turnover": float(backtest["turnover"].mean()),
    }
