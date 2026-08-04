from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.switch_backend("Agg")


COLORS = {
    "strategy": "#087E8B",
    "benchmark": "#5C677D",
    "excess": "#C44536",
    "train": "#087E8B",
    "valid": "#C44536",
    "ic": "#5C677D",
    "rank_ic": "#D18B00",
}


def _prepare_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )


def _format_date_axis(axis: plt.Axes) -> None:
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def plot_training_history(history_path: str | Path, output_path: str | Path) -> bool:
    source = Path(history_path)
    if not source.exists():
        return False
    history = json.loads(source.read_text(encoding="utf-8")).get("history", [])
    if not history:
        return False
    frame = pd.DataFrame(history)
    epochs = frame["epoch"] + 1
    _prepare_style()
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.1))
    axes[0].plot(epochs, frame["train_loss"], label="Train", color=COLORS["train"])
    axes[0].plot(epochs, frame["validation_loss"], label="Validation", color=COLORS["valid"])
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="MSE")
    axes[0].legend(frameon=False)

    axes[1].plot(
        epochs,
        frame["validation_rank_ic_mean"],
        color=COLORS["rank_ic"],
    )
    axes[1].axhline(0.0, color="#333333", linewidth=0.8)
    axes[1].set(title="Validation Rank IC", xlabel="Epoch", ylabel="Rank IC")

    if "samples_per_second" in frame:
        axes[2].plot(epochs, frame["samples_per_second"], color=COLORS["strategy"])
        axes[2].set(title="Training Throughput", xlabel="Epoch", ylabel="Samples / second")
    else:
        axes[2].plot(epochs, frame["learning_rate"], color=COLORS["strategy"])
        axes[2].set(title="Learning Rate", xlabel="Epoch", ylabel="LR")
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)
    return True


def plot_prediction_diagnostics(
    predictions: pd.DataFrame,
    daily_ic: pd.DataFrame,
    output_path: str | Path,
) -> None:
    _prepare_style()
    prediction_frame = predictions.dropna(subset=["prediction", "target"]).copy()
    ic_frame = daily_ic.copy()
    ic_frame["date"] = pd.to_datetime(ic_frame["date"])
    figure, axes = plt.subplots(2, 2, figsize=(13, 8))

    axes[0, 0].plot(ic_frame["date"], ic_frame["ic"], color=COLORS["ic"], alpha=0.55)
    axes[0, 0].plot(
        ic_frame["date"],
        ic_frame["ic"].rolling(20, min_periods=5).mean(),
        color=COLORS["strategy"],
        label="20-day mean",
    )
    axes[0, 0].axhline(0.0, color="#333333", linewidth=0.8)
    axes[0, 0].set(title="Daily IC", xlabel="Date", ylabel="Pearson IC")
    axes[0, 0].legend(frameon=False)
    _format_date_axis(axes[0, 0])

    axes[0, 1].plot(
        ic_frame["date"],
        ic_frame["rank_ic"].fillna(0.0).cumsum(),
        color=COLORS["rank_ic"],
    )
    axes[0, 1].set(title="Cumulative Rank IC", xlabel="Date", ylabel="Cumulative value")
    _format_date_axis(axes[0, 1])

    scatter = prediction_frame
    if len(scatter) > 50_000:
        scatter = scatter.sample(50_000, random_state=42)
    axes[1, 0].hexbin(
        scatter["prediction"],
        scatter["target"],
        gridsize=55,
        mincnt=1,
        cmap="viridis",
    )
    axes[1, 0].set(title="Prediction vs. Target", xlabel="Prediction", ylabel="Realized return")

    prediction_frame["score_percentile"] = prediction_frame.groupby(
        "date", observed=True
    )["prediction"].rank(pct=True, method="first")
    prediction_frame["decile"] = np.minimum(
        9,
        np.floor(prediction_frame["score_percentile"] * 10).astype(int),
    )
    decile_returns = (
        prediction_frame.groupby("decile", observed=True)["target"].mean().reindex(range(10))
    )
    axes[1, 1].bar(
        decile_returns.index + 1,
        decile_returns.values,
        color=COLORS["strategy"],
    )
    axes[1, 1].set(
        title="Mean Return by Prediction Decile",
        xlabel="Prediction decile (low to high)",
        ylabel="Mean realized return",
    )
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)


def plot_portfolio_report(report: pd.DataFrame, output_path: str | Path) -> None:
    _prepare_style()
    frame = report.copy()
    frame.index = pd.to_datetime(frame.index)
    net_return = frame["return"] - frame.get("cost", 0.0)
    strategy_equity = (1.0 + net_return).cumprod()
    benchmark_equity = (1.0 + frame["bench"]).cumprod()
    drawdown = strategy_equity / strategy_equity.cummax() - 1.0
    excess = (1.0 + net_return - frame["bench"]).cumprod()

    figure, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes[0, 0].plot(frame.index, strategy_equity, color=COLORS["strategy"], label="Strategy")
    axes[0, 0].plot(
        frame.index,
        benchmark_equity,
        color=COLORS["benchmark"],
        label="Benchmark",
    )
    axes[0, 0].set(title="Portfolio Net Value", xlabel="Date", ylabel="Net value")
    axes[0, 0].legend(frameon=False)
    _format_date_axis(axes[0, 0])

    axes[0, 1].plot(frame.index, excess, color=COLORS["excess"])
    axes[0, 1].axhline(1.0, color="#333333", linewidth=0.8)
    axes[0, 1].set(title="Compounded Excess Return", xlabel="Date", ylabel="Net value")
    _format_date_axis(axes[0, 1])

    axes[1, 0].fill_between(frame.index, drawdown, 0.0, color=COLORS["valid"], alpha=0.7)
    axes[1, 0].set(title="Strategy Drawdown", xlabel="Date", ylabel="Drawdown")
    _format_date_axis(axes[1, 0])

    turnover = frame.get("turnover", pd.Series(0.0, index=frame.index))
    axes[1, 1].plot(frame.index, turnover, color=COLORS["rank_ic"], alpha=0.8)
    axes[1, 1].set(title="Portfolio Turnover", xlabel="Date", ylabel="Turnover")
    _format_date_axis(axes[1, 1])
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)


def plot_comparison(
    comparison: pd.DataFrame,
    output_path: str | Path,
) -> None:
    required = [
        "rank_ic_mean",
        "rank_icir_annualized",
        "excess_with_cost_annualized_return",
        "excess_with_cost_information_ratio",
    ]
    available = [column for column in required if column in comparison]
    if not available:
        return
    _prepare_style()
    figure, axes = plt.subplots(1, len(available), figsize=(4.2 * len(available), 4.5))
    axes_array = np.atleast_1d(axes)
    for axis, metric in zip(axes_array, available):
        values = comparison.set_index("experiment")[metric].sort_values(ascending=False)
        axis.bar(values.index, values.values, color=COLORS["strategy"])
        axis.axhline(0.0, color="#333333", linewidth=0.8)
        axis.set_title(metric.replace("_", " ").title(), fontsize=10)
        axis.tick_params(axis="x", rotation=55)
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)


def plot_significance(significance: pd.DataFrame, output_path: str | Path) -> None:
    if significance.empty:
        return
    _prepare_style()
    frame = significance.sort_values("mean_rank_ic_difference").reset_index(drop=True)
    positions = np.arange(len(frame))
    lower_error = frame["mean_rank_ic_difference"] - frame["ci_lower"]
    upper_error = frame["ci_upper"] - frame["mean_rank_ic_difference"]
    figure, axis = plt.subplots(figsize=(8, max(3.5, len(frame) * 0.55)))
    axis.errorbar(
        frame["mean_rank_ic_difference"],
        positions,
        xerr=np.vstack((lower_error, upper_error)),
        fmt="o",
        color=COLORS["strategy"],
        ecolor=COLORS["benchmark"],
        capsize=4,
    )
    axis.axvline(0.0, color=COLORS["excess"], linewidth=1.0)
    axis.set_yticks(positions, frame["experiment"])
    axis.set(
        title="Paired Block Bootstrap: Rank IC Difference",
        xlabel="Mean Rank IC difference vs. reference (95% CI)",
        ylabel="Experiment",
    )
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)


def flatten_summary(prefix: str, payload: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in payload.items():
        name = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            output.update(flatten_summary(name, value))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            output[name] = value
    return output
