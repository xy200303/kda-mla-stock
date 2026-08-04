from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kda_mla_stock.configuration import ModelConfig
from kda_mla_stock.modeling import build_model, count_parameters
from kda_mla_stock.reporting import plot_comparison, plot_significance


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _experiment_identity(run_dir: Path) -> tuple[str, str]:
    if run_dir.name.startswith("seed-"):
        return run_dir.parent.name, run_dir.name.removeprefix("seed-")
    return run_dir.name, "single"


def _load_run(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "evaluation_test" / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"evaluation summary not found: {summary_path}")
    summary = _read_json(summary_path)
    model_config = _read_json(run_dir / "model_config.json")
    train_summary = _read_json(run_dir / "train_summary.json")
    parameters = train_summary.get("trainable_parameters")
    if parameters is None:
        parameters = count_parameters(build_model(ModelConfig.from_dict(model_config)))
    experiment, seed = _experiment_identity(run_dir)
    row: dict[str, Any] = {
        "experiment": experiment,
        "seed": seed,
        "architecture": model_config.get("architecture", "kda_mla"),
        "parameters": parameters,
        "best_epoch": train_summary.get("best_epoch"),
    }
    prediction_metrics = summary.get("prediction_metrics", {})
    row.update(prediction_metrics)
    qlib_metrics = summary.get("qlib_backtest", {}).get("risk_metrics", {})
    row.update(qlib_metrics)
    trade_metrics = summary.get("qlib_backtest", {}).get("trade_indicators", {})
    row.update({f"trade_{key}": value for key, value in trade_metrics.items()})
    return row


def _aggregate_runs(runs: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        column
        for column in runs.select_dtypes(include="number").columns
        if column not in {"seed"}
    ]
    rows = []
    for experiment, group in runs.groupby("experiment", sort=False, observed=True):
        row: dict[str, Any] = {
            "experiment": experiment,
            "architecture": group["architecture"].iloc[0],
            "runs": len(group),
        }
        for column in numeric_columns:
            row[column] = group[column].mean()
            row[f"{column}_std"] = group[column].std(ddof=1) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _block_bootstrap_mean(
    differences: list[np.ndarray],
    samples: int,
    block_size: int,
    seed: int = 42,
) -> tuple[float, float, float, float]:
    if samples <= 0 or block_size <= 0:
        raise ValueError("bootstrap samples and block size must be positive")
    if not differences or any(len(values) == 0 for values in differences):
        raise ValueError("bootstrap differences must contain non-empty arrays")
    rng = np.random.default_rng(seed)
    observed = float(np.mean([values.mean() for values in differences]))
    bootstrapped = np.empty(samples, dtype=np.float64)
    null_bootstrapped = np.empty(samples, dtype=np.float64)
    centered_differences = [values - values.mean() for values in differences]
    for sample_index in range(samples):
        seed_means = []
        null_seed_means = []
        for values, centered_values in zip(differences, centered_differences):
            current_block_size = min(block_size, len(values))
            block_count = math.ceil(len(values) / current_block_size)
            maximum_start = len(values) - current_block_size
            starts = rng.integers(0, maximum_start + 1, size=block_count)
            sampled = np.concatenate(
                [values[start : start + current_block_size] for start in starts]
            )[: len(values)]
            seed_means.append(sampled.mean())
            null_sampled = np.concatenate(
                [
                    centered_values[start : start + current_block_size]
                    for start in starts
                ]
            )[: len(centered_values)]
            null_seed_means.append(null_sampled.mean())
        bootstrapped[sample_index] = np.mean(seed_means)
        null_bootstrapped[sample_index] = np.mean(null_seed_means)
    lower, upper = np.quantile(bootstrapped, [0.025, 0.975])
    extreme_count = np.count_nonzero(np.abs(null_bootstrapped) >= abs(observed))
    p_value = (extreme_count + 1.0) / (samples + 1.0)
    return observed, float(lower), float(upper), float(p_value)


def _rank_ic_significance(
    run_dirs: list[Path],
    reference: str,
    bootstrap_samples: int,
    block_size: int,
) -> pd.DataFrame:
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    for run_dir in run_dirs:
        experiment, seed = _experiment_identity(run_dir)
        path = run_dir / "evaluation_test" / "daily_ic.csv"
        if path.exists():
            frame = pd.read_csv(path, usecols=["date", "rank_ic"])
            frame["date"] = pd.to_datetime(frame["date"])
            frames[(experiment, seed)] = frame
    experiments = sorted({experiment for experiment, _seed in frames})
    rows = []
    for experiment in experiments:
        differences = []
        for (current_experiment, seed), frame in frames.items():
            if current_experiment != experiment or (reference, seed) not in frames:
                continue
            reference_frame = frames[(reference, seed)]
            paired = frame.merge(reference_frame, on="date", suffixes=("", "_reference")).dropna()
            if not paired.empty:
                differences.append(
                    (paired["rank_ic"] - paired["rank_ic_reference"]).to_numpy()
                )
        if not differences:
            continue
        observed, lower, upper, p_value = _block_bootstrap_mean(
            differences,
            bootstrap_samples,
            block_size,
        )
        rows.append(
            {
                "experiment": experiment,
                "reference": reference,
                "seeds": len(differences),
                "paired_dates": sum(len(values) for values in differences),
                "mean_rank_ic_difference": observed,
                "ci_lower": lower,
                "ci_upper": upper,
                "two_sided_p_value": p_value,
            }
        )
    return pd.DataFrame(rows)


def _write_markdown(frame: pd.DataFrame, path: Path) -> None:
    preferred = [
        "experiment",
        "parameters",
        "rank_ic_mean",
        "rank_icir_annualized",
        "excess_with_cost_annualized_return",
        "excess_with_cost_information_ratio",
        "excess_with_cost_max_drawdown",
    ]
    columns = [column for column in preferred if column in frame]
    table = frame.loc[:, columns].copy()
    for column in table.select_dtypes(include="number"):
        table[column] = table[column].map(lambda value: f"{value:.6f}")
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in table.to_numpy())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate paper experiment metrics and figures")
    parser.add_argument("--run-dirs", nargs="*", default=None)
    parser.add_argument("--output-root", default="outputs/paper")
    parser.add_argument("--reference", default="kda-mla-fast")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--block-size", type=int, default=20)
    args = parser.parse_args()

    if args.bootstrap_samples <= 0 or args.block_size <= 0:
        parser.error("--bootstrap-samples and --block-size must be positive")

    output_root = Path(args.output_root)
    if args.run_dirs:
        run_dirs = [Path(path) for path in args.run_dirs]
    else:
        summaries = output_root.glob("**/evaluation_test/summary.json")
        run_dirs = sorted(path.parent.parent for path in summaries)
    if not run_dirs:
        raise ValueError("no evaluated experiment directories were found")
    runs = pd.DataFrame(_load_run(path) for path in run_dirs)
    comparison = _aggregate_runs(runs)
    output_root.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output_root / "comparison_runs.csv", index=False)
    comparison.to_csv(output_root / "comparison.csv", index=False)
    _write_markdown(comparison, output_root / "comparison.md")
    plot_comparison(comparison, output_root / "model_comparison.png")
    available_experiments = set(runs["experiment"])
    reference = (
        args.reference
        if args.reference in available_experiments
        else runs["experiment"].iloc[0]
    )
    significance = _rank_ic_significance(
        run_dirs,
        reference,
        args.bootstrap_samples,
        args.block_size,
    )
    significance.to_csv(output_root / "rank_ic_significance.csv", index=False)
    plot_significance(significance, output_root / "rank_ic_significance.png")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
