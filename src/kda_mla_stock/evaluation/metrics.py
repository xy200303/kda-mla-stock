from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _valid_correlation(left: pd.Series, right: pd.Series) -> float:
    if len(left) < 2 or left.nunique() < 2 or right.nunique() < 2:
        return float("nan")
    return float(left.corr(right))


def daily_information_coefficients(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "prediction", "target"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions are missing columns: {', '.join(sorted(missing))}")
    rows = []
    for date, group in predictions.groupby("date", sort=True, observed=True):
        clean = group.loc[:, ["prediction", "target"]].dropna()
        ic = _valid_correlation(clean["prediction"], clean["target"])
        rank_ic = _valid_correlation(
            clean["prediction"].rank(method="average"),
            clean["target"].rank(method="average"),
        )
        rows.append({"date": date, "ic": ic, "rank_ic": rank_ic, "count": len(clean)})
    return pd.DataFrame(rows, columns=["date", "ic", "rank_ic", "count"])


def evaluate_predictions(
    predictions: pd.DataFrame,
    periods_per_year: float = 252.0,
) -> dict[str, float | int]:
    clean = predictions.loc[:, ["date", "prediction", "target"]].dropna()
    if clean.empty:
        raise ValueError("no finite predictions are available for evaluation")
    error = clean["prediction"] - clean["target"]
    daily_ic = daily_information_coefficients(clean)
    valid_rank_ic = daily_ic["rank_ic"].dropna()
    valid_ic = daily_ic["ic"].dropna()
    rank_ic_std = float(valid_rank_ic.std(ddof=1)) if len(valid_rank_ic) > 1 else float("nan")
    rank_ic_mean = float(valid_rank_ic.mean()) if len(valid_rank_ic) else float("nan")
    icir = (
        rank_ic_mean / rank_ic_std * math.sqrt(periods_per_year)
        if np.isfinite(rank_ic_std) and rank_ic_std > 0
        else float("nan")
    )
    return {
        "samples": int(len(clean)),
        "dates": int(clean["date"].nunique()),
        "mse": float(np.mean(np.square(error))),
        "mae": float(np.mean(np.abs(error))),
        "direction_accuracy": float(
            np.mean(np.sign(clean["prediction"]) == np.sign(clean["target"]))
        ),
        "ic_mean": float(valid_ic.mean()) if len(valid_ic) else float("nan"),
        "rank_ic_mean": rank_ic_mean,
        "rank_ic_std": rank_ic_std,
        "rank_icir_annualized": icir,
    }
