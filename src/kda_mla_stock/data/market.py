from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("date", "symbol", "open", "high", "low", "close", "volume")
FEATURE_COLUMNS = (
    "return_1d",
    "return_5d",
    "return_20d",
    "intraday_return",
    "high_low_range",
    "volume_change_1d",
    "volume_zscore_20d",
    "volatility_20d",
    "close_to_ma5",
    "close_to_ma20",
)


@dataclass
class NormalizationStats:
    feature_names: list[str]
    mean: list[float]
    std: list[float]
    clip: float = 10.0

    def save_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def from_json(cls, path: str | Path) -> NormalizationStats:
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def read_market_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"market CSV is missing required columns: {', '.join(missing)}")
    frame = frame.loc[:, list(REQUIRED_COLUMNS)].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise", utc=True).dt.tz_localize(None)
    frame["symbol"] = frame["symbol"].astype(str).str.strip()
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if frame["symbol"].eq("").any():
        raise ValueError("symbol must not be empty")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("OHLC prices must be positive")
    if (frame["volume"] < 0).any():
        raise ValueError("volume must be non-negative")
    if frame.duplicated(["date", "symbol"]).any():
        raise ValueError("market CSV contains duplicate date/symbol rows")
    return cast(
        pd.DataFrame,
        frame.sort_values(["symbol", "date"], kind="stable").reset_index(drop=True),
    )


def engineer_features(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    output = frame.sort_values(["symbol", "date"], kind="stable").copy()
    grouped = output.groupby("symbol", sort=False, observed=True)

    output["return_1d"] = grouped["close"].pct_change(1, fill_method=None)
    output["return_5d"] = grouped["close"].pct_change(5, fill_method=None)
    output["return_20d"] = grouped["close"].pct_change(20, fill_method=None)
    output["intraday_return"] = output["close"] / output["open"] - 1.0
    output["high_low_range"] = output["high"] / output["low"] - 1.0
    log_volume = np.log1p(output["volume"])
    output["volume_change_1d"] = log_volume.groupby(output["symbol"], sort=False).diff()

    rolling_volume_mean = grouped["volume"].transform(
        lambda values: values.rolling(20, min_periods=20).mean()
    )
    rolling_volume_std = grouped["volume"].transform(
        lambda values: values.rolling(20, min_periods=20).std(ddof=0)
    )
    output["volume_zscore_20d"] = np.where(
        rolling_volume_std > 0.0,
        (output["volume"] - rolling_volume_mean) / rolling_volume_std,
        0.0,
    )
    output["volatility_20d"] = grouped["return_1d"].transform(
        lambda values: values.rolling(20, min_periods=20).std(ddof=0)
    )
    moving_average_5 = grouped["close"].transform(
        lambda values: values.rolling(5, min_periods=5).mean()
    )
    moving_average_20 = grouped["close"].transform(
        lambda values: values.rolling(20, min_periods=20).mean()
    )
    output["close_to_ma5"] = output["close"] / moving_average_5 - 1.0
    output["close_to_ma20"] = output["close"] / moving_average_20 - 1.0

    output["target"] = grouped["close"].shift(-horizon) / output["close"] - 1.0
    output["target_date"] = grouped["date"].shift(-horizon)
    output.loc[:, list(FEATURE_COLUMNS)] = output.loc[:, list(FEATURE_COLUMNS)].replace(
        [np.inf, -np.inf], np.nan
    )
    return output.reset_index(drop=True)


def load_and_engineer_market_data(path: str | Path, horizon: int) -> pd.DataFrame:
    return engineer_features(read_market_csv(path), horizon)


def fit_normalization_stats(
    frame: pd.DataFrame,
    train_end: str | pd.Timestamp,
    feature_names: tuple[str, ...] = FEATURE_COLUMNS,
    clip: float = 10.0,
) -> NormalizationStats:
    cutoff = pd.Timestamp(train_end)
    train_features = frame.loc[frame["date"] <= cutoff, list(feature_names)].replace(
        [np.inf, -np.inf], np.nan
    )
    finite_rows = train_features.notna().all(axis=1)
    train_features = train_features.loc[finite_rows]
    if train_features.empty:
        raise ValueError("no complete training features are available before train_end")
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0, ddof=0).clip(lower=1e-8)
    return NormalizationStats(
        feature_names=list(feature_names),
        mean=mean.astype(float).tolist(),
        std=std.astype(float).tolist(),
        clip=clip,
    )


def apply_normalization(frame: pd.DataFrame, stats: NormalizationStats) -> pd.DataFrame:
    output = frame.copy()
    names = stats.feature_names
    mean = pd.Series(stats.mean, index=names)
    std = pd.Series(stats.std, index=names)
    normalized = (output.loc[:, names] - mean) / std
    output.loc[:, names] = normalized.clip(-stats.clip, stats.clip)
    return output
