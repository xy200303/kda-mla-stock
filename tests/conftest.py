from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def market_frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2020-01-01", periods=100)
    frames = []
    for symbol_index in range(5):
        returns = rng.normal(0.0003, 0.012, size=len(dates))
        returns[1:] += 0.1 * returns[:-1]
        close = (30.0 + symbol_index * 5.0) * np.exp(np.cumsum(returns))
        open_price = np.concatenate(([close[0]], close[:-1])) * np.exp(
            rng.normal(0.0, 0.002, size=len(dates))
        )
        spread = np.abs(rng.normal(0.006, 0.002, size=len(dates)))
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "symbol": f"S{symbol_index:02d}",
                    "open": open_price,
                    "high": np.maximum(open_price, close) * (1.0 + spread),
                    "low": np.minimum(open_price, close) / (1.0 + spread),
                    "close": close,
                    "volume": rng.integers(100_000, 1_000_000, size=len(dates)),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)
