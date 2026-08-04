from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_market(days: int, symbols: int, start: str, seed: int) -> pd.DataFrame:
    if days < 30 or symbols < 2:
        raise ValueError("days must be at least 30 and symbols must be at least 2")
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=days)
    market = np.zeros(days, dtype=np.float64)
    market_noise = rng.normal(0.0002, 0.007, size=days)
    for index in range(1, days):
        market[index] = 0.08 * market[index - 1] + market_noise[index]

    rows = []
    for symbol_index in range(symbols):
        ticker = f"S{symbol_index:04d}"
        beta = rng.uniform(0.7, 1.3)
        returns = np.zeros(days, dtype=np.float64)
        idiosyncratic = rng.normal(0.0, rng.uniform(0.008, 0.018), size=days)
        for index in range(2, days):
            returns[index] = (
                beta * market[index]
                + 0.12 * returns[index - 1]
                - 0.05 * returns[index - 2]
                + idiosyncratic[index]
            )
        close = rng.uniform(15.0, 120.0) * np.exp(np.cumsum(returns))
        overnight = rng.normal(0.0, 0.003, size=days)
        open_price = np.concatenate(([close[0]], close[:-1])) * np.exp(overnight)
        spread = np.abs(rng.normal(0.006, 0.003, size=days))
        high = np.maximum(open_price, close) * (1.0 + spread)
        low = np.minimum(open_price, close) / (1.0 + spread)
        base_volume = rng.uniform(2e5, 3e6)
        volume = base_volume * np.exp(
            8.0 * np.abs(returns) + rng.normal(0.0, 0.25, size=days)
        )
        rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "symbol": ticker,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume.round().astype(np.int64),
                }
            )
        )
    return pd.concat(rows, ignore_index=True).sort_values(["date", "symbol"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reproducible synthetic OHLCV data")
    parser.add_argument("--output", default="data/synthetic.csv")
    parser.add_argument("--days", type=int, default=1800)
    parser.add_argument("--symbols", type=int, default=64)
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = generate_market(args.days, args.symbols, args.start, args.seed)
    frame.to_csv(output, index=False)
    print(f"saved {len(frame):,} rows for {args.symbols} symbols to {output}")


if __name__ == "__main__":
    main()
