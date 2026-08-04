from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _ticker_frame(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        if ticker in raw.columns.get_level_values(0):
            frame = raw.xs(ticker, axis=1, level=0).copy()
        elif ticker in raw.columns.get_level_values(1):
            frame = raw.xs(ticker, axis=1, level=1).copy()
        else:
            return pd.DataFrame()
    else:
        frame = raw.copy()
    frame = frame.reset_index()
    frame.columns = [str(column).lower().replace(" ", "_") for column in frame.columns]
    if "date" not in frame.columns:
        frame = frame.rename(columns={frame.columns[0]: "date"})
    frame["symbol"] = ticker
    return frame.loc[:, ["date", "symbol", "open", "high", "low", "close", "volume"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download adjusted OHLCV data from Yahoo Finance")
    parser.add_argument("tickers", nargs="+", help="Ticker symbols, for example AAPL MSFT NVDA")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--output", default="data/yfinance.csv")
    args = parser.parse_args()

    try:
        import yfinance as yf
    except ImportError as error:
        raise SystemExit("install the data extra first: pip install -e '.[data]'") from error

    tickers = [ticker.upper() for ticker in args.tickers]
    raw = yf.download(
        tickers=tickers,
        start=args.start,
        end=args.end,
        auto_adjust=True,
        group_by="ticker",
        progress=False,
        threads=True,
    )
    if raw.empty:
        raise SystemExit("Yahoo Finance returned no rows")
    frames = [_ticker_frame(raw, ticker) for ticker in tickers]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise SystemExit("none of the requested tickers returned usable rows")
    output_frame = pd.concat(frames, ignore_index=True).dropna()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output_frame.sort_values(["date", "symbol"]).to_csv(output, index=False)
    print(f"saved {len(output_frame):,} rows for {len(tickers)} symbols to {output}")


if __name__ == "__main__":
    main()
