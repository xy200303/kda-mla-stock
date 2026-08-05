from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_qlib_market(
    provider_uri: str | Path,
    market: str,
    region: str,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    try:
        import qlib
        from qlib.constant import REG_CN, REG_US
        from qlib.data import D
    except ImportError as error:
        raise RuntimeError("install the Qlib extra first: pip install -e '.[qlib]'") from error

    region_value = REG_CN if region == "cn" else REG_US
    qlib.init(provider_uri=str(Path(provider_uri).expanduser()), region=region_value)
    fields = ["$open", "$high", "$low", "$close", "$volume"]
    frame = D.features(
        D.instruments(market),
        fields,
        start_time=start,
        end_time=end,
        freq="day",
    ).reset_index()
    frame = frame.rename(
        columns={
            "datetime": "date",
            "instrument": "symbol",
            "$open": "open",
            "$high": "high",
            "$low": "low",
            "$close": "close",
            "$volume": "volume",
        }
    )
    columns = ["date", "symbol", "open", "high", "low", "close", "volume"]
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise RuntimeError(f"Qlib export is missing columns: {', '.join(missing)}")
    output = frame.loc[:, columns].dropna(subset=["date", "symbol", "close"])
    return output.sort_values(["date", "symbol"], kind="stable").reset_index(drop=True)
