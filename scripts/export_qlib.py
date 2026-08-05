from __future__ import annotations

import argparse
from pathlib import Path

from kda_mla_stock.data.qlib import export_qlib_market


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a Qlib market to the standard OHLCV CSV")
    parser.add_argument("--provider-uri", required=True, help="Qlib binary data directory")
    parser.add_argument("--market", default="csi300")
    parser.add_argument("--region", choices=["cn", "us"], default="cn")
    parser.add_argument("--start", default="2008-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--output", default="data/qlib-market.csv")
    args = parser.parse_args()

    frame = export_qlib_market(
        args.provider_uri,
        args.market,
        args.region,
        args.start,
        args.end,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"saved {len(frame):,} Qlib rows to {output}")


if __name__ == "__main__":
    main()
