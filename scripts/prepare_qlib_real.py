from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from kda_mla_stock.core.config import TrainingConfig
from kda_mla_stock.data.qlib import export_qlib_market


def _automatic_boundaries(
    dates: pd.Series,
    validation_days: int,
    test_days: int,
) -> tuple[str, str]:
    unique_dates = pd.DatetimeIndex(pd.to_datetime(dates).dropna().unique()).sort_values()
    minimum_days = validation_days + test_days + 252
    if len(unique_dates) < minimum_days:
        raise ValueError(
            f"the exported market has {len(unique_dates)} trading days; at least "
            f"{minimum_days} are required for the requested splits"
        )
    train_end = unique_dates[-(validation_days + test_days + 1)]
    valid_end = unique_dates[-(test_days + 1)]
    return train_end.strftime("%Y-%m-%d"), valid_end.strftime("%Y-%m-%d")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download real Qlib data, export a market, and create time splits"
    )
    parser.add_argument("--provider-uri", default="~/.qlib/qlib_data/cn_data")
    parser.add_argument("--market", default="csi300")
    parser.add_argument("--region", choices=["cn", "us"], default="cn")
    parser.add_argument("--output", default="data/qlib-csi300.csv")
    parser.add_argument("--base-config", default="configs/train.json")
    parser.add_argument("--train-config-output", default="data/train-real.json")
    parser.add_argument("--validation-days", type=int, default=252)
    parser.add_argument("--test-days", type=int, default=252)
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Replace an existing Qlib data directory with a fresh download",
    )
    args = parser.parse_args()

    try:
        from qlib.tests.data import GetData
    except ImportError as error:
        raise SystemExit("install the Qlib extra first: pip install -e '.[qlib]'") from error

    provider_uri = Path(args.provider_uri).expanduser()
    downloader = GetData(delete_zip_file=True)
    downloader.qlib_data(
        name="qlib_data",
        target_dir=str(provider_uri),
        interval="1d",
        region=args.region,
        delete_old=args.force_download,
        exists_skip=not args.force_download,
    )
    frame = export_qlib_market(provider_uri, args.market, args.region)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

    train_end, valid_end = _automatic_boundaries(
        frame["date"],
        args.validation_days,
        args.test_days,
    )
    base_config = TrainingConfig.from_json(args.base_config)
    real_config = replace(
        base_config,
        data_path=str(output),
        train_end=train_end,
        valid_end=valid_end,
    )
    real_config.validate()
    real_config.save_json(args.train_config_output)
    date_min = pd.Timestamp(frame["date"].min()).date()
    date_max = pd.Timestamp(frame["date"].max()).date()
    print(
        f"saved {len(frame):,} real rows ({date_min} to {date_max}) to {output}\n"
        f"train_end={train_end}, valid_end={valid_end}\n"
        f"training config: {args.train_config_output}"
    )


if __name__ == "__main__":
    main()
