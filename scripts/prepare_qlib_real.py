from __future__ import annotations

import argparse
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from kda_mla_stock.core.config import TrainingConfig
from kda_mla_stock.data.qlib import export_qlib_market

DEFAULT_ARCHIVE_DIR = Path("data/qlib_archives")


def _find_local_archive(
    archive_dir: str | Path,
    region: str,
    interval: str = "1d",
) -> Path | None:
    directory = Path(archive_dir).expanduser()
    if not directory.is_dir():
        return None

    prefix = f"qlib_data_{region.lower()}_{interval.lower()}_"
    latest = directory / f"{prefix}latest.zip"
    if latest.is_file():
        return latest

    candidates = sorted(
        directory.glob(f"{prefix}*.zip"),
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    return candidates[0] if candidates else None


def _extract_local_archive(
    downloader: Any,
    archive_path: str | Path,
    target_dir: str | Path,
    delete_old: bool,
) -> None:
    archive = Path(archive_path).expanduser()
    destination = Path(target_dir).expanduser()
    if not archive.is_file():
        raise ValueError(f"local Qlib archive does not exist: {archive}")
    if not zipfile.is_zipfile(archive):
        raise ValueError(f"local Qlib archive is not a valid ZIP file: {archive}")

    destination.mkdir(exist_ok=True, parents=True)
    print(f"using local Qlib archive: {archive.resolve()}")
    downloader._unzip(archive, destination, delete_old)


def _prepare_qlib_data(
    downloader: Any,
    provider_uri: str | Path,
    archive_dir: str | Path,
    region: str,
    force_download: bool,
    data_exists: bool,
) -> str:
    destination = Path(provider_uri).expanduser()
    if data_exists and not force_download:
        print(f"Qlib data already exists, reusing: {destination}")
        return "existing"

    archive = _find_local_archive(archive_dir, region)
    if archive is not None:
        _extract_local_archive(
            downloader,
            archive,
            destination,
            delete_old=force_download,
        )
        return "local"

    print("local Qlib archive not found; using the official PyQLib downloader")
    downloader.qlib_data(
        name="qlib_data",
        target_dir=str(destination),
        interval="1d",
        region=region,
        delete_old=force_download,
        exists_skip=not force_download,
    )
    return "official"


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
        help="Replace existing Qlib data using a local archive or official download",
    )
    parser.add_argument(
        "--archive-dir",
        default=str(DEFAULT_ARCHIVE_DIR),
        help=(
            "Directory containing manually downloaded Qlib ZIP files "
            f"(default: {DEFAULT_ARCHIVE_DIR})"
        ),
    )
    args = parser.parse_args()

    try:
        from qlib.tests.data import GetData
        from qlib.utils import exists_qlib_data
    except ImportError as error:
        raise SystemExit("install the Qlib extra first: pip install -e '.[qlib]'") from error

    provider_uri = Path(args.provider_uri).expanduser()
    downloader = GetData(delete_zip_file=True)
    try:
        _prepare_qlib_data(
            downloader,
            provider_uri,
            args.archive_dir,
            args.region,
            args.force_download,
            data_exists=exists_qlib_data(provider_uri),
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

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
