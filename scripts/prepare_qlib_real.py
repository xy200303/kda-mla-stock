from __future__ import annotations

import argparse
import os
import time
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pandas as pd

from kda_mla_stock.core.config import TrainingConfig
from kda_mla_stock.data.qlib import export_qlib_market

DEFAULT_GITHUB_MIRROR = "https://gh-proxy.com"


def _apply_github_mirror(official_url: str, mirror: str | None) -> str:
    if mirror is None:
        return official_url
    normalized = mirror.strip().rstrip("/")
    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("GitHub mirror must be an absolute HTTP(S) URL")
    return f"{normalized}/{official_url}"


def _remote_size(content_range: str | None) -> int | None:
    if not content_range or "/" not in content_range:
        return None
    total = content_range.rsplit("/", maxsplit=1)[-1]
    return int(total) if total.isdigit() else None


def _resolve_proxies(url: str, proxy: str | None) -> tuple[dict[str, str] | None, str]:
    from requests.utils import get_environ_proxies

    if proxy:
        return {"http": proxy, "https": proxy}, "explicit"
    return None, "system" if get_environ_proxies(url) else "direct"


def _remote_exists(
    url: str,
    *,
    retries: int,
    timeout: float,
    proxy: str | None,
) -> bool:
    import requests

    proxies, proxy_mode = _resolve_proxies(url, proxy)
    for attempt in range(1, retries + 1):
        try:
            with requests.get(
                url,
                headers={"Range": "bytes=0-0"},
                proxies=proxies,
                stream=True,
                timeout=timeout,
            ) as response:
                if response.status_code == 404:
                    return False
                response.raise_for_status()
                print(f"Qlib dataset probe succeeded (proxy={proxy_mode})")
                return True
        except requests.RequestException as error:
            if attempt == retries:
                raise RuntimeError(f"Qlib dataset probe failed after {retries} attempts") from error
            delay = min(30, 2 ** (attempt - 1))
            print(f"dataset probe interrupted ({error}); retrying in {delay}s")
            time.sleep(delay)
    return False


def _download_with_resume(
    url: str,
    target_path: str | Path,
    *,
    retries: int,
    timeout: float,
    proxy: str | None = None,
    chunk_size: int = 1024 * 1024,
) -> None:
    import requests
    from tqdm import tqdm

    destination = Path(target_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    proxies, proxy_mode = _resolve_proxies(url, proxy)

    for attempt in range(1, retries + 1):
        downloaded = destination.stat().st_size if destination.exists() else 0
        headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}
        try:
            with requests.get(
                url,
                headers=headers,
                proxies=proxies,
                stream=True,
                timeout=timeout,
            ) as response:
                if response.status_code == 416:
                    total = _remote_size(response.headers.get("Content-Range"))
                    if total is not None and downloaded == total:
                        return
                    destination.write_bytes(b"")
                    raise requests.exceptions.RequestException(
                        "the remote file changed while resuming; restarting the download"
                    )

                response.raise_for_status()
                resumed = downloaded > 0 and response.status_code == 206
                if resumed:
                    content_range = response.headers.get("Content-Range")
                    if not content_range or not content_range.startswith(f"bytes {downloaded}-"):
                        destination.write_bytes(b"")
                        raise requests.exceptions.RequestException(
                            "the server returned an invalid byte range; restarting"
                        )
                    mode = "ab"
                    total = _remote_size(content_range)
                    if total is None:
                        remaining = int(response.headers.get("Content-Length", 0))
                        total = downloaded + remaining if remaining else None
                    initial = downloaded
                else:
                    mode = "wb"
                    total_header = int(response.headers.get("Content-Length", 0))
                    total = total_header or None
                    initial = 0

                action = "resuming" if resumed else "downloading"
                print(
                    f"{action} {destination.name} "
                    f"(attempt {attempt}/{retries}, proxy={proxy_mode}, "
                    f"{initial:,} bytes already saved)"
                )
                with tqdm(
                    total=total,
                    initial=initial,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=destination.name,
                ) as progress:
                    with destination.open(mode) as output_file:
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if not chunk:
                                continue
                            output_file.write(chunk)
                            progress.update(len(chunk))

                actual_size = destination.stat().st_size
                if total is not None and actual_size != total:
                    raise requests.exceptions.ChunkedEncodingError(
                        f"incomplete download: received {actual_size:,} of {total:,} bytes"
                    )
                return
        except (OSError, requests.RequestException) as error:
            if attempt == retries:
                raise RuntimeError(
                    f"Qlib data download failed after {retries} attempts; "
                    f"keep {destination} and rerun the command to resume"
                ) from error
            delay = min(30, 2 ** (attempt - 1))
            saved = destination.stat().st_size if destination.exists() else 0
            print(f"download interrupted ({error}); retrying in {delay}s from {saved:,} bytes")
            time.sleep(delay)


def _download_data_resumably(
    downloader: Any,
    file_name: str,
    target_dir: str | Path,
    delete_old: bool,
    *,
    retries: int,
    timeout: float,
    proxy: str | None,
) -> None:
    destination_dir = Path(target_dir).expanduser()
    destination_dir.mkdir(exist_ok=True, parents=True)
    partial_path = destination_dir / f".{os.path.basename(file_name)}.part"
    if not partial_path.exists():
        legacy_files = sorted(
            destination_dir.glob(f"*_{os.path.basename(file_name)}"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if legacy_files:
            legacy_files[0].replace(partial_path)
            print(f"reusing interrupted Qlib download: {partial_path}")

    url = downloader.merge_remote_url(file_name)
    _download_with_resume(
        url,
        partial_path,
        retries=retries,
        timeout=timeout,
        proxy=proxy,
    )
    try:
        with zipfile.ZipFile(partial_path) as archive:
            archive.infolist()
    except zipfile.BadZipFile:
        print("downloaded archive is invalid; restarting it once from byte 0")
        partial_path.write_bytes(b"")
        _download_with_resume(
            url,
            partial_path,
            retries=retries,
            timeout=timeout,
            proxy=proxy,
        )
        with zipfile.ZipFile(partial_path) as archive:
            archive.infolist()

    downloader._unzip(partial_path, destination_dir, delete_old)
    if downloader.delete_zip_file:
        partial_path.unlink(missing_ok=True)


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
    parser.add_argument(
        "--download-retries",
        type=int,
        default=8,
        help="Maximum download attempts; interrupted attempts resume automatically",
    )
    parser.add_argument(
        "--download-timeout",
        type=float,
        default=120.0,
        help="HTTP connection and read timeout in seconds",
    )
    parser.add_argument(
        "--proxy",
        default=None,
        help="Override the system HTTP(S) proxy for the Qlib download",
    )
    parser.add_argument(
        "--github-mirror",
        default=DEFAULT_GITHUB_MIRROR,
        help=f"GitHub URL mirror (default: {DEFAULT_GITHUB_MIRROR})",
    )
    parser.add_argument(
        "--no-github-mirror",
        action="store_true",
        help="Download directly from the official GitHub Release URL",
    )
    args = parser.parse_args()

    if args.download_retries < 1:
        parser.error("--download-retries must be at least 1")
    if args.download_timeout <= 0:
        parser.error("--download-timeout must be positive")
    github_mirror = None if args.no_github_mirror else args.github_mirror
    try:
        _apply_github_mirror("https://github.com/example/release.zip", github_mirror)
    except ValueError as error:
        parser.error(str(error))

    try:
        from qlib.tests.data import GetData
    except ImportError as error:
        raise SystemExit("install the Qlib extra first: pip install -e '.[qlib]'") from error

    provider_uri = Path(args.provider_uri).expanduser()
    print(f"Qlib download route: {'official GitHub' if github_mirror is None else 'mirror'}")

    class ResumableGetData(GetData):
        def merge_remote_url(self, file_name: str) -> str:
            official_url = super().merge_remote_url(file_name)
            return _apply_github_mirror(official_url, github_mirror)

        def check_dataset(self, file_name: str) -> bool:
            return _remote_exists(
                self.merge_remote_url(file_name),
                retries=args.download_retries,
                timeout=args.download_timeout,
                proxy=args.proxy,
            )

        def download_data(
            self,
            file_name: str,
            target_dir: str | Path,
            delete_old: bool = True,
        ) -> None:
            _download_data_resumably(
                self,
                file_name,
                target_dir,
                delete_old,
                retries=args.download_retries,
                timeout=args.download_timeout,
                proxy=args.proxy,
            )

    downloader = ResumableGetData(delete_zip_file=True)
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
