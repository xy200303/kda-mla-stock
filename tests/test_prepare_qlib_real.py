from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pytest

from scripts.prepare_qlib_real import (
    _extract_local_archive,
    _find_local_archive,
    _prepare_qlib_data,
)


class RecordingDownloader:
    def __init__(self) -> None:
        self.unzip_calls: list[tuple[Path, Path, bool]] = []
        self.download_calls: list[dict[str, object]] = []

    def _unzip(
        self,
        archive: str | Path,
        destination: str | Path,
        delete_old: bool,
    ) -> None:
        self.unzip_calls.append((Path(archive), Path(destination), delete_old))

    def qlib_data(self, **kwargs: object) -> None:
        self.download_calls.append(kwargs)


def _write_qlib_archive(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w") as archive:
        archive.writestr("calendars/day.txt", "2026-08-05")


def test_find_local_archive_prefers_latest_package(tmp_path: Path) -> None:
    archive_dir = tmp_path / "archives"
    archive_dir.mkdir()
    versioned = archive_dir / "qlib_data_cn_1d_0.9.7.zip"
    latest = archive_dir / "qlib_data_cn_1d_latest.zip"
    versioned.write_bytes(b"versioned")

    assert _find_local_archive(archive_dir, "cn") == versioned

    latest.write_bytes(b"latest")
    assert _find_local_archive(archive_dir, "cn") == latest
    assert _find_local_archive(archive_dir, "us") is None


def test_extract_local_archive_uses_official_unzip(tmp_path: Path) -> None:
    archive = tmp_path / "qlib_data_cn_1d_latest.zip"
    destination = tmp_path / "provider"
    _write_qlib_archive(archive)
    downloader = RecordingDownloader()

    _extract_local_archive(downloader, archive, destination, delete_old=True)

    assert downloader.unzip_calls == [(archive, destination, True)]
    assert downloader.download_calls == []


def test_extract_local_archive_rejects_invalid_zip(tmp_path: Path) -> None:
    archive = tmp_path / "qlib_data_cn_1d_latest.zip"
    archive.write_bytes(b"not a zip")

    with pytest.raises(ValueError, match="not a valid ZIP"):
        _extract_local_archive(RecordingDownloader(), archive, tmp_path / "provider", False)


def test_prepare_qlib_data_prefers_local_archive(tmp_path: Path) -> None:
    archive_dir = tmp_path / "archives"
    archive = archive_dir / "qlib_data_cn_1d_latest.zip"
    destination = tmp_path / "provider"
    _write_qlib_archive(archive)
    downloader = RecordingDownloader()

    source = _prepare_qlib_data(
        downloader,
        destination,
        archive_dir,
        region="cn",
        force_download=True,
        data_exists=True,
    )

    assert source == "local"
    assert downloader.unzip_calls == [(archive, destination, True)]
    assert downloader.download_calls == []


def test_prepare_qlib_data_falls_back_to_official_downloader(tmp_path: Path) -> None:
    destination = tmp_path / "provider"
    downloader = RecordingDownloader()

    source = _prepare_qlib_data(
        downloader,
        destination,
        tmp_path / "missing-archives",
        region="cn",
        force_download=False,
        data_exists=False,
    )

    assert source == "official"
    assert downloader.unzip_calls == []
    assert downloader.download_calls == [
        {
            "name": "qlib_data",
            "target_dir": str(destination),
            "interval": "1d",
            "region": "cn",
            "delete_old": False,
            "exists_skip": True,
        }
    ]


def test_prepare_qlib_data_reuses_existing_provider(tmp_path: Path) -> None:
    downloader = RecordingDownloader()

    source = _prepare_qlib_data(
        downloader,
        tmp_path / "provider",
        tmp_path / "archives",
        region="cn",
        force_download=False,
        data_exists=True,
    )

    assert source == "existing"
    assert downloader.unzip_calls == []
    assert downloader.download_calls == []
