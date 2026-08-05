from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest

from scripts.prepare_qlib_real import (
    _download_data_resumably,
    _download_with_resume,
    _remote_exists,
    _remote_size,
    _resolve_proxies,
)


def test_remote_size() -> None:
    assert _remote_size("bytes 10-19/100") == 100
    assert _remote_size("bytes */100") == 100
    assert _remote_size(None) is None


def test_system_proxy_is_delegated_to_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://system-proxy.invalid:8080")
    system_proxies, system_mode = _resolve_proxies("https://github.com/file.zip", None)
    explicit_proxies, explicit_mode = _resolve_proxies(
        "https://github.com/file.zip",
        "http://explicit-proxy.invalid:3128",
    )

    assert system_proxies is None
    assert system_mode == "system"
    assert explicit_proxies == {
        "http": "http://explicit-proxy.invalid:3128",
        "https": "http://explicit-proxy.invalid:3128",
    }
    assert explicit_mode == "explicit"


def test_download_resumes_partial_file(tmp_path: Path) -> None:
    payload = bytes(range(256)) * 4096
    partial_size = 123_456
    destination = tmp_path / "dataset.zip.part"
    destination.write_bytes(payload[:partial_size])
    observed_ranges: list[str | None] = []

    class RangeHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            range_header = self.headers.get("Range")
            observed_ranges.append(range_header)
            start = int(range_header.removeprefix("bytes=").removesuffix("-"))
            body = payload[start:]
            self.send_response(206)
            self.send_header("Content-Length", str(len(body)))
            self.send_header(
                "Content-Range",
                f"bytes {start}-{len(payload) - 1}/{len(payload)}",
            )
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        _download_with_resume(
            f"http://{host}:{port}/dataset.zip",
            destination,
            retries=2,
            timeout=5,
        )
    finally:
        server.shutdown()
        thread.join()
        server.server_close()

    assert observed_ranges == [f"bytes={partial_size}-"]
    assert destination.read_bytes() == payload


def test_remote_probe_uses_range_request() -> None:
    observed_ranges: list[str | None] = []

    class ProbeHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            observed_ranges.append(self.headers.get("Range"))
            self.send_response(206)
            self.send_header("Content-Length", "1")
            self.send_header("Content-Range", "bytes 0-0/100")
            self.end_headers()
            self.wfile.write(b"x")

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), ProbeHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        exists = _remote_exists(
            f"http://{host}:{port}/dataset.zip",
            retries=1,
            timeout=5,
            proxy=None,
        )
    finally:
        server.shutdown()
        thread.join()
        server.server_close()

    assert exists is True
    assert observed_ranges == ["bytes=0-0"]


def test_download_data_reuses_legacy_partial_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file_name = "qlib_data_cn_1d_latest.zip"
    legacy = tmp_path / f"20260805120000_{file_name}"
    legacy.write_bytes(b"partial")

    class Downloader:
        delete_zip_file = False

        @staticmethod
        def merge_remote_url(name: str) -> str:
            assert name == file_name
            return "https://example.invalid/dataset.zip"

        @staticmethod
        def _unzip(file_path: Path, target_dir: Path, delete_old: bool) -> None:
            raise AssertionError("this test stops before extraction")

    def stop_after_migration(*args: object, **kwargs: object) -> None:
        partial = tmp_path / f".{file_name}.part"
        assert partial.read_bytes() == b"partial"
        raise AssertionError("migration complete")

    monkeypatch.setattr("scripts.prepare_qlib_real._download_with_resume", stop_after_migration)
    with pytest.raises(AssertionError, match="migration complete"):
        _download_data_resumably(
            Downloader(),
            file_name,
            tmp_path,
            False,
            retries=1,
            timeout=1,
            proxy=None,
        )

    assert not legacy.exists()
