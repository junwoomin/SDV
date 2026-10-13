"""Small local archives and HTTP responses; never download the CARLA payload."""
import io
import os
from pathlib import Path
import stat
import tarfile
import threading
import urllib.error
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from sdv.runtime import install as installer


def archive_at(path, members):
    with tarfile.open(str(path), "w:gz") as archive:
        for name, kind, data, mode in members:
            member = tarfile.TarInfo(name)
            member.mode = mode
            if kind == "file":
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
            else:
                member.type = {"directory": tarfile.DIRTYPE, "symlink": tarfile.SYMTYPE,
                               "hardlink": tarfile.LNKTYPE, "fifo": tarfile.FIFOTYPE}[kind]
                if kind in {"symlink", "hardlink"}:
                    member.linkname = data
                archive.addfile(member)
    return path


@contextmanager
def download_server(payload, mode="normal"):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            requested = self.headers.get("Range")
            requests.append(requested)
            start = int(requested.split("=")[1].split("-")[0]) if requested else 0
            ranged = bool(requested) and mode != "ignore_range"
            if not ranged:
                start = 0
            body = payload[start:]
            self.send_response(206 if ranged else 200)
            self.send_header("Content-Length", str(len(body)))
            if ranged:
                claimed = start + 1 if mode == "wrong_range" else start
                total = len(payload) + 7 if mode == "wrong_total" else len(payload)
                self.send_header("Content-Range", "bytes {}-{}/{}".format(claimed, len(payload)-1, total))
            self.end_headers()
            if mode == "truncated":
                body = body[:max(1, len(body)//2)]
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield "http://127.0.0.1:{}/carla.tar.gz".format(server.server_port), requests
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def test_extract_regular_executable_internal_and_forward_links(tmp_path):
    archive = archive_at(tmp_path / "package.tar.gz", [
        ("./", "directory", b"", 0o755),
        ("./bin", "directory", b"", 0o755),
        ("bin/hard-before", "hardlink", "bin/program", 0o755),
        ("bin/program", "file", b"#!/bin/sh\necho carla\n", 0o755),
        ("lib", "directory", b"", 0o755),
        ("lib/program", "symlink", "../bin/program", 0o777),
        ("bin/hard-after", "hardlink", "bin/program", 0o755),
    ])
    output = tmp_path / "out"
    installer.extract(archive, output, threading.Event(), lambda *args: None)
    executable = output / "bin/program"
    assert executable.read_bytes().startswith(b"#!/bin/sh")
    assert stat.S_IMODE(executable.stat().st_mode) == 0o755
    assert (output / "lib/program").is_symlink()
    assert (output / "lib/program").read_bytes() == executable.read_bytes()
    assert executable.stat().st_ino == (output / "bin/hard-before").stat().st_ino
    assert executable.stat().st_ino == (output / "bin/hard-after").stat().st_ino


@pytest.mark.parametrize("member", [
    ("../escape", "file", b"bad", 0o644),
    ("/absolute", "file", b"bad", 0o644),
    ("C:/absolute", "file", b"bad", 0o644),
    ("link", "symlink", "../escape", 0o777),
    ("link", "symlink", "/tmp/escape", 0o777),
    ("link", "hardlink", "../escape", 0o644),
    ("pipe", "fifo", b"", 0o644),
])
def test_extract_rejects_unsafe_archive_headers(tmp_path, member):
    archive = archive_at(tmp_path / "unsafe.tar.gz", [member])
    with pytest.raises((ValueError, tarfile.TarError)):
        installer.extract(archive, tmp_path / "out", threading.Event(), lambda *args: None)
    assert not (tmp_path / "escape").exists()


def test_extract_cannot_follow_preexisting_external_symlink(tmp_path):
    external = tmp_path / "external"
    external.mkdir()
    output = tmp_path / "out"
    output.mkdir()
    (output / "nested").symlink_to(external, target_is_directory=True)
    archive = archive_at(tmp_path / "unsafe.tar.gz", [("nested/payload", "file", b"bad", 0o644)])
    with pytest.raises(ValueError):
        installer.extract(archive, output, threading.Event(), lambda *args: None)
    assert not (external / "payload").exists()


def test_extraction_replaces_preexisting_hardlink_without_modifying_external_file(tmp_path):
    external = tmp_path / "external"
    external.write_bytes(b"keep external contents")
    output = tmp_path / "out"
    output.mkdir()
    os.link(str(external), str(output / "payload"))
    archive = archive_at(tmp_path / "safe.tar.gz", [("payload", "file", b"new contents", 0o644)])
    installer.extract(archive, output, threading.Event(), lambda *args: None)
    assert external.read_bytes() == b"keep external contents"
    assert (output / "payload").read_bytes() == b"new contents"


def test_extraction_cancel_keeps_archive_without_finishing_file(tmp_path):
    archive = archive_at(tmp_path / "package.tar.gz", [
        ("large", "file", os.urandom(2 * 1024 * 1024), 0o644),
        ("after", "file", b"must not exist", 0o644),
    ])
    cancel = threading.Event()
    with pytest.raises(installer.Cancelled):
        installer.extract(archive, tmp_path / "out", cancel, lambda *args: cancel.set())
    assert archive.is_file()
    assert 0 < (tmp_path / "out/large").stat().st_size < 2 * 1024 * 1024
    assert not (tmp_path / "out/after").exists()


@pytest.mark.parametrize("mode", ["normal", "ignore_range"])
def test_download_resumes_or_safely_restarts(tmp_path, mode):
    payload = b"real archive data" * 100
    destination = tmp_path / "download.part"
    destination.write_bytes(payload[:31])
    with download_server(payload, mode) as (url, requests):
        installer.fetch(url, destination, threading.Event(), lambda *args: None, len(payload))
    assert requests == ["bytes=31-"]
    assert destination.read_bytes() == payload


@pytest.mark.parametrize("mode", ["wrong_range", "wrong_total"])
def test_download_bad_range_does_not_overwrite_partial(tmp_path, mode):
    payload = b"real archive data" * 100
    destination = tmp_path / "download.part"
    destination.write_bytes(payload[:31])
    with download_server(payload, mode) as (url, requests):
        with pytest.raises(IOError):
            installer.fetch(url, destination, threading.Event(), lambda *args: None, len(payload))
    assert destination.read_bytes() == payload[:31]


def test_download_cancel_preserves_resumable_bytes(tmp_path):
    payload = b"0123456789" * 220000
    destination = tmp_path / "download.part"
    cancelled = threading.Event()
    with download_server(payload) as (url, requests):
        with pytest.raises(installer.Cancelled):
            installer.fetch(url, destination, cancelled, lambda *args: cancelled.set(), len(payload))
        partial_size = destination.stat().st_size
        assert 0 < partial_size < len(payload)
        cancelled.clear()
        installer.fetch(url, destination, cancelled, lambda *args: None, len(payload))
    assert requests == [None, "bytes={}-".format(partial_size)]
    assert destination.read_bytes() == payload


def test_download_failure_preserves_partial(tmp_path):
    payload = b"archive" * 100
    destination = tmp_path / "download.part"
    with download_server(payload, "truncated") as (url, requests):
        with pytest.raises(IOError):
            installer.fetch(url, destination, threading.Event(), lambda *args: None, len(payload))
    assert 0 < destination.stat().st_size < len(payload)


def valid_installation(path):
    path.mkdir(parents=True)
    (path / "VERSION").write_text("0.9.15\n")
    (path / "CarlaUE4.sh").write_text("#!/bin/sh\n")
    (path / "CarlaUE4.sh").chmod(0o755)
    binary = path / "CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"carla binary fixture")
    binary.chmod(0o755)


def test_valid_existing_installation_reused_without_download(tmp_path, monkeypatch):
    target = tmp_path / "carla"
    valid_installation(target)
    monkeypatch.setattr(installer, "fetch", lambda *args: pytest.fail("existing installation downloaded again"))
    progress = []
    installer.install(threading.Event(), lambda value, text: progress.append(value), root=target)
    status = installer.installation_status(target)
    assert status["installed"] and status["version"] == "0.9.15"
    assert progress[-1] == 1


def test_nonempty_existing_installation_is_preserved(tmp_path, monkeypatch):
    target = tmp_path / "carla"
    target.mkdir()
    (target / "my-data").write_bytes(b"preserve this")
    monkeypatch.setattr(installer, "fetch", lambda *args: pytest.fail("must validate target before download"))
    with pytest.raises((ValueError, FileExistsError)):
        installer.install(threading.Event(), lambda *args: None, root=target)
    assert (target / "my-data").read_bytes() == b"preserve this"


def test_failed_install_keeps_archive_and_never_publishes_partial_tree(tmp_path, monkeypatch):
    archive = archive_at(tmp_path / "fixture.tar.gz", [("VERSION", "file", b"wrong-version", 0o644)])
    monkeypatch.setattr(installer, "SIZE", archive.stat().st_size)
    monkeypatch.setattr(installer.shutil, "disk_usage", lambda path: type("Usage", (), {"free": 100*2**30})())
    def fetch_fixture(url, destination, cancel, progress, expected):
        destination.write_bytes(archive.read_bytes())
    monkeypatch.setattr(installer, "fetch", fetch_fixture)
    target = tmp_path / "carla"
    with pytest.raises(ValueError):
        installer.install(threading.Event(), lambda *args: None, root=target)
    assert not target.exists()
    assert installer.installation_status(target)["download_bytes"] == archive.stat().st_size


def test_completed_install_publishes_only_verified_executable_distribution(tmp_path, monkeypatch):
    archive = archive_at(tmp_path / "fixture.tar.gz", [
        ("VERSION", "file", b"0.9.15\n", 0o644),
        ("CarlaUE4.sh", "file", b"#!/bin/sh\n", 0o755),
        ("CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping", "file", b"binary fixture", 0o755),
    ])
    monkeypatch.setattr(installer, "SIZE", archive.stat().st_size)
    monkeypatch.setattr(installer.shutil, "disk_usage", lambda path: type("Usage", (), {"free": 100*2**30})())
    target = tmp_path / "carla"
    observed = []
    with download_server(archive.read_bytes()) as (url, requests):
        monkeypatch.setattr(installer, "URL", url)
        installer.install(threading.Event(), lambda value, message: observed.append((value, target.exists())), root=target)
    assert installer.installation_status(target)["installed"]
    assert observed[-1] == (1, True)
    assert all(not visible for progress, visible in observed[:-1])
    assert all(0 <= progress <= 1 for progress, visible in observed)


def test_primary_403_uses_verified_distribution_mirror(tmp_path, monkeypatch):
    archive = archive_at(tmp_path / "fixture.tar.gz", [
        ("VERSION", "file", b"0.9.15\n", 0o644),
        ("CarlaUE4.sh", "file", b"#!/bin/sh\n", 0o755),
        ("CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping", "file", b"binary fixture", 0o755),
    ])
    monkeypatch.setattr(installer, "SIZE", archive.stat().st_size)
    monkeypatch.setattr(installer.shutil, "disk_usage", lambda path: type("Usage", (), {"free": 100*2**30})())
    requests = []
    def fetch_fixture(url, destination, cancel, progress, expected):
        requests.append(url)
        if url == installer.URL:
            raise urllib.error.HTTPError(url, 403, "Forbidden", {}, None)
        destination.write_bytes(archive.read_bytes())
    monkeypatch.setattr(installer, "fetch", fetch_fixture)
    installer.install(threading.Event(), lambda *args: None, root=tmp_path / "carla")
    assert requests == [installer.URL, installer.MIRROR_URL]
    assert installer.installation_status(tmp_path / "carla")["installed"]


def test_default_target_reuses_original_runtime_archive(tmp_path, monkeypatch):
    monkeypatch.setattr(installer, "ROOT", tmp_path)
    monkeypatch.setattr(installer, "RUNTIME", tmp_path / ".runtime")
    archive = tmp_path / ".runtime/downloads/CARLA_0.9.15.tar.gz.part"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"already downloaded")
    status = installer.installation_status(tmp_path / "carla")
    assert status["path"] == str(tmp_path / "carla")
    assert status["download_bytes"] == 18
