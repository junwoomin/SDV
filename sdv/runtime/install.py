"""Cancellable project-local CARLA install, including Python 3.7 support."""
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import urllib.error
import urllib.request

from sdv.paths import ROOT, RUNTIME

URL = "https://downloads.carlasim.com/Linux/CARLA_0.9.15.tar.gz"
MIRROR_URL = "https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz"
SIZE = 8386636048
VERSION = "0.9.15"
REQUIRED_FREE_BYTES = 45 * 2**30
CHUNK_BYTES = 1024 * 1024


class Cancelled(Exception):
    pass


def _cancelled(cancel):
    if cancel.is_set():
        raise Cancelled("설치를 중단했습니다. 다운로드 파일은 이어받기용으로 보관합니다.")


def _paths(root=None):
    target = Path(root).expanduser().resolve() if root is not None else (ROOT / "carla").resolve()
    if target == (ROOT / "carla").resolve():
        # Keep downloads created by the original desktop installer reusable.
        workspace = RUNTIME
        archive = workspace / "downloads/CARLA_0.9.15.tar.gz.part"
        stage = workspace / "carla-extract"
    else:
        workspace = target.parent / ("." + target.name + "-install")
        archive = workspace / "CARLA_0.9.15.tar.gz.part"
        stage = workspace / "extract"
    return target, archive, stage, workspace


def _existing_parent(path):
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def installation_status(root=None):
    target, archive, stage, workspace = _paths(root)
    result = {"installed": False, "path": str(target), "version": None,
              "download_bytes": 0, "total_bytes": SIZE,
              "required_free_bytes": REQUIRED_FREE_BYTES, "free_bytes": None,
              "source_url": URL, "fallback_url": MIRROR_URL, "error": None}
    try:
        result["download_bytes"] = archive.stat().st_size if archive.is_file() else 0
        result["required_free_bytes"] = max(0, REQUIRED_FREE_BYTES - min(SIZE, result["download_bytes"]))
        result["free_bytes"] = shutil.disk_usage(_existing_parent(target.parent)).free
        if not target.exists():
            return result
        if not target.is_dir():
            result["error"] = "CARLA 설치 경로가 폴더가 아닙니다."
            return result
        version_file = target / "VERSION"
        if version_file.is_file():
            result["version"] = version_file.read_text(encoding="utf-8").strip()
        launcher = target / "CarlaUE4.sh"
        binary = target / "CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping"
        result["installed"] = (result["version"] == VERSION and launcher.is_file()
                               and binary.is_file() and os.access(launcher, os.X_OK)
                               and os.access(binary, os.X_OK))
        if not result["installed"] and any(target.iterdir()):
            result["error"] = "기존 CARLA 폴더가 비어 있지 않거나 0.9.15 실행 파일이 불완전합니다. 기존 carla 폴더를 확인하세요. 파일은 덮어쓰지 않습니다."
    except (OSError, UnicodeError) as error:
        result["error"] = str(error)
    return result


def fetch(url, destination, cancel, progress, expected=None):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _cancelled(cancel)
    offset = destination.stat().st_size if destination.exists() else 0
    if expected is not None and offset > expected:
        raise IOError("기존 다운로드가 배포 크기보다 큽니다. 파일을 확인한 뒤 다시 시도하세요.")
    if expected is not None and offset == expected:
        progress(1.0, "기존 다운로드 파일을 확인합니다.")
        return
    headers = {"Range": "bytes={}-".format(offset)} if offset else {}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=15) as response:
        status = response.getcode()
        length_header = response.headers.get("Content-Length")
        length = int(length_header) if length_header is not None else None
        if status == 206:
            matched = re.fullmatch(r"bytes ([0-9]+)-([0-9]+)/([0-9]+)", response.headers.get("Content-Range", ""))
            if not matched:
                raise IOError("이어받기 응답에 올바른 Content-Range가 없습니다.")
            start, end, total = map(int, matched.groups())
            if start != offset or end < start or end >= total or end != total - 1:
                raise IOError("이어받기 응답 범위가 일치하지 않습니다. 기존 다운로드는 보존했습니다.")
            if length is not None and length != end - start + 1:
                raise IOError("이어받기 응답 길이가 일치하지 않습니다.")
            if expected is not None and total != expected:
                raise IOError("이어받기 응답의 배포 크기가 다릅니다. 기존 다운로드는 보존했습니다.")
        elif status == 200:
            # A server may ignore Range. Validate before replacing the partial file.
            if expected is not None and length is not None and length != expected:
                raise IOError("서버의 배포 크기가 예상과 다릅니다. 기존 다운로드는 보존했습니다.")
            offset = 0
            total = expected if expected is not None else length
        else:
            raise IOError("지원하지 않는 다운로드 응답: HTTP {}".format(status))
        _cancelled(cancel)
        with destination.open("ab" if offset else "wb") as output:
            while True:
                _cancelled(cancel)
                chunk = response.read(CHUNK_BYTES)
                if not chunk:
                    break
                if total is not None and offset + len(chunk) > total:
                    raise IOError("다운로드 응답이 배포 크기를 초과했습니다.")
                output.write(chunk)
                offset += len(chunk)
                fraction = offset / total if total else 0.0
                progress(min(1.0, fraction), "기본 CARLA 다운로드 {:.2f} / {:.2f} GiB".format(offset/2**30, (total or 0)/2**30))
        _cancelled(cancel)
        if total is not None and offset != total:
            raise IOError("다운로드가 중간에 끝났습니다. 다시 시도하면 이어받습니다.")


def _inside(root, path):
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError("압축 파일에 설치 폴더를 벗어나는 경로가 있습니다: " + str(path))
    return resolved


def _member_path(root, name):
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or ".." in path.parts or "\\" in name
            or re.match(r"^[A-Za-z]:", name)):
        raise ValueError("압축 파일의 경로를 허용할 수 없습니다: " + name)
    target = root.joinpath(*path.parts)
    _inside(root, target)
    return target


def _link_target(root, target, member):
    link = member.linkname
    path = PurePosixPath(link)
    if not link or path.is_absolute() or "\\" in link or re.match(r"^[A-Za-z]:", link):
        raise ValueError("압축 파일의 링크 경로를 허용할 수 없습니다: " + link)
    origin = target.parent if member.issym() else root
    return _inside(root, origin.joinpath(*path.parts))


def extract(archive, destination, cancel, progress):
    """Stream trusted distribution data through explicit path/type validation.

    Python 3.7 has no tarfile.data_filter. The same checks are used on every
    supported Python; tarfile never writes files or creates links itself.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    pending_links, directory_modes = [], []
    compressed_size = max(1, Path(archive).stat().st_size)
    with open(str(archive), "rb") as source:
        with tarfile.open(fileobj=source, mode="r|*") as tar:
            for number, member in enumerate(tar, 1):
                _cancelled(cancel)
                target = _member_path(root, member.name)
                if not (member.isdir() or member.isreg() or member.issym() or member.islnk()):
                    raise ValueError("압축 파일의 특수 파일은 허용하지 않습니다: " + member.name)
                if member.issym() or member.islnk():
                    link_target = _link_target(root, target, member)
                if target == root:
                    if not member.isdir():
                        raise ValueError("압축 루트 경로에는 디렉터리만 올 수 있습니다.")
                else:
                    _inside(root, target.parent)
                    target.parent.mkdir(parents=True, exist_ok=True)
                if member.isdir():
                    if target.is_symlink():
                        raise ValueError("폴더를 링크 위에 덮어쓸 수 없습니다: " + member.name)
                    target.mkdir(exist_ok=True)
                    directory_modes.append((target, member.mode & 0o777))
                elif member.isreg():
                    if target.is_symlink():
                        raise ValueError("파일을 링크 위에 덮어쓸 수 없습니다: " + member.name)
                    # An existing regular file could itself be a hard link to a
                    # file outside the extraction tree. Replace its entry first.
                    if target.is_file():
                        target.unlink()
                    stream = tar.extractfile(member)
                    if stream is None:
                        raise ValueError("압축 파일 데이터를 읽을 수 없습니다: " + member.name)
                    with stream, target.open("wb") as output:
                        while True:
                            _cancelled(cancel)
                            block = stream.read(CHUNK_BYTES)
                            if not block:
                                break
                            output.write(block)
                            progress(min(0.999, source.tell()/compressed_size), "압축 해제: " + member.name)
                    target.chmod(member.mode & 0o777)
                elif member.issym():
                    if target.exists() or target.is_symlink():
                        raise ValueError("기존 압축 파일 경로를 링크로 덮어쓸 수 없습니다: " + member.name)
                    target.symlink_to(member.linkname)
                else:
                    # Tar hard links may refer to a regular file appearing later.
                    pending_links.append((target, member.linkname))
                progress(min(0.999, source.tell()/compressed_size), "압축 해제 {}개 파일: {}".format(number, member.name))
    while pending_links:
        remaining = []
        for target, linkname in pending_links:
            _cancelled(cancel)
            _inside(root, target)
            link_target = _inside(root, root.joinpath(*PurePosixPath(linkname).parts))
            if not link_target.is_file():
                remaining.append((target, linkname))
                continue
            if target.exists() or target.is_symlink():
                raise ValueError("하드 링크 경로를 덮어쓸 수 없습니다: " + str(target))
            os.link(str(link_target), str(target))
        if len(remaining) == len(pending_links):
            raise ValueError("압축 파일의 하드 링크 대상이 없습니다.")
        pending_links = remaining
    for target, mode in sorted(directory_modes, key=lambda item: len(item[0].parts), reverse=True):
        _inside(root, target)
        target.chmod(mode)
    _cancelled(cancel)
    progress(1.0, "압축 해제 완료")


def install(cancel, progress, root=None):
    target, archive, stage, workspace = _paths(root)
    status = installation_status(target)
    if status["installed"]:
        progress(1.0, "기존 CARLA 0.9.15 설치를 사용합니다.")
        return
    if status["error"]:
        raise ValueError(status["error"])
    _cancelled(cancel)
    if status["free_bytes"] is None or status["free_bytes"] < status["required_free_bytes"]:
        raise OSError("CARLA 설치에는 약 {:.1f} GiB의 추가 여유 공간이 필요합니다.".format(status["required_free_bytes"]/2**30))
    workspace.mkdir(parents=True, exist_ok=True)
    # Prevent standalone desktop and web callers from mutating one download.
    import fcntl
    with (workspace / "carla-install.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("같은 경로에 CARLA 설치가 이미 진행 중입니다.")
        archive.parent.mkdir(parents=True, exist_ok=True)
        download_progress = lambda value, text: progress(value * 0.65, text)
        try:
            fetch(URL, archive, cancel, download_progress, SIZE)
        except urllib.error.URLError as error:
            if isinstance(error, urllib.error.HTTPError) and error.code not in (403, 404, 500, 502, 503, 504):
                raise
            _cancelled(cancel)
            progress(0.65 * min(1, archive.stat().st_size / SIZE) if archive.exists() else 0,
                     "주 다운로드 주소에 연결할 수 없어 CARLA 공식 배포 미러에서 이어받습니다.")
            fetch(MIRROR_URL, archive, cancel, download_progress, SIZE)
        _cancelled(cancel)
        if stage.is_symlink():
            raise ValueError("임시 설치 폴더가 링크입니다. 경로를 확인하세요.")
        if stage.exists():
            shutil.rmtree(str(stage))
        stage.mkdir(parents=True)
        extract(archive, stage, cancel, lambda value, text: progress(0.65 + value * 0.34, text))
        verified = installation_status(stage)
        if not verified["installed"]:
            raise ValueError("배포 버전 또는 실행 파일이 올바르지 않습니다. 다운로드 파일을 확인하세요.")
        _cancelled(cancel)
        if target.exists():
            if not target.is_dir() or any(target.iterdir()):
                raise FileExistsError("기존 CARLA 폴더를 덮어쓰지 않습니다. 새 설치 경로를 선택하세요.")
            target.rmdir()
        target.parent.mkdir(parents=True, exist_ok=True)
        stage.rename(target)
        progress(1.0, "CARLA 0.9.15 설치 완료. 서버를 시작할 수 있습니다.")
