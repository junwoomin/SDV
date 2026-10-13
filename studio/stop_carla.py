"""Stop this user's local CARLA servers before Studio starts (stdlib, Python 3.7).

Use executable identity, never command-line substring matching or process groups.
The procfs path and the signal PID are kept separate for namespaced Linux hosts.
"""
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path


BINARIES = frozenset(("CarlaUE4-Linux-Shipping", "CarlaUnreal-Linux-Shipping"))
WRAPPERS = frozenset(("CarlaUE4.sh", "CarlaUnreal.sh"))
SHELLS = frozenset(("bash", "sh", "dash", "zsh", "ksh"))


@dataclass(frozen=True)
class Process:
    proc_pid: int
    pid: int
    started: int
    uid: int
    executable: str
    script: str = ""


def _status_fields(text):
    return dict(line.split(":", 1) for line in text.splitlines() if ":" in line)


def _stat_fields(text):
    # comm may contain whitespace and ')'; fields after its final ')' are stable.
    return text[text.rfind(")") + 2:].split()


def _wrapper(executable, argv):
    if Path(executable).name not in SHELLS:
        return ""
    args = argv[1:]
    while args and args[0].startswith("-"):
        option = args.pop(0)
        if option == "--":
            break
        # Never match shell snippets, including `bash -c '... CarlaUE4.sh ...'`.
        if "c" in option or option.startswith("--") or "o" in option:
            return ""
    return args[0] if args and Path(args[0]).name in WRAPPERS else ""


def read_process(proc_pid, proc_root=Path("/proc"), uid=None):
    """Return an exact CARLA identity, or None for other users/processes/zombies."""
    proc_root = Path(proc_root)
    path = proc_root / str(proc_pid)
    uid = os.geteuid() if uid is None else uid
    try:
        fields = _status_fields((path / "status").read_text())
        if int(fields["Uid"].split()[0]) != uid:
            return None
        stat = _stat_fields((path / "stat").read_text())
        if stat[0] in ("Z", "X"):
            return None
        argv = [item.decode("utf-8", "surrogateescape") for item in
                (path / "cmdline").read_bytes().split(b"\0") if item]
        try:
            executable = os.readlink(str(path / "exe"))
        except PermissionError:
            if argv and (Path(argv[0]).name in BINARIES or _wrapper(argv[0], argv)):
                raise RuntimeError("CARLA PID {}의 실행파일을 확인할 권한이 없습니다.".format(proc_pid))
            return None
        executable = executable[:-10] if executable.endswith(" (deleted)") else executable
        script = _wrapper(executable, argv)
        if Path(executable).name not in BINARIES and not script:
            return None
        # Only CARLA candidates need namespace metadata; protected unrelated
        # system processes may expose cmdline but deny exe/ns read access.
        if (path / "ns/pid").stat().st_ino != (proc_root / "self/ns/pid").stat().st_ino:
            return None
        pid = int(fields.get("NSpid", str(proc_pid)).split()[-1])
        return Process(int(proc_pid), pid, int(stat[19]), uid, executable, script)
    except (FileNotFoundError, ProcessLookupError):
        return None
    except (KeyError, IndexError, ValueError):
        return None


def find_carla(proc_root=Path("/proc")):
    """Discover same-user CARLA servers, excluding this launcher and ancestors."""
    proc_root = Path(proc_root)
    if not (proc_root / "self/stat").is_file():
        raise RuntimeError("CARLA 자동 종료에는 Linux /proc 파일시스템이 필요합니다.")
    own_stat = (proc_root / "self/stat").read_text()
    protected = {int(own_stat.split(" ", 1)[0])}
    parent = int(_stat_fields(own_stat)[1])
    while parent > 0 and parent not in protected:
        protected.add(parent)
        try:
            parent = int(_stat_fields((proc_root / str(parent) / "stat").read_text())[1])
        except (OSError, ValueError, IndexError):
            break
    result = []
    for path in proc_root.iterdir():
        if path.name.isdigit() and int(path.name) not in protected:
            try:
                process = read_process(int(path.name), proc_root)
            except PermissionError:
                # procfs ownership still distinguishes another user's hidden entry.
                if path.stat().st_uid != os.geteuid():
                    continue
                raise RuntimeError("PID {}를 확인할 권한이 없어 CARLA 자동 종료를 완료할 수 없습니다.".format(path.name))
            if process is not None:
                result.append(process)
    return result


def _same_process(process):
    return read_process(process.proc_pid) == process


def _send(process, sig):
    # Check start time, executable, user and wrapper immediately before signalling.
    if not _same_process(process):
        return False
    try:
        os.kill(process.pid, sig)
    except ProcessLookupError:
        return False
    except PermissionError as exc:
        raise RuntimeError("CARLA PID {}를 종료할 권한이 없습니다: {}".format(process.pid, exc))
    return True


def _wait(processes, timeout):
    deadline = time.monotonic() + timeout
    remaining = [process for process in processes if _same_process(process)]
    while remaining and time.monotonic() < deadline:
        time.sleep(min(.05, max(0, deadline - time.monotonic())))
        remaining = [process for process in remaining if _same_process(process)]
    return remaining


def stop_all(term_timeout=5.0, kill_timeout=2.0, discover=find_carla):
    processes = discover()
    print("CARLA 자동 종료: 현재 사용자 프로세스 {}개를 확인했습니다.".format(len(processes)), flush=True)
    errors = []
    for process in processes:
        print("  종료 요청 · PID {} · {}".format(process.pid, process.script or process.executable), flush=True)
        try:
            _send(process, signal.SIGTERM)
        except RuntimeError as exc:
            errors.append(str(exc))
    remaining = _wait(processes, term_timeout)
    for process in remaining:
        try:
            _send(process, signal.SIGKILL)
        except RuntimeError as exc:
            errors.append(str(exc))
    remaining = _wait(remaining, kill_timeout)
    # A wrapper may have launched a child just as it received TERM. Stop that
    # newly discovered exact CARLA process too, without touching unrelated peers.
    known = set(processes)
    late = [process for process in discover() if process not in known]
    for process in late:
        try:
            _send(process, signal.SIGTERM)
        except RuntimeError as exc:
            errors.append(str(exc))
    late_remaining = _wait(late, term_timeout) if late else []
    for process in late_remaining:
        try:
            _send(process, signal.SIGKILL)
        except RuntimeError as exc:
            errors.append(str(exc))
    remaining += _wait(late_remaining, kill_timeout) if late_remaining else []
    final = discover()
    if errors or remaining or final:
        detail = "; ".join(dict.fromkeys(errors))
        if remaining or final:
            detail += " 남은 CARLA PID: {}. 자동 재시작 서비스를 확인하세요.".format(
                ", ".join(str(pid) for pid in sorted({item.pid for item in remaining + final})))
        raise RuntimeError("CARLA 자동 종료 실패. " + detail)
    count = len(set(processes + late))
    print("CARLA 자동 종료 완료: {}개. 웹 서버 준비를 계속합니다.".format(count), flush=True)
    return count


def main():
    try:
        stop_all()
        return 0
    except (OSError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
