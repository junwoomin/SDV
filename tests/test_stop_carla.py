"""Controlled Linux child fixtures; no test signals unrelated CARLA processes."""
import os
import shutil
import signal
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import pytest

from studio import stop_carla


@pytest.fixture
def children(tmp_path):
    spawned = []

    def start(name, ignore_term=False):
        executable = tmp_path / name
        shutil.copy2(shutil.which("sleep"), str(executable))
        process = subprocess.Popen(
            [str(executable), "60"],
            preexec_fn=(lambda: signal.signal(signal.SIGTERM, signal.SIG_IGN)) if ignore_term else None,
        )
        spawned.append(process)
        return process

    yield start
    for process in spawned:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)


def fixture_identity(child):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        identity = next((item for item in stop_carla.find_carla() if item.pid == child.pid), None)
        if identity:
            return identity
        time.sleep(.01)
    pytest.fail("Controlled CARLA-named child was not discovered")


def test_exact_executable_termination_preserves_unrelated_process(children, capsys):
    carla = children("CarlaUE4-Linux-Shipping")
    unrelated = children("not-CarlaUE4-Linux-Shipping")
    identity = fixture_identity(carla)
    assert all(item.pid != unrelated.pid for item in stop_carla.find_carla())
    # Restrict the mutation to the fixture even if a developer has real CARLA open.
    discover = lambda: [identity] if stop_carla._same_process(identity) else []
    assert stop_carla.stop_all(.3, .3, discover=discover) == 1
    assert carla.wait(timeout=3) == -signal.SIGTERM
    assert unrelated.poll() is None
    assert "1개" in capsys.readouterr().out


def test_unresponsive_fixture_escalates_to_kill(children):
    carla = children("CarlaUnreal-Linux-Shipping", ignore_term=True)
    identity = fixture_identity(carla)
    discover = lambda: [identity] if stop_carla._same_process(identity) else []
    assert stop_carla.stop_all(.05, .3, discover=discover) == 1
    assert carla.wait(timeout=3) == -signal.SIGKILL


def test_reused_pid_identity_never_receives_signal(monkeypatch):
    original = stop_carla.Process(100, 100, 10, os.geteuid(), "/fixture/CarlaUE4-Linux-Shipping")
    replacement = replace(original, started=11)
    sent = []
    monkeypatch.setattr(stop_carla, "read_process", lambda pid: replacement)
    monkeypatch.setattr(stop_carla.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    assert stop_carla._send(original, signal.SIGTERM) is False
    assert sent == []


def test_permission_denial_is_explicit(monkeypatch):
    fixture = stop_carla.Process(100, 100, 10, os.geteuid(), "/fixture/CarlaUE4-Linux-Shipping")
    monkeypatch.setattr(stop_carla, "_same_process", lambda process: True)
    def denied(pid, sig):
        raise PermissionError("controlled test denial")
    monkeypatch.setattr(stop_carla.os, "kill", denied)
    with pytest.raises(RuntimeError, match="종료할 권한"):
        stop_carla.stop_all(0, 0, discover=lambda: [fixture])


def test_new_carla_child_is_stopped_without_broad_process_group_signal(monkeypatch):
    old = stop_carla.Process(100, 100, 10, os.geteuid(), "/fixture/CarlaUE4-Linux-Shipping")
    late = replace(old, pid=101, proc_pid=101)
    scans = iter(([old], [late], []))
    sent = []
    monkeypatch.setattr(stop_carla, "_send", lambda process, sig: sent.append((process.pid, sig)))
    monkeypatch.setattr(stop_carla, "_wait", lambda processes, timeout: [])
    assert stop_carla.stop_all(discover=lambda: next(scans)) == 2
    assert sent == [(100, signal.SIGTERM), (101, signal.SIGTERM)]


@pytest.mark.parametrize("executable, argv, expected", [
    ("/usr/bin/bash", ["bash", "/some folder/CarlaUE4.sh", "-RenderOffScreen"], "/some folder/CarlaUE4.sh"),
    ("/bin/dash", ["sh", "-e", "--", "./CarlaUE4.sh"], "./CarlaUE4.sh"),
    ("/bin/bash", ["bash", "-c", "./CarlaUE4.sh"], ""),
    ("/bin/bash", ["bash", "-lc", "exec ./CarlaUE4.sh"], ""),
    ("/usr/bin/python", ["python", "CarlaUE4.sh"], ""),
    ("/bin/bash", ["bash", "script.sh", "CarlaUE4.sh"], ""),
])
def test_wrapper_detection_uses_exact_script_argument(executable, argv, expected):
    assert stop_carla._wrapper(executable, argv) == expected


def test_other_user_is_excluded_before_process_inspection(tmp_path):
    process = tmp_path / "100"
    process.mkdir()
    (process / "status").write_text("Uid:\t12345\t12345\t12345\t12345\n")
    assert stop_carla.read_process(100, tmp_path, uid=23456) is None


def test_protected_unrelated_process_does_not_block_startup(tmp_path, monkeypatch):
    process = tmp_path / "100"
    process.mkdir()
    (process / "status").write_text("Uid:\t12345\t12345\t12345\t12345\n")
    (process / "stat").write_text("100 (protected daemon) S 1 " + "0 " * 20)
    (process / "cmdline").write_bytes(b"rsync\0--daemon\0")
    def denied(path):
        raise PermissionError("controlled protected system process")
    monkeypatch.setattr(stop_carla.os, "readlink", denied)
    assert stop_carla.read_process(100, tmp_path, uid=12345) is None


def test_start_script_preserves_independently_started_carla():
    script = Path(__file__).resolve().parents[1].joinpath("start.sh").read_text()
    assert '"$sdv_python" "$sdv_root/studio/stop_carla.py"' not in script
    assert 'pkill' not in script
    assert 'exec "$sdv_web_python" -m studio' in script


def test_start_script_never_invokes_global_cleanup_helper(tmp_path):
    script = Path(__file__).resolve().parents[1] / "start.sh"
    calls = tmp_path / "interpreter-calls"
    interpreter = tmp_path / "controlled-python"
    interpreter.write_text(
        '#!/bin/sh\n'
        'printf "%s\\n" "$1" >> "$SDV_TEST_CALLS"\n'
        'case "$1" in\n'
        '  -c) exit 0 ;;\n'
        '  */studio/stop_carla.py) echo "controlled cleanup failure" >&2; exit 41 ;;\n'
        '  *) exit 99 ;;\n'
        'esac\n'
    )
    interpreter.chmod(0o700)
    env = dict(os.environ, SDV_WEB_PYTHON=str(interpreter), SDV_TEST_CALLS=str(calls))
    result = subprocess.run(["bash", str(script)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert result.returncode == 1
    assert all("stop_carla" not in call for call in calls.read_text().splitlines())
    assert b"controlled cleanup failure" not in result.stderr
