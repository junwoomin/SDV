"""Small, dependency-free bridge from research subprocesses to the web monitor.

Only observed values are emitted. One append is protected by a process lock;
readers may ignore an incomplete trailing line after abrupt termination.
"""
import json
import math
import os
import threading
import time
from pathlib import Path

_lock = threading.Lock()


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def emit(event, _mirror_path=None, **values):
    record = _json_value({"event": event, "type": event, "timestamp": time.time(), **values})
    encoded = (json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    path = os.environ.get("SDV_EVENTS_PATH")
    with _lock:
        targets = {str(value) for value in (path, _mirror_path) if value}
        for target in targets:
            destination = Path(target)
            destination.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(destination, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
            try:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX)
                remaining = memoryview(encoded)
                while remaining:
                    written = os.write(fd, remaining)
                    remaining = remaining[written:]
            finally:
                os.close(fd)
        if os.environ.get("SDV_EVENTS_STDOUT", "1") == "1":
            print("SDV_EVENT " + encoded.decode("utf-8").rstrip(), flush=True)
    return record


def configured_path(variable, default):
    """Explicit worker overrides can refer to user-selected storage outside SDV."""
    from sdv.paths import ROOT
    value = Path(os.environ.get(variable, str(default))).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()
