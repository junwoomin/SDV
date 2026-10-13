"""Hardware-free lifecycle tests using real, isolated Python child processes.

The short commands below are test fixtures, never application demonstrations or
training metrics. They exercise OS process management without CARLA or a GPU.
"""
from __future__ import annotations

import json
import os
import sys
import signal
import time
import types
import re

import psutil
import studio.jobs as jobs_module
from pathlib import Path

import pytest

from studio.config import Config
from studio.jobs import JobManager

TERMINAL = {"Completed", "Failed", "Stopped"}


@pytest.fixture(autouse=True)
def namespace_psutil_adapter(monkeypatch):
    """Adapt this executor's host-mounted /proc, only when PID namespaces differ.

    Production code is untouched. Starts/signals/waits still use real OS child
    PIDs. Only psutil metadata reads translate to the host procfs PID. A normal
    Linux checkout takes the unmodified code path.
    """
    self_stat = Path("/proc/self/stat")
    if not self_stat.exists() or int(self_stat.read_text().split()[0]) == os.getpid():
        return
    namespace = Path("/proc/self/ns/pid").stat().st_ino
    real_process = psutil.Process

    def host_pid(pid):
        if pid == os.getpid():
            return int(self_stat.read_text().split()[0])
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                if (entry / "ns/pid").stat().st_ino != namespace:
                    continue
                match = re.search(r"^NSpid:\s+(.+)$", (entry / "status").read_text(), re.M)
                if match and int(match.group(1).split()[-1]) == pid:
                    return int(entry.name)
            except (OSError, ValueError):
                continue
        raise psutil.NoSuchProcess(pid)

    class ProcessMetadata:
        def __init__(self, pid=None):
            self.pid = os.getpid() if pid is None else pid
            self._process = real_process(host_pid(self.pid))
        def children(self, recursive=False):
            children = []
            for item in self._process.children(recursive=recursive):
                try:
                    status = Path(f"/proc/{item.pid}/status").read_text()
                    match = re.search(r"^NSpid:\s+(.+)$", status, re.M)
                    if match:
                        children.append(ProcessMetadata(int(match.group(1).split()[-1])))
                except (OSError, psutil.Error):
                    continue
            return children
        def create_time(self):
            return self._process.create_time()
        def status(self):
            return self._process.status()
        def is_running(self):
            try:
                os.kill(self.pid, 0)
                return True
            except ProcessLookupError:
                return False

    adapter = types.SimpleNamespace(Process=ProcessMetadata, Error=psutil.Error, NoSuchProcess=psutil.NoSuchProcess, STATUS_ZOMBIE=psutil.STATUS_ZOMBIE)
    monkeypatch.setattr(jobs_module, "psutil", adapter)


def wait_for(manager, job_id, predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    latest = None
    while time.monotonic() < deadline:
        latest = manager.get(job_id)
        if predicate(latest):
            return latest
        time.sleep(0.025)
    pytest.fail(f"Job did not reach expected state: {latest!r}")


def command(manager, monkeypatch, source):
    monkeypatch.setattr(manager, "_command", lambda *args, **kwargs: [sys.executable, "-u", "-c", source])


@pytest.fixture
def manager(tmp_path):
    instance = JobManager(Config(tmp_path))
    yield instance
    for job in instance.list():
        if job["status"] == "Running":
            try:
                instance.stop(job["id"])
                wait_for(instance, job["id"], lambda item: item["status"] in TERMINAL)
            except (KeyError, ValueError, RuntimeError):
                pass
    instance.shutdown()


def test_real_process_completion_and_persisted_history(manager, monkeypatch):
    command(manager, monkeypatch, "print('fixture completed', flush=True)")
    started = manager.start("collection", "QA completion fixture")
    ended = wait_for(manager, started["id"], lambda job: job["status"] in TERMINAL)
    assert ended["status"] == "Completed"
    assert ended["return_code"] == 0
    assert any(item["id"] == started["id"] for item in manager.list())


def test_real_nonzero_process_is_failed(manager, monkeypatch):
    command(manager, monkeypatch, "import sys; print('deliberate QA fixture failure', flush=True); sys.exit(7)")
    started = manager.start("training", "QA failure fixture")
    ended = wait_for(manager, started["id"], lambda job: job["status"] in TERMINAL)
    assert ended["status"] == "Failed"
    assert ended["return_code"] == 7


def test_silent_process_stop_and_duplicate_rejection(manager, monkeypatch):
    # No stdout: stop must not wait for a readline() that can never finish.
    command(manager, monkeypatch, "import time; time.sleep(60)")
    started = manager.start("collection", "QA silent fixture")
    with pytest.raises((ValueError, RuntimeError)):
        manager.start("training", "QA duplicate must not run")
    before = time.monotonic()
    manager.stop(started["id"])
    ended = wait_for(manager, started["id"], lambda job: job["status"] in TERMINAL, timeout=8)
    assert ended["status"] == "Stopped"
    assert time.monotonic() - before < 8


def test_saved_configuration_is_snapshot_not_live_setting(manager, monkeypatch):
    command(manager, monkeypatch, "import time; time.sleep(60)")
    started = manager.start("collection", "QA snapshot fixture")
    manager.config.save_settings({"project_name": "Changed after launch"})
    snap = manager.get(started["id"])
    assert snap["settings"]["project_name"] == "SDV Research"
    assert manager.config.settings()["project_name"] == "Changed after launch"


def test_global_bev_job_environment_uses_frozen_asset_settings(manager, monkeypatch):
    manager.config.save_settings({'bev_generator': {'map_root': 'assets/original', 'global_ppm': 5}})
    command(manager, monkeypatch, "import os,json; print('BEV_ENV '+json.dumps({'root':os.environ['SDV_BEV_MAP_ROOT'],'settings':json.loads(os.environ['SDV_BEV_GENERATOR'])}),flush=True)")
    started = manager.start('collection', 'QA frozen PNG map environment')
    manager.config.save_settings({'bev_generator': {'map_root': 'assets/changed', 'global_ppm': 10}})
    ended = wait_for(manager, started['id'], lambda job: job['status'] in TERMINAL)
    assert ended['status'] == 'Completed'
    line = next(line for line in Path(ended['log_path']).read_text().splitlines() if line.startswith('BEV_ENV '))
    value = json.loads(line[len('BEV_ENV '):])
    assert value['root'] == str(manager.config.root/'assets/original')
    assert value['settings']['global_ppm'] == 5
    assert ended['settings']['bev_generator']['map_root'] == 'assets/original'


def test_terminal_history_survives_new_manager(tmp_path, monkeypatch):
    config = Config(tmp_path)
    first = JobManager(config)
    command(first, monkeypatch, "print('QA durable fixture', flush=True)")
    started = first.start("collection", "QA durable fixture")
    ended = wait_for(first, started["id"], lambda job: job["status"] in TERMINAL)
    assert ended["status"] == "Completed"
    first.shutdown()
    second = JobManager(Config(tmp_path))
    try:
        recovered = second.get(started["id"])
        assert recovered["status"] == "Completed"
        assert recovered["id"] == started["id"]
    finally:
        second.shutdown()


def test_fresh_manager_observes_and_stops_running_job(manager, monkeypatch):
    command(manager, monkeypatch, "import time; time.sleep(60)")
    started = manager.start("collection", "QA browser-refresh fixture")
    observer = JobManager(Config(manager.config.root))
    try:
        assert observer.get(started["id"])["status"] == "Running"
        command(observer, monkeypatch, "raise SystemExit('must never run')")
        with pytest.raises((ValueError, RuntimeError)):
            observer.start("training", "Duplicate across managers")
        observer.stop(started["id"])
        ended = wait_for(manager, started["id"], lambda job: job["status"] in TERMINAL)
        assert ended["status"] == "Stopped"
    finally:
        observer.shutdown()


def test_stop_cleans_child_process_after_parent_exits(manager, monkeypatch):
    ready = manager.config.root / "qa_child_ready"
    child = "import os,signal,time; from pathlib import Path; signal.signal(signal.SIGINT,signal.SIG_IGN); Path(%r).write_text(str(os.getpid())); time.sleep(60)" % str(ready)
    parent = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',%r]); time.sleep(60)" % child
    command(manager, monkeypatch, parent)
    started = manager.start("collection", "QA process tree fixture")
    deadline = time.monotonic() + 5
    while not ready.exists() and time.monotonic() < deadline:
        time.sleep(.025)
    assert ready.exists(), manager.logs(started["id"])
    child_pid = int(ready.read_text())
    try:
        manager.stop(started["id"])
        ended = wait_for(manager, started["id"], lambda job: job["status"] in TERMINAL)
        assert ended["status"] == "Stopped"
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try:
                process = jobs_module.psutil.Process(child_pid)
                child_alive = process.is_running() and process.status() != psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:
                child_alive = False
            if not child_alive:
                break
            time.sleep(.025)
        assert not child_alive, "Cancelled job left a child process running after its parent exited"
    finally:
        try:
            os.kill(child_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_observed_telemetry_only_and_partial_tail_is_ignored(manager, monkeypatch):
    # These explicit fixture values test ingestion; they are never shipped as results.
    event = {"event": "training_epoch", "epoch": 2, "epochs": 4, "progress": 0.5, "metrics": {"train_loss": 0.75}}
    payload = json.dumps(event) + "\n{malformed fixture}\n" + '{"event":"training_epoch","epoch":3'
    source = "import os,time; from pathlib import Path; Path(os.environ['SDV_EVENTS_PATH']).write_text(%r); time.sleep(60)" % payload
    command(manager, monkeypatch, source)
    started = manager.start("training", "QA telemetry fixture")
    observed = wait_for(manager, started["id"], lambda job: job["progress_pct"] == 50)
    assert observed["status"] == "Running"
    assert observed["progress_current"] == 2
    assert observed["progress_total"] == 4
    assert observed["metrics"] == [event]
    observer = JobManager(Config(manager.config.root))
    try:
        # Re-reading a live run must not replay an already stored metric twice.
        deadline = time.monotonic() + 0.6
        while time.monotonic() < deadline:
            assert observer.get(started["id"])["metrics"] == [event]
            time.sleep(.025)
    finally:
        observer.shutdown()
    manager.stop(started["id"])
    ended = wait_for(manager, started["id"], lambda job: job["status"] in TERMINAL)
    assert ended["status"] == "Stopped"
    assert ended["metrics"] == [event]


@pytest.mark.parametrize("code, expected", [(0, "Completed"), (7, "Failed")])
def test_worker_receipt_recovers_lost_final_database_update(tmp_path, monkeypatch, code, expected):
    first = JobManager(Config(tmp_path))
    command(first, monkeypatch, f"raise SystemExit({code})")
    started = first.start("collection", "QA durable exit receipt")
    ended = wait_for(first, started["id"], lambda job: job["status"] in TERMINAL)
    first.shutdown()
    receipt = Path(ended["receipt_path"])
    assert receipt.is_file()
    assert json.loads(receipt.read_text())["return_code"] == code
    # Model the exact crash window: worker receipt exists, final DB write was lost.
    stale = dict(ended)
    stale.update(status="Running", return_code=None, finished_at=None)
    first._save(stale)
    recovered_manager = JobManager(Config(tmp_path))
    try:
        recovered = recovered_manager.get(started["id"])
        assert recovered["status"] == expected
        assert recovered["return_code"] == code
    finally:
        recovered_manager.shutdown()


@pytest.mark.parametrize('code, status', [(0, 'Completed'), (7, 'Failed')])
def test_collection_keeps_actual_global_counts_on_process_exit_and_recovery(tmp_path, monkeypatch, code, status):
    # Explicit telemetry fixtures represent two completed Town01 samples and one
    # Town02 sample. A zero exit code alone must not invent the fourth sample.
    events = [
        {'event':'collection_started','saved':0,'total':4,'progress':0},
        {'event':'collection_progress','town':'Town01','town_saved':2,'town_target':2,'saved':2,'total':4,'progress':.5},
        {'event':'collection_town_started','town':'Town02','town_saved':0,'town_target':2,'saved':2,'total':4,'progress':.5},
        {'event':'collection_progress','town':'Town02','town_saved':1,'town_target':2,'saved':3,'total':4,'progress':.5},
    ]
    payload = ''.join(json.dumps(event)+'\n' for event in events)
    source = "import os; from pathlib import Path; Path(os.environ['SDV_EVENTS_PATH']).write_text(%r); raise SystemExit(%d)" % (payload, code)
    first = JobManager(Config(tmp_path))
    command(first, monkeypatch, source)
    try:
        started = first.start('collection', 'QA global counts fixture')
        ended = wait_for(first, started['id'], lambda job: job['status'] in TERMINAL)
        assert ended['status'] == status
        assert (ended['progress_current'], ended['progress_total'], ended['progress_pct']) == (3,4,75)
        assert ended['last_event']['town_saved'] == 1
    finally:
        first.shutdown()
    stale = dict(ended)
    stale.update(status='Running',return_code=None,finished_at=None)
    first._save(stale)
    observer = JobManager(Config(tmp_path))
    try:
        recovered = observer.get(started['id'])
        assert recovered['status'] == status
        assert (recovered['progress_current'],recovered['progress_total'],recovered['progress_pct']) == (3,4,75)
    finally:
        observer.shutdown()


def test_collection_stop_preserves_last_committed_sample_count(manager, monkeypatch):
    event={'event':'collection_progress','town':'Town02','town_saved':1,'town_target':2,'saved':3,'total':4,'progress':.75}
    payload=json.dumps(event)+'\n'
    source="import os,time; from pathlib import Path; Path(os.environ['SDV_EVENTS_PATH']).write_text(%r); time.sleep(60)" % payload
    command(manager,monkeypatch,source)
    started=manager.start('collection','QA partial stop fixture')
    observed=wait_for(manager,started['id'],lambda job:job['progress_current']==3)
    assert observed['progress_pct']==75
    manager.stop(started['id'])
    ended=wait_for(manager,started['id'],lambda job:job['status'] in TERMINAL)
    assert ended['status']=='Stopped'
    assert (ended['progress_current'],ended['progress_total'],ended['progress_pct'])==(3,4,75)


def test_collection_completed_event_reports_actual_count(manager, monkeypatch):
    events=[
        {'event':'collection_progress','town':'Town01','saved':2,'total':4,'progress':.5},
        {'event':'collection_progress','town':'Town02','saved':1,'total':4,'progress':.25},
        {'event':'collection_completed','town':'Town02','saved':4,'total':4,'progress':1},
    ]
    payload=''.join(json.dumps(event)+'\n' for event in events)
    source="import os; from pathlib import Path; Path(os.environ['SDV_EVENTS_PATH']).write_text(%r)" % payload
    command(manager,monkeypatch,source)
    started=manager.start('collection','QA completed events fixture')
    ended=wait_for(manager,started['id'],lambda job:job['status'] in TERMINAL)
    assert ended['status']=='Completed'
    assert (ended['progress_current'],ended['progress_total'],ended['progress_pct'])==(4,4,100)


def test_collection_rejects_invalid_or_regressing_global_counts(manager, tmp_path):
    events = tmp_path/'collection-events.jsonl'
    invalid = [(1,4), (-1,4), (5,4), (True,4), (3,0), (3.5,4)]
    records = [{'event':'collection_progress','saved':saved,'total':total,'progress':1} for saved,total in invalid]
    events.write_text(''.join(json.dumps(event)+'\n' for event in records))
    job = {'kind':'collection','events_path':str(events),'progress_current':2,'progress_total':4,'progress_pct':50}
    manager._events(job,0)
    assert (job['progress_current'],job['progress_total'],job['progress_pct'])==(2,4,50)
