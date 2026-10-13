"""CARLA management tests using real OS children, never a GPU/CARLA server."""
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import types

import psutil
import pytest

from studio.config import Config, atomic_json
import studio.carla_service as service_module
from studio.carla_service import CarlaService, identity, process_alive, read_state, update_state
import studio.carla_worker as worker_module


@pytest.fixture(autouse=True)
def process_metadata_for_executor(monkeypatch):
    """Only translate procfs metadata when this executor mounts host /proc."""
    self_stat = Path('/proc/self/stat')
    if not self_stat.exists() or int(self_stat.read_text().split()[0]) == os.getpid():
        return
    namespace = Path('/proc/self/ns/pid').stat().st_ino
    original = psutil.Process

    def host_pid(pid):
        if pid == os.getpid():
            return int(self_stat.read_text().split()[0])
        for path in Path('/proc').iterdir():
            if not path.name.isdigit():
                continue
            try:
                if (path / 'ns/pid').stat().st_ino != namespace:
                    continue
                match = re.search(r'^NSpid:\s+(.+)$', (path / 'status').read_text(), re.M)
                if match and int(match.group(1).split()[-1]) == pid:
                    return int(path.name)
            except (OSError, ValueError):
                pass
        raise psutil.NoSuchProcess(pid)

    class Process:
        def __init__(self, pid=None):
            self.pid = os.getpid() if pid is None else pid
            self.original = original(host_pid(self.pid))
        def create_time(self):
            return self.original.create_time()
        def status(self):
            return self.original.status()
        def is_running(self):
            try:
                os.kill(self.pid, 0)
                return True
            except ProcessLookupError:
                return False
        def children(self, recursive=False):
            result = []
            for child in self.original.children(recursive=recursive):
                try:
                    text = Path('/proc/{}/status'.format(child.pid)).read_text()
                    match = re.search(r'^NSpid:\s+(.+)$', text, re.M)
                    if match:
                        result.append(Process(int(match.group(1).split()[-1])))
                except (OSError, psutil.Error):
                    pass
            return result
    monkeypatch.setattr(service_module, 'psutil', types.SimpleNamespace(
        Process=Process, Error=psutil.Error, NoSuchProcess=psutil.NoSuchProcess, STATUS_ZOMBIE=psutil.STATUS_ZOMBIE))


def eventually(function, predicate, timeout=6):
    deadline = time.monotonic() + timeout
    value = None
    while time.monotonic() < deadline:
        value = function()
        if predicate(value):
            return value
        time.sleep(.025)
    pytest.fail('Expected state not reached: {!r}'.format(value))


def seed(service, operation='prepare'):
    run = service.directory / 'fixture'
    run.mkdir(exist_ok=True)
    state = {'task': {'id': 'fixture', 'operation': operation, 'status': 'Running', 'phase': 'queued',
                     'progress_pct': None, 'message': '', 'started_at': None, 'updated_at': None, 'stop_requested': False},
             'engine': {'running': False, 'ready': False, 'owned': False, 'host': '127.0.0.1', 'port': 2000,
                        'pid': None, 'created': None, 'descendants': []},
             'worker': identity(os.getpid()), 'run_dir': str(run), 'settings': service.config.settings(), 'town': 'Town01'}
    atomic_json(service.directory / 'state.json', state)
    return state


@pytest.fixture
def service(tmp_path):
    instance = CarlaService(Config(tmp_path))
    yield instance
    # Worker tests supervise in-process; do not signal the pytest worker.
    value = read_state(instance.directory)
    if value.get('worker', {}).get('pid') == os.getpid():
        return
    instance.shutdown()


def worker_setup(service, monkeypatch, command=None, answer=None, operation='prepare'):
    import studio.carla_diagnostics as diagnostic_module
    monkeypatch.setattr(diagnostic_module,'diagnostics',lambda *a,**kw:{'ready':True,'warnings':[]})
    import sdv.runtime.install as installer
    def installed(cancel, progress, root=None):
        Path(root).mkdir(parents=True, exist_ok=True)
        progress(1, 'isolated test fixture installation')
    monkeypatch.setattr(installer, 'install', installed)
    monkeypatch.setattr(worker_module, 'ensure_client', lambda *args: None)
    monkeypatch.setattr(worker_module, 'check_ports', lambda port: None)
    monkeypatch.setattr(worker_module, 'engine_command', lambda *args: command or [sys.executable, '-u', '-c', "import time; print('fixture engine output', flush=True); time.sleep(60)"])
    monkeypatch.setattr(worker_module, 'probe_rpc', lambda *args, **kwargs: answer or {'connected': True})
    state = seed(service, operation)
    cancel = worker_module.Cancellation(Path(state['run_dir']) / 'stop.json')
    thread = threading.Thread(target=worker_module.run_task, args=(service.config, 'fixture', cancel))
    thread.start()
    return thread, cancel


def test_empty_state_installation_and_local_host_rule(service):
    status = service.status()
    assert status['task']['status'] == 'Idle'
    assert status['engine']['running'] is False
    assert status['local_allowed'] is True
    service.config.save_settings({'carla_host': '192.0.2.10'})
    with pytest.raises(ValueError, match='loopback'):
        service.start('prepare')
    assert not read_state(service.directory)


def test_symlink_installation_outside_project_is_rejected(service, tmp_path):
    external = tmp_path.parent / (tmp_path.name + '-outside')
    external.mkdir()
    (tmp_path / 'carla').symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match='프로젝트 폴더 내부'):
        service.start('install')
    assert list(external.iterdir()) == []
    assert service.status()['installed'] is False
    assert service.status()['local_allowed'] is False
    assert service.status()['local_reason']


def test_real_worker_install_receipt_and_refresh(service, monkeypatch):
    # Installation-only worker executes as a genuine separate Python process.
    # Empty valid local distribution makes installer return without networking.
    root = service.config.root / 'carla'
    binary = root / 'CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping'
    binary.parent.mkdir(parents=True)
    binary.write_text('#!/bin/sh\nexit 0\n')
    binary.chmod(0o755)
    (root / 'CarlaUE4.sh').write_text('#!/bin/sh\nexit 0\n')
    (root / 'CarlaUE4.sh').chmod(0o755)
    (root / 'VERSION').write_text('0.9.15\n')
    started = service.start('install')
    ended = eventually(service.status, lambda item: item['task']['status'] in ('Completed', 'Failed'))
    assert ended['task']['status'] == 'Completed', service.logs()
    assert ended['engine']['running'] is False
    assert '설치' in service.logs()
    fresh = CarlaService(Config(service.config.root))
    try:
        assert fresh.status()['task']['id'] == started['task']['id']
        assert fresh.status()['task']['status'] == 'Completed'
        receipt = Path(read_state(service.directory)['run_dir']) / 'exit.json'
        assert json.loads(receipt.read_text())['status'] == 'Completed'
    finally:
        fresh.shutdown()


def test_prepare_keeps_ready_engine_and_stop_cleans_real_child(service, monkeypatch):
    thread, cancel = worker_setup(service, monkeypatch)
    try:
        ready = eventually(service.status, lambda item: item['engine']['ready'])
        assert ready['task']['status'] == 'Completed'
        assert service.busy() is False
        assert ready['engine']['owned'] is True
        assert process_alive(read_state(service.directory)['engine'])
        # The fake readiness probe can complete before the real child gets
        # scheduled to flush stdout, especially during parallel CPU checks.
        assert 'fixture engine output' in eventually(service.logs, lambda value: 'fixture engine output' in value)
        fresh = CarlaService(Config(service.config.root))
        assert fresh.status()['engine']['pid'] == ready['engine']['pid']
        with pytest.raises(ValueError, match='이미 동작'):
            fresh.start('prepare')
        # Avoid a reaper thread targeting the in-process supervisor fixture.
        fresh._request_stop(read_state(service.directory))
        thread.join(5)
        assert not thread.is_alive()
        ended = service.status()
        assert ended['task']['status'] == 'Stopped'
        assert ended['engine']['running'] is False
        assert not process_alive(read_state(service.directory)['engine'])
    finally:
        cancel.set()
        thread.join(8)


def test_readiness_timeout_fails_and_terminates_engine(service, monkeypatch):
    monkeypatch.setattr(worker_module, 'READINESS_TIMEOUT', .2)
    thread, cancel = worker_setup(service, monkeypatch, answer={'connected': False, 'error': 'fixture RPC unavailable'})
    try:
        thread.join(5)
        assert not thread.is_alive()
        status = service.status()
        assert status['task']['status'] == 'Failed'
        assert 'RPC' in status['task']['message']
        assert not status['engine']['running']
        assert not process_alive(read_state(service.directory)['engine'])
    finally:
        cancel.set()
        thread.join(8)


def test_engine_nonzero_exit_is_failed(service, monkeypatch):
    command = [sys.executable, '-c', "import sys; print('fixture GPU failure', flush=True); sys.exit(7)"]
    thread, cancel = worker_setup(service, monkeypatch, command=command, answer={'connected': False, 'error': 'starting'})
    try:
        thread.join(5)
        assert not thread.is_alive()
        assert service.status()['task']['status'] == 'Failed'
        assert '7' in service.status()['task']['message']
        assert 'fixture GPU failure' in service.logs()
    finally:
        cancel.set()
        thread.join(8)


def test_external_listener_is_preserved():
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        with pytest.raises(RuntimeError, match='사용 중'):
            worker_module.check_ports(port)
        with socket.create_connection(('127.0.0.1', port), timeout=1):
            pass


def test_pid_reuse_fingerprint_never_signals_external_child():
    process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
    try:
        stale = identity(process.pid)
        stale['created'] -= 10
        assert service_module.signal_owned_group(stale, signal.SIGKILL) is False
        assert process.poll() is None
    finally:
        process.terminate()
        process.wait(timeout=3)


def test_lost_worker_marks_failed_and_reaps_owned_engine(service):
    process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
    try:
        state = seed(service)
        engine = dict(state['engine'], running=True, owned=True, ready=True, **identity(process.pid))
        update_state(service.directory, 'fixture', engine=engine, worker={'pid': os.getpid(), 'created': 0})
        ended = eventually(service.status, lambda item: item['task']['status'] == 'Failed')
        assert ended['engine']['running'] is False
        process.wait(timeout=3)
        assert process.returncode != 0
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=3)


def test_saved_receipt_recovers_terminal_status(service):
    state = seed(service, operation='install')
    atomic_json(Path(state['run_dir']) / 'exit.json', {'status': 'Completed', 'return_code': 0,
                'task': {'status': 'Completed', 'phase': 'installed', 'progress_pct': 100, 'message': 'fixture installed'}})
    update_state(service.directory, 'fixture', worker={'pid': os.getpid(), 'created': 0})
    recovered = CarlaService(Config(service.config.root)).status()
    assert recovered['task']['status'] == 'Completed'
    assert recovered['task']['progress_pct'] == 100


def test_real_long_worker_duplicate_refresh_stop_and_shutdown(service, monkeypatch):
    source = (
        "import sys,time; from pathlib import Path; "
        "sys.path.insert(0, %r); "
        "from studio.config import Config; "
        "from studio.carla_service import state_directory,read_state,update_state; "
        "directory=state_directory(Config(%r)); "
        "value=read_state(directory); task_id=value['task']['id']; "
        "stop=Path(value['run_dir'])/'stop.json'; "
        "update_state(directory,task_id,task={'phase':'installing','progress_pct':25}); "
        "print('isolated long worker fixture',flush=True)\n"
        "while not stop.exists(): time.sleep(.025)\n"
        "update_state(directory,task_id,task={'status':'Stopped','phase':'stopped'})\n"
    ) % (str(Path(service_module.__file__).resolve().parents[1]), str(service.config.root))
    monkeypatch.setattr(service, '_worker_command', lambda task_id: [sys.executable, '-u', '-c', source])
    service.start('install')
    eventually(service.status, lambda item: item['task']['progress_pct'] == 25)
    observer = CarlaService(Config(service.config.root))
    assert observer.busy()
    with pytest.raises(ValueError, match='이미 동작'):
        observer.start('install')
    observer.stop()
    ended = eventually(service.status, lambda item: item['task']['status'] == 'Stopped')
    assert ended['task']['stop_requested']
    eventually(service.status, lambda status: not service._owned)
    service.start('install')
    eventually(service.status, lambda item: item['task']['progress_pct'] == 25)
    service.shutdown()
    assert service.status()['task']['status'] == 'Stopped'
    assert not service._owned


def test_broken_client_fails_before_install_download(service, monkeypatch):
    monkeypatch.setattr(worker_module, "check_ports", lambda *args: None)
    import sdv.runtime.install as installer
    installed = []
    monkeypatch.setattr(installer, 'install', lambda *args, **kwargs: installed.append(True))
    def missing(*args):
        raise RuntimeError('fixture missing carla client')
    monkeypatch.setattr(worker_module, 'ensure_client', missing)
    seed(service)
    assert worker_module.run_task(service.config, 'fixture') == 1
    assert installed == []
    assert service.status()['task']['status'] == 'Failed'
    assert 'missing carla client' in service.status()['task']['message']


@pytest.fixture
def research_python(service):
    import venv
    path = service.config.root / 'research-venv'
    # Match `python -m venv` on Linux. Standalone interpreters need their
    # executable symlink to retain the original libpython/stdlib location.
    venv.EnvBuilder(with_pip=True, symlinks=True).create(str(path))
    python = path / 'bin/python'
    service.config.save_settings({'python_executable': str(python)})
    return python


def client_fixture_wheel(folder):
    """Minimal isolated wheel tests pip integration; it is not a CARLA mock UI."""
    import zipfile
    folder.mkdir(parents=True, exist_ok=True)
    wheel = folder / 'carla-0.9.15-py3-none-any.whl'
    files = {
        'carla/__init__.py': "# QA fixture, no simulator functionality.\nclass Client:\n def __init__(self,host,port): pass\n def get_client_version(self): return '0.9.15'\n",
        'carla-0.9.15.dist-info/METADATA': 'Metadata-Version: 2.1\nName: carla\nVersion: 0.9.15\nSummary: isolated SDV pip integration fixture\n',
        'carla-0.9.15.dist-info/WHEEL': 'Wheel-Version: 1.0\nGenerator: SDV-isolated-QA\nRoot-Is-Purelib: true\nTag: py3-none-any\n',
    }
    files['carla-0.9.15.dist-info/RECORD'] = '\n'.join(name + ',,' for name in list(files) + ['carla-0.9.15.dist-info/RECORD']) + '\n'
    with zipfile.ZipFile(wheel, 'w') as archive:
        for name, contents in files.items():
            archive.writestr(name, contents)
    return wheel


def test_missing_client_installs_real_wheel_and_continues_engine(service, research_python, monkeypatch):
    import studio.carla_diagnostics as diagnostic_module
    monkeypatch.setattr(diagnostic_module,'diagnostics',lambda *a,**kw:{'ready':True,'warnings':[]})
    import sdv.runtime.install as installer
    settings = service.config.settings()
    # A downloaded engine directory must not masquerade as an installed Python client.
    (service.config.root / 'carla').mkdir()
    assert worker_module.inspect_client(service.config, settings)['state'] == 'missing'
    wheelhouse = service.config.root / 'fixture-wheels'
    client_fixture_wheel(wheelhouse)
    monkeypatch.setenv('PIP_NO_INDEX', '1')
    monkeypatch.setenv('PIP_FIND_LINKS', str(wheelhouse))
    commands = []
    original = worker_module._run_client_install
    def record_install(config, command, cancel, **kwargs):
        commands.append(command)
        return original(config, command, cancel, **kwargs)
    monkeypatch.setattr(worker_module, '_run_client_install', record_install)
    def installed(cancel, progress, root=None):
        Path(root).mkdir(parents=True, exist_ok=True)
        progress(1, 'fixture engine distribution')
    monkeypatch.setattr(installer, 'install', installed)
    monkeypatch.setattr(worker_module, 'check_ports', lambda *args: None)
    monkeypatch.setattr(worker_module, 'engine_command', lambda *args: [sys.executable, '-c', 'import time;time.sleep(60)'])
    monkeypatch.setattr(worker_module, 'probe_rpc', lambda *args, **kwargs: {'connected': True})
    state = seed(service)
    cancel = worker_module.Cancellation(Path(state['run_dir']) / 'stop.json')
    thread = threading.Thread(target=worker_module.run_task, args=(service.config, 'fixture', cancel))
    thread.start()
    try:
        eventually(service.status, lambda status: status['engine']['ready'], timeout=30)
        assert service.status()['task']['status'] == 'Completed'
        assert worker_module.inspect_client(service.config, settings)['version'] == '0.9.15'
        assert len(commands) == 1
        assert commands[0][:3] == [str(research_python), '-m', 'pip']
        assert '--no-deps' in commands[0]
        assert '--only-binary=:all:' in commands[0]
        assert '--upgrade' not in commands[0]
        # A second request reuses the valid client without pip/network activity.
        worker_module.ensure_client(service.config, settings, cancel)
        assert len(commands) == 1
    finally:
        cancel.set()
        thread.join(8)
    assert not thread.is_alive()
    assert service.status()['task']['status'] == 'Stopped'


@pytest.mark.parametrize('source, expected', [
    ("class Client:\n def __init__(self,*args):pass\n def get_client_version(self):return '0.9.14'\n", 'mismatch'),
    ("raise ImportError('fixture missing libcarla.so')\n", 'broken'),
])
def test_existing_other_or_broken_client_is_not_overwritten(service, research_python, monkeypatch, source, expected):
    # cwd appears on Python's import path and models an existing research module.
    (service.config.root / 'carla.py').write_text(source)
    settings = service.config.settings()
    assert worker_module.inspect_client(service.config, settings)['state'] == expected
    commands = []
    monkeypatch.setattr(worker_module, '_run_client_install', lambda *args, **kwargs: commands.append(args))
    cancel = worker_module.Cancellation(service.directory / 'not-requested')
    with pytest.raises(RuntimeError, match='자동 변경하지 않았습니다'):
        worker_module.ensure_client(service.config, settings, cancel)
    assert commands == []
    assert (service.config.root / 'carla.py').read_text() == source


def test_client_install_cancel_terminates_real_subprocess(service):
    marker = service.config.root / 'pip-fixture-pid'
    command = [sys.executable, '-c', "import os,time;from pathlib import Path;Path(%r).write_text(str(os.getpid()));time.sleep(60)" % str(marker)]
    cancel = worker_module.Cancellation(service.directory / 'cancel-fixture')
    outcomes = []
    def run():
        try:
            worker_module._run_client_install(service.config, command, cancel)
        except worker_module.StopRequested:
            outcomes.append('stopped')
    thread = threading.Thread(target=run)
    thread.start()
    try:
        eventually(lambda: marker.exists(), bool)
        record = identity(int(marker.read_text()))
        cancel.set()
        thread.join(5)
        assert not thread.is_alive()
        assert outcomes == ['stopped']
        assert not process_alive(record)
    finally:
        cancel.set()
        thread.join(8)


def test_client_install_timeout_terminates_real_subprocess(service, monkeypatch):
    monkeypatch.setattr(worker_module, 'CLIENT_INSTALL_TIMEOUT', .2)
    marker = service.config.root / 'pip-timeout-pid'
    command = [sys.executable, '-c', "import os,time;from pathlib import Path;Path(%r).write_text(str(os.getpid()));time.sleep(60)" % str(marker)]
    cancel = worker_module.Cancellation(service.directory / 'not-requested')
    with pytest.raises(TimeoutError, match='10분'):
        worker_module._run_client_install(service.config, command, cancel)
    assert marker.exists()
    with pytest.raises(psutil.NoSuchProcess):
        identity(int(marker.read_text()))


def test_missing_pip_uses_cached_runner_without_changing_research_pip(service, monkeypatch):
    import zipfile
    commands = []
    selected = service.config.settings()['python_executable']
    # Force fallback even when this test interpreter happens to have pip.
    calls = []
    def pip_info(config, python):
        calls.append(python)
        return {'available': False} if len(calls) == 1 else {'available': True, 'version': '26.2'}
    monkeypatch.setattr(worker_module, '_pip_info', pip_info)
    def download(config, command, cancel, **kwargs):
        commands.append(command)
        target = Path(command[command.index('--dest') + 1]) / 'pip-24.0-py3-none-any.whl'
        with zipfile.ZipFile(target, 'w') as archive:
            archive.writestr('pip/__main__.py', '# QA runner selection only\n')
            archive.writestr('pip-24.0.dist-info/METADATA', 'Name: pip\nVersion: 24.0\n')
    monkeypatch.setattr(worker_module, '_run_client_install', download)
    cancel = worker_module.Cancellation(service.directory / 'not-requested')
    command = worker_module._pip_command(service.config, service.config.settings(), cancel)
    assert len(commands) == 1
    assert 'download' in commands[0]
    assert commands[0][-1] == 'pip==24.0'
    assert command[:2] == [selected, '-c']
    assert 'run_module' in command[2]
    assert command[-1].endswith('pip-24.0-py3-none-any.whl')
    assert '--python' not in command
    # The runner is cached locally; no installation or upgrade of target pip.
    monkeypatch.setattr(worker_module, '_pip_info', lambda *args: {'available': False})
    assert worker_module._pip_command(service.config, service.config.settings(), cancel) == command
    assert len(commands) == 1
