"""Durable, ownership-scoped CARLA installation and local engine supervision.

Research jobs are separate: a ready CARLA engine can stay alive while collection
or training runs. Only a PID with the recorded process creation time is signaled.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import ipaddress
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import uuid

import psutil

from .config import atomic_json


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def identity(pid):
    process = psutil.Process(int(pid))
    return {'pid': int(pid), 'created': process.create_time()}


def process_alive(record):
    if not record or not record.get('pid') or record.get('created') is None:
        return False
    try:
        process = psutil.Process(int(record['pid']))
        return (abs(process.create_time() - float(record['created'])) < 0.01
                and process.is_running() and process.status() != psutil.STATUS_ZOMBIE)
    except (psutil.Error, ValueError, TypeError):
        return False


def signal_owned_group(record, signum):
    """Never signal an old PID or an external process discovered by port/name."""
    members = [record] + list(record.get('descendants', []))
    verified = False
    for member in members:
        try:
            if process_alive(member) and os.getpgid(int(member['pid'])) == int(record['pid']):
                verified = True
                break
        except (ProcessLookupError, PermissionError, ValueError, TypeError):
            continue
    if not verified:
        return False
    try:
        os.killpg(int(record['pid']), signum)
        return True
    except (ProcessLookupError, PermissionError, ValueError, TypeError):
        return False


def descendants(record):
    found = {item['pid']: item for item in record.get('descendants', [])}
    if process_alive(record):
        try:
            for process in psutil.Process(record['pid']).children(recursive=True):
                try:
                    found[process.pid] = identity(process.pid)
                except psutil.Error:
                    pass
        except psutil.Error:
            pass
    return list(found.values())


def group_alive(record):
    return any(process_alive(item) for item in [record] + list(record.get('descendants', [])))


def local_permission(host):
    if host == 'localhost':
        return True, ''
    try:
        if ipaddress.ip_address(host).is_loopback:
            return True, ''
    except ValueError:
        pass
    return False, '로컬 CARLA 실행은 서버 주소를 localhost 또는 loopback IP로 설정한 경우에만 가능합니다. 외부 서버는 연결 확인을 사용하세요.'


def state_directory(config):
    directory = config.state / 'carla'
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@contextmanager
def state_lock(directory, name='state.lock', blocking=True):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / name).open('a') as handle:
        mode = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(handle.fileno(), mode)
        except BlockingIOError:
            raise ValueError('CARLA 작업 시작 또는 종료 처리가 진행 중입니다.') from None
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def read_state(directory):
    path = directory / 'state.json'
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('CARLA 실행 상태 파일이 올바른 JSON 객체가 아닙니다.')
    return value


def update_state(directory, task_id, task=None, engine=None, **fields):
    with state_lock(directory):
        value = read_state(directory)
        if value.get('task', {}).get('id') != task_id:
            return None
        if task:
            value['task'].update(task)
        if engine:
            value['engine'].update(engine)
        value.update(fields)
        value['task']['updated_at'] = timestamp()
        atomic_json(directory / 'state.json', value)
        return value


class CarlaService:
    def __init__(self, config):
        self.config = config
        self.directory = state_directory(config)
        self._owned = {}
        self._stopping = set()
        self._closing = False
        self._lock = threading.RLock()

    def _installation(self):
        from sdv.runtime.install import installation_status, URL, SIZE, REQUIRED_FREE_BYTES
        try:
            from .carla_diagnostics import engine_root
            target = engine_root(self.config)
        except ValueError as error:
            return {'installed': False, 'path': str(self.config.root / 'carla'), 'version': None,
                    'source_url': URL, 'download_bytes': 0, 'total_bytes': SIZE,
                    'required_free_bytes': REQUIRED_FREE_BYTES, 'free_bytes': None, 'error': str(error), '_path_error': True}
        return installation_status(target)

    def _reap(self):
        for task_id, process in list(self._owned.items()):
            if process.poll() is not None:
                self._owned.pop(task_id, None)

    def _reconcile(self):
        self._reap()
        value = read_state(self.directory)
        if not value:
            return value
        task = value['task']
        worker = value.get('worker', {})
        # The launcher owns a short window before publishing its child PID.
        if not worker.get('pid') and time.time() - value.get('launch_unix', 0) < 5:
            if process_alive(value.get('launcher', {})):
                return value
        if process_alive(worker):
            return value
        engine = value.get('engine', {})
        if engine.get('owned') and group_alive(engine):
            # A crashed supervisor must not leave a GPU engine orphaned.
            signal_owned_group(engine, signal.SIGKILL)
            if group_alive(engine):
                update_state(self.directory, task['id'], task={'message': '종료된 CARLA 감독 프로세스의 엔진을 정리하는 중입니다.'}, engine={'ready': False})
                return read_state(self.directory)
        if task['status'] == 'Running' or engine.get('running'):
            stopped = task.get('stop_requested', False)
            terminal = {
                'status': 'Stopped' if stopped else 'Failed',
                'phase': 'stopped' if stopped else 'failed',
                'message': 'CARLA 작업을 중지했습니다.' if stopped else 'CARLA 감독 프로세스가 예기치 않게 종료되었습니다. 로그를 확인하세요.',
            }
            receipt = Path(value['run_dir']) / 'exit.json'
            if receipt.is_file():
                try:
                    saved = json.loads(receipt.read_text(encoding='utf-8'))
                    if saved.get('status') in ('Completed', 'Failed', 'Stopped'):
                        terminal.update(saved.get('task', {}))
                        terminal['status'] = 'Stopped' if stopped else saved['status']
                except (OSError, ValueError, TypeError):
                    pass
            update_state(self.directory, task['id'], task=terminal, engine={'running': False, 'ready': False})
        return read_state(self.directory)

    def status(self):
        with self._lock:
            value = self._reconcile()
        settings = self.config.settings()
        allowed, reason = local_permission(settings['carla_host'])
        info = self._installation()
        info['install_path'] = info['path']
        from sdv.runtime.install import VERSION
        info['required_version'] = VERSION
        if info.pop('_path_error', False):
            allowed, reason = False, info['error']
        if not value:
            task = {'id': None, 'operation': None, 'status': 'Idle', 'phase': 'idle', 'progress_pct': None,
                    'message': 'CARLA 설치 또는 실행을 시작할 수 있습니다.', 'started_at': None, 'updated_at': None, 'stop_requested': False}
            engine = {'running': False, 'ready': False, 'owned': False, 'host': settings['carla_host'], 'port': settings['carla_port'], 'pid': None}
        else:
            task = dict(value['task'])
            engine = {key: value['engine'].get(key) for key in ('running', 'ready', 'owned', 'host', 'port', 'pid','state','exit_code','last_error','profile','last_stage','rpc')}
        engine['last_pid']=engine.get('pid')
        if not engine.get('running'):engine['pid']=None
        engine.setdefault('state','Stopped')
        if engine.get('state') is None:engine['state']='Stopped'
        return dict(info, local_allowed=allowed, local_reason=reason, task=task, engine=engine)

    def busy(self):
        return self.status()['task']['status'] == 'Running'

    def _worker_command(self, task_id):
        return [sys.executable, '-u', str(Path(__file__).with_name('carla_worker.py')),
                '--root', str(self.config.root), '--task', task_id]

    def start(self, operation='prepare'):
        if operation not in ('prepare', 'install','test'):
            raise ValueError('지원 CARLA 작업은 prepare(설치 후 실행), install(설치만)입니다.')
        if self._closing:
            raise ValueError('서버가 종료 중입니다.')
        from .carla_diagnostics import engine_root
        engine_root(self.config)
        with state_lock(self.directory, 'launch.lock', blocking=False):
            current = self.status()
            if current['task']['status'] == 'Running' or current['engine']['running']:
                raise ValueError('CARLA 작업 또는 이 서버에서 실행한 엔진이 이미 동작 중입니다.')
            if operation in ('prepare','test') and not current['local_allowed']:
                raise ValueError(current['local_reason'])
            settings = self.config.settings()
            previous=read_state(self.directory)
            if previous.get('crash_streak',0)>=2 and previous.get('settings',{}).get('carla_render_profile','offscreen')==settings.get('carla_render_profile','offscreen'):
                raise ValueError('동일 프로파일에서 Segmentation Fault가 반복되어 재시작을 중단했습니다. 진단 로그를 확인하고 프로파일을 변경하세요.')
            task_id = uuid.uuid4().hex[:16]
            run = self.directory / task_id
            run.mkdir()
            task = {'id': task_id, 'operation': operation, 'status': 'Running', 'phase': 'queued', 'progress_pct': None,
                    'message': 'CARLA 준비 작업을 시작합니다.', 'started_at': timestamp(), 'updated_at': timestamp(), 'stop_requested': False}
            value = {'task': task, 'engine': {'running': False, 'ready': False, 'owned': False,
                     'host': settings['carla_host'], 'port': settings['carla_port'], 'pid': None, 'created': None, 'descendants': []},
                     'crash_streak':previous.get('crash_streak',0) if previous.get('settings',{}).get('carla_render_profile')==settings.get('carla_render_profile') else 0,
                     'settings': settings, 'town': 'Town01' if operation=='test' else self.config.sensors()['towns'][0],
                     'worker': {}, 'launcher': identity(os.getpid()), 'launch_unix': time.time(), 'run_dir': str(run)}
            with state_lock(self.directory):
                atomic_json(self.directory / 'state.json', value)
            try:
                with (run / 'worker.log').open('ab', buffering=0) as log:
                    process = subprocess.Popen(self._worker_command(task_id), cwd=self.config.root,
                                               stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                self._owned[task_id] = process
                try:
                    worker_identity = identity(process.pid)
                except psutil.Error:
                    if process.poll() is None:
                        process.terminate()
                    raise RuntimeError('CARLA 감독 프로세스 식별 정보를 확인할 수 없습니다.')
                update_state(self.directory, task_id, worker=worker_identity)
            except Exception as error:
                update_state(self.directory, task_id, task={'status': 'Failed', 'phase': 'failed', 'message': 'CARLA 작업 실행 실패: ' + str(error)})
        return self.status()

    def _request_stop(self, value):
        task_id = value['task']['id']
        atomic_json(Path(value['run_dir']) / 'stop.json', {'requested_at': timestamp()})
        update_state(self.directory, task_id, task={'stop_requested': True, 'message': 'CARLA 작업과 소유 엔진을 종료하는 중입니다.'})
        return task_id

    def stop(self):
        with state_lock(self.directory, 'launch.lock'):
            value = self._reconcile()
            if not value or (value['task']['status'] != 'Running' and not value['engine']['running']):
                return self.status()
            task_id = self._request_stop(value)
            with self._lock:
                if task_id not in self._stopping:
                    self._stopping.add(task_id)
                    threading.Thread(target=self._stop_watch, args=(task_id,), daemon=True).start()
        return self.status()

    def _stop_watch(self, task_id):
        started = time.monotonic()
        try:
            while True:
                value = read_state(self.directory)
                if value.get('task', {}).get('id') != task_id:
                    return
                worker = value.get('worker', {})
                if not process_alive(worker):
                    self._reconcile()
                    return
                elapsed = time.monotonic() - started
                if elapsed >= 22:
                    engine = value.get('engine', {})
                    if engine.get('owned'):
                        signal_owned_group(engine, signal.SIGKILL)
                    signal_owned_group(worker, signal.SIGKILL)
                    self._reconcile()
                    return
                if elapsed >= 17:
                    signal_owned_group(worker, signal.SIGTERM)
                time.sleep(.2)
        finally:
            self._stopping.discard(task_id)

    def logs(self, limit=60000):
        value = read_state(self.directory)
        if not value:
            return ''
        run = Path(value['run_dir'])
        result = []
        for filename in ('worker.log', 'engine.log','unreal.log'):
            path = run / filename
            if not path.is_file():
                continue
            with path.open('rb') as stream:
                stream.seek(max(0, path.stat().st_size - limit // 2))
                result.append('[' + filename + ']\n' + stream.read(limit // 2).decode('utf-8', errors='replace'))
        return '\n'.join(result)

    def shutdown(self):
        self._closing = True
        self.stop()
        deadline = time.monotonic() + 24
        while time.monotonic() < deadline:
            value = self._reconcile()
            if not value or not process_alive(value.get('worker', {})):
                break
            time.sleep(.1)
        self._reap()
