"""Durable supervisor for a draft sensor layout preview in the research Python.

The worker owns CARLA actors and frame production. This service owns process
metadata and commands; it never rewrites the worker's status or saved settings.
"""
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
import uuid

from .config import CAMERAS, atomic_json, integer
from .carla_service import (descendants, group_alive, identity, process_alive,
                            signal_owned_group, state_lock, timestamp)

VIEWS = frozenset(('perspective', 'top', 'side'))
CHANNELS = frozenset(('lidar',)) | frozenset(name + '_rgb' for name in CAMERAS + ('front_cam_pv',))
TERMINAL = frozenset(('Completed', 'Failed', 'Stopped'))


def _read_json(path, default=None):
    if not path.exists():
        return {} if default is None else default
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('미리보기 상태 파일은 JSON 객체여야 합니다: ' + path.name)
    return value


def _view(value):
    if not isinstance(value, str) or value not in VIEWS:
        raise ValueError('미리보기 시점은 perspective, top, side 중 하나여야 합니다.')
    return value


def _has_frame(item, revision):
    frame = item.get('frame')
    return (item.get('revision') == revision and isinstance(frame, (int, float))
            and not isinstance(frame, bool) and frame >= 0)


class SensorPreviewService:
    def __init__(self, config):
        self.config = config
        self.directory = config.path('.runtime/studio/sensor-preview')
        self.directory.mkdir(parents=True, exist_ok=True)
        self._owned = {}
        self._stopping = set()
        self._closing = False
        self._lock = threading.RLock()

    def _current(self):
        pointer = _read_json(self.directory / 'current.json')
        session_id = pointer.get('session_id')
        if session_id is None:
            return None, None
        if not isinstance(session_id, str) or not re.fullmatch(r'[a-f0-9]{16}', session_id):
            raise ValueError('저장된 센서 미리보기 세션 ID가 올바르지 않습니다.')
        run = self.directory / session_id
        if run.is_symlink() or run.resolve().parent != self.directory.resolve():
            raise ValueError('허용되지 않은 센서 미리보기 세션 경로입니다.')
        return session_id, run

    def _metadata(self, run):
        return _read_json(run / 'metadata.json')

    def _update(self, session_id, **fields):
        with state_lock(self.directory):
            current_id, run = self._current()
            if current_id != session_id:
                return None
            metadata = self._metadata(run)
            metadata.update(fields)
            metadata['updated_at'] = timestamp()
            atomic_json(run / 'metadata.json', metadata)
            return metadata

    def _worker_status(self, run):
        try:
            return _read_json(run / 'status.json')
        except (OSError, ValueError) as error:
            return {'message': '센서 미리보기 상태를 읽을 수 없습니다: ' + str(error), 'error': str(error)}

    def _reap(self):
        for session_id, process in list(self._owned.items()):
            return_code = process.poll()
            if return_code is not None:
                self._update(session_id, return_code=return_code)
                self._owned.pop(session_id, None)

    def _reconcile(self):
        self._reap()
        session_id, run = self._current()
        if run is None:
            return None, {}, {}
        metadata = self._metadata(run)
        worker = self._worker_status(run)
        if not metadata:
            return run, {'session_id': session_id, 'status': 'Failed', 'phase': 'failed',
                         'message': '센서 미리보기 프로세스 기록이 없습니다.', 'error': 'metadata.json missing'}, worker
        record = metadata.get('process', {})
        if not record.get('pid') and time.time() - metadata.get('launch_unix', 0) < 5:
            if process_alive(metadata.get('launcher', {})):
                return run, metadata, worker
        if process_alive(record):
            children = descendants(record)
            if children != record.get('descendants', []):
                record = dict(record, descendants=children)
                metadata = self._update(session_id, process=record) or metadata
            return run, metadata, worker
        if group_alive(record):
            signal_owned_group(record, signal.SIGKILL)
            if group_alive(record):
                metadata = self._update(session_id, status='Running', phase='cleaning',
                                        message='센서 미리보기의 남은 프로세스를 정리하는 중입니다.') or metadata
                return run, metadata, worker
        if metadata.get('status') == 'Running':
            requested = metadata.get('stop_requested', False)
            worker_terminal = worker.get('status')
            if metadata.get('forced_stop'):
                status, phase = 'Stopped', 'stopped'
                message = '센서 미리보기 프로세스를 강제 종료했습니다. CARLA 액터 정리 완료 여부는 확인하지 못했습니다.'
                error = message
            elif worker_terminal in TERMINAL:
                status = worker_terminal
                phase = worker.get('phase') or status.lower()
                message = worker.get('message') or ('센서 미리보기를 종료했습니다.' if requested else '센서 미리보기 프로세스가 종료되었습니다.')
                error = worker.get('error')
            else:
                status, phase = 'Failed', 'failed'
                message = '센서 미리보기 프로세스가 정리 완료 상태를 남기지 않고 종료되었습니다. 로그를 확인하세요.'
                lines = [line.strip() for line in self.logs(limit=2000).splitlines() if line.strip()]
                if lines:
                    message += ' ' + lines[-1][:1000]
                error = message
            metadata = self._update(session_id, status=status, phase=phase, message=message, error=error,
                                    finished_at=timestamp()) or metadata
        return run, metadata, worker

    def status(self):
        with self._lock:
            run, metadata, worker = self._reconcile()
        if run is None:
            settings = self.config.settings()
            return {'session_id': None, 'status': 'Idle', 'phase': 'idle', 'message': '센서 위치를 확인하려면 미리보기를 시작하세요.',
                    'error': None, 'busy': False, 'ready': False, 'revision': 0, 'requested_revision': 0,
                    'view': 'perspective', 'requested_view': 'perspective', 'sensors': None, 'channels': [],
                    'preview_fps': settings.get('preview_fps', 24), 'requested_fps': settings.get('preview_fps', 24),
                    'host': settings['carla_host'], 'port': settings['carla_port'], 'started_at': None,
                    'updated_at': None, 'stop_requested': False}
        active = metadata.get('status') == 'Running'
        phase = worker.get('phase', metadata.get('phase', 'starting')) if active else metadata.get('phase', 'failed')
        message = worker.get('message') or metadata.get('message', '') if active else metadata.get('message', '')
        error = worker.get('error') if active else metadata.get('error')
        if active and metadata.get('stop_requested'):
            phase, message = 'stopping', '센서와 미리보기 차량을 정리하는 중입니다.'
        if active and worker.get('status') in TERMINAL:
            phase, message = 'cleaning', '센서 정리를 마친 작업 프로세스가 종료되는 중입니다.'
        revision = worker.get('revision', 0)
        requested_revision = metadata.get('requested_revision', 0)
        revision_matches = revision == requested_revision
        if active and not revision_matches and phase not in ('stopping', 'cleaning'):
            phase, message = 'applying', '요청한 센서 설정을 적용하고 새 프레임을 기다리는 중입니다.'
        channels = []
        for item in worker.get('channels', []):
            if isinstance(item, dict) and item.get('id') in CHANNELS:
                row = dict(item)
                row['path'] = 'frames/' + item['id'] + '.jpg'
                row['ready'] = False
                if revision_matches and _has_frame(item, requested_revision):
                    try:
                        row['ready'] = self._image_path(run, item['id']).is_file()
                    except (OSError, ValueError):
                        pass
                if not row['ready']:
                    # The previous camera pose must not appear to be the new one.
                    row.update(frame=None, timestamp=None, received_at=None, measured_fps=None, received_fps=None)
                channels.append(row)
        return {'session_id': metadata.get('session_id'), 'status': metadata.get('status', 'Failed'),
                'phase': phase, 'message': message, 'error': error, 'busy': active,
                'ready': active and phase == 'live' and bool(channels) and all(row['ready'] for row in channels),
                'revision': revision, 'requested_revision': requested_revision,
                'view': worker.get('view', metadata.get('initial_view', 'perspective')),
                'requested_view': metadata.get('requested_view', 'perspective'), 'sensors': metadata.get('requested_sensors'),
                'channels': channels, 'host': metadata.get('host'), 'port': metadata.get('port'),
                'preview_fps': worker.get('preview_fps', metadata.get('preview_fps', 24)),
                'requested_fps': metadata.get('requested_fps', metadata.get('preview_fps', 24)),
                'frame_rate_note': worker.get('frame_rate_note'),
                'map': worker.get('map'), 'actor_ids': worker.get('actor_ids', []),
                'started_at': metadata.get('started_at'), 'updated_at': worker.get('updated_at') or metadata.get('updated_at'),
                'stop_requested': metadata.get('stop_requested', False), 'return_code': metadata.get('return_code')}

    def busy(self):
        return self.status()['busy']

    def _worker_command(self, run):
        settings = _read_json(run / 'snapshot.json')['settings']
        return [settings['python_executable'], '-u', str(Path(__file__).with_name('sensor_preview_worker.py')),
                '--run', str(run)]

    def start(self, sensors=None, view='perspective', owns_clock=False):
        if self._closing:
            raise ValueError('웹 서버가 종료 중입니다.')
        draft = self.config.validate_sensors(sensors if sensors is not None else self.config.sensors())
        view = _view(view)
        if type(owns_clock) is not bool:
            raise ValueError('미리보기 엔진의 clock 소유 정보가 올바르지 않습니다.')
        with state_lock(self.directory, 'launch.lock', blocking=False):
            if self.busy():
                raise ValueError('센서 미리보기가 이미 실행 중입니다. 설정 적용 또는 중지를 사용하세요.')
            settings = dict(self.config.settings())
            settings['preview_owns_clock'] = owns_clock
            session_id = uuid.uuid4().hex[:16]
            run = self.directory / session_id
            run.mkdir()
            snapshot = {'session_id': session_id, 'settings': settings, 'sensors': draft, 'revision': 0,
                        'view': view, 'exclusive_tick': owns_clock}
            atomic_json(run / 'snapshot.json', snapshot)
            metadata = {'session_id': session_id, 'status': 'Running', 'phase': 'starting',
                        'message': '연구 Python에서 센서 미리보기를 시작합니다.', 'error': None,
                        'started_at': timestamp(), 'updated_at': timestamp(), 'launch_unix': time.time(),
                        'launcher': identity(os.getpid()), 'process': {}, 'stop_requested': False,
                        'requested_revision': 0, 'requested_view': view, 'initial_view': view,
                        'preview_fps': settings.get('preview_fps', 24), 'requested_fps': settings.get('preview_fps', 24),
                        'requested_sensors': draft, 'host': settings['carla_host'], 'port': settings['carla_port']}
            with state_lock(self.directory):
                atomic_json(run / 'metadata.json', metadata)
                atomic_json(self.directory / 'current.json', {'session_id': session_id})
            try:
                with (run / 'worker.log').open('ab', buffering=0) as log:
                    process = subprocess.Popen(self._worker_command(run), cwd=self.config.root, stdin=subprocess.DEVNULL,
                                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                self._owned[session_id] = process
                try:
                    record = identity(process.pid)
                except Exception:
                    if process.poll() is None:
                        process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=3)
                    raise
                self._update(session_id, process=record)
            except Exception as error:
                self._update(session_id, status='Failed', phase='failed', message='센서 미리보기 실행 실패: ' + str(error), error=str(error))
        return self.status()

    def apply(self, sensors, view=None):
        draft = self.config.validate_sensors(sensors)
        if view is not None:
            _view(view)
        with state_lock(self.directory, 'launch.lock'):
            run, metadata, worker = self._reconcile()
            if run is None or metadata.get('status') != 'Running':
                raise ValueError('설정을 적용하려면 센서 미리보기를 먼저 시작하세요.')
            if metadata.get('stop_requested') or worker.get('status') in TERMINAL:
                raise ValueError('미리보기가 종료 중이어서 설정을 적용할 수 없습니다.')
            selected_view = view if view is not None else metadata['requested_view']
            revision = metadata.get('requested_revision', 0) + 1
            fps = integer(self.config.settings().get('preview_fps', 24), '프리뷰 FPS', 1, 60)
            atomic_json(run / 'command.json', {'revision': revision, 'sensors': draft, 'view': selected_view,
                                              'preview_fps': fps})
            self._update(metadata['session_id'], requested_revision=revision, requested_sensors=draft,
                         requested_view=selected_view, requested_fps=fps,
                         message='새 센서 구성과 프리뷰 FPS를 미리보기에 적용합니다.')
        return self.status()

    def stop(self):
        with state_lock(self.directory, 'launch.lock'):
            run, metadata, worker = self._reconcile()
            if run is not None and metadata.get('status') == 'Running':
                session_id = metadata['session_id']
                atomic_json(run / 'stop.json', {'requested_at': timestamp()})
                self._update(session_id, stop_requested=True, message='센서 미리보기를 중지하고 액터를 정리합니다.')
                with self._lock:
                    if session_id not in self._stopping:
                        self._stopping.add(session_id)
                        threading.Thread(target=self._stop_watch, args=(session_id,), daemon=True).start()
        return self.status()

    def _stop_watch(self, session_id):
        started = time.monotonic()
        signaled = False
        try:
            while True:
                run, metadata, worker = self._reconcile()
                if run is None or metadata.get('session_id') != session_id or metadata.get('status') != 'Running':
                    return
                record = metadata.get('process', {})
                elapsed = time.monotonic() - started
                if elapsed >= 15:
                    self._update(session_id, forced_stop=True)
                    signal_owned_group(record, signal.SIGKILL)
                elif elapsed >= 5 and not signaled:
                    signal_owned_group(record, signal.SIGTERM)
                    signaled = True
                time.sleep(.1)
        finally:
            self._stopping.discard(session_id)

    def logs(self, limit=60000):
        session_id, run = self._current()
        if run is None or not (run / 'worker.log').is_file():
            return ''
        path = run / 'worker.log'
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size - limit))
            return stream.read(limit).decode('utf-8', errors='replace')

    def _image_path(self, run, channel):
        directory = run / 'frames'
        if directory.is_symlink():
            raise ValueError('센서 미리보기 프레임 폴더는 심볼릭 링크일 수 없습니다.')
        path = directory / (channel + '.jpg')
        if path.is_symlink() or path.resolve().parent != directory.resolve():
            raise ValueError('허용되지 않은 센서 미리보기 이미지 경로입니다.')
        return path

    def frame(self, channel):
        if not isinstance(channel, str) or channel not in CHANNELS:
            raise ValueError('지원하지 않는 센서 미리보기 채널입니다.')
        with state_lock(self.directory, 'launch.lock'):
            session_id, run = self._current()
            if run is None:
                raise FileNotFoundError('아직 센서 미리보기가 시작되지 않았습니다.')
            path = self._image_path(run, channel)
            metadata = self._metadata(run)
            worker = self._worker_status(run)
            revision = metadata.get('requested_revision', 0)
            item = next((item for item in worker.get('channels', [])
                         if isinstance(item, dict) and item.get('id') == channel), {})
            if worker.get('revision') != revision or not _has_frame(item, revision) or not path.is_file():
                raise FileNotFoundError('현재 센서 설정에 해당하는 새 프레임을 기다리는 중입니다.')
            return path

    def shutdown(self):
        self._closing = True
        self.stop()
        deadline = time.monotonic() + 19
        while time.monotonic() < deadline:
            if not self.busy():
                break
            time.sleep(.1)
        self._reap()
