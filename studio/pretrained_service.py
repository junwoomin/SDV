"""Prepare registry-selected upstream weights without importing torch.

Only ensure() starts work. status() reads local state/cache and never downloads.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone

from .config import atomic_json
from sdv.training.model_registry import BACKBONES

WEIGHTS = {name: dict(spec['weights']) for name, spec in BACKBONES.items()}
DOWNLOAD_TIMEOUT = 600.0
SOCKET_TIMEOUT = 15.0
MAX_BYTES = 192 * 1024 * 1024
CHUNK_BYTES = 1024 * 1024
REDIRECT_HOSTS = frozenset(('download.pytorch.org', 'github.com', 'release-assets.githubusercontent.com', 'objects.githubusercontent.com'))


class Cancelled(Exception):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


class UpstreamRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        target = urllib.parse.urlsplit(newurl)
        if target.scheme != 'https' or target.hostname not in REDIRECT_HOSTS or target.username or target.password:
            raise ValueError('공식 가중치 배포 호스트 외부로 이동하는 다운로드를 중단했습니다.')
        return super().redirect_request(request, fp, code, message, headers, newurl)


class PretrainedService:
    def __init__(self, config):
        self.config = config
        self.directory = config.path('.runtime/studio/pretrained')
        self.cache = config.path('.runtime/cache/torch/hub/checkpoints')
        self._lock = threading.RLock()
        self._validation = {}
        self._threads = {}
        self._closing = False

    def _spec(self, encoder):
        if not isinstance(encoder, str) or encoder not in WEIGHTS:
            raise ValueError('지원되는 사전학습 인코더를 선택하세요: ' + ', '.join(WEIGHTS))
        return WEIGHTS[encoder]

    def _record_key(self, encoder):
        # Aliases share state and a cross-process lock, not only a target file.
        # Keep the first (legacy) key so existing ResNet.json records still work.
        filename = self._spec(encoder)['filename']
        return next(name for name, spec in WEIGHTS.items() if spec['filename'] == filename)

    def _paths(self, encoder):
        spec = self._spec(encoder)
        target = self.cache / spec['filename']
        self.config.path(str(target))
        if target.is_symlink():
            raise ValueError('사전학습 가중치 캐시가 심볼릭 링크입니다. 실제 파일 경로를 확인하세요: ' + str(target))
        key = self._record_key(encoder)
        return target, self.directory / (key + '.json'), self.directory / (key + '.lock')

    def _base(self, encoder):
        spec = self._spec(encoder)
        return dict(spec, encoder=encoder, path=str(self.cache / spec['filename']), status='Idle', phase='missing',
                    ready=False, progress_pct=None, downloaded_bytes=0, total_bytes=None,
                    message='사전학습 가중치가 없습니다. 학습 준비 확인에서 자동으로 준비합니다.', error=None,
                    started_at=None, updated_at=None)

    def _read(self, path):
        if not path.exists():
            return {}
        value = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(value, dict):
            raise ValueError('사전학습 가중치 상태 파일 형식이 잘못되었습니다.')
        return value

    def _write(self, encoder, state, **fields):
        state.update(fields)
        state['updated_at'] = now()
        atomic_json(self.directory / (self._record_key(encoder) + '.json'), state)

    def _locked(self, path):
        if not path.exists():
            return False
        with path.open('rb') as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            return False

    def _signature(self, path):
        item = path.stat()
        return (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns)

    def _digest(self, path, cancel=None, deadline=None):
        signature = self._signature(path)
        cached = self._validation.get(str(path))
        if cached and cached[0] == signature:
            return cached[1]
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            while True:
                self._check_cancel(cancel, deadline)
                chunk = stream.read(CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
        if self._signature(path) != signature:
            raise ValueError('검사 중 가중치 파일이 변경되었습니다. 파일 쓰기가 끝난 뒤 다시 확인하세요.')
        result = digest.hexdigest()
        self._validation[str(path)] = (signature, result)
        return result

    def _check_cancel(self, cancel, deadline):
        if cancel is not None and cancel.is_set():
            raise Cancelled('서버 종료 요청으로 가중치 준비를 중지했습니다. 다시 준비하면 재시도합니다.')
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError('사전학습 가중치 준비가 10분 안에 끝나지 않았습니다. 네트워크를 확인한 뒤 다시 준비하세요.')

    def status(self, encoder=None):
        if encoder is None:
            models = {name: self.status(name) for name in WEIGHTS}
            return {'models': models, 'busy': any(item['status'] == 'Running' for item in models.values())}
        result = self._base(encoder)
        try:
            target, record, lock_path = self._paths(encoder)
            saved = self._read(record)
            dynamic = {'status', 'phase', 'ready', 'progress_pct', 'downloaded_bytes', 'total_bytes',
                       'message', 'error', 'started_at', 'updated_at', 'sha256', 'task_id'}
            result.update({key: value for key, value in saved.items() if key in dynamic})
            result['ready'] = False
            if target.exists():
                if not target.is_file():
                    raise ValueError('사전학습 가중치 경로가 파일이 아닙니다: ' + str(target))
                digest = self._digest(target)
                if not digest.startswith(result['hash_prefix']):
                    raise ValueError('기존 가중치 SHA256 검증 실패: {}. 기존 파일은 보존했습니다. 해당 파일을 다른 이름으로 옮긴 후 다시 준비하세요.'.format(target.name))
                size = target.stat().st_size
                result.update(status='Completed', phase='ready', ready=True, progress_pct=100,
                              downloaded_bytes=size, total_bytes=size, sha256=digest, error=None,
                              message='검증된 사전학습 가중치를 사용할 수 있습니다.')
            elif result['status'] == 'Running' and not self._locked(lock_path):
                result.update(status='Failed', phase='failed', ready=False,
                              error='이전 가중치 준비 프로세스가 종료되었습니다.', message='이전 가중치 준비가 중단되었습니다. 준비 확인을 다시 실행하면 재시도합니다.')
            elif result['status'] == 'Completed':
                result.update(status='Idle', phase='missing', ready=False, progress_pct=None,
                              error=None, message='기존에 준비한 가중치 파일이 없어 다시 준비해야 합니다.')
        except (OSError, ValueError) as error:
            result.update(status='Failed', phase='failed', ready=False, error=str(error), message=str(error))
        return result

    def ensure(self, encoder):
        self._spec(encoder)
        with self._lock:
            if self._closing:
                raise ValueError('웹 서버가 종료 중이어서 가중치 준비를 시작할 수 없습니다.')
            current = self.status(encoder)
            if current['ready'] or current['status'] == 'Running':
                return current
            try:
                target, record, lock_path = self._paths(encoder)
            except ValueError:
                return current
            # Never overwrite a corrupt/unknown user file under a known filename.
            if target.exists():
                return current
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
                self.cache.mkdir(parents=True, exist_ok=True)
                handle = lock_path.open('a+b')
            except OSError as error:
                current.update(status='Failed', phase='failed', ready=False, message=str(error), error=str(error))
                return current
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                state = self.status(encoder)
                if state['status'] != 'Running' and not state['ready']:
                    state.update(status='Running', phase='checking', message='다른 요청에서 가중치를 준비하고 있습니다.')
                return state
            state = self._base(encoder)
            state.update(task_id=uuid.uuid4().hex[:16], status='Running', phase='checking',
                         started_at=now(), message='공식 사전학습 가중치 다운로드를 준비합니다.')
            cancel = threading.Event()
            try:
                self._write(encoder, state)
                thread = threading.Thread(target=self._prepare, args=(encoder, state, handle, cancel), daemon=True,
                                          name='sdv-pretrained-' + encoder)
                self._threads[encoder] = (thread, cancel)
                thread.start()
            except Exception:
                self._threads.pop(encoder, None)
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()
                raise
            return dict(state)

    def _open(self, spec):
        # URLs are selected from WEIGHTS, never accepted from an HTTP request.
        allowed = {item['source_url'] for item in WEIGHTS.values()}
        if spec['source_url'] not in allowed:
            raise ValueError('지원하지 않는 사전학습 가중치 URL입니다.')
        request = urllib.request.Request(spec['source_url'], headers={'User-Agent': 'SDV-Studio/1.0', 'Accept-Encoding': 'identity'})
        return urllib.request.build_opener(UpstreamRedirect()).open(request, timeout=SOCKET_TIMEOUT)

    def _prepare(self, encoder, state, lock_handle, cancel):
        temporary = None
        try:
            target, _, _ = self._paths(encoder)
            spec = self._spec(encoder)
            deadline = time.monotonic() + DOWNLOAD_TIMEOUT
            self._check_cancel(cancel, deadline)
            # Recheck under the interprocess lock, including a just-created file.
            if target.exists():
                if not self._digest(target, cancel, deadline).startswith(spec['hash_prefix']):
                    raise ValueError('다른 작업에서 만든 기존 가중치의 SHA256 검증에 실패했습니다. 파일은 덮어쓰지 않았습니다.')
            else:
                fd, name = tempfile.mkstemp(prefix=spec['filename'] + '.', suffix='.part', dir=str(self.cache))
                temporary = Path(name)
                total, received, last_write = None, 0, 0
                with os.fdopen(fd, 'wb') as output:
                    with self._open(spec) as response:
                        code = response.getcode()
                        if code != 200:
                            raise IOError('가중치 서버가 HTTP {}를 반환했습니다.'.format(code))
                        raw_length = response.headers.get('Content-Length')
                        total = int(raw_length) if raw_length is not None else None
                        if total is not None and (total <= 0 or total > MAX_BYTES):
                            raise ValueError('가중치 응답 크기가 허용 범위를 벗어났습니다.')
                        self._write(encoder, state, phase='downloading', total_bytes=total,
                                    message='공식 {} 가중치를 다운로드합니다.'.format(spec['model']))
                        read = getattr(response, 'read1', response.read)
                        while True:
                            self._check_cancel(cancel, deadline)
                            chunk = read(CHUNK_BYTES)
                            if not chunk:
                                break
                            received += len(chunk)
                            if received > MAX_BYTES or (total is not None and received > total):
                                raise ValueError('가중치 다운로드가 예상 크기를 초과했습니다.')
                            output.write(chunk)
                            current = time.monotonic()
                            if current - last_write >= .25:
                                last_write = current
                                self._write(encoder, state, downloaded_bytes=received,
                                            progress_pct=95 * received / total if total else None)
                    output.flush()
                    os.fsync(output.fileno())
                if total is not None and received != total:
                    raise IOError('가중치 다운로드 크기가 일치하지 않습니다: {} / {} bytes'.format(received, total))
                self._write(encoder, state, phase='verifying', progress_pct=95, downloaded_bytes=received,
                            total_bytes=total or received, message='다운로드한 가중치 SHA256을 검증합니다.')
                digest = self._digest(temporary, cancel, deadline)
                if not digest.startswith(spec['hash_prefix']):
                    raise ValueError('다운로드한 가중치 SHA256이 공식 파일과 일치하지 않습니다. 검증되지 않은 파일은 사용하지 않습니다.')
                self._check_cancel(cancel, deadline)
                try:
                    # Publish a verified file atomically without ever replacing an
                    # existing file. Both names are on the same cache filesystem.
                    os.link(str(temporary), str(target))
                except FileExistsError:
                    if not self._digest(target, cancel, deadline).startswith(spec['hash_prefix']):
                        raise ValueError('다운로드 도중 다른 가중치 파일이 생겼습니다. 기존 파일은 보존했습니다.')
                temporary.unlink()
                temporary = None
            digest = self._digest(target, cancel, deadline)
            size = target.stat().st_size
            self._write(encoder, state, status='Completed', phase='ready', ready=True, progress_pct=100,
                        downloaded_bytes=size, total_bytes=size, sha256=digest, error=None,
                        message='사전학습 가중치 준비와 SHA256 검증을 완료했습니다.')
        except Cancelled as error:
            self._write(encoder, state, status='Stopped', phase='stopped', ready=False, message=str(error), error=None)
        except Exception as error:
            self._write(encoder, state, status='Failed', phase='failed', ready=False, message=str(error), error=str(error))
        finally:
            if temporary is not None:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()

    def shutdown(self):
        with self._lock:
            self._closing = True
            jobs = list(self._threads.values())
        for thread, cancel in jobs:
            cancel.set()
        deadline = time.monotonic() + SOCKET_TIMEOUT + 3
        for thread, cancel in jobs:
            thread.join(max(0, deadline - time.monotonic()))
