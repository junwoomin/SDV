"""Project-local caches and ownership-scoped, bounded subprocesses."""
import atexit
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import uuid

from sdv.paths import ROOT, RUNTIME
from .actors import _cleanup_note, destroy_owned_actors
OWNED = []
CLEANUPS = []

def prepare():
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.chdir(ROOT)
    for name in ('home', 'cache', 'tmp', 'logs'):
        (RUNTIME / name).mkdir(parents=True, exist_ok=True)
    os.environ.update(XDG_CACHE_HOME=str(RUNTIME / 'cache'),
                      TMPDIR=str(RUNTIME / 'tmp'), PYTHONDONTWRITEBYTECODE='1',
                      PYTHONUNBUFFERED='1', TORCH_HOME=str(RUNTIME / 'cache/torch'), MPLCONFIGDIR=str(RUNTIME / 'cache/matplotlib'),
                      SDL_AUDIODRIVER='dummy', PYGAME_HIDE_SUPPORT_PROMPT='1',
                      OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    atexit.register(stop_all)
    def shutdown(signum, frame):
        stop_all()
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

class Job:
    def __init__(self, command, label, cpu=1, memory_gib=2, seconds=90, vram_mib=0):
        if not shutil_which('systemd-run'):
            raise RuntimeError('systemd 사용자 자원 제한을 사용할 수 없습니다.')
        self.label, self.reason = label, ''
        self.started = time.monotonic()
        self.gpu_baseline = 0
        if vram_mib:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3)
            if result.returncode:
                raise RuntimeError('GPU 감시를 사용할 수 없어 CARLA를 시작하지 않습니다.')
            self.gpu_baseline = sum(int(v) for v in result.stdout.splitlines())
        self.unit = 'sdv-' + uuid.uuid4().hex + '.scope'
        self.log_path = RUNTIME / 'logs' / (label + '-' + self.unit[4:12] + '.log')
        self.output = open(self.log_path, 'w', encoding='utf-8')
        self.stopped = False
        self.command = ['systemd-run', '--user', '--scope', '--quiet', '--unit=' + self.unit,
                        '-p', 'CPUQuota=' + str(cpu * 100) + '%', '-p', 'MemoryMax=' + str(memory_gib) + 'G',
                        '-p', 'MemorySwapMax=0', 'nice', '-n', '10', 'timeout', '--signal=TERM',
                        '--kill-after=5s', str(seconds) + 's'] + list(map(str, command))
        self.proc = subprocess.Popen(self.command, cwd=ROOT, env=os.environ.copy(),
                                     stdout=self.output, stderr=subprocess.STDOUT, start_new_session=True)
        self.pid = self.proc.pid
        OWNED.append(self)
        if vram_mib:
            threading.Thread(target=self._guard, args=(vram_mib,), daemon=True).start()
    def poll(self):
        return self.proc.poll()
    def _guard(self, limit):
        import psutil
        while self.poll() is None and not self.stopped:
            try:
                pids = {str(p.pid) for p in psutil.Process(self.pid).children(recursive=True)}
                result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                         '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3)
                if result.returncode:
                    self.reason = 'GPU 사용량 감시 실패'
                    self.stop()
                    return
                used = sum(int(row.split(',')[1].strip()) for row in result.stdout.splitlines()
                           if row.split(',')[0].strip() in pids)
                total = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3)
                if total.returncode:
                    raise RuntimeError('GPU 전체 메모리 감시 실패')
                # Graphics-only processes may be absent from query-compute-apps.
                used = max(used, sum(int(v) for v in total.stdout.splitlines()) - self.gpu_baseline)
                if used > limit:
                    self.reason = f'GPU 메모리 제한 초과: {used}MiB / {limit}MiB'
                    self.stop()
                    return
            except Exception as error:
                self.reason = 'GPU 감시 오류: ' + str(error)
                self.stop()
                return
            time.sleep(1)
    def stop(self):
        if self.stopped:
            return
        self.stopped = True
        # Unit is created by this Job; never match or terminate unrelated processes.
        try:
            subprocess.run(['systemctl', '--user', 'stop', self.unit], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=8, check=True)
        except (subprocess.SubprocessError, OSError) as error:
            _cleanup_note({'unit': self.unit, 'stop_error': str(error)})
            try:
                subprocess.run(['systemctl', '--user', 'kill', '--kill-whom=all', '--signal=SIGKILL', self.unit],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2)
            except (subprocess.SubprocessError, OSError): pass
        if self.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=3)
        self.output.close()
        import json
        with (RUNTIME / 'logs/process-events.jsonl').open('a',encoding='utf-8') as output:
            output.write(json.dumps({'job':self.label,'unit':self.unit,'reason':self.reason,'elapsed':round(time.monotonic()-self.started,2),'returncode':self.poll()},ensure_ascii=False)+'\n')
    terminate = stop
    kill = stop

from shutil import which as shutil_which

def _cleanup_callbacks(callbacks):
    for cleanup in list(callbacks):
        try: cleanup()
        except Exception as error: _cleanup_note({'callback_error': str(error)})

def stop_all():
    _cleanup_callbacks(CLEANUPS)
    CLEANUPS.clear()
    for job in list(OWNED):
        try: job.stop()
        except Exception as error: _cleanup_note({'job': job.label, 'stop_error': str(error)})

def stop_carla():
    jobs = [job for job in OWNED if job.label == 'carla']
    callbacks = [callback for callback in CLEANUPS
                 if any(getattr(getattr(callback, '__self__', None), 'server_job', None) is job for job in jobs)]
    _cleanup_callbacks(callbacks)
    for callback in callbacks:
        if callback in CLEANUPS: CLEANUPS.remove(callback)
    for job in jobs:
        try: job.stop()
        except Exception as error: _cleanup_note({'job': job.label, 'stop_error': str(error)})

def launch_carla(carla_root, port, town="Town01"):
    root = Path(carla_root).resolve()
    if ROOT not in root.parents:
        raise ValueError('CARLA 경로는 SDV_V1 안에 있어야 합니다.')
    binary = root / 'CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping'
    if not binary.is_file() or (root / 'VERSION').read_text().strip() != '0.9.15':
        raise ValueError('CARLA 0.9.15 배포 파일을 확인하세요.')
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3)
    if result.returncode or min(int(v) for v in result.stdout.splitlines()) < 8192:
        raise RuntimeError('GPU 여유 메모리 8GiB 이상이 필요합니다. 실행 중인 작업을 확인하세요.')
    import socket
    for used_port in (port, port + 1, port + 2):
        check = socket.socket()
        try:
            check.bind(('0.0.0.0', used_port))
        except OSError:
            raise RuntimeError(f'포트 {used_port}가 사용 중입니다. 다른 서비스를 종료하지 않고 실행을 보류합니다.')
        finally: check.close()
    from sdv.config.validation import TOWNS
    if town not in TOWNS or not (root / 'CarlaUE4/Content/Carla/Maps' / (town + '.umap')).is_file():
        raise ValueError('선택한 타운의 로컬 맵 파일을 확인하세요.')
    from studio.carla_diagnostics import PROFILES
    profile=os.environ.get('SDV_CARLA_RENDER_PROFILE','offscreen')
    if profile not in PROFILES:raise ValueError('Unknown CARLA rendering profile')
    return Job([binary, 'CarlaUE4', '/Game/Carla/Maps/' + town] + PROFILES[profile] + ['-ResX=640', '-ResY=360',
                '-benchmark', '-fps=10', '-nosound', f'-carla-rpc-port={port}', '-stdout', '-FullStdOutLogOutput', '-log', f'-Abslog={RUNTIME / "logs/carla-engine.log"}'], 'carla',
               cpu=4, memory_gib=8, seconds=3600, vram_mib=0)

def connect_carla(job, port, seconds=75, cancel=None):
    """Do not reuse an RPC client whose initial TCP connection was refused."""
    import carla
    import socket
    deadline = min(job.started + 85, time.monotonic() + seconds)
    last_error = '서버 포트 준비 중'
    connection_log = RUNTIME / 'logs/connection.log'
    while time.monotonic() < deadline:
        if cancel and cancel.is_set():
            raise RuntimeError('CARLA 연결을 취소했습니다.')
        if job.poll() is not None:
            raise RuntimeError(f'CARLA 종료({job.poll()}): {job.reason} 로그 {job.log_path.name}')
        probe = socket.socket()
        probe.settimeout(.25)
        ready = probe.connect_ex(('127.0.0.1', port)) == 0
        probe.close()
        if ready:
            client = carla.Client('127.0.0.1', port, worker_threads=2)
            client.set_files_base_folder(str(RUNTIME / 'cache/carla'))
            client.set_timeout(2.0)
            try:
                version = client.get_server_version()
                if version != client.get_client_version():
                    raise ValueError(f'CARLA 버전 불일치: 서버 {version}, Python {client.get_client_version()}')
                world = client.get_world()
                client.set_timeout(5.0)
                with connection_log.open('a', encoding='utf-8') as output:
                    output.write(f'{time.monotonic()-job.started:.2f}s server={version} world={world.id} connected\n')
                return client, world
            except RuntimeError as error:
                last_error = str(error)
                with connection_log.open('a', encoding='utf-8') as output:
                    output.write(f'{time.monotonic()-job.started:.2f}s {last_error}\n')
        if cancel: cancel.wait(.25)
        else: time.sleep(.25)
    raise TimeoutError('CARLA 연결 시간 초과: ' + last_error + ' · 다시 시도하거나 로그를 확인하세요.')
