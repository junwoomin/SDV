"""Dedicated CARLA preparation worker; independent from browser connections."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from studio.config import Config
from studio.carla_service import (descendants, group_alive, identity, local_permission, process_alive,
                                  read_state, signal_owned_group, state_directory,
                                  timestamp, update_state)

READINESS_TIMEOUT = 120.0
CLIENT_INSTALL_TIMEOUT = 600.0
CLIENT_SCRIPT = r'''
import glob,importlib.util,json,os,sys
try:
 spec=importlib.util.find_spec('carla')
 if spec is None or (spec.origin is None and spec.submodule_search_locations is not None):
  # The local engine folder 'carla/' can appear as a namespace package even
  # when the Python wheel is absent. A real CARLA wheel has an __init__.py.
  # A damaged installed distribution is not permission to replace its version.
  distributions=[]
  for path in sys.path:
   if not path or not os.path.isdir(path): continue
   for metadata in glob.glob(os.path.join(path,'carla-*.dist-info','METADATA'))+glob.glob(os.path.join(path,'carla*.egg-info','PKG-INFO')):
    distributions.append(metadata)
  if distributions:
   print(json.dumps({'available':False,'state':'broken','error':'carla 설치 정보는 있지만 모듈이 없습니다: '+distributions[0]}));sys.exit(1)
  print(json.dumps({'available':False,'state':'missing','error':'No module named carla'}));sys.exit(1)
 import carla
 version=carla.Client('127.0.0.1',int(sys.argv[1])).get_client_version()
 if version!='0.9.15':
  print(json.dumps({'available':False,'state':'mismatch','version':version,'error':'Python carla 버전 '+version+' / 필요한 버전 0.9.15'}));sys.exit(1)
 print(json.dumps({'available':True,'state':'ready','version':version}))
except Exception as error:
 print(json.dumps({'available':False,'state':'broken','error':str(error)}));sys.exit(1)
'''
PIP_SCRIPT = r'''
import json
try:
 import pip
 print(json.dumps({'available':True,'version':pip.__version__}))
except Exception as error:
 print(json.dumps({'available':False,'error':str(error)}))
'''
RPC_SCRIPT = r'''
import json,sys
try:
 import carla
except Exception as error:
 print(json.dumps({'connected':False,'fatal':True,'error':'연구 Python의 carla 패키지를 불러올 수 없습니다: '+str(error)}));sys.exit(1)
try:
 client=carla.Client(sys.argv[1],int(sys.argv[2]));client.set_timeout(2.0)
 server=client.get_server_version();local=client.get_client_version()
 if server!=local:
  print(json.dumps({'connected':False,'fatal':True,'error':'CARLA 버전 불일치: 서버 '+server+' / Python '+local}));sys.exit(1)
 world=client.get_world();snapshot=world.get_snapshot()
 print(json.dumps({'connected':True,'version':server,'map':world.get_map().name,'frame':snapshot.frame,'timestamp_s':snapshot.timestamp.elapsed_seconds}))
except Exception as error:
 print(json.dumps({'connected':False,'fatal':False,'error':str(error)}));sys.exit(1)
'''

MAP_SCRIPT = r'''
import carla,json,sys
try:
 c=carla.Client(sys.argv[1],int(sys.argv[2]));c.set_timeout(90)
 w=c.get_world()
 if w.get_map().name.split('/')[-1]!=sys.argv[3]:w=c.load_world(sys.argv[3])
 snap=w.get_snapshot()
 print(json.dumps({'connected':True,'version':c.get_server_version(),'map':w.get_map().name,'frame':snap.frame,'timestamp_s':snap.timestamp.elapsed_seconds}))
except Exception as e:
 print(json.dumps({'connected':False,'fatal':True,'error':str(e)}));sys.exit(1)
'''


class StopRequested(Exception):
    pass


class Cancellation:
    def __init__(self, path):
        self.path = path
        self.event = threading.Event()

    def is_set(self):
        return self.event.is_set() or self.path.exists()

    def set(self):
        self.event.set()

    def wait(self, seconds):
        deadline = time.monotonic() + seconds
        while not self.is_set() and time.monotonic() < deadline:
            self.event.wait(min(.1, max(0, deadline - time.monotonic())))
        return self.is_set()


def check_cancel(cancel):
    if cancel.is_set():
        raise StopRequested('CARLA 작업을 중지했습니다.')


def engine_command(config, state):
    from studio.carla_diagnostics import engine_root, PROFILES
    root = engine_root(config, state['settings'])
    binary = root / 'CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping'
    if not binary.is_file():
        raise FileNotFoundError('CARLA 실행 파일이 없습니다. 설치를 완료한 후 다시 실행하세요: ' + str(binary))
    if not os.access(str(binary),os.X_OK):raise PermissionError('CARLA 실행 권한이 없습니다: '+str(binary))
    if (root/'VERSION').read_text().strip()!='0.9.15':raise ValueError('CARLA 0.9.15 설치가 필요합니다.')
    town = state['town']
    from studio.config import TOWNS
    if town not in TOWNS:
        raise ValueError('지원하지 않는 CARLA 타운입니다: ' + str(town))
    if not (root / 'CarlaUE4/Content/Carla/Maps' / (town + '.umap')).is_file():
        raise FileNotFoundError('설치된 CARLA에 선택한 타운 맵이 없습니다: ' + town)
    profile=state['settings'].get('carla_render_profile','offscreen')
    if profile not in PROFILES:raise ValueError('지원하지 않는 렌더링 프로파일입니다.')
    if profile in ('windowed','low') and not os.environ.get('DISPLAY'):raise RuntimeError('화면 렌더링에는 DISPLAY가 필요합니다.')
    return [str(binary), 'CarlaUE4', '/Game/Carla/Maps/' + town] + PROFILES[profile] + [
            '-ResX=640', '-ResY=360', '-nosound', '-carla-rpc-port=' + str(state['settings']['carla_port']),
            '-stdout', '-FullStdOutLogOutput', '-log']


def check_ports(port):
    if not 1 <= port <= 65533:
        raise ValueError('로컬 CARLA는 RPC 포트와 다음 두 포트를 사용합니다. RPC 포트는 1~65533으로 설정하세요.')
    for number in (port, port + 1, port + 2):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('0.0.0.0', number))
        except OSError as error:
            raise RuntimeError('포트 {}가 사용 중이어서 CARLA를 시작하지 않았습니다. 기존 서버는 연결 확인으로 사용하거나 포트 설정을 바꾸세요.'.format(number)) from error
        finally:
            sock.close()


def inspect_client(config, settings):
    """Distinguish absence from a broken import or an existing other version."""
    try:
        result = subprocess.run([settings['python_executable'], '-c', CLIENT_SCRIPT, str(settings['carla_port'])],
                                cwd=config.root, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                universal_newlines=True, timeout=10)
        lines = result.stdout.strip().splitlines()
        answer = json.loads(lines[-1]) if lines else {'available': False, 'state': 'broken', 'error': result.stderr[-1500:]}
        if not isinstance(answer, dict):
            raise ValueError('CARLA 클라이언트 확인 응답이 올바르지 않습니다.')
        if result.returncode != 0 and answer.get('available'):
            raise ValueError('CARLA 클라이언트 확인 프로세스가 실패했습니다.')
        return answer
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        return {'available': False, 'state': 'broken', 'error': str(error)}


def check_client(config, settings):
    answer = inspect_client(config, settings)
    if not answer.get('available'):
        raise RuntimeError('연구 Python의 CARLA 클라이언트를 확인할 수 없습니다: ' + answer.get('error', '알 수 없는 오류'))
    return answer


def _pip_info(config, python):
    try:
        result = subprocess.run([python, '-c', PIP_SCRIPT], cwd=config.root, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, universal_newlines=True, timeout=10)
        lines = result.stdout.strip().splitlines()
        info = json.loads(lines[-1]) if lines else {}
        return info if result.returncode == 0 and isinstance(info, dict) else {}
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return {}


def _version_tuple(value):
    # pip version comparison only needs major/minor; development suffixes are ignored.
    import re
    parts = re.match(r'^(\d+)\.(\d+)', str(value))
    return tuple(map(int, parts.groups())) if parts else (0, 0)


def _pip_command(config, settings, cancel, deadline=None):
    selected = settings['python_executable']
    selected_pip = _pip_info(config, selected)
    if selected_pip.get('available') and _version_tuple(selected_pip.get('version')) >= (20, 3):
        return [selected, '-m', 'pip']
    # A modern web pip cannot always execute under an older target Python.
    # Cache a compatible runner instead of upgrading the research environment.
    runner_cache = config.path('.runtime/cache/pip-client/runner')
    runner_cache.mkdir(parents=True, exist_ok=True)
    wheel = runner_cache / 'pip-24.0-py3-none-any.whl'
    import zipfile
    valid = False
    if wheel.is_file():
        try:
            with zipfile.ZipFile(wheel) as package:
                valid = 'pip/__main__.py' in package.namelist() and 'pip-24.0.dist-info/METADATA' in package.namelist()
        except (OSError, zipfile.BadZipFile):
            pass
    if not valid:
        if wheel.exists():
            wheel.unlink()
        web_pip = _pip_info(config, sys.executable)
        if not web_pip.get('available'):
            raise RuntimeError('연구 Python의 pip를 대신할 설치 도구를 준비하려면 웹 Python의 pip가 필요합니다. start.sh로 웹 환경을 다시 준비한 뒤 실행하세요.')
        _run_client_install(config, [sys.executable, '-m', 'pip', 'download', '--only-binary=:all:',
                            '--no-deps', '--no-input', '--disable-pip-version-check', '--dest', str(runner_cache), 'pip==24.0'], cancel, deadline=deadline)
        if not wheel.is_file():
            raise RuntimeError('Python 3.7 호환 pip 실행 도구를 다운로드하지 못했습니다.')
    runner = "import runpy,sys;sys.path.insert(0,sys.argv.pop(1));runpy.run_module('pip',run_name='__main__')"
    return [selected, '-c', runner, str(wheel)]


def _terminate_installer(process, record):
    # Installers share the worker's process group, so a forced worker shutdown
    # also cleans them. Normal cancellation targets this child tree only.
    record['descendants'] = descendants(record)
    def signal_children(signum):
        for child in record.get('descendants', []):
            if process_alive(child):
                try:
                    os.kill(child['pid'], signum)
                except ProcessLookupError:
                    pass
    signal_children(signal.SIGTERM)
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)
    signal_children(signal.SIGKILL)


def _run_client_install(config, command, cancel, deadline=None):
    cache = config.path('.runtime/cache/pip-client')
    cache.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(PIP_CACHE_DIR=str(cache), PYTHONUNBUFFERED='1', PIP_NO_INPUT='1', PIP_DISABLE_PIP_VERSION_CHECK='1')
    print('Python CARLA 클라이언트 설치: ' + ' '.join(command), flush=True)
    process = subprocess.Popen(command, cwd=config.root, env=env, stdin=subprocess.DEVNULL,
                               stdout=None, stderr=subprocess.STDOUT)
    try:
        record = identity(process.pid)
    except Exception:
        record = {}
    deadline = deadline if deadline is not None else time.monotonic() + CLIENT_INSTALL_TIMEOUT
    try:
        while process.poll() is None:
            record['descendants'] = descendants(record)
            check_cancel(cancel)
            if time.monotonic() >= deadline:
                raise TimeoutError('Python CARLA 클라이언트 설치가 10분 안에 끝나지 않았습니다. 네트워크와 프록시를 확인한 뒤 다시 실행하세요.')
            cancel.wait(.1)
        check_cancel(cancel)
        if process.returncode:
            raise RuntimeError('Python carla==0.9.15 wheel 설치가 실패했습니다(exit {}). 작업 로그에서 네트워크·Python 버전·설치 권한 오류를 확인하세요. 다른 연구 패키지는 변경하지 않았습니다.'.format(process.returncode))
    finally:
        _terminate_installer(process, record)


def ensure_client(config, settings, cancel, progress=None):
    """Install only a missing CARLA wheel into the selected research interpreter."""
    check_cancel(cancel)
    answer = inspect_client(config, settings)
    check_cancel(cancel)
    if answer.get('available'):
        return answer
    if answer.get('state') != 'missing':
        raise RuntimeError('기존 CARLA Python 환경을 자동 변경하지 않았습니다: ' + answer.get('error', '모듈을 불러올 수 없습니다.') + ' · CARLA 0.9.15가 정상 동작하는 연구 Python을 선택하거나 해당 환경을 복구하세요.')
    message = '선택한 연구 Python에 CARLA 0.9.15 클라이언트를 설치합니다. 기존 연구 패키지는 유지합니다.'
    if progress:
        progress(message)
    else:
        print(message, flush=True)
    deadline = time.monotonic() + CLIENT_INSTALL_TIMEOUT
    command = _pip_command(config, settings, cancel, deadline=deadline) + ['install', '--only-binary=:all:', '--no-deps', '--no-input',
               '--disable-pip-version-check', 'carla==0.9.15']
    check_cancel(cancel)
    _run_client_install(config, command, cancel, deadline=deadline)
    check_cancel(cancel)
    answer = check_client(config, settings)
    if progress:
        progress('Python CARLA 0.9.15 클라이언트 설치와 실제 import 검증을 완료했습니다.')
    return answer


def probe_rpc(config, settings, timeout=4, town=None):
    try:
        command=[settings['python_executable'], '-c', MAP_SCRIPT if town else RPC_SCRIPT, settings['carla_host'],str(settings['carla_port'])]
        if town:command.append(town)
        result = subprocess.run(command,
                                cwd=config.root, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                universal_newlines=True, timeout=timeout)
        lines = result.stdout.strip().splitlines()
        answer = json.loads(lines[-1]) if lines else {'connected': False, 'error': result.stderr[-1500:] or 'CARLA RPC 응답이 없습니다.'}
        if not isinstance(answer, dict):
            raise ValueError('CARLA RPC 응답 형식이 올바르지 않습니다.')
        return answer
    except subprocess.TimeoutExpired:
        return {'connected': False, 'error': 'CARLA RPC 응답 대기 중입니다.'}
    except (OSError, ValueError) as error:
        return {'connected': False, 'fatal': True, 'error': str(error)}


def terminate_engine(process, record):
    if process is None:
        return
    record['descendants'] = descendants(record)
    signal_owned_group(record, signal.SIGTERM)
    deadline = time.monotonic() + 6
    while time.monotonic() < deadline:
        process.poll()
        if not group_alive(record):
            break
        time.sleep(.1)
    if group_alive(record):
        signal_owned_group(record, signal.SIGKILL)
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        # The Popen object still owns this unreaped child, even if metadata fails.
        process.kill()
        process.wait(timeout=2)


def run_task(config, task_id, cancel=None):
    from sdv.runtime.install import Cancelled, install
    directory = state_directory(config)
    state = read_state(directory)
    if state.get('task', {}).get('id') != task_id:
        return 1
    run = Path(state['run_dir'])
    cancel = cancel or Cancellation(run / 'stop.json')
    process, engine_log = None, None
    engine = dict(state['engine'])
    terminal = {'status': 'Failed', 'phase': 'failed', 'message': 'CARLA 작업이 예기치 않게 중단되었습니다.'}
    return_code = 1
    last_progress = [0]

    def progress(ratio, message):
        # Download invokes this per MiB. Persist at most twice per second.
        current = time.monotonic()
        if current - last_progress[0] < .5 and ratio < 1 and not cancel.is_set():
            return
        last_progress[0] = current
        pct = max(0, min(100, float(ratio) * 100))
        update_state(directory, task_id, task={'phase': 'installing', 'progress_pct': pct, 'message': str(message)})
        print(message, flush=True)

    try:
        check_cancel(cancel)
        settings = state['settings']
        operation = state['task']['operation']
        if operation in ('prepare','test'):
            allowed, reason = local_permission(settings['carla_host'])
            if not allowed:
                raise ValueError(reason)
            update_state(directory, task_id, task={'phase': 'checking', 'message': '연구 Python의 CARLA 0.9.15 클라이언트를 확인합니다.'})
            check_ports(int(settings['carla_port']))
            def client_progress(message):
                update_state(directory, task_id, task={'phase': 'client_installing', 'progress_pct': None, 'message': message})
                print(message, flush=True)
            ensure_client(config, settings, cancel, client_progress)
        update_state(directory, task_id, task={'phase': 'installing', 'message': 'CARLA 설치 파일을 확인합니다.', 'progress_pct': None})
        from studio.carla_diagnostics import engine_root, diagnostics, crash_evidence
        root=engine_root(config, settings)
        if settings.get('carla_install_path'):
            from sdv.runtime.install import installation_status
            info=installation_status(root)
            if not info['installed']:raise RuntimeError('지정한 기존 CARLA 설치가 유효하지 않습니다: '+str(info.get('error')))
            print('기존 CARLA 0.9.15 설치를 재사용합니다: '+str(root),flush=True)
        else:
            install(cancel, progress, root=config.path('carla'))
        check_cancel(cancel)
        if operation == 'install':
            terminal = {'status': 'Completed', 'phase': 'installed', 'progress_pct': 100,
                        'message': 'CARLA 설치를 완료했습니다. 설치 후 실행으로 엔진을 시작할 수 있습니다.'}
            return_code = 0
            return 0
        check_ports(int(settings['carla_port']))
        preflight=diagnostics(config,settings)
        from studio.config import atomic_json
        atomic_json(run/'preflight.json',preflight)
        if not preflight['ready']:raise RuntimeError('GPU/Vulkan/CARLA 사전 검사 실패: '+ '; '.join(preflight['warnings']))
        command = engine_command(config, state)
        command += ['-Abslog='+str(run/'unreal.log')]
        atomic_json(run/'launch.json',{'command':command,'cwd':str(root),'settings':settings,'environment':{k:os.environ.get(k) for k in ('DISPLAY','WAYLAND_DISPLAY','XDG_SESSION_TYPE','VK_ICD_FILENAMES','VK_DRIVER_FILES','CUDA_VISIBLE_DEVICES','LD_LIBRARY_PATH')}})
        print('CARLA 실행 명령: '+repr(command),flush=True)
        update_state(directory, task_id, task={'phase': 'starting', 'progress_pct': None, 'message': 'CARLA 엔진을 시작합니다. RPC 준비를 최대 120초 동안 기다립니다.'})
        engine_log = (run / 'engine.log').open('ab', buffering=0)
        process = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL,
                                   stdout=engine_log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            engine.update(identity(process.pid))
        except Exception:
            process.terminate()
            process.wait(timeout=3)
            raise
        engine.update(running=True, ready=False, owned=True, descendants=[],state='Starting',exit_code=None,profile=settings.get('carla_render_profile','offscreen'),last_stage='Starting')
        update_state(directory, task_id, engine=engine)
        deadline = time.monotonic() + READINESS_TIMEOUT
        last_error = '서버 시작 중입니다.'
        while time.monotonic() < deadline:
            check_cancel(cancel)
            if process.poll() is not None:
                raise RuntimeError('CARLA 엔진이 RPC 준비 전에 종료되었습니다(exit {}). 엔진 로그에서 GPU/Vulkan/권한 오류를 확인하세요.'.format(process.returncode))
            engine['descendants'] = descendants(engine)
            update_state(directory, task_id, engine=engine)
            answer = probe_rpc(config, settings, timeout=max(.1, min(4, deadline - time.monotonic())))
            if answer.get('connected'):
                if answer.get('map') and answer['map'].split('/')[-1]!=state['town']:
                    update_state(directory,task_id,task={'phase':'loading_map','message':'선택한 지도 로딩 중: '+state['town']},engine={'state':'Loading Map','last_stage':'Loading Map '+state['town']})
                    answer=probe_rpc(config,settings,timeout=95,town=state['town'])
                    if process.poll() is not None:raise RuntimeError('지도 로딩 중 CARLA 종료(exit {}).'.format(process.returncode))
                    if not answer.get('connected'):raise RuntimeError('선택한 지도 초기화 실패: '+answer.get('error','알 수 없는 오류'))
                engine['ready'] = True
                engine.update(state='Ready',last_stage='Ready',rpc=answer)
                update_state(directory, task_id, task={'status': 'Completed', 'phase': 'ready', 'progress_pct': 100,
                             'message': 'CARLA RPC 연결 준비가 완료되었습니다. 데이터 수집을 시작할 수 있습니다.'}, engine=engine)
                break
            last_error = answer.get('error', last_error)
            if answer.get('fatal'):
                raise RuntimeError(last_error)
            update_state(directory, task_id, task={'phase': 'starting', 'message': 'CARLA RPC 준비 중: ' + str(last_error)})
            cancel.wait(min(1, max(0, deadline - time.monotonic())))
        if not engine.get('ready'):
            raise TimeoutError('CARLA RPC가 120초 안에 준비되지 않아 소유 엔진을 종료했습니다: ' + str(last_error))
        if operation=='test':
            terminal={'status':'Completed','phase':'tested','message':'최소 엔진 테스트 통과: CARLA 0.9.15, World/Snapshot 확인. 센서·수집 검증은 별도입니다.'}
            return_code=0
            return 0
        # The preparation task is complete, but this worker remains its owner.
        while True:
            check_cancel(cancel)
            if process.poll() is not None:
                raise RuntimeError('실행 중이던 CARLA 엔진이 종료되었습니다(exit {}). 엔진 로그를 확인하세요.'.format(process.returncode))
            engine['descendants'] = descendants(engine)
            update_state(directory, task_id, engine={'descendants':engine['descendants']})
            cancel.wait(1)
    except (StopRequested, Cancelled) as error:
        print(str(error), flush=True)
        terminal = {'status': 'Stopped', 'phase': 'stopped', 'stop_requested': True, 'message': str(error)}
        return_code = 0
    except Exception as error:
        actual_exit=process.poll() if process is not None else None
        if process is not None and actual_exit is not None:
            latest=read_state(directory)
            crash_evidence(config,settings,run,actual_exit,latest.get('engine',{}).get('last_stage','Starting'))
            streak=(latest.get('crash_streak',0)+1) if actual_exit==-11 else 0
            update_state(directory,task_id,engine={'state':'Crashed','exit_code':actual_exit,'last_error':str(error)},crash_streak=streak)
        print('CARLA 준비 실패: ' + str(error), flush=True)
        terminal = {'status': 'Failed', 'phase': 'failed', 'message': str(error)}
        return_code = 1
    finally:
        if process is not None:
            terminate_engine(process, engine)
        if engine_log is not None:
            engine_log.close()
        if cancel.is_set():
            terminal = {'status': 'Stopped', 'phase': 'stopped', 'stop_requested': True, 'message': 'CARLA 작업과 소유 엔진을 중지했습니다.'}
            return_code = 0
        from studio.config import atomic_json
        atomic_json(run / 'exit.json', {'return_code': return_code, 'finished_at': timestamp(), 'status': terminal['status'], 'task': terminal,'engine_exit_code':process.returncode if process is not None else None})
        update_state(directory, task_id, task=terminal, engine={'running': False, 'ready': False,'state':'Stopped' if terminal['status']=='Stopped' else 'Crashed' if terminal['status']=='Failed' else 'Stopped','exit_code':process.returncode if process is not None else None})
    return return_code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--task', required=True)
    args = parser.parse_args()
    config = Config(args.root)
    state = read_state(state_directory(config))
    if state.get('task', {}).get('id') != args.task:
        return 1
    cancel = Cancellation(Path(state['run_dir']) / 'stop.json')
    signal.signal(signal.SIGINT, lambda *_: cancel.set())
    signal.signal(signal.SIGTERM, lambda *_: cancel.set())
    return run_task(config, args.task, cancel)


if __name__ == '__main__':
    sys.exit(main())
