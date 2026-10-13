"""Research-worker lifecycle and draft isolation using real fixture processes."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from studio.config import Config, atomic_json
from studio.carla_service import identity, process_alive
from studio.sensor_preview_service import SensorPreviewService
from test_carla_service import process_metadata_for_executor


def wait_for(service, predicate, timeout=6):
    deadline = time.monotonic() + timeout
    status = None
    while time.monotonic() < deadline:
        status = service.status()
        if predicate(status):
            return status
        time.sleep(.025)
    pytest.fail('Preview state not reached: {!r}; log={!r}'.format(status, service.logs()))


@pytest.fixture
def preview(tmp_path):
    service = SensorPreviewService(Config(tmp_path))
    yield service
    service.shutdown()


def fake_worker(preview, monkeypatch, mode='live', cleanup=.15):
    source = r'''
import json,sys,time
from pathlib import Path
run=Path(sys.argv[1]);mode=sys.argv[2];cleanup=float(sys.argv[3])
def write(value):
 p=run/'status.json'; t=run/'status.tmp'; t.write_text(json.dumps(value)); t.replace(p)
snapshot=json.loads((run/'snapshot.json').read_text())
print('preview fixture started',flush=True)
if mode=='fail':
 print('fixture missing carla',flush=True);sys.exit(7)
(run/'frames').mkdir()
(run/'frames/front_cam_rgb.jpg').write_bytes(b'\xff\xd8fixture-jpeg\xff\xd9')
state={'status':'Running','phase':'live','revision':0,'view':snapshot['view'],'channels':[{'id':'front_cam_rgb','label':'fixture front_cam_rgb','kind':'rgb','revision':0,'frame':1,'timestamp':time.time(),'width':2,'height':2,'path':'untrusted path ignored'}]}
write(state)
while not (run/'stop.json').exists():
 command=run/'command.json'
 if command.exists() and mode!='pause_apply':
  value=json.loads(command.read_text())
  if value['revision']>state['revision']:
   state.update(revision=value['revision'],view=value['view'])
   for channel in state['channels']:channel.update(revision=value['revision'],frame=value['revision']+1)
   write(state)
 time.sleep(.02)
state.update(phase='stopping',message='fixture actors cleaning');write(state)
time.sleep(cleanup)
state.update(status='Stopped',phase='stopped',message='fixture actors cleaned');write(state)
if mode=='late_exit':time.sleep(.4)
'''
    monkeypatch.setattr(preview, '_worker_command', lambda run: [sys.executable, '-u', '-c', source, str(run), mode, str(cleanup)])


def test_idle_and_invalid_drafts_do_not_launch(preview):
    assert preview.status()['status'] == 'Idle'
    assert not preview.busy()
    with pytest.raises(ValueError, match='시점'):
        preview.start(view='outside')
    with pytest.raises(ValueError):
        preview.start(sensors={'sensors': {}})
    with pytest.raises(ValueError, match='먼저 시작'):
        preview.apply(preview.config.sensor_defaults())
    assert preview.status()['session_id'] is None


def test_live_draft_apply_revisions_and_refresh_never_save_settings(preview, monkeypatch):
    fake_worker(preview, monkeypatch)
    original = preview.config.sensors()
    draft = preview.config.sensor_defaults()
    draft['sensors']['front_cam']['x'] = 2.8
    start = preview.start(draft, view='side', owns_clock=True)
    ready = wait_for(preview, lambda item: item['ready'])
    assert ready['view'] == 'side'
    assert ready['sensors']['sensors']['front_cam']['x'] == 2.8
    assert preview.config.sensors() == original
    session_id, run = preview._current()
    snapshot = json.loads((run / 'snapshot.json').read_text())
    assert snapshot['exclusive_tick'] is True
    assert snapshot['settings']['preview_owns_clock'] is True
    assert snapshot['settings']['python_executable'] == preview.config.settings()['python_executable']
    observer = SensorPreviewService(Config(preview.config.root))
    assert observer.status()['session_id'] == start['session_id']
    with pytest.raises(ValueError, match='이미 실행'):
        observer.start(draft)
    draft['sensors']['front_cam']['z'] = 3.4
    observer.apply(draft, view='top')
    changed = wait_for(preview, lambda item: item['revision'] == 1)
    assert changed['requested_revision'] == 1
    assert changed['view'] == 'top'
    assert preview.config.sensors() == original
    observer.apply(draft)
    assert wait_for(preview, lambda item: item['revision'] == 2)['requested_view'] == 'top'
    assert preview.frame('front_cam_rgb').is_file()
    assert changed['channels'][0]['path'] == 'frames/front_cam_rgb.jpg'
    assert 'preview fixture started' in observer.logs()
    observer.stop()
    ended = wait_for(preview, lambda item: not item['busy'])
    assert ended['status'] == 'Stopped'
    observer.shutdown()


def test_stop_stays_busy_until_actor_cleanup_and_process_exit(preview, monkeypatch):
    fake_worker(preview, monkeypatch, mode='late_exit', cleanup=.4)
    preview.start()
    wait_for(preview, lambda item: item['ready'])
    state = preview.stop()
    assert state['busy']
    assert state['phase'] == 'stopping'
    with pytest.raises(ValueError, match='종료 중'):
        preview.apply(preview.config.sensor_defaults())
    time.sleep(.12)
    assert preview.busy()
    with pytest.raises(ValueError, match='이미 실행'):
        preview.start()
    cleaning = wait_for(preview, lambda item: item['phase'] == 'cleaning')
    assert cleaning['busy']
    assert not cleaning['ready']
    ended = wait_for(preview, lambda item: not item['busy'])
    assert ended['status'] == 'Stopped'


def test_apply_passes_new_saved_fps_without_rewriting_initial_snapshot(preview, monkeypatch):
    fake_worker(preview, monkeypatch, mode='pause_apply')
    preview.config.save_settings({'preview_fps': 2})
    preview.start()
    wait_for(preview, lambda item: item['ready'])
    session_id, run = preview._current()
    assert json.loads((run / 'snapshot.json').read_text())['settings']['preview_fps'] == 2
    assert preview.config.save_settings({'preview_fps': 24})['preview_fps'] == 24
    state = preview.apply(preview.config.sensor_defaults())
    command = json.loads((run / 'command.json').read_text())
    assert command['preview_fps'] == 24
    assert state['requested_fps'] == 24
    assert state['preview_fps'] == 2  # The worker has not applied the new revision yet.
    assert json.loads((run / 'snapshot.json').read_text())['settings']['preview_fps'] == 2


def test_unexpected_worker_exit_is_failed_and_logs_real_reason(preview, monkeypatch):
    fake_worker(preview, monkeypatch, mode='fail')
    preview.start()
    ended = wait_for(preview, lambda item: not item['busy'])
    assert ended['status'] == 'Failed'
    assert ended['error']
    assert 'fixture missing carla' in preview.logs()
    observer = SensorPreviewService(Config(preview.config.root))
    assert observer.status()['status'] == 'Failed'
    observer.shutdown()


def test_frames_block_unknown_channel_and_symlink_escape(preview, monkeypatch, tmp_path):
    fake_worker(preview, monkeypatch)
    preview.start()
    wait_for(preview, lambda item: item['ready'])
    for channel in ('../metadata', '/etc/passwd', 'front_cam_rgb/../../metadata', 'unknown', 'overview'):
        with pytest.raises(ValueError):
            preview.frame(channel)
    with pytest.raises(FileNotFoundError):
        preview.frame('left_cam_rgb')
    session_id, run = preview._current()
    private = tmp_path / 'private.jpg'
    private.write_bytes(b'private fixture')
    (run / 'frames/left_cam_rgb.jpg').symlink_to(private)
    with pytest.raises(ValueError):
        preview.frame('left_cam_rgb')
    (run / 'frames/left_cam_rgb.jpg').unlink()
    (run / 'frames').rename(run / 'real-frames')
    (run / 'frames').symlink_to(run / 'real-frames', target_is_directory=True)
    with pytest.raises(ValueError):
        preview.frame('front_cam_rgb')


def test_shutdown_stops_only_preview_and_preserves_external_process(preview, monkeypatch):
    fake_worker(preview, monkeypatch)
    external = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'], start_new_session=True)
    try:
        preview.start()
        wait_for(preview, lambda item: item['ready'])
        session_id, run = preview._current()
        owned = json.loads((run / 'metadata.json').read_text())['process']
        preview.shutdown()
        assert preview.status()['status'] == 'Stopped'
        assert not process_alive(owned)
        assert external.poll() is None
    finally:
        external.terminate()
        external.wait(timeout=3)


def test_pid_reuse_does_not_signal_external_process(preview):
    external = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'], start_new_session=True)
    try:
        session_id = 'a' * 16
        run = preview.directory / session_id
        run.mkdir()
        record = identity(external.pid)
        record['created'] -= 10
        atomic_json(run / 'metadata.json', {'session_id': session_id, 'status': 'Running', 'process': record})
        atomic_json(preview.directory / 'current.json', {'session_id': session_id})
        assert preview.status()['status'] == 'Failed'
        preview.stop()
        assert external.poll() is None
    finally:
        external.terminate()
        external.wait(timeout=3)


def test_non_owned_server_clock_is_default(preview, monkeypatch):
    fake_worker(preview, monkeypatch)
    preview.start()
    session_id, run = preview._current()
    snapshot = json.loads((run / 'snapshot.json').read_text())
    assert snapshot['exclusive_tick'] is False
    assert snapshot['settings']['preview_owns_clock'] is False


def test_ready_and_frames_require_current_revision_and_actual_samples(preview, monkeypatch):
    fake_worker(preview, monkeypatch, mode='pause_apply')
    preview.start()
    wait_for(preview, lambda item: item['ready'])
    session_id, run = preview._current()
    worker = json.loads((run / 'status.json').read_text())
    worker['channels'][0]['frame'] = None
    atomic_json(run / 'status.json', worker)
    assert not preview.status()['ready']
    assert preview.status()['channels'][0]['frame'] is None
    with pytest.raises(FileNotFoundError, match='새 프레임'):
        preview.frame('front_cam_rgb')
    worker['channels'][0]['frame'] = 1
    atomic_json(run / 'status.json', worker)
    assert preview.status()['ready']
    preview.apply(preview.config.sensor_defaults(), 'top')
    pending = preview.status()
    assert pending['phase'] == 'applying'
    assert not pending['ready']
    assert pending['channels'][0]['frame'] is None
    with pytest.raises(FileNotFoundError):
        preview.frame('front_cam_rgb')
    # Worker acknowledges revision 1 before that camera's first new frame.
    worker.update(revision=1, view='top')
    atomic_json(run / 'status.json', worker)
    assert not preview.status()['ready']
    with pytest.raises(FileNotFoundError):
        preview.frame('front_cam_rgb')
    worker['channels'][0].update(revision=1, frame=2)
    atomic_json(run / 'status.json', worker)
    assert preview.status()['ready']
    assert preview.frame('front_cam_rgb').is_file()
