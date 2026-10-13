"""Web preview ownership and draft boundaries; no CARLA server is used."""
import copy
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
import studio.api as module

HEADERS = {'X-SDV-Request': '1'}


class PreviewFixture:
    def __init__(self, config):
        self.config = config
        self.closed = False
        self.owns_clock = None
        self.value = {'status': 'Idle', 'busy': False, 'channels': [], 'revision': 0}

    def status(self):
        return copy.deepcopy(self.value)

    def busy(self):
        return self.value['busy']

    def start(self, sensors=None, view='perspective', owns_clock=False):
        if self.busy():
            raise ValueError('이미 실행 중입니다.')
        self.owns_clock = owns_clock
        self.value.update(status='Running', busy=True, revision=1, sensors=sensors, view=view)
        return self.status()

    def apply(self, sensors, view=None):
        self.value.update(sensors=sensors, revision=self.value['revision'] + 1)
        return self.status()

    def stop(self):
        self.value.update(status='Stopped', busy=False)
        return self.status()

    def logs(self):
        return 'explicit preview test fixture'

    def frame(self, channel):
        if channel != 'front_cam_rgb':
            raise ValueError('Invalid channel')
        from PIL import Image
        path = self.config.state / 'fixture.jpg'
        Image.new('RGB', (4, 4)).save(str(path))
        return path

    def shutdown(self):
        self.closed = True


@pytest.fixture
def browser(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'SensorPreviewService', PreviewFixture, raising=False)
    app = module.create_app(tmp_path)
    monkeypatch.setattr(app.state.runtime, 'probe_carla', lambda: {'connected': True})
    monkeypatch.setattr(app.state.runtime, 'preflight', lambda *args: {'ready': True, 'blockers': []})
    with TestClient(app, base_url='http://127.0.0.1:8765') as client:
        yield app, client
    assert app.state.sensor_preview.closed


def test_preview_endpoints_keep_draft_separate_from_saved_settings(browser):
    app, client = browser
    saved = client.get('/api/config/sensors').json()
    draft = copy.deepcopy(saved)
    draft['sensors']['front_cam']['x'] = 3.0
    result = client.post('/api/sensor-preview/start', json={'sensors': draft}, headers=HEADERS)
    assert result.status_code == 200, result.text
    assert result.json()['sensors'] == draft
    assert client.get('/api/config/sensors').json() == saved
    assert client.get('/api/status').json()['sensor_preview']['busy'] is True
    assert client.get('/api/bootstrap').json()['sensor_preview']['revision'] == 1
    draft['sensors']['front_cam']['x'] = 4.0
    assert client.put('/api/sensor-preview', json={'sensors': draft}, headers=HEADERS).json()['revision'] == 2
    assert client.get('/api/sensor-preview/logs').json()['text'] == 'explicit preview test fixture'
    frame = client.get('/api/sensor-preview/frame?channel=front_cam_rgb')
    assert frame.status_code == 200
    assert frame.headers['content-type'] == 'image/jpeg'
    assert frame.headers['cache-control'] == 'no-store'
    assert client.get('/api/sensor-preview/frame?channel=front_cam_rgb&revision=0').status_code == 404
    assert client.get('/api/sensor-preview/frame?channel=front_cam_rgb&session_id=previous-session').status_code == 404
    assert client.get('/api/sensor-preview/frame?channel=../../secret').status_code == 422
    assert client.post('/api/sensor-preview/stop', headers=HEADERS).json()['status'] == 'Stopped'


def test_preview_control_requires_csrf_and_valid_draft(browser):
    app, client = browser
    assert client.post('/api/sensor-preview/start', json={}).status_code == 403
    assert client.put('/api/sensor-preview', json={}).status_code == 403
    assert client.post('/api/sensor-preview/stop').status_code == 403
    assert client.post('/api/sensor-preview/start', json={'view': 'shell'}, headers=HEADERS).status_code == 422
    assert client.post('/api/sensor-preview/start', json={'sensors': {'unexpected': True}}, headers=HEADERS).status_code == 422


def test_preview_blocks_research_engine_stop_and_connection_changes(browser):
    app, client = browser
    app.state.sensor_preview.value.update(status='Running', busy=True)
    assert client.post('/api/jobs', json={'kind': 'collection'}, headers=HEADERS).status_code == 409
    assert client.post('/api/jobs', json={'kind': 'training'}, headers=HEADERS).status_code == 409
    assert client.post('/api/carla/local/start', json={'operation': 'prepare'}, headers=HEADERS).status_code == 409
    assert client.post('/api/carla/local/stop', headers=HEADERS).status_code == 409
    assert client.post('/api/carla/disconnect', headers=HEADERS).status_code == 409
    assert client.put('/api/settings', json={'carla_port': 2120}, headers=HEADERS).status_code == 409
    # Draft sensor saving is permitted without silently respawning a live sensor.
    saved = client.get('/api/config/sensors').json()
    assert client.put('/api/config/sensors', json=saved, headers=HEADERS).status_code == 200


def test_saved_two_fps_can_be_raised_while_preview_runs(browser):
    app, client = browser
    assert client.put('/api/settings', json={'preview_fps': 2}, headers=HEADERS).status_code == 200
    app.state.sensor_preview.value.update(status='Running', busy=True)
    for target in (24, 60, 1, 24):
        response = client.put('/api/settings', json={'preview_fps': target}, headers=HEADERS)
        assert response.status_code == 200, response.text
        assert response.json()['preview_fps'] == target
        assert client.get('/api/settings').json()['preview_fps'] == target
    for invalid in (0, 1.5, 61, True):
        response = client.put('/api/settings', json={'preview_fps': invalid}, headers=HEADERS)
        assert response.status_code == 422
        assert client.get('/api/settings').json()['preview_fps'] == 24


def test_running_research_blocks_preview(browser, monkeypatch):
    app, client = browser
    monkeypatch.setattr(app.state.jobs, 'active', lambda: {'kind': 'collection', 'status': 'Running'})
    assert client.post('/api/sensor-preview/start', json={}, headers=HEADERS).status_code == 409


def test_inverted_lidar_vertical_field_is_rejected_before_apply(browser):
    app, client = browser
    draft = app.state.config.sensors()
    draft['sensors']['lidar'] = {'x': 0, 'y': 0, 'z': 2, 'pitch': 0, 'yaw': 0, 'roll': 0,
                                  'upper_fov': -40, 'lower_fov': 10}
    response = client.post('/api/sensor-preview/start', json={'sensors': draft}, headers=HEADERS)
    assert response.status_code == 422
    assert 'upper_fov' in response.json()['detail']


@pytest.mark.parametrize('owned', [False, True])
def test_only_managed_engine_grants_tick_ownership(browser, monkeypatch, owned):
    app, client = browser
    monkeypatch.setattr(app.state.local_carla, 'status', lambda: {'task': {'status': 'Completed'}, 'engine': {'owned': owned, 'running': True, 'ready': True}})
    result = client.post('/api/sensor-preview/start', json={'owns_clock': True, 'exclusive_tick': True}, headers=HEADERS)
    assert result.status_code == 200, result.text
    assert app.state.sensor_preview.owns_clock is owned
