"""Exercise the web control boundary without downloading or starting CARLA."""
import copy
import threading

import pytest
from fastapi.testclient import TestClient
import studio.api as api_module

HEADERS = {'X-SDV-Request': '1'}


class LocalFixture:
    def __init__(self, config):
        self.started = []
        self.stopped = 0
        self.closed = False
        self.value = {
            'installed': False, 'install_path': str(config.root / 'carla'),
            'local_allowed': True, 'local_reason': '',
            'task': {'id': None, 'status': 'Idle', 'phase': 'idle'},
            'engine': {'running': False, 'ready': False, 'owned': False},
        }

    def status(self):
        return copy.deepcopy(self.value)

    def busy(self):
        return self.value['task']['status'] == 'Running'

    def start(self, operation='prepare'):
        if self.busy() or self.value['engine']['running']:
            raise ValueError('이미 실행 중입니다.')
        self.started.append(operation)
        self.value['task'] = {'id': 'fixture', 'operation': operation,
                              'status': 'Running', 'phase': 'download'}
        return self.status()

    def stop(self):
        self.stopped += 1
        self.value['task']['status'] = 'Stopped'
        self.value['engine'].update(running=False, ready=False)
        return self.status()

    def logs(self):
        return 'isolated lifecycle fixture log'

    def shutdown(self):
        self.closed = True


@pytest.fixture
def local_app(tmp_path, monkeypatch):
    monkeypatch.setattr(api_module, 'CarlaService', LocalFixture)
    app = api_module.create_app(tmp_path)
    with TestClient(app, base_url='http://127.0.0.1:8765') as client:
        yield app, client
    assert app.state.local_carla.closed


def test_local_control_status_and_logs_are_connected(local_app):
    app, client = local_app
    assert client.get('/api/carla/local').json()['installed'] is False
    response = client.post('/api/carla/local/start', json={'operation': 'prepare'}, headers=HEADERS)
    assert response.status_code == 200, response.text
    assert app.state.local_carla.started == ['prepare']
    assert client.get('/api/status').json()['carla_local']['task']['status'] == 'Running'
    assert client.get('/api/bootstrap').json()['carla_local']['task']['id'] == 'fixture'
    assert client.get('/api/carla/local/logs').json()['text'] == 'isolated lifecycle fixture log'
    assert client.post('/api/carla/local/stop', headers=HEADERS).json()['task']['status'] == 'Stopped'
    assert app.state.local_carla.stopped == 1


def test_install_operation_and_duplicate_start(local_app):
    app, client = local_app
    assert client.post('/api/carla/local/start', json={'operation': 'install'}, headers=HEADERS).status_code == 200
    assert app.state.local_carla.started == ['install']
    assert client.post('/api/carla/local/start', json={'operation': 'prepare'}, headers=HEADERS).status_code == 409
    assert client.post('/api/carla/connect', headers=HEADERS).status_code == 409
    assert client.post('/api/jobs', json={'kind': 'collection'}, headers=HEADERS).status_code == 409


def test_lifecycle_posts_require_csrf_and_known_operation(local_app):
    app, client = local_app
    assert client.post('/api/carla/local/start', json={'operation': 'prepare'}).status_code == 403
    assert client.post('/api/carla/local/stop').status_code == 403
    assert client.post('/api/carla/local/start', json={'operation': 'shell'}, headers=HEADERS).status_code == 422
    assert app.state.local_carla.started == []


def test_cannot_start_engine_or_stop_it_during_collection(local_app, monkeypatch):
    app, client = local_app
    monkeypatch.setattr(app.state.jobs, 'active', lambda: {'kind': 'collection', 'status': 'Running'})
    assert client.post('/api/carla/local/start', json={'operation': 'prepare'}, headers=HEADERS).status_code == 409
    assert client.post('/api/carla/local/stop', headers=HEADERS).status_code == 409
    assert app.state.local_carla.started == []
    assert app.state.local_carla.stopped == 0


def test_running_engine_allows_collection_settings_but_locks_connection(local_app):
    app, client = local_app
    app.state.local_carla.value['engine']['running'] = True
    current = client.get('/api/settings').json()
    # Collection's save action must remain usable after starting the engine.
    assert client.put('/api/settings', json={'fps': 20, 'weather': 'cycle',
        'driving': {'agent': 'existing', 'control_hz': 20}}, headers=HEADERS).status_code == 200
    assert client.put('/api/settings', json=current, headers=HEADERS).status_code == 200
    response = client.put('/api/settings', json={'carla_port': 2180}, headers=HEADERS)
    assert response.status_code == 409
    assert client.get('/api/settings').json()['carla_port'] == current['carla_port']


def test_ready_engine_triggers_connection_once_and_respects_disconnect(local_app, monkeypatch):
    app, client = local_app
    service = app.state.local_carla
    service.value['task'].update(id='ready-fixture', status='Completed')
    service.value['engine'].update(running=True, ready=True, owned=True)
    calls = []
    completed = threading.Event()

    def probe():
        calls.append(1)
        app.state.runtime._carla_probing = False
        app.state.runtime.carla.update(connected=True, checked_unix=api_module.time.time())
        completed.set()

    monkeypatch.setattr(app.state.runtime, 'probe_carla', probe)
    client.get('/api/status')
    assert completed.wait(2)
    assert client.get('/api/status').json()['carla']['connected'] is True
    assert len(calls) == 1
    assert client.post('/api/carla/disconnect', headers=HEADERS).status_code == 200
    assert client.get('/api/status').json()['carla']['connected'] is False
    assert len(calls) == 1


@pytest.mark.parametrize('status', ['Completed', 'Failed', 'Stopped'])
def test_client_setup_invalidates_dependency_cache_once(local_app, status):
    app, client = local_app
    runtime = app.state.runtime
    runtime._deps = {'dependencies': {'carla': {'available': False}}}
    app.state.local_carla.value['task'].update(id='client-fixture', operation='prepare', status=status)
    client.get('/api/status')
    assert runtime._deps is None
    # Polls after a fresh probe must not discard its newly observed package state.
    refreshed = {'dependencies': {'carla': {'available': True}}}
    runtime._deps = refreshed
    client.get('/api/status')
    assert runtime._deps is refreshed
