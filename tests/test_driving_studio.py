"""Driving UI/API contracts; fixtures do not claim CARLA driving validation."""
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from studio.api import create_app
from studio.config import Config
from studio.sequences import normalize_scene

HEADERS = {'X-SDV-Request': '1'}


def test_roach_defaults_and_physical_sampling_gate(tmp_path):
    config = Config(tmp_path)
    assert config.settings()['fps'] == 30
    assert config.settings()['driving']['agent'] == 'roach'
    assert config.settings()['driving']['control_hz'] == 30
    assert config.save_settings({'driving': {'control_hz': 10}})['driving']['control_hz'] == 10
    for invalid in ({'fps': 20}, {'driving': {'policy_ppm': 10}},
                    {'driving': {'control_hz': 7}}, {'driving': {'deterministic': 'false'}},
                    {'driving': {'checkpoint': '../untrusted.pth'}}, {'driving': {'agent': 'random'}}):
        with pytest.raises(ValueError):
            config.save_settings(invalid)
    assert config.save_settings({'fps': 20, 'driving': {'agent': 'existing', 'control_hz': 20}})['fps'] == 20


def test_missing_checkpoint_inspection_never_reports_loaded(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        result = client.post('/api/driving/inspect', json={}, headers=HEADERS)
        assert result.status_code == 409
        assert 'Checkpoint' in result.json()['detail']
        assert client.get('/api/driving/checkpoints').json()['checkpoints'][0]['exists'] is False


def test_manual_commands_are_mode_bound_fresh_ordered_and_csrf_protected(tmp_path, monkeypatch):
    app = create_app(tmp_path)
    run_dir = tmp_path / 'results' / 'manual-test'
    job = {'id': 'test', 'kind': 'collection', 'status': 'Running', 'stop_requested': False,
           'run_dir': str(run_dir), 'settings': {'driving': {'agent': 'manual'}}}
    monkeypatch.setattr(app.state.jobs, 'get', lambda _id: job)
    command = {'sequence': 1, 'sent_unix': time.time(), 'throttle': .2, 'brake': 0, 'steer': -.1}
    with TestClient(app) as client:
        assert client.post('/api/jobs/test/manual-control', json=command).status_code == 403
        result = client.post('/api/jobs/test/manual-control', json=command, headers=HEADERS)
        assert result.status_code == 200, result.text
        record = json.loads((run_dir / 'manual_control.json').read_text())
        assert record['job_id'] == 'test' and record['throttle'] == .2
        assert result.json()['expires_unix'] - record['issued_unix'] == .5
        assert client.post('/api/jobs/test/manual-control', json=command, headers=HEADERS).status_code == 409
        command.update(sequence=2, sent_unix=time.time()-1)
        assert client.post('/api/jobs/test/manual-control', json=command, headers=HEADERS).status_code == 409
        command.update(sequence=3, sent_unix=time.time(), throttle=0, brake=1)
        result = client.post('/api/jobs/test/manual-control', json=command, headers=HEADERS)
        assert result.status_code == 200
        assert json.loads((run_dir/'manual_control.json').read_text())['brake'] == 1
        command.update(sequence=4, sent_unix=time.time(), throttle=.5)
        assert client.post('/api/jobs/test/manual-control', json=command, headers=HEADERS).status_code == 422
        command.update(throttle=0)
        job['settings']['driving']['agent'] = 'roach'
        assert client.post('/api/jobs/test/manual-control', json=command, headers=HEADERS).status_code == 409


def test_manual_endpoint_obeys_configured_auth(tmp_path, monkeypatch):
    monkeypatch.setenv('SDV_AUTH_TOKEN', 'test-application-token')
    with TestClient(create_app(tmp_path)) as client:
        result=client.post('/api/jobs/test/manual-control', json={}, headers=HEADERS)
        assert result.status_code == 401


def test_sequence_keeps_policy_action_and_navigation_without_fabricating_state():
    policy={'name':'roach','raw_action':[.2,-.1],'applied_control':{'throttle':0,'brake':1,'steer':0},
            'inference_ms':6.2,'history_ready':True,'safety_intervened':True}
    scene=normalize_scene({'sdv_metadata':{'policy':policy,'navigation':{'distance_to_route_m':.4,
            'route_completion_pct':21.5},'safety':{'intervened':True,'reason':'policy failure'}}},{},'0001')
    assert all(scene['policy'][key] == value for key, value in policy.items())
    assert scene['navigation']['distance_to_route_m'] == .4
    assert scene['navigation']['route_completion_pct'] == 21.5
    assert scene['safety']['reason'] == 'policy failure'
    assert normalize_scene({}, {}, '0')['policy'] == {}
