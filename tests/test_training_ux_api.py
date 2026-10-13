"""Web configuration and preparation boundaries; no CARLA/CUDA claims."""
import json
import pytest
from studio.config import Config


def test_preview_fps_default_and_integer_steps(tmp_path):
    config = Config(tmp_path)
    assert config.settings()["preview_fps"] == 24
    for value in (1, 24, 25, 60):
        assert config.save_settings({"preview_fps": value})["preview_fps"] == value
    for value in (0, 0.1, 1.5, 61, True):
        with pytest.raises(ValueError):
            config.save_settings({"preview_fps": value})


def test_bev_decoder_matches_enabled_task(tmp_path):
    config = Config(tmp_path)
    value = config.training_defaults()
    value["model"]["bev_decoder"] = "EfficientNet"
    assert config.validate_training(value)["model"]["bev_decoder"] is None
    value["model_conf"]["BEV"] = True
    value["sensor_use"].update(left_cam=True, right_cam=True)
    assert config.validate_training(value)["model"]["bev_decoder"] == "EfficientNet"
    value["model"]["bev_decoder"] = None
    with pytest.raises(ValueError):
        config.validate_training(value)


def test_legacy_fractional_fps_migrates_without_rewriting_user_file(tmp_path):
    config = Config(tmp_path)
    file = config.state / 'settings.json'
    file.write_text(json.dumps({'preview_fps': 0.5}))
    assert config.settings()['preview_fps'] == 24
    assert json.loads(file.read_text())['preview_fps'] == 0.5
    file.write_text(json.dumps({'preview_fps': 2}))
    assert config.settings()['preview_fps'] == 2


def test_saved_bev_off_configuration_reads_as_none(tmp_path):
    config = Config(tmp_path)
    value = config.training_defaults()
    value['model']['bev_decoder'] = 'ResNet'
    (config.state / 'training.json').write_text(json.dumps(value))
    assert config.training()['model']['bev_decoder'] is None
    assert config.save_training(value)['model']['bev_decoder'] is None


@pytest.fixture
def ux_browser(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from studio.api import create_app
    app = create_app(tmp_path)
    monkeypatch.setattr(app.state.runtime, 'capabilities', lambda *a, **k: {'dependencies': {}, 'cuda_available': False})
    with TestClient(app, base_url='http://127.0.0.1:8765') as browser:
        yield app, browser


HEADERS = {'X-SDV-Request': '1'}


def test_options_uses_draft_paths_and_never_saves_them(ux_browser):
    from PIL import Image
    app, client = ux_browser
    root = app.state.config.root
    previous = app.state.config.settings()
    for split in ('train', 'val'):
        for modality in ('front_cam_rgb', 'front_cam_seg'):
            path = root / 'draft' / split / 'scene' / modality / '0001.png'
            path.parent.mkdir(parents=True)
            Image.new('RGB', (32, 32)).save(str(path))
    body = {'train_data': 'draft/train', 'val_data': 'draft/val', 'sensor_use': {'front_cam': True}}
    result = client.post('/api/training/options', json=body, headers=HEADERS)
    assert result.status_code == 200, result.text
    options = result.json()
    assert options['sensors']['front_cam']['available']
    assert not options['sensors']['left_cam']['available']
    assert options['tasks']['Seg']['available']
    assert not options['tasks']['Depth']['available']
    assert app.state.config.settings() == previous
    assert not client.get('/api/bootstrap').json()['training_options']['sensors']['front_cam']['available']


def test_options_validates_paths_types_and_csrf(ux_browser):
    app, client = ux_browser
    assert client.post('/api/training/options', json={}).status_code == 403
    for body in ({'train_data': '../outside'}, {'val_data': '/tmp/outside'}, {'sensor_use': []},
                 {'sensor_use': {'front_cam': 'true'}}, {'command': 'ignored'}):
        assert client.post('/api/training/options', json=body, headers=HEADERS).status_code in (400, 422)


def test_weights_download_is_explicit_and_preflight_reports_progress(ux_browser, monkeypatch):
    app, client = ux_browser
    requests = []
    def ensure(encoder):
        requests.append(encoder)
        return {'encoder': encoder, 'filename': 'resnet34-b627a593.pth', 'status': 'Running',
                'ready': False, 'progress_pct': 15, 'downloaded_bytes': 15, 'total_bytes': 100}
    monkeypatch.setattr(app.state.pretrained, 'ensure', ensure)
    assert client.get('/api/bootstrap').status_code == 200
    assert client.get('/api/status').status_code == 200
    check = client.post('/api/preflight', json={'kind': 'training'}, headers=HEADERS)
    assert check.status_code == 200 and not requests
    result = client.post('/api/preflight', json={'kind': 'training', 'prepare_weights': True}, headers=HEADERS)
    assert result.status_code == 200, result.text
    body = result.json()
    assert requests == ['ResNet'] and not body['ready']
    assert body['pretrained']['progress_pct'] == 15
    assert any('준비 중' in message for message in body['blockers'])
    assert client.post('/api/preflight', json={'kind': 'training', 'prepare_weights': 'false'}, headers=HEADERS).status_code == 400
    assert client.post('/api/preflight', json={'kind': 'training', 'prepare_weights': True}).status_code == 403
    assert requests == ['ResNet']


def test_failed_pretrained_cause_is_exposed_without_automatic_retry(ux_browser, monkeypatch):
    app, client = ux_browser
    monkeypatch.setattr(app.state.pretrained, 'status', lambda encoder: {'encoder': encoder, 'filename': 'fixture.pth',
                        'status': 'Failed', 'ready': False, 'error': 'Controlled network error'})
    monkeypatch.setattr(app.state.pretrained, 'ensure', lambda encoder: pytest.fail('Quiet check must not download'))
    result = client.post('/api/preflight', json={'kind': 'training', 'prepare_weights': False}, headers=HEADERS)
    assert result.status_code == 200
    assert any('Controlled network error' in item for item in result.json()['blockers'])
