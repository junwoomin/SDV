"""Model selection and readiness boundaries without importing torch."""
import pytest
from studio.api import create_app
from sdv.training.model_registry import BACKBONES, IMAGE_DECODERS


@pytest.fixture
def environment(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    app = create_app(tmp_path)
    caps = {'dependencies': {'torchvision': {'available': True, 'version': '0.17.2'},
                             'efficientnet_pytorch': {'available': True}},
            'cuda_available': False,
            'model_builders': {spec['constructor']: True for spec in BACKBONES.values()
                               if spec['provider'] == 'torchvision'}}
    monkeypatch.setattr(app.state.runtime, 'capabilities', lambda *a, **k: caps)
    with TestClient(app) as client:
        yield app, client, caps


def test_catalog_drives_bootstrap_and_selectors(environment):
    app, client, _ = environment
    catalog = client.get('/api/training/models').json()
    assert {row['id'] for row in catalog['encoders']} == set(BACKBONES)
    assert {row['id'] for row in catalog['decoders']} == set(IMAGE_DECODERS)
    assert all(row['available'] for row in catalog['encoders'])
    assert client.get('/api/bootstrap').json()['model_catalog'] == catalog


def test_unavailable_constructor_explains_required_environment(environment):
    app, client, caps = environment
    constructor = BACKBONES['ConvNeXt_Tiny']['constructor']
    caps['model_builders'][constructor] = False
    catalog = client.get('/api/training/models').json()
    row = next(row for row in catalog['encoders'] if row['id'] == 'ConvNeXt_Tiny')
    assert not row['available'] and 'torchvision' in row['reason']
    assert 'torchvision' in row['reason_en']
    draft = app.state.config.training_defaults()
    draft['model']['encoder'] = 'ConvNeXt_Tiny'
    result = app.state.runtime.preflight('training', draft)
    assert not result['ready']
    assert any('ConvNeXt' in reason and 'torchvision' in reason for reason in result['blockers'])


def test_missing_legacy_dependency_only_disables_its_encoder(environment):
    _, client, caps = environment
    caps['dependencies']['efficientnet_pytorch']['available'] = False
    rows = {row['id']: row for row in client.get('/api/training/models').json()['encoders']}
    assert not rows['EfficientNet']['available']
    assert rows['EfficientNet_B0']['available']
    assert rows['ResNet']['available']
    bev = {row['id']: row for row in client.get('/api/training/models').json()['bev_decoders']}
    assert not bev['EfficientNet']['available']
    assert bev['UNet']['available']


def test_new_encoder_explicit_preflight_prepares_its_own_weights(environment, monkeypatch):
    app, _, _ = environment
    calls = []
    def ensure(encoder):
        calls.append(encoder)
        return {'encoder': encoder, 'filename': 'fixture.pth', 'ready': False,
                'status': 'Running', 'progress_pct': 12}
    monkeypatch.setattr(app.state.pretrained, 'ensure', ensure)
    draft = app.state.config.training_defaults()
    draft['model']['encoder'] = 'MobileNetV3_Small'
    result = app.state.runtime.preflight('training', draft, prepare_weights=True)
    assert calls == ['MobileNetV3_Small']
    assert result['pretrained']['encoder'] == 'MobileNetV3_Small'


def test_explicit_random_initialization_never_downloads(environment, monkeypatch):
    app, _, _ = environment
    monkeypatch.setattr(app.state.pretrained, 'ensure', lambda encoder: pytest.fail('Unexpected download'))
    draft = app.state.config.training_defaults()
    draft['model']['pretrained'] = False
    result = app.state.runtime.preflight('training', draft, prepare_weights=True)
    assert 'pretrained' not in result


@pytest.mark.parametrize('url', ['/', '/static/app.js', '/static/i18n.js', '/static/style.css'])
def test_updated_ui_does_not_reuse_previous_constraints(environment, url):
    _, client, _ = environment
    response = client.get(url)
    assert response.status_code == 200
    assert response.headers['cache-control'] == 'no-store'
