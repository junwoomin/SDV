"""Weight preparation contract, hash integrity, atomicity and cancellation."""
import hashlib
import io
import json
import threading
import time
import urllib.request
from pathlib import Path

import pytest

from studio.config import Config, atomic_json
import studio.pretrained_service as module
from studio.pretrained_service import PretrainedService


class Response(io.BytesIO):
    def __init__(self, data, length=None):
        super().__init__(data)
        self.headers = {'Content-Length': str(len(data) if length is None else length)}
    def getcode(self):
        return 200


def wait_for(service, predicate, encoder='ResNet', timeout=5):
    deadline = time.monotonic() + timeout
    state = None
    while time.monotonic() < deadline:
        state = service.status(encoder)
        if predicate(state):
            return state
        time.sleep(.01)
    pytest.fail('Weight state not reached: {!r}'.format(state))


@pytest.fixture
def fixture_weights(monkeypatch):
    content = b'isolated weight integrity fixture\n' * 1000
    specs = {key: dict(value, hash_prefix=hashlib.sha256(content).hexdigest()[:8])
             for key, value in module.WEIGHTS.items()}
    monkeypatch.setattr(module, 'WEIGHTS', specs)
    return content


@pytest.fixture
def service(tmp_path, fixture_weights):
    instance = PretrainedService(Config(tmp_path))
    yield instance
    instance.shutdown()


def test_status_is_read_only_and_never_downloads(service, monkeypatch):
    monkeypatch.setattr(service, '_open', lambda spec: pytest.fail('GET/status attempted a download'))
    result = service.status()
    assert not result['busy']
    assert set(result['models']) == set(module.WEIGHTS)
    assert all(item['phase'] == 'missing' for item in result['models'].values())
    assert not service.directory.exists()
    assert not service.cache.exists()
    for unsupported in ('resnet18', '../ResNet', None, {}):
        with pytest.raises(ValueError):
            service.ensure(unsupported)


def test_download_checks_hash_and_reuses_valid_cache(service, fixture_weights, monkeypatch):
    calls = []
    monkeypatch.setattr(service, '_open', lambda spec: calls.append(spec['source_url']) or Response(fixture_weights))
    first = service.ensure('ResNet')
    assert first['status'] == 'Running'
    result = wait_for(service, lambda item: item['status'] == 'Completed')
    assert result['ready']
    assert result['progress_pct'] == 100
    assert Path(result['path']).read_bytes() == fixture_weights
    assert result['sha256'] == hashlib.sha256(fixture_weights).hexdigest()
    assert not list(service.cache.glob('*.part'))
    assert service.ensure('ResNet')['ready']
    assert len(calls) == 1
    fresh = PretrainedService(Config(service.config.root))
    try:
        assert fresh.status('ResNet')['ready']
        assert fresh.ensure('ResNet')['ready']
    finally:
        fresh.shutdown()


def test_corrupt_existing_weight_is_preserved_and_not_downloaded(service, monkeypatch):
    target, record, lock = service._paths('ResNet')
    target.parent.mkdir(parents=True)
    target.write_bytes(b'unknown existing weight fixture')
    monkeypatch.setattr(service, '_open', lambda spec: pytest.fail('Corrupt existing file was overwritten'))
    result = service.ensure('ResNet')
    assert result['status'] == 'Failed'
    assert 'SHA256' in result['error']
    assert target.read_bytes() == b'unknown existing weight fixture'


def test_hash_failure_never_publishes_checkpoint(service, monkeypatch):
    monkeypatch.setattr(service, '_open', lambda spec: Response(b'wrong download fixture'))
    service.ensure('ResNet')
    failed = wait_for(service, lambda item: item['status'] == 'Failed')
    assert 'SHA256' in failed['error']
    assert not Path(failed['path']).exists()
    assert not list(service.cache.glob('*.part'))


def test_truncated_download_is_failed_and_retry_succeeds(service, fixture_weights, monkeypatch):
    monkeypatch.setattr(service, '_open', lambda spec: Response(fixture_weights[:10], length=len(fixture_weights)))
    service.ensure('ResNet')
    failed = wait_for(service, lambda item: item['status'] == 'Failed')
    assert '크기' in failed['error']
    # Wait for lock cleanup, which follows the terminal status write.
    service._threads['ResNet'][0].join(2)
    monkeypatch.setattr(service, '_open', lambda spec: Response(fixture_weights))
    service.ensure('ResNet')
    assert wait_for(service, lambda item: item['ready'])['status'] == 'Completed'


def test_cross_instance_duplicate_is_blocked_and_publication_is_atomic(service, fixture_weights, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    calls = []
    def slow(spec):
        calls.append(spec['source_url'])
        entered.set()
        release.wait(3)
        return Response(fixture_weights)
    monkeypatch.setattr(service, '_open', slow)
    service.ensure('ResNet')
    assert entered.wait(2)
    current = service.status('ResNet')
    assert not Path(current['path']).exists()
    other = PretrainedService(Config(service.config.root))
    monkeypatch.setattr(other, '_open', lambda spec: pytest.fail('Duplicate download started'))
    try:
        assert other.ensure('ResNet')['status'] == 'Running'
        assert len(calls) == 1
    finally:
        release.set()
    assert wait_for(other, lambda item: item['ready'])['status'] == 'Completed'
    other.shutdown()


def test_resnet_aliases_share_progress_lock_and_verified_file(service, fixture_weights, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    calls = []
    def slow(spec):
        calls.append(spec['source_url'])
        entered.set()
        release.wait(3)
        return Response(fixture_weights)
    monkeypatch.setattr(service, '_open', slow)
    assert service._paths('ResNet') == service._paths('ResNet34')
    service.ensure('ResNet34')
    assert entered.wait(2)
    other = PretrainedService(Config(service.config.root))
    monkeypatch.setattr(other, '_open', lambda spec: pytest.fail('Alias started duplicate download'))
    try:
        current = other.ensure('ResNet')
        assert current['status'] == 'Running'
        assert current['encoder'] == 'ResNet'
        assert current['task_id'] == service.status('ResNet34')['task_id']
        assert len(calls) == 1
        release.set()
        original = wait_for(other, lambda item: item['ready'], encoder='ResNet')
        alias = wait_for(service, lambda item: item['ready'], encoder='ResNet34')
        assert original['path'] == alias['path']
        assert original['sha256'] == alias['sha256']
        assert other.ensure('ResNet34')['ready']
        assert len(calls) == 1
    finally:
        release.set()
        other.shutdown()


@pytest.mark.parametrize('encoder', ['EfficientNet_B1', 'MobileNetV3_Small', 'ConvNeXt_Tiny', 'ResNet101'])
def test_new_backbones_prepare_only_selected_weight(service, fixture_weights, monkeypatch, encoder):
    calls = []
    monkeypatch.setattr(service, '_open', lambda spec: calls.append(spec['filename']) or Response(fixture_weights))
    service.ensure(encoder)
    state = wait_for(service, lambda item: item['ready'], encoder=encoder)
    assert calls == [module.WEIGHTS[encoder]['filename']]
    assert Path(state['path']).read_bytes() == fixture_weights
    assert not Path(service.status('ResNet')['path']).exists()


def test_external_file_created_during_download_is_never_overwritten(service, fixture_weights, monkeypatch):
    target, record, lock = service._paths('ResNet')
    def appeared(spec):
        target.write_bytes(b'external research weight fixture')
        return Response(fixture_weights)
    monkeypatch.setattr(service, '_open', appeared)
    service.ensure('ResNet')
    service._threads['ResNet'][0].join(3)
    assert service.status('ResNet')['status'] == 'Failed'
    assert target.read_bytes() == b'external research weight fixture'
    assert not list(service.cache.glob('*.part'))


def test_shutdown_cancels_stream_and_removes_partial(service, fixture_weights, monkeypatch):
    entered = threading.Event()
    class SlowResponse(Response):
        def read1(self, size):
            entered.set()
            time.sleep(.05)
            return b'x'
    monkeypatch.setattr(service, '_open', lambda spec: SlowResponse(fixture_weights))
    service.ensure('ResNet')
    assert entered.wait(2)
    service.shutdown()
    result = service.status('ResNet')
    assert result['status'] == 'Stopped'
    assert not Path(result['path']).exists()
    assert not list(service.cache.glob('*.part'))


def test_deadline_fails_without_publishing(service, fixture_weights, monkeypatch):
    class SlowResponse(Response):
        def read1(self, size):
            time.sleep(.05)
            return b'x'
    monkeypatch.setattr(service, '_open', lambda spec: SlowResponse(fixture_weights))
    monkeypatch.setattr(module, 'DOWNLOAD_TIMEOUT', .1)
    service.ensure('ResNet')
    result = wait_for(service, lambda item: item['status'] == 'Failed')
    assert '10분' in result['error']
    assert not Path(result['path']).exists()


def test_stale_running_state_reports_failure_without_writing_or_download(service, monkeypatch):
    service.directory.mkdir(parents=True)
    target, record, lock = service._paths('ResNet')
    atomic_json(record, {'status': 'Running', 'phase': 'downloading'})
    before = record.read_bytes()
    monkeypatch.setattr(service, '_open', lambda spec: pytest.fail('Read triggered download'))
    assert service.status('ResNet')['status'] == 'Failed'
    assert record.read_bytes() == before


def test_state_file_cannot_replace_fixed_url_hash_or_destination(service, monkeypatch):
    service.directory.mkdir(parents=True)
    target, record, lock = service._paths('ResNet')
    atomic_json(record, {'hash_prefix': '', 'source_url': 'https://evil.example/a', 'filename': '../x', 'path': '/etc/passwd'})
    state = service.status('ResNet')
    assert state['hash_prefix'] == module.WEIGHTS['ResNet']['hash_prefix']
    assert state['source_url'] == module.WEIGHTS['ResNet']['source_url']
    assert state['path'] == str(target)


def test_modified_cache_is_revalidated(service, fixture_weights, monkeypatch):
    monkeypatch.setattr(service, '_open', lambda spec: Response(fixture_weights))
    service.ensure('ResNet')
    ready = wait_for(service, lambda item: item['ready'])
    Path(ready['path']).write_bytes(b'changed cached fixture')
    assert service.status('ResNet')['status'] == 'Failed'


def test_redirects_only_accept_upstream_https_hosts():
    handler = module.UpstreamRedirect()
    request = urllib.request.Request(module.WEIGHTS['EfficientNet']['source_url'])
    for url in ('http://github.com/file', 'https://evil.example/weights', 'file:///tmp/weights'):
        with pytest.raises(ValueError):
            handler.redirect_request(request, None, 302, 'redirect', {}, url)
    result = handler.redirect_request(request, None, 302, 'redirect', {}, 'https://release-assets.githubusercontent.com/github-production-release-asset/file')
    assert result.full_url.startswith('https://release-assets.githubusercontent.com/')
