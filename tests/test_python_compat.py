"""Exercise compatibility paths on every supported Python runtime."""
import asyncio
import contextvars
import json
import threading

import pytest
from fastapi.testclient import TestClient

from studio.api import create_app
from studio.catalog import Catalog, _series_from_events
from studio.compat import to_thread


def test_background_call_preserves_context_kwargs_and_thread_boundary():
    trace = contextvars.ContextVar('trace', default='unset')
    main_thread = threading.get_ident()

    def blocking(value, suffix):
        return value + suffix, trace.get(), threading.get_ident()

    async def exercise():
        trace.set('request-context')
        return await to_thread(blocking, 'value', suffix='!')

    value, context, worker_thread = asyncio.run(exercise())
    assert value == 'value!'
    assert context == 'request-context'
    assert worker_thread != main_thread


def test_background_call_propagates_errors():
    def failing():
        raise ValueError('background failure')

    with pytest.raises(ValueError, match='background failure'):
        asyncio.run(to_thread(failing))


def test_bearer_authorization_and_lifespan_shutdown(tmp_path, monkeypatch):
    monkeypatch.setenv('SDV_AUTH_TOKEN', 'compatibility-test-token')
    app = create_app(tmp_path)
    shutdown_threads = []
    monkeypatch.setattr(app.state.jobs, 'shutdown', lambda: shutdown_threads.append(threading.get_ident()))
    with TestClient(app, base_url='http://127.0.0.1:8765') as client:
        assert client.get('/api/settings').status_code == 401
        assert client.get('/api/settings', headers={'Authorization': 'Bearer wrong'}).status_code == 401
        response = client.get('/api/settings', headers={'Authorization': 'Bearer compatibility-test-token'})
        assert response.status_code == 200
    assert len(shutdown_threads) == 1
    assert shutdown_threads[0] != threading.get_ident()


def test_status_event_stream_reports_real_empty_state(tmp_path):
    app = create_app(tmp_path)
    endpoint = next(route.endpoint for route in app.routes if getattr(route, 'path', '') == '/api/events')

    class ConnectedRequest:
        async def is_disconnected(self):
            return False

    async def first_event():
        response = await endpoint(ConnectedRequest())
        try:
            return await response.body_iterator.__anext__()
        finally:
            await response.body_iterator.aclose()

    event = asyncio.run(first_event())
    assert event.startswith('event: status\ndata: ')
    status = json.loads(event.split('data: ', 1)[1])
    assert status['jobs'] == []
    assert status['active_job'] is None
    assert status['carla']['connected'] is False


def test_prefixed_and_plain_event_logs_keep_real_metrics(tmp_path):
    path = tmp_path / 'events.jsonl'
    events = [
        {'event': 'training_epoch', 'epoch': 1, 'metrics': {'train_loss': 0.4}},
        {'event': 'training_epoch', 'epoch': 2, 'metrics': {'train_loss': 0.3}},
    ]
    path.write_text('SDV_EVENT ' + json.dumps(events[0]) + '\n' + json.dumps(events[1]) + '\n')
    assert _series_from_events(path)[0]['points'] == [{'x': 1.0, 'y': 0.4}, {'x': 2.0, 'y': 0.3}]


def test_failed_export_cleans_temporary_archive(tmp_path, monkeypatch):
    run = tmp_path / 'results' / 'run'
    run.mkdir(parents=True)
    (run / 'train_metrics.csv').write_text('epoch,train_loss\n1,0.4\n')
    catalog = Catalog(tmp_path, {'results_root': 'results'})
    identifier = catalog.experiments()[0]['id']

    def fail_artifact(*args):
        raise OSError('artifact removed during export')

    monkeypatch.setattr(catalog, 'artifact', fail_artifact)
    with pytest.raises(OSError, match='artifact removed'):
        catalog.export(identifier)
    assert not list((tmp_path / '.runtime/studio/exports').glob('*.zip'))
