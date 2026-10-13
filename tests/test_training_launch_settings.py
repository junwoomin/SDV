"""Allowed next-model settings remain editable during the actual launch lock."""
import threading

from test_monitor_api import HEADERS, enqueue, monitor, wait_status
from test_studio_jobs import namespace_psutil_adapter


def test_next_dataset_settings_save_while_running_launch_holds_resource_guard(monitor, monkeypatch):
    app, client, gate = monitor
    jobs = app.state.jobs
    entered, release = threading.Event(), threading.Event()
    original = jobs._launch

    def waiting_launch(job):
        # Dispatcher has reserved Running and holds resource_guard. There is
        # intentionally no child PID yet, exactly the startup race being tested.
        entered.set()
        if not release.wait(10):
            raise RuntimeError('QA launch was not released')
        return original(job)

    monkeypatch.setattr(jobs, '_launch', waiting_launch)
    gate['ready'] = True
    job = enqueue(app, client, 'Launch settings race fixture')
    try:
        assert entered.wait(5), 'Dispatcher never entered the reserved launch'
        reserved = client.get('/api/jobs/' + job['id']).json()
        assert reserved['status'] == 'Running' and reserved['pid'] is None
        changed = {'train_data': 'data/next_train', 'val_data': 'data/next_val'}
        response = client.put('/api/settings', json=changed, headers=HEADERS)
        assert response.status_code == 200, response.text
        assert response.json()['train_data'] == 'data/next_train'
        # The original job and the next job's draft cannot share mutable paths.
        assert jobs.get(job['id'])['settings']['train_data'] == 'data/train'
        assert jobs.get(job['id'])['settings']['val_data'] == 'data/val'
        denied = client.put('/api/settings', json={'results_root': 'other-results'}, headers=HEADERS)
        assert denied.status_code in (409, 422)
        assert app.state.config.settings()['results_root'] == 'results'
    finally:
        release.set()
    running = wait_status(client, job['id'], {'Running'})
    # Running is a reserved status too; use the PID receipt to prove that a real
    # OS child launched before exercising its ordinary stop path.
    import time
    deadline = time.monotonic() + 5
    while running.get('pid') is None and time.monotonic() < deadline:
        time.sleep(.02)
        running = client.get('/api/jobs/' + job['id']).json()
    assert running['pid']
    assert client.post('/api/jobs/' + job['id'] + '/stop', headers=HEADERS).status_code == 200
    assert wait_status(client, job['id'], {'Stopped'})['status'] == 'Stopped'
