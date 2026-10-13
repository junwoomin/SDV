"""Monitor HTTP boundaries and real subprocess queue lifecycle.

Generated metadata/images and the epoch-receipt child below are explicit QA
fixtures. These tests make no CARLA/CUDA or research accuracy claims.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import pytest
from fastapi.testclient import TestClient

# Reuse the existing executor-only procfs adapter. Production process management
# and all child start/signal/receipt operations remain real OS operations.
from test_studio_jobs import namespace_psutil_adapter

from studio.api import create_app


HEADERS = {'X-SDV-Request': '1'}
TERMINAL = {'Completed', 'Failed', 'Stopped', 'Paused'}
CHILD = r'''
import json,os,sys,time
from pathlib import Path
events=Path(os.environ['SDV_EVENTS_PATH'])
output=Path(os.environ['SDV_TRAIN_OUTPUT']);output.mkdir(parents=True,exist_ok=True)
control=Path(os.environ['SDV_TRAIN_CONTROL'])
def emit(kind,**values):
    with events.open('a') as stream:
        stream.write(json.dumps(dict(event=kind,timestamp=time.time(),**values))+'\n')
resume=os.environ.get('SDV_TRAIN_RESUME')
if resume:
    saved=json.loads(Path(resume).read_text())
    for epoch in range(saved['epoch']+1,4):
        emit('training_epoch',epoch=epoch,epochs=3,global_step=epoch*3,progress=epoch/3.0,metrics={'val_loss':1.0/epoch})
    print('QA lifecycle resumed from completed epoch '+str(saved['epoch']),flush=True)
    sys.exit(0)
emit('training_started',epochs=3,steps_per_epoch=3,train_samples=6,val_samples=6)
emit('training_step',epoch=1,epochs=3,global_step=1,progress=1.0/9,metrics={'train_batch_loss':.5})
while True:
    if control.exists() and json.loads(control.read_text()).get('action')=='pause':
        checkpoint=output/'resume_checkpoint.pth'
        checkpoint.write_text(json.dumps({'epoch':1,'global_step':3}))
        emit('training_epoch',epoch=1,epochs=3,global_step=3,progress=1.0/3,metrics={'val_loss':1.0})
        emit('training_checkpoint',epoch=1,epochs=3,global_step=3,resume_checkpoint=str(checkpoint))
        emit('training_paused',epoch=1,epochs=3,global_step=3,resume_checkpoint=str(checkpoint))
        sys.exit(75)
    time.sleep(.02)
'''


@pytest.fixture
def monitor(tmp_path, monkeypatch, namespace_psutil_adapter):
    app = create_app(tmp_path)
    runtime, jobs = app.state.runtime, app.state.jobs
    monkeypatch.setattr(runtime, 'capabilities', lambda *a, **k: {'dependencies': {}, 'cuda_available': False})
    monkeypatch.setattr(runtime, 'preflight', lambda *a, **k: {'ready': True, 'kind': a[0] if a else 'training', 'blockers': []})
    monkeypatch.setattr(jobs, '_command', lambda *a, **k: [sys.executable, '-u', '-c', CHILD])
    gate = {'ready': False}
    jobs.set_training_gate(lambda job: {'ready': gate['ready'], 'blockers': [] if gate['ready'] else ['QA dispatcher gate']})
    with TestClient(app, base_url='http://127.0.0.1:8765') as client:
        yield app, client, gate


def wait_status(client, jid, wanted, timeout=12):
    deadline = time.monotonic()+timeout
    last = None
    while time.monotonic()<deadline:
        last = client.get('/api/jobs/'+jid).json()
        if last.get('status') in wanted:
            return last
        time.sleep(.03)
    pytest.fail('Job did not reach {}: {}'.format(wanted, last))


def enqueue(app, client, name='QA queue'):
    config = app.state.config.training_defaults()
    config['Epochs'] = 3
    response = client.post('/api/training/queue', json={'name': name, 'config': config}, headers=HEADERS)
    assert response.status_code == 200, response.text
    return response.json()


def seed_artifacts(app):
    config = app.state.config.training_defaults()
    config['model_conf'].update(Depth=True, Seg=True)
    config['opt_checkboxes'].update(Adam=False, AdamW=True)
    job = app.state.jobs._build_job('training', 'QA prediction artifacts', config, 'Completed')
    output = Path(job['output_dir'])
    output.mkdir(parents=True)
    from PIL import Image
    entries = []
    for epoch in (1,2,3):
        for sample in range(4):
            record = {'id': '{}:{}'.format(epoch,sample), 'capture_id':epoch, 'split':'val', 'epoch':epoch,
                      'global_step':epoch*100, 'sample_id':str(sample), 'dataset_fingerprint':'qa-fixed-content',
                      'task':'Seg', 'camera':'front_cam', 'relative_frame':'scene/front_cam_rgb/{}.png'.format(sample)}
            for key in ('input','gt','pred'):
                relative = 'predictions/epoch_{}/sample_{}/{}.png'.format(epoch,sample,key)
                path = output/relative
                path.parent.mkdir(parents=True,exist_ok=True)
                Image.new('RGB',(8,8),(sample*40,epoch*40,20)).save(str(path))
                record[key] = relative
            entries.append(record)
    manifest = {'version':1,'entries':entries,'captures':3,'datasets':{'val':{'count':4,'samples':[{'sample_id':str(i)} for i in range(4)],'dataset_fingerprint':'qa-fixed-content'}}}
    (output/'predictions/index.json').write_text(json.dumps(manifest))
    (output/'train_metrics.csv').write_text('epoch,train_loss,val_loss,val_seg_miou,val_depth_abs_rel\n1,1.1,0.9,0.4,0.2\n2,0.8,0.7,0.6,0.1\n3,0.7,0.8,0.5,0.15\n')
    with app.state.jobs.db() as db:
        db.execute('INSERT INTO jobs(id,status,created_at,payload) VALUES(?,?,?,?)',(job['id'],job['status'],job['created_at'],json.dumps(job)))
        app.state.jobs._save(job,db)
    app.state.catalog.invalidate()
    return job


def test_queue_snapshot_survives_settings_and_training_edits(monitor):
    app, client, gate = monitor
    first = enqueue(app,client,'QA immutable queued model')
    saved = client.put('/api/settings',json={'train_data':'data/new_train','val_data':'data/new_val'},headers=HEADERS)
    assert saved.status_code == 200, saved.text
    changed = app.state.config.training_defaults()
    changed['Epochs'] = 9
    changed['model']['encoder'] = 'ResNet18'
    assert client.put('/api/config/training',json=changed,headers=HEADERS).status_code == 200
    durable = client.get('/api/jobs/'+first['id']).json()
    assert durable['config']['Epochs'] == 3
    assert durable['config']['model']['encoder'] == 'ResNet'
    assert durable['settings']['train_data'] != 'data/new_train'
    assert durable['status'] == 'Queued'


def test_queue_reorder_requires_exact_waiting_permutation_and_csrf(monitor):
    app, client, gate = monitor
    jobs = [enqueue(app,client,'QA waiting '+str(i)) for i in range(3)]
    ids = [job['id'] for job in jobs]
    response = client.post('/api/training/queue/reorder',json={'ids':ids[::-1]},headers=HEADERS)
    assert response.status_code == 200,response.text
    queue = client.get('/api/training/queue').json()['jobs']
    assert [job['id'] for job in queue if job['status']=='Queued'] == ids[::-1]
    for attack in (ids[:-1],ids+[ids[0]],ids[:-1]+['not-a-job']):
        assert client.post('/api/training/queue/reorder',json={'ids':attack},headers=HEADERS).status_code in (400,409,422)
    assert client.post('/api/training/queue/reorder',json={'ids':ids}).status_code == 403


def test_real_pause_receipt_resume_and_queue_delete_preserve_artifacts(monitor):
    app, client, gate = monitor
    gate['ready'] = True
    job = enqueue(app,client,'QA real child pause/resume')
    running = wait_status(client,job['id'],{'Running'})
    assert running.get('eta_remaining_seconds') is None
    gate['ready'] = False
    assert client.delete('/api/jobs/'+job['id'],headers=HEADERS).status_code in (409,422)
    response = client.post('/api/jobs/'+job['id']+'/pause',headers=HEADERS)
    assert response.status_code == 200,response.text
    paused = wait_status(client,job['id'],{'Paused'})
    assert paused['return_code'] == 75 and paused['last_completed_epoch'] == 1
    checkpoint = Path(paused['checkpoint'])
    assert checkpoint.is_file() and json.loads(checkpoint.read_text())['epoch'] == 1
    assert client.get('/api/jobs/'+job['id']).json()['status'] == 'Paused'
    resumed = client.post('/api/jobs/'+job['id']+'/resume',headers=HEADERS)
    assert resumed.status_code == 200,resumed.text
    assert resumed.json()['status'] == 'Queued'
    gate['ready'] = True
    done = wait_status(client,job['id'],{'Completed'})
    assert done['last_completed_epoch'] == 3
    assert done['resume_count'] == 1
    assert 'resumed from completed epoch 1' in client.get('/api/jobs/'+job['id']+'/logs').json()['text']
    assert client.delete('/api/jobs/'+job['id'],headers=HEADERS).status_code == 200
    assert job['id'] not in [row['id'] for row in client.get('/api/training/queue').json()['jobs']]
    assert checkpoint.is_file() and Path(done['run_dir']).is_dir()
    assert any(row.get('job_id')==job['id'] for row in client.get('/api/experiments').json()['experiments'])


def test_can_add_and_save_next_model_while_real_training_child_runs(monitor):
    app, client, gate = monitor
    gate['ready'] = True
    active = enqueue(app,client,'QA active model')
    wait_status(client,active['id'],{'Running'})
    next_config = app.state.config.training_defaults()
    next_config['model']['encoder']='ResNet18'
    assert client.put('/api/config/training',json=next_config,headers=HEADERS).status_code==200
    assert client.put('/api/settings',json={'train_data':'data/next_train','val_data':'data/next_val'},headers=HEADERS).status_code==200
    response=client.post('/api/training/queue',json={'name':'QA next model','config':next_config},headers=HEADERS)
    assert response.status_code==200,response.text
    assert response.json()['status']=='Queued'
    assert client.get('/api/jobs/'+active['id']).json()['config']['model']['encoder']=='ResNet'
    current_python=client.get('/api/settings').json()['python_executable']
    unchanged=client.put('/api/settings',json={'python_executable':current_python},headers=HEADERS)
    assert unchanged.status_code==200,unchanged.text
    assert unchanged.json()['python_executable']==current_python
    assert client.get('/api/settings').json()['python_executable']==current_python
    for body in ({'carla_port':2100},{'python_executable':current_python+'.different'},{'results_root':'next-results'}):
        assert client.put('/api/settings',json=body,headers=HEADERS).status_code in (409,422)
    client.post('/api/jobs/'+active['id']+'/stop',headers=HEADERS)
    wait_status(client,active['id'],{'Stopped'})


def test_prediction_api_filters_fixed_samples_pagination_and_images(monitor):
    app, client, gate = monitor
    job = seed_artifacts(app)
    base='/api/jobs/'+job['id']+'/predictions'
    response=client.get(base,params={'split':'val','epoch':2,'limit':2,'offset':1})
    assert response.status_code==200,response.text
    gallery=response.json()
    assert gallery['total']==4 and len(gallery['entries'])==2
    assert gallery['fixed_sample_count']==4
    assert gallery['dataset_fingerprint']=='qa-fixed-content'
    assert all(row['epoch']==2 and row['split']=='val' for row in gallery['entries'])
    sample=gallery['entries'][0]
    for relative in sample['images'].values():
        image=client.get(base+'/frame',params={'path':relative})
        assert image.status_code==200 and image.headers['content-type'].startswith('image/')
        assert image.content.startswith(b'\x89PNG')
    assert client.get(base,params={'offset':99}).json()['entries']==[]
    for query in ({'offset':-1},{'limit':0},{'limit':201},{'split':'private'}):
        assert client.get(base,params=query).status_code in (400,422)


def test_prediction_frame_rejects_traversal_unlisted_files_and_escaping_link(monitor):
    app, client, gate=monitor
    job=seed_artifacts(app)
    base='/api/jobs/'+job['id']+'/predictions/frame'
    output=Path(job['output_dir'])
    from PIL import Image
    Image.new('RGB',(8,8)).save(str(output/'unlisted.png'))
    secret=app.state.config.root/'private.png'
    Image.new('RGB',(8,8),(255,0,0)).save(str(secret))
    (output/'escape.png').symlink_to(secret)
    for attack in ('../private.png',str(secret),'..\\private.png','unlisted.png','output/unlisted.png','escape.png','output/escape.png'):
        result=client.get(base,params={'path':attack})
        assert result.status_code in (400,403,404,422),attack


def test_experiment_prediction_gallery_and_model_optimizer_metadata(monitor):
    app, client, gate=monitor
    job=seed_artifacts(app)
    experiments=client.get('/api/experiments').json()['experiments']
    row=next(item for item in experiments if item['job_id']==job['id'])
    assert row['optimizer']=='AdamW'
    assert row['encoder']=='ResNet' and row['decoder']=='FPN'
    base='/api/experiments/'+row['id']+'/predictions'
    gallery=client.get(base).json()
    assert gallery['total']==12 and len(gallery['fixed_samples'])==4
    assert client.get(base+'/frame',params={'path':gallery['entries'][0]['images']['pred']}).status_code==200


def test_metrics_registry_and_best_table_respect_metric_direction(monitor):
    app, client, gate=monitor
    job=seed_artifacts(app)
    response=client.get('/api/training/metrics')
    assert response.status_code==200,response.text
    registry=response.json()
    registry=registry.get('metrics',registry)
    assert registry['val_seg_miou']['direction']=='max'
    assert registry['val_depth_abs_rel']['direction']=='min'
    experiments=client.get('/api/experiments').json()['experiments']
    eid=next(item['id'] for item in experiments if item['job_id']==job['id'])
    detail=client.get('/api/experiments/'+eid).json()
    best=detail['best_metrics']
    assert best['val_loss']['value']==.7 and best['val_loss']['epoch']==2
    assert best['val_seg_miou']['value']==.6 and best['val_seg_miou']['epoch']==2
    assert best['val_depth_abs_rel']['value']==.1 and best['val_depth_abs_rel']['epoch']==2


def test_queue_lifecycle_mutations_require_csrf_and_unknown_jobs_are_404(monitor):
    app, client, gate=monitor
    assert client.post('/api/training/queue',json={}).status_code==403
    for verb,path in (('post','/api/jobs/unknown/pause'),('post','/api/jobs/unknown/resume'),('delete','/api/jobs/unknown')):
        assert getattr(client,verb)(path).status_code==403
        assert getattr(client,verb)(path,headers=HEADERS).status_code==404


def test_prediction_endpoints_use_remote_auth_boundary(monitor,monkeypatch):
    app, client, gate=monitor
    job=seed_artifacts(app)
    monkeypatch.setenv('SDV_AUTH_TOKEN','QA-test-secret')
    base='/api/jobs/'+job['id']+'/predictions'
    assert client.get(base).status_code==401
    allowed={'Authorization':'Bearer QA-test-secret'}
    gallery=client.get(base,headers=allowed)
    assert gallery.status_code==200
    relative=gallery.json()['entries'][0]['images']['pred']
    assert client.get(base+'/frame',params={'path':relative}).status_code==401
    assert client.get(base+'/frame',params={'path':relative},headers=allowed).status_code==200


@pytest.mark.parametrize('monitor_config', [
    {'train_preview_every':-1}, {'train_preview_every':True}, {'train_preview_every':1.5},
    {'val_preview_every':0}, {'preview_samples':3}, {'command':'not-supported'},
])
def test_monitor_period_bounds_reject_invalid_settings_without_overwrite(monitor,monitor_config):
    app, client, gate=monitor
    original=client.get('/api/config/training').json()
    value=app.state.config.training_defaults()
    value['monitor'].update(monitor_config)
    response=client.put('/api/config/training',json=value,headers=HEADERS)
    assert response.status_code in (400,422),response.text
    assert client.get('/api/config/training').json()==original


def test_queue_draft_settings_and_preflight_allow_busy_are_constrained(monitor):
    app, client, gate=monitor
    for settings in ({'carla_port':2100},{'python_executable':'python'},{'train_data':'../outside'},[]):
        response=client.post('/api/training/queue',json={'settings':settings},headers=HEADERS)
        assert response.status_code in (400,422),response.text
    response=client.post('/api/preflight',json={'kind':'collection','allow_busy':True},headers=HEADERS)
    assert response.status_code==400
    response=client.post('/api/preflight',json={'kind':'training','allow_busy':'true'},headers=HEADERS)
    assert response.status_code==400
