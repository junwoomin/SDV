"""Durable training queue tests with real OS children, never synthetic UI results."""
import json
import os
import sqlite3
import time
from pathlib import Path

import pytest

from studio.config import Config
from studio.jobs import JobManager
from test_studio_jobs import namespace_psutil_adapter, command, wait_for


@pytest.fixture
def manager(tmp_path):
    instance=JobManager(Config(tmp_path))
    yield instance
    instance.shutdown()


def manual_dispatch(manager,ready=True,blockers=None):
    manager._training_gate=lambda job: {'ready':ready,'blockers':blockers or []}
    return manager._dispatch_once()


def test_enqueue_while_running_has_immutable_config_and_paths(manager,monkeypatch):
    command(manager,monkeypatch,"import time;time.sleep(60)")
    current=manager.start('training','Current model')
    queued=manager.enqueue_training('Next model')
    assert queued['status']=='Queued'
    old=dict(queued['config'])
    manager.config.save_settings({'project_name':'Edited after enqueue'})
    changed=manager.config.training();changed['Epochs']=9
    manager.config.save_training(changed)
    assert manager.get(queued['id'])['config']==old
    assert manager.get(queued['id'])['settings']['project_name']=='SDV Research'
    assert json.loads((Path(queued['run_dir'])/'config/settings.json').read_text())['project_name']=='SDV Research'
    assert manual_dispatch(manager) is None
    assert manager.active()['id']==current['id']


def test_reorder_is_exact_atomic_permutation(manager):
    a=manager.enqueue_training('A');b=manager.enqueue_training('B');c=manager.enqueue_training('C')
    ordered=manager.reorder([c['id'],a['id'],b['id']])
    assert [j['id'] for j in ordered]==[c['id'],a['id'],b['id']]
    with pytest.raises(ValueError):manager.reorder([a['id'],b['id']])
    with pytest.raises(ValueError):manager.reorder([a['id'],a['id'],b['id']])
    assert [j['id'] for j in manager.queue()]==[c['id'],a['id'],b['id']]


def test_blocked_head_does_not_skip_to_later_job(manager,monkeypatch):
    command(manager,monkeypatch,"print('must wait')")
    a=manager.enqueue_training('A');b=manager.enqueue_training('B')
    manager._training_gate=lambda job: {'ready':job['id']==b['id'],'blockers':['A dataset missing']}
    assert manager._dispatch_once() is None
    assert manager.get(a['id'])['blocked_reason']==['A dataset missing']
    assert manager.get(b['id'])['status']=='Queued'
    manager.reorder([b['id'],a['id']])
    started=manager._dispatch_once()
    assert started['id']==b['id']
    wait_for(manager,b['id'],lambda j:j['status']=='Completed')


def test_automatic_dispatch_next_after_completion(manager,monkeypatch):
    command(manager,monkeypatch,"import time;time.sleep(.15)")
    a=manager.enqueue_training('A');b=manager.enqueue_training('B')
    manager.set_training_gate(lambda job:{'ready':True,'blockers':[]})
    first=wait_for(manager,a['id'],lambda j:j['status']=='Completed')
    second=wait_for(manager,b['id'],lambda j:j['status']=='Completed')
    assert first['finished_at']<=second['started_at']


PAUSE_SOURCE='''import os,json,time,sys
from pathlib import Path
out=Path(os.environ['SDV_TRAIN_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
events=Path(os.environ['SDV_EVENTS_PATH'])
def emit(event):
    with events.open('a') as stream:stream.write(json.dumps(event)+'\\n')
if os.environ.get('SDV_TRAIN_RESUME'):
    checkpoint=Path(os.environ['SDV_TRAIN_RESUME'])
    state=json.loads(checkpoint.read_text())
    (out/'resume_observed.json').write_text(json.dumps({'epoch':state['epoch'],'path':str(checkpoint)}))
    emit({'event':'training_epoch','epoch':2,'epochs':20,'global_step':10,'progress':.1,'duration_seconds':1,'metrics':{'val_loss':.2}})
    sys.exit(0)
(out/'child_ready').write_text('ready')
control=Path(os.environ['SDV_TRAIN_CONTROL'])
while not control.exists():time.sleep(.02)
assert json.loads(control.read_text())['pause_requested']
checkpoint=out/'resume_checkpoint.pth';checkpoint.write_text(json.dumps({'epoch':1,'global_step':5}))
emit({'event':'training_epoch','epoch':1,'epochs':20,'global_step':5,'progress':.05,'duration_seconds':1,'metrics':{'val_loss':.3}})
emit({'event':'training_checkpoint','epoch':1,'next_epoch':1,'global_step':5,'resume_checkpoint':str(checkpoint)})
emit({'event':'training_paused','epoch':1,'next_epoch':1,'global_step':5,'resume_checkpoint':str(checkpoint)})
sys.exit(75)
'''


def paused_job(manager,monkeypatch):
    command(manager,monkeypatch,PAUSE_SOURCE)
    started=manager.start('training','Pause fixture')
    ready=Path(started['output_dir'])/'child_ready'
    end=time.monotonic()+5
    while not ready.exists() and time.monotonic()<end:time.sleep(.02)
    assert ready.exists()
    pending=manager.pause(started['id'])
    assert pending['status']=='Pausing'
    return wait_for(manager,started['id'],lambda j:j['status']=='Paused')


def test_pause_resume_restores_same_output_and_next_epoch(manager,monkeypatch):
    paused=paused_job(manager,monkeypatch)
    assert paused['last_completed_epoch']==1
    assert paused['global_step']==5
    assert paused['return_code']==75
    first_elapsed=paused['elapsed_seconds']
    original_events=len(paused['metrics'])
    resumed=manager.resume(paused['id'])
    assert resumed['status']=='Queued'
    assert resumed['output_dir']==paused['output_dir']
    assert resumed['config']==paused['config']
    assert resumed['resume_count']==1
    manual_dispatch(manager)
    ended=wait_for(manager,paused['id'],lambda j:j['status']=='Completed')
    assert ended['elapsed_seconds']>=first_elapsed
    assert len(ended['metrics'])==original_events+1
    state=json.loads((Path(ended['output_dir'])/'resume_observed.json').read_text())
    assert state['epoch']==1
    assert state['path']==paused['checkpoint']
    assert [e['epoch'] for e in ended['metrics'] if e['event']=='training_epoch']==[1,2]
    assert ended['best_metrics']['val_loss']=={'value':.2,'epoch':2,'direction':'min'}


def test_pausing_keeps_exclusive_slot_until_real_exit(manager,monkeypatch):
    command(manager,monkeypatch,"import time;time.sleep(60)")
    started=manager.start('training','Nonresponsive pause fixture')
    manager.pause(started['id'])
    assert manager.active()['status']=='Pausing'
    queued=manager.enqueue_training('Later')
    assert manual_dispatch(manager) is None
    with pytest.raises(ValueError):manager.start('collection','Must not overlap')
    with pytest.raises(ValueError):manager.remove(started['id'])
    manager.stop(started['id'])
    wait_for(manager,started['id'],lambda j:j['status']=='Stopped')
    command(manager,monkeypatch,"print('next started')")
    # Queue command is immutable: queued fixture remains sleeping, stop in teardown.
    assert manual_dispatch(manager)['id']==queued['id']


def test_exit_75_without_pause_receipt_is_failure(manager,monkeypatch):
    command(manager,monkeypatch,"raise SystemExit(75)")
    started=manager.start('training','Missing pause receipt fixture')
    ended=wait_for(manager,started['id'],lambda j:j['status']=='Failed')
    assert ended['return_code']==75
    with pytest.raises(ValueError):manager.resume(started['id'])


@pytest.mark.parametrize('location',['missing','outside','empty'])
def test_pause_requires_checkpoint_inside_owned_output(manager,monkeypatch,location):
    outside=manager.config.root/'outside.pth';outside.write_bytes(b'not an owned checkpoint')
    source="""import os,json,sys
from pathlib import Path
out=Path(os.environ['SDV_TRAIN_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
p=out/'resume_checkpoint.pth'
LOCATION
Path(os.environ['SDV_EVENTS_PATH']).write_text(json.dumps({'event':'training_paused','epoch':1,'resume_checkpoint':str(p)})+'\\n')
sys.exit(75)
"""
    statement={'missing':'pass','outside':'p=Path(%r)'%str(outside),'empty':"p.write_bytes(b'')"}[location]
    command(manager,monkeypatch,source.replace('LOCATION',statement))
    started=manager.start('training','Invalid checkpoint fixture')
    assert wait_for(manager,started['id'],lambda j:j['status']=='Failed')['return_code']==75


def test_remove_queue_retains_configuration_and_result_directory(manager):
    job=manager.enqueue_training('Removed queued experiment')
    removed=manager.remove(job['id'])
    assert removed['status']=='Stopped'
    assert removed['queue_removed']
    assert manager.queue()==[]
    assert any(j['id']==job['id'] for j in manager.list())
    assert (Path(job['run_dir'])/'config/train_config.yaml').is_file()


def test_remove_paused_retains_checkpoint_and_blocks_resume(manager,monkeypatch):
    paused=paused_job(manager,monkeypatch)
    manager.remove(paused['id'])
    assert Path(paused['checkpoint']).is_file()
    assert manager.queue()==[]
    with pytest.raises(ValueError):manager.resume(paused['id'])


def test_paused_state_survives_restart_and_receipt_recovery(tmp_path,monkeypatch):
    first=JobManager(Config(tmp_path))
    try:paused=paused_job(first,monkeypatch)
    finally:first.shutdown()
    # Simulate server crash between durable worker receipt and last SQL update.
    stale=dict(paused);stale.update(status='Pausing',pause_event=None,return_code=None,finished_at=None,events_position=0,metrics=[],best_metrics={})
    first._save(stale)
    observer=JobManager(Config(tmp_path))
    try:
        restored=observer.get(paused['id'])
        assert restored['status']=='Paused'
        assert restored['last_completed_epoch']==1
        assert restored['checkpoint']==paused['checkpoint']
        assert observer.queue()[0]['status']=='Paused'
    finally:observer.shutdown()


def test_queued_order_and_snapshot_survive_restart(tmp_path):
    first=JobManager(Config(tmp_path))
    a=first.enqueue_training('A');b=first.enqueue_training('B')
    first.reorder([b['id'],a['id']]);first.shutdown()
    second=JobManager(Config(tmp_path))
    try:
        assert [j['id'] for j in second.queue()]==[b['id'],a['id']]
        assert second.get(a['id'])['config']==a['config']
    finally:second.shutdown()


def test_best_metrics_retains_earliest_best_after_curve_trim(manager):
    job=manager.enqueue_training('Long metrics history fixture')
    job['metrics']=[{'event':'training_epoch','epoch':i,'duration_seconds':2,'metrics':{'val_loss':float(i),'val_seg_miou':1.0/i}} for i in range(1,251)]
    job.update(status='Completed',progress_current=250,progress_total=250,progress_pct=100)
    job['best_metrics']={}
    for event in job['metrics']:manager._record_best(job,event)
    manager._save(job)
    summary=manager.summaries()[0]
    assert len(summary['metrics'])==200
    assert summary['best_metrics']['val_loss']['epoch']==1
    assert summary['best_metrics']['val_seg_miou']['epoch']==1


def test_metric_directions_and_nonfinite_values_ignored(manager):
    job=manager.enqueue_training('Metric directions fixture')
    manager._record_best(job,{'epoch':1,'metrics':{'val_loss':.4,'val_seg_miou':.2,'val_depth_rmse':4,'val_depth_valid_pixels':100}})
    manager._record_best(job,{'epoch':2,'metrics':{'val_loss':.5,'val_seg_miou':.5,'val_depth_rmse':3,'val_depth_valid_pixels':200}})
    manager._record_best(job,{'epoch':3,'metrics':{'val_loss':float('nan'),'val_seg_miou':float('inf'),'unknown':1}})
    assert job['best_metrics']['val_loss']['epoch']==1
    assert job['best_metrics']['val_seg_miou']['epoch']==2
    assert job['best_metrics']['val_depth_rmse']['epoch']==2
    assert 'val_depth_valid_pixels' not in job['best_metrics']
    assert 'unknown' not in job['best_metrics']


def test_eta_uses_actual_epoch_durations_excluding_pause(manager):
    job=manager.enqueue_training('Timing fixture')
    job.update(status='Running',progress_total=20,last_completed_epoch=2,active_elapsed_seconds=8,segment_started_unix=time.time()-2,metrics=[{'event':'training_epoch','epoch':1,'duration_seconds':3},{'event':'training_epoch','epoch':2,'duration_seconds':5}])
    summary,_=manager._compact(job)
    assert 10<=summary['elapsed_seconds']<11
    assert summary['eta_remaining_seconds']==72
    job['status']='Paused';job['segment_started_unix']=None
    paused,_=manager._compact(job)
    assert paused['elapsed_seconds']==8
    assert paused['eta_remaining_seconds'] is None


def test_shutdown_does_not_launch_queued_job(manager,monkeypatch):
    marker=manager.config.root/'unexpected_start'
    command(manager,monkeypatch,"from pathlib import Path;Path(%r).write_text('bad')"%str(marker))
    queued=manager.enqueue_training('Queued on shutdown')
    manager.shutdown()
    manager._training_gate=lambda job:{'ready':True}
    assert manager._dispatch_once() is None
    assert manager.get(queued['id'])['status']=='Queued'
    assert not marker.exists()


def test_enqueue_settings_override_preserved_without_changing_draft(manager):
    current=manager.config.settings()
    queued=manager.enqueue_training('Dataset snapshot',settings_override={'train_data':'data/another-train','val_data':'data/another-val'})
    assert queued['settings']['train_data']=='data/another-train'
    assert queued['settings']['val_data']=='data/another-val'
    assert manager.config.settings()==current
    assert json.loads((Path(queued['run_dir'])/'config/experiment.json').read_text())['name']=='Dataset snapshot'


def test_invalid_settings_override_leaves_no_queue_record(manager):
    with pytest.raises(ValueError):manager.enqueue_training('Invalid path',settings_override={'train_data':'../outside'})
    assert manager.queue()==[]
    assert not (manager.config.root/'results').exists()


def test_paused_job_frees_slot_for_next_queued_model(manager,monkeypatch):
    command(manager,monkeypatch,PAUSE_SOURCE)
    running=manager.start('training','Model A pause fixture')
    queued=manager.enqueue_training('Model B fixture')
    manager.set_training_gate(lambda job:{'ready':True,'blockers':[]})
    manager.pause(running['id'])
    paused=wait_for(manager,running['id'],lambda j:j['status']=='Paused')
    following=wait_for(manager,queued['id'],lambda j:j['status']=='Running')
    assert paused['finished_at']<=following['started_at']
    assert manager.active()['id']==queued['id']


def test_eta_ticks_from_last_completed_epoch_and_ignores_idle_pause(manager):
    job=manager.enqueue_training('Timing during resumed epoch')
    current=time.time()
    job.update(status='Running',progress_total=10,last_completed_epoch=2,
               active_elapsed_seconds=10,segment_started_unix=current-1,
               metrics=[{'event':'training_epoch','epoch':2,'duration_seconds':5,'timestamp':current-100}])
    summary,_=manager._compact(job)
    # The 99 seconds spent paused is excluded; only this one-second segment counts.
    assert 38<summary['eta_remaining_seconds']<=39


def test_long_readiness_check_does_not_block_draft_or_enqueue(manager,monkeypatch):
    import threading
    entered=threading.Event();release=threading.Event()
    command(manager,monkeypatch,"print('fixture completed')")
    first=manager.enqueue_training('Long gate fixture')
    busy={'value':True}
    def gate(job):
        entered.set();release.wait(5)
        return {'ready':True,'blockers':[]}
    def resources(job):return {'ready':not busy['value'],'blockers':['Preview is busy']}
    manager.set_training_gate(gate,resources)
    try:
        assert entered.wait(2)
        before=time.monotonic()
        second=manager.enqueue_training('Additional model')
        with manager.resource_guard():manager.config.save_settings({'project_name':'Draft remains editable'})
        assert time.monotonic()-before<2
        release.set()
        blocked=wait_for(manager,first['id'],lambda j:j.get('blocked_reason')==['Preview is busy'])
        assert blocked['status']=='Queued'
        assert manager.get(second['id'])['status']=='Queued'
        assert manager.active() is None
        busy['value']=False;manager._dispatch_wake.set()
        wait_for(manager,first['id'],lambda j:j['status']=='Completed')
    finally:release.set()


def test_active_job_not_lost_after_more_than_200_recent_queue_items(manager,monkeypatch):
    command(manager,monkeypatch,"import time;time.sleep(60)")
    running=manager.start('training','Older still running fixture')
    queued=manager.enqueue_training('Recent queued fixture')
    # Seed many history rows to exercise paging without launching test models.
    with manager.db() as db:
        for index in range(205):
            record=dict(queued);record['id']='pagingfixture{}'.format(index)
            db.execute('INSERT INTO jobs(id,status,created_at,payload) VALUES(?,?,?,?)',(record['id'],record['status'],record['created_at'],json.dumps(record)))
            manager._save(record,db)
    assert manager.active()['id']==running['id']
    summary=manager.summaries()
    assert any(j['id']==running['id'] and j['status']=='Running' for j in summary)
    assert len(summary)==201
    assert len(manager.queue())==207
    assert all(j['metrics']==[] for j in manager.queue())
