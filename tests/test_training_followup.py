import json
from pathlib import Path
import pytest
from studio.config import Config
from studio.dataset_choices import dataset_choices
from studio.jobs import JobManager
from studio.predictions import prediction_index
from sdv.training.preview_config import should_capture
from test_training_queue import paused_job, manual_dispatch
from test_studio_jobs import command, wait_for
from test_predictions import saved_manifest, _torch_fixture


def test_pairs_disambiguate_empty_missing_and_saved_custom(tmp_path):
    for parent in ['prepared/a/same','prepared/b/same','prepared/empty','prepared/missing']:
        for split in ['train','val']:
            if parent.endswith('missing') and split=='val':continue
            folder=tmp_path/'data'/parent/split/'scene/front_cam_rgb';folder.mkdir(parents=True)
            if not parent.endswith('empty'):(folder/'1.png').write_bytes(b'actual file')
    items=dataset_choices(tmp_path,{'train_data':'data/custom_train','val_data':'data/custom_val'})['datasets']
    same=[d for d in items if d['name']=='same']
    assert len(same)==2 and same[0]['id']!=same[1]['id'] and same[0]['path']!=same[1]['path']
    assert all(item['splits']['train']['samples']==1 for item in same)
    assert not next(d for d in items if d['name']=='empty')['available']
    assert not next(d for d in items if d['name']=='missing')['available']
    assert any(d['name']=='현재 저장 경로' for d in items)


def test_pair_scanner_never_follows_external_links(tmp_path):
    outside=tmp_path.parent/(tmp_path.name+'_external');outside.mkdir();(outside/'train').mkdir()
    (tmp_path/'data').mkdir();(tmp_path/'data/linked').symlink_to(outside,target_is_directory=True)
    assert dataset_choices(tmp_path,{'train_data':str(outside/'train'),'val_data':str(outside/'val')})['datasets']==[]


def test_epoch_cadence_and_legacy_global_cadence():
    cfg={'train_preview_every':2,'train_preview_basis':'epoch'}
    assert should_capture(cfg,2,7) and not should_capture(cfg,1,6)
    assert should_capture({'train_preview_every':2},1,6)
    assert not should_capture({'train_preview_every':0},2,6)


def test_timeline_preserves_missing_local_step_and_real_indices(tmp_path):
    saved_manifest(tmp_path,epochs=(1,2,3))
    data=prediction_index(tmp_path,epoch=1,global_step=10)
    assert len(data['entries'])==1
    assert [e['global_step'] for e in data['timeline']]==[10,20,30]
    assert all(e['epoch_step'] is None for e in data['timeline'])


def test_stopped_resume_and_fresh_restart_do_not_share_outputs(tmp_path,monkeypatch):
    manager=JobManager(Config(tmp_path))
    try:
        job=paused_job(manager,monkeypatch);job.update(status='Stopped');manager._save(job)
        assert manager.get(job['id'])['can_resume']
        continued=manager.resume(job['id']);assert continued['output_dir']==job['output_dir']
        with pytest.raises(ValueError):manager.resume(job['id'])
        manual_dispatch(manager);wait_for(manager,job['id'],lambda j:j['status']=='Completed')
        fresh=manager.restart(job['id']);double=manager.restart(job['id'])
        assert fresh['id']==double['id'] and fresh['id']!=job['id']
        assert fresh['output_dir']!=job['output_dir'] and fresh['last_completed_epoch']==0
        assert not fresh.get('resume_from')
        assert Path(job['checkpoint']).is_file()
        assert fresh['restart_of']==job['id']
    finally:manager.shutdown()


def test_prediction_budget_does_not_delete_previous_capture(tmp_path):
    torch,np,dataset,collater=_torch_fixture(tmp_path)
    from sdv.training.predictions import PredictionRecorder
    class Model(torch.nn.Module):
        def __init__(self):super().__init__();self.p=torch.nn.Parameter(torch.ones(1))
        def forward(self,images,*args):return (None,None,None,None,None,images[:,:,:1]+10,None,None)
    model=Model();recorder=PredictionRecorder(tmp_path/'output',tmp_path,dataset)
    first=recorder.capture(model,dataset,collater,split='train',epoch=1,global_step=2,epoch_step=2,steps_per_epoch=3)
    assert first['count']==4
    before=recorder.index_path.read_bytes();files={str(p):p.read_bytes() for p in recorder.folder.rglob('*.png')}
    recorder.max_bytes=recorder.used_bytes+1
    limited=recorder.capture(model,dataset,collater,split='train',epoch=2,global_step=5,epoch_step=2,steps_per_epoch=3)
    assert limited['budget_exhausted'] and recorder.index_path.read_bytes()==before
    smaller=recorder.capture(model,dataset,collater,split='val',epoch=2,global_step=5,indices=[0])
    assert smaller['budget_exhausted'] and recorder.index_path.read_bytes()==before
    assert {str(p):p.read_bytes() for p in recorder.folder.rglob('*.png')}==files
    assert model.training
    new=PredictionRecorder(tmp_path/'output',tmp_path,dataset)
    new.capture(model,dataset,collater,split='train',epoch=2,global_step=5,epoch_step=2,steps_per_epoch=3)
    data=prediction_index(tmp_path/'output',split='train')
    assert [(e['epoch_step'],e['global_step']) for e in data['timeline']]==[(2,2),(2,5)]


def test_checkpoint_rewind_archives_partial_predictions_without_deleting_images(tmp_path):
    torch,np,dataset,collater=_torch_fixture(tmp_path)
    from sdv.training.predictions import PredictionRecorder
    class Model(torch.nn.Module):
        def __init__(self):super().__init__();self.p=torch.nn.Parameter(torch.ones(1))
        def forward(self,images,*args):return (None,None,None,None,None,images[:,:,:1]+10,None,None)
    model=Model();recorder=PredictionRecorder(tmp_path/'output',tmp_path,dataset)
    recorder.capture(model,dataset,collater,split='train',epoch=1,global_step=2,epoch_step=2)
    recorder.capture(model,dataset,collater,split='train',epoch=2,global_step=3,epoch_step=1)
    before={str(p):p.read_bytes() for p in recorder.folder.rglob('*.png')}
    resumed=PredictionRecorder(tmp_path/'output',tmp_path,dataset,resume_epoch=1,resume_global_step=2)
    assert prediction_index(tmp_path/'output',split='train')['epochs']==[1]
    assert any(e.get('superseded') for e in resumed.index['entries'])
    assert all(Path(p).read_bytes()==value for p,value in before.items())
    resumed.capture(model,dataset,collater,split='train',epoch=2,global_step=3,epoch_step=1)
    data=prediction_index(tmp_path/'output',split='train')
    assert [(e['epoch'],e['global_step']) for e in data['timeline']]==[(1,2),(2,3)]
