"""Observed-class availability and cache invalidation with controlled labels."""
import json
import numpy as np
from PIL import Image
from studio.config import Config
from studio.task_catalog import task_catalog, selection_blockers
from sdv.data_contract import new_task_settings


def sample(root,split,key,class_id):
    scene=root/'data'/split/'scene'
    for folder in ('front_cam_rgb','front_cam_seg_ids','front_cam_bbs'):
        (scene/folder).mkdir(parents=True,exist_ok=True)
    Image.new('RGB',(8,8),(22,44,66)).save(str(scene/'front_cam_rgb'/(key+'.png')))
    Image.new('L',(8,8),class_id).save(str(scene/'front_cam_seg_ids'/(key+'.png')))
    np.save(str(scene/'front_cam_bbs'/(key+'.npy')),np.empty((0,),dtype=np.float32))
    return scene


def test_catalog_only_reports_observed_ids_and_invalidates_rgb_changes(tmp_path):
    scene=sample(tmp_path,'train','00000_a',1)
    sample(tmp_path,'train','00001_a',14)
    cfg=Config(tmp_path)
    first=task_catalog(tmp_path,cfg.settings())
    assert [v['id'] for v in first['segmentation']['classes']]==[1,14]
    assert first['detection']['classes']==[]
    assert first['detection']['empty_frames']==2
    assert first['recommended_task_settings']['detection']['class_ids']==[]
    (scene/'front_cam_rgb/00001_a.png').unlink()
    second=task_catalog(tmp_path,cfg.settings())
    assert [v['id'] for v in second['segmentation']['classes']]==[1]
    assert second['segmentation']['frames']==1


def test_validation_only_class_cannot_silently_train(tmp_path):
    sample(tmp_path,'train','00000_a',1)
    sample(tmp_path,'val','00000_a',14)
    cfg=Config(tmp_path)
    draft=cfg.training_defaults()
    draft['task_settings']=new_task_settings([1,14],[])
    checked=cfg.validate_training(draft)
    issues=selection_blockers(tmp_path,cfg.settings(),checked)
    assert any('outside the training split' in v and '14' in v for v in issues)


def test_legacy_config_absence_retains_checkpoint_contract(tmp_path):
    cfg=Config(tmp_path)
    assert 'task_settings' not in cfg.validate_training(cfg.training_defaults())
    draft=cfg.training_defaults()
    draft['task_settings']=new_task_settings([1],[])
    checked=cfg.validate_training(draft)
    assert checked['task_settings']['segmentation']['mode']=='carla'
    assert checked['task_settings']['detection']['class_ids']==[]
    assert json.dumps(checked,allow_nan=False)
