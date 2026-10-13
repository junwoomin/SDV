"""Launch preflight matches the frame-keyed, selected-task research loader."""
from pathlib import Path
import pytest
import yaml
from PIL import Image
from studio.dataset_tools import validate_scene, validate_training_inputs


def scene(root, name='scene'):
    path=root/name
    (path/'front_cam_rgb').mkdir(parents=True)
    for stem in ('00000','00001'):
        Image.new('RGB',(400,300)).save(path/'front_cam_rgb'/(stem+'.png'))
    config={'sensors':{'front_cam':dict(width=400,height=300,fov=100,x=1.3,y=0,z=2.3,pitch=0,roll=0,yaw=0)}}
    (path/'sensor_config.yaml').write_text(yaml.safe_dump(config))
    return path


def test_disabled_optional_labels_do_not_block_selected_tasks(tmp_path):
    path=scene(tmp_path)
    (path/'front_cam_seg').mkdir()
    Image.new('RGB',(400,300)).save(path/'front_cam_seg/00000.png')
    config={'sensor_use':{'front_cam':True},'model_conf':{}}
    assert validate_scene(path,config)['samples']==2
    config['model_conf']['Seg']=True
    with pytest.raises(ValueError,match='frame ID'):validate_scene(path,config)


def test_selected_raw_segmentation_with_palette_fallback(tmp_path):
    path=scene(tmp_path)
    for folder,stem in [('front_cam_seg_ids','00000'),('front_cam_seg','00001')]:
        (path/folder).mkdir()
        Image.new('L' if folder.endswith('_ids') else 'RGB',(400,300)).save(path/folder/(stem+'.png'))
    cfg={'sensor_use':{'front_cam':True},'model_conf':{'Seg':True},'task_settings':{'segmentation':{'mode':'carla','class_ids':[1]}}}
    assert validate_scene(path,cfg)['samples']==2


def test_single_bev_head_requires_only_its_native_layer(tmp_path):
    path=scene(tmp_path)
    (path/'das').mkdir()
    for stem in ('00000','00001'):Image.new('L',(500,500)).save(path/'das'/(stem+'.png'))
    cfg={'sensor_use':{'front_cam':True},'model_conf':{'BEV':True},'task_settings':{'bev':{'layers':['das']}}}
    assert validate_scene(path,cfg)['modalities']==['das','front_cam_rgb']
    cfg['task_settings']['bev']['layers'].append('vehicle')
    with pytest.raises(ValueError,match='vehicle_mask'):validate_scene(path,cfg)


def test_recursive_scene_preflight_preserves_leakage_detection(tmp_path):
    train=scene(tmp_path/'train/Town01/session','train_scene')
    val=scene(tmp_path/'val/Town02/session','val_scene')
    settings={'train_data':'train','val_data':'val','sensor_config':str(train/'sensor_config.yaml')}
    config={'sensor_use':{'front_cam':True},'model_conf':{}}
    assert validate_training_inputs(tmp_path,settings,config)==[]
    settings['val_data']='train'
    assert any('같은 원본' in item for item in validate_training_inputs(tmp_path,settings,config))
