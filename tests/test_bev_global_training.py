"""Schema-4 explicit crop-affine integration; controlled pixels, not CARLA runs."""
import json
from pathlib import Path
import numpy as np
import pytest

from sdv.bev.geometry import BEVGeometry,world_to_local
from sdv.bev.io import semantic_from_channels
from sdv.bev.training import (frame_bev_schema,model_ground_truth,validate_raster_affine,
    validate_training_layers,load_bev_ground_truth)
from test_bev_training import fixture_frame,save_arrays,dataset_setup,config,web_scene


def global_frame(root,rotate=True,yaw=90.,width=500,height=400,mpp=.2):
    cv2=pytest.importorskip('cv2')
    scene,arrays,metadata=fixture_frame(root,width=width,height=height,mpp=mpp)
    pose={'x':123.,'y':-34.,'z':7.,'yaw':yaw}
    effective=dict(pose,yaw=yaw if rotate else 0.)
    geometry=BEVGeometry(width,height,mpp)
    origin=world_to_local((0.,0.),effective,geometry)
    matrix=np.column_stack((world_to_local((1.,0.),effective,geometry)-origin,
        world_to_local((0.,1.),effective,geometry)-origin,origin))
    metadata.update(schema_version=4,ego_pose=pose,label_source='carla_simulator_ground_truth',
        static_label_source='provided_global_raster_png',online_input=False,
        available_static_channels=['road','drivable','lane'],surface_table={'low':{'index':0}},
        actors=[{'actor_id':2,'included':True}])
    metadata['geometry'].update(world_to_bev_pixel=matrix.tolist(),
        orientation='ego_forward_up_ego_right_right' if rotate else 'world_x_up_world_y_right',
        rotation_yaw_degrees=effective['yaw'])
    arrays['elevation_m'][:]=np.nan
    arrays['elevation_validity']=np.zeros((height,width),np.uint8)
    # The same front-right rectangle in physical ego coordinates for ON/OFF.
    p=np.array([[6.,2.],[12.,2.],[12.,6.],[6.,6.]])
    c,s=np.cos(np.radians(yaw)),np.sin(np.radians(yaw))
    world=p @ np.array([[c,s],[-s,c]])+(pose['x'],pose['y'])
    uv=world @ matrix[:,:2].T+matrix[:,2]
    polygon=np.rint(uv).astype(np.int32)
    cv2.fillConvexPoly(arrays['drivable'],polygon,1)
    cv2.fillConvexPoly(arrays['vehicle'],polygon,1)
    arrays['actor_id'][arrays['vehicle']>0]=2
    arrays['semantic']=semantic_from_channels(arrays)
    save_arrays(scene,arrays)
    info=json.loads((scene/'info/1.json').read_text());info['bev_gt']=metadata
    packet=info['sdv_metadata']['synchronization']['sensors']['front_cam_rgb']
    packet['world_transform']={'position':{'x':123.+c,'y':-34.+s,'z':9.},
        'rotation_deg':{'yaw':yaw,'pitch':0.,'roll':0.}}
    (scene/'info/1.json').write_text(json.dumps(info))
    return scene,arrays,metadata


@pytest.mark.parametrize('rotate',[True,False])
@pytest.mark.parametrize('yaw',[0.,90.,-45.])
def test_global_png_crop_uses_actual_ego_yaw_even_when_rotation_off(tmp_path,rotate,yaw):
    scene,arrays,metadata=global_frame(tmp_path,rotate=rotate,yaw=yaw)
    assert frame_bev_schema(scene,'1')==4
    labels=model_ground_truth(scene,'1')
    assert labels['vehicle'][0,120,145]==1.  # physical ego x9.1,y4.1
    assert labels['vehicle'][0,80,145]==0.   # x9.1,y-3.9
    assert labels['vehicle'][0,120,55]==0.   # behind
    assert labels['validity'].any()
    assert not labels['elevation_validity'].any()
    assert np.isnan(labels['elevation_m']).all()
    assert set(np.unique(labels['das'][np.isfinite(labels['das'])]))<={0.,1.}


def test_global_unknown_metric_height_is_warning_not_semantic_failure(tmp_path):
    from sdv.bev.quality import validate_frame
    scene,arrays,metadata=global_frame(tmp_path)
    issues=validate_frame(arrays,metadata,100,5.)
    assert len(issues)==1
    assert issues[0]['code']=='bev_metric_elevation_unavailable'
    assert issues[0]['severity']=='warning'
    del arrays['elevation_validity'];save_arrays(scene,arrays)
    with pytest.raises(ValueError,match='elevation_validity'):
        load_bev_ground_truth(scene,'1')


@pytest.mark.parametrize('field', ['affine','scale','anchor','orientation'])
def test_invalid_affine_never_silently_uses_old_center_crop(tmp_path,field):
    scene,arrays,metadata=global_frame(tmp_path,rotate=False)
    if field=='affine':del metadata['geometry']['world_to_bev_pixel']
    elif field=='scale':metadata['geometry']['meters_per_pixel']=.1
    elif field=='anchor':metadata['geometry']['anchor_x']=.25
    else:metadata['geometry']['orientation']='ego_forward_up_ego_right_right'
    with pytest.raises(ValueError):validate_raster_affine(metadata)


def test_missing_static_layer_cannot_train_zero_masks(tmp_path):
    scene,arrays,metadata=global_frame(tmp_path)
    metadata['available_static_channels'].remove('lane')
    with pytest.raises(ValueError,match='disabled during collection'):
        validate_training_layers(metadata,['lane'])
    validate_training_layers(metadata,['das','vehicle','walker'])


def test_new_contract_signature_does_not_change_legacy_or_v3(tmp_path):
    from sdv.training.checkpointing import experiment_signature,BEV_GT_CONTRACT_VERSION,GLOBAL_RASTER_CONTRACT_VERSION
    train=tmp_path/'train';val=tmp_path/'val';train.mkdir();val.mkdir()
    sensor=tmp_path/'sensor.yaml';sensor.write_text('sensors: {}')
    legacy=experiment_signature(config(),sensor,train,val)
    old=experiment_signature(config(),sensor,train,val,bev_schema=3)
    new=experiment_signature(config(),sensor,train,val,bev_schema=4)
    assert old.pop('bev_ground_truth_contract')==BEV_GT_CONTRACT_VERSION
    assert new.pop('bev_ground_truth_contract')==GLOBAL_RASTER_CONTRACT_VERSION
    assert legacy==old==new


def test_schema4_actual_cpu_model_loss_backward_and_web_preflight(tmp_path,monkeypatch):
    torch=pytest.importorskip('torch');torch.set_num_threads(1)
    from sdv.training.modeling.model import models
    from sdv.training.modeling.loss import SimpleLoss
    from studio.dataset_tools import validate_scene
    scene,arrays,metadata=global_frame(tmp_path,rotate=False,yaw=90.)
    data=dataset_setup(tmp_path,monkeypatch)
    dataset=data.bev_Dataset(True,config())
    sample=dataset[0]
    assert dataset.bev_schema==4 and len(sample)==18
    assert np.isnan(dataset.get_bev_aux(0)['elevation_m'].numpy()).all()
    checked=validate_scene(scene,config())
    assert checked['bev_schema']==4 and checked['label_format']=='global_raster_numeric_v4'
    model=models(config(),pretrained=False).eval()
    outputs=model(sample[0][None],sample[13][None],sample[12][None],sample[11][None],
        sample[9][None],sample[8][None],None,None)
    loss=sum(SimpleLoss(2.13)(outputs[i],sample[i+2][None]) for i in range(4))
    assert torch.isfinite(loss)
    loss.backward()
    assert model.bevencode.das_head.weight.grad is not None


def test_catalog_reports_v4_without_claiming_absent_static_or_metric_height(tmp_path):
    from studio.task_catalog import task_catalog
    web_scene(tmp_path/'data/train')
    scene,arrays,metadata=global_frame(tmp_path/'data/train')
    metadata['available_static_channels'].remove('lane')
    info=json.loads((scene/'info/1.json').read_text());info['bev_gt']=metadata
    (scene/'info/1.json').write_text(json.dumps(info))
    result=task_catalog(tmp_path,{'train_data':'data/train','val_data':'data/val'})['bev']
    assert result['schema_version']==4 and result['schema_versions']==[4]
    assert result['height_channels']==[]
    layers={row['id']:row for row in result['layers']}
    assert layers['das']['available'] and not layers['lane']['available']
