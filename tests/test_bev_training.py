"""Explicit geometry fixtures: numeric BEV labels, calibration and real CPU loss.

These are controlled geometry tests, not claims of CARLA runtime validation.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from sdv.bev.schema import BINARY_CHANNELS, HEIGHT_CHANNELS, bev_settings_defaults
from sdv.bev.training import (camera_optical_to_ego, frame_has_numeric_bev,
    load_bev_ground_truth, model_ground_truth, model_sampling_indices, model_to_display,
    camera_optical_to_bev_frame, validate_training_layers)


def fixture_frame(root, stem='1', width=500, height=500, mpp=.2, anchor=(.5,.5)):
    scene = Path(root) / 'scene'
    (scene / 'bev_gt').mkdir(parents=True, exist_ok=True)
    (scene / 'info').mkdir(exist_ok=True)
    shape = (height, width)
    arrays = {key: np.zeros(shape, np.uint8) for key in BINARY_CHANNELS}
    arrays.update({key: np.full(shape, np.nan, np.float32) for key in HEIGHT_CHANNELS})
    arrays.update(semantic=np.zeros(shape, np.uint8), actor_id=np.zeros(shape, np.int32),
                  surface_id=np.zeros(shape, np.int32))
    arrays['validity'][:] = 1
    arrays['occupancy_validity'][:] = 1
    arrays['lane_validity'][:] = 1
    arrays['elevation_m'][:] = 7.
    geometry = {'width':width,'height':height,'meters_per_pixel':mpp,
                'anchor_x':anchor[0],'anchor_y':anchor[1]}
    metadata = {'schema_version':3,'geometry':geometry,
                'ego_pose':{'x':0.,'y':0.,'z':7.,'yaw':0.}, 'frame_id':100,
                'timestamp':5., 'source':'controlled_test_fixture', 'settings':bev_settings_defaults()}
    sync = {'carla_frame':100,'sensors':{'front_cam_rgb':{'frame':100,'timestamp_s':5.,
        'world_transform':{'position':{'x':1.,'y':0.,'z':9.},'rotation_deg':{'pitch':0.,'roll':0.,'yaw':0.}}}}}
    (scene/'info'/(stem+'.json')).write_text(json.dumps({'bev_gt':metadata,'sdv_metadata':{'synchronization':sync}}))
    np.savez_compressed(scene/'bev_gt'/(stem+'.npz'), **arrays)
    return scene, arrays, metadata


def save_arrays(scene, arrays, stem='1'):
    np.savez_compressed(scene/'bev_gt'/(stem+'.npz'), **arrays)


def test_physical_grid_is_y_rows_x_columns_and_forward_up_render(tmp_path):
    scene, arrays, metadata = fixture_frame(tmp_path)
    # A front-right 4 x 2 m vehicle: x=8..12m, y=4..6m.
    arrays['vehicle'][190:211,270:281] = 1
    arrays['object_height_m'][190:211,270:281] = 3.
    save_arrays(scene, arrays)
    value = model_ground_truth(scene, '1')
    assert value['vehicle'].shape == (1,200,200)
    rows, cols = np.where(value['vehicle'][0] == 1)
    assert 119 <= rows.min() <= 121 and 129 <= rows.max() <= 131
    assert 139 <= cols.min() <= 141 and 159 <= cols.max() <= 161
    display = model_to_display(value['vehicle'])[0]
    rows, cols = np.where(display == 1)
    assert rows.max() < 100 and cols.min() > 100
    assert np.nanmax(value['object_height_m']) == 3.
    assert np.all(value['elevation_m'] == 7.)


@pytest.mark.parametrize('mpp,anchor,width,height', [(.1,(.25,.75),700,800),(.5,(.75,.25),100,90)])
def test_nondefault_geometry_nearest_sampling_and_coverage(tmp_path,mpp,anchor,width,height):
    scene, arrays, metadata = fixture_frame(tmp_path,width=width,height=height,mpp=mpp,anchor=anchor)
    u,v = np.meshgrid(np.arange(width),np.arange(height))
    arrays['drivable'] = ((u+v)%2).astype(np.uint8)
    save_arrays(scene, arrays)
    output = model_ground_truth(scene,'1')
    yy,xx,inside = model_sampling_indices(metadata['geometry'])
    valid = output['validity'][0]
    np.testing.assert_array_equal(valid,inside)
    np.testing.assert_array_equal(output['das'][0][valid],arrays['drivable'][yy,xx][inside])
    assert set(np.unique(output['das'][np.isfinite(output['das'])])) <= {0.,1.}
    assert np.isnan(output['das'][0][~valid]).all()
    # Nearest sampling coordinate error is bounded by half a source pixel.
    x = -20. + (np.arange(200)+.5)*.2
    y = x.copy()
    true_u = width*anchor[0]+y[:,None]/mpp
    true_v = height*anchor[1]-x[None,:]/mpp
    assert np.max(np.abs((xx-true_u)[inside])*mpp) <= .5*mpp+1e-8
    assert np.max(np.abs((yy-true_v)[inside])*mpp) <= .5*mpp+1e-8


def test_static_and_occupancy_validity_are_not_conflated(tmp_path):
    scene, arrays, metadata = fixture_frame(tmp_path)
    arrays['occupancy_validity'][:] = 0
    arrays['drivable'][:] = 1
    arrays['lane_validity'][:] = 0
    arrays['vehicle'][:] = 1  # Must not silently become valid GT.
    save_arrays(scene, arrays)
    value = model_ground_truth(scene,'1')
    assert np.all(value['das'] == 1.)
    assert np.isnan(value['vehicle']).all()
    assert np.isnan(value['lane']).all()
    assert not value['occupancy_validity'].any()
    assert np.all(value['actor_id'] == -1)


@pytest.mark.parametrize('change', ['missing','dtype','shape','nonbinary','badclass','infinite'])
def test_malformed_archive_rejected(tmp_path,change):
    scene, arrays, metadata = fixture_frame(tmp_path)
    if change == 'missing': del arrays['validity']
    elif change == 'dtype': arrays['vehicle'] = arrays['vehicle'].astype(np.float32)
    elif change == 'shape': arrays['lane'] = arrays['lane'][:-1]
    elif change == 'nonbinary': arrays['lane'][0,0] = 255
    elif change == 'badclass': arrays['semantic'][0,0] = 255
    elif change == 'infinite': arrays['elevation_m'][0,0] = np.inf
    save_arrays(scene,arrays)
    with pytest.raises(ValueError): load_bev_ground_truth(scene,'1')


def test_declared_new_frame_never_falls_back_to_legacy_png(tmp_path):
    scene,arrays,metadata = fixture_frame(tmp_path)
    (scene/'bev_gt/1.npz').unlink()
    with pytest.raises(ValueError,match='Missing declared'): frame_has_numeric_bev(scene,'1')
    (scene/'info/1.json').unlink()
    assert frame_has_numeric_bev(scene,'1') is False


def test_optical_basis_and_carla_pitch_roll_yaw():
    transform,rotation,translation = camera_optical_to_ego([1,2,3,0,0,0])
    np.testing.assert_allclose(rotation @ [0,0,1],[1,0,0])  # camera forward
    np.testing.assert_allclose(rotation @ [1,0,0],[0,1,0])  # camera right
    np.testing.assert_allclose(rotation @ [0,1,0],[0,0,-1])  # camera down
    np.testing.assert_allclose(transform[:3,3],[1,2,3])
    _,rotation,_ = camera_optical_to_ego([0,0,0,0,0,90])
    np.testing.assert_allclose(rotation @ [0,0,1],[0,1,0],atol=1e-12)
    _,rotation,_ = camera_optical_to_ego([0,0,0,30,0,0])
    np.testing.assert_allclose(rotation @ [0,0,1],[np.sqrt(3)/2,0,.5],atol=1e-12)
    _,rotation,_ = camera_optical_to_ego([0,0,0,0,30,0])
    np.testing.assert_allclose(rotation @ [1,0,0],[0,np.sqrt(3)/2,-.5],atol=1e-12)


def test_saved_camera_world_pose_preserves_slope_in_yaw_leveled_bev(tmp_path):
    scene,arrays,metadata=fixture_frame(tmp_path)
    path=scene/'info/1.json'
    info=json.loads(path.read_text())
    info['bev_gt']['ego_pose']['yaw']=90.
    packet=info['sdv_metadata']['synchronization']['sensors']['front_cam_rgb']
    packet['world_transform']['position']={'x':0.,'y':1.,'z':9.}
    packet['world_transform']['rotation_deg']={'yaw':90.,'pitch':30.,'roll':0.}
    path.write_text(json.dumps(info))
    transform,rotation,translation=camera_optical_to_bev_frame(scene,'1','front_cam')
    # Actual sensor's central 10m ray rises on a 30deg slope; BEV remains horizontal.
    projected=transform @ [0.,0.,10.,1.]
    np.testing.assert_allclose(projected,[1.+10*np.sqrt(3)/2,0.,7.,1.],atol=1e-12)
    packet['frame']=101
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError,match='frame mismatch'):
        camera_optical_to_bev_frame(scene,'1','front_cam')


@pytest.mark.parametrize('task,key,removed',[('das','semantic_layers','drivable'),('lane','semantic_layers','lane'),
    ('vehicle','actor_classes','vehicle'),('walker','actor_classes','pedestrian')])
def test_disabled_collection_classes_are_not_negative_labels(tmp_path,task,key,removed):
    scene,arrays,metadata=fixture_frame(tmp_path)
    metadata['settings'][key].remove(removed)
    with pytest.raises(ValueError,match='disabled during collection'):
        validate_training_layers(metadata,[task])


def test_masked_loss_gradient_metrics_and_preview():
    torch = pytest.importorskip('torch')
    from sdv.training.modeling.loss import SimpleLoss
    from sdv.training.task_metrics import BinaryMetrics
    from sdv.training.predictions import _mask
    truth = torch.tensor([[[[1.,0.],[float('nan'),float('nan')]]]])
    logits = torch.tensor([[[[2.,-2.],[-999.,999.]]]],requires_grad=True)
    loss = SimpleLoss(2.13)(logits,truth)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits.flatten()[:2],truth.flatten()[:2],pos_weight=torch.tensor([2.13]))
    torch.testing.assert_close(loss,expected)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad[0,0,1]) == 0
    metrics = BinaryMetrics();metrics.update(logits,truth)
    assert metrics.compute()['iou'] == metrics.compute()['dice'] == 1.
    metrics = BinaryMetrics();metrics.update(logits,torch.full_like(truth,float('nan')))
    assert metrics.compute() == {}
    logits.grad.zero_()
    empty_loss = SimpleLoss(2.13)(logits,torch.full_like(truth,float('nan')))
    empty_loss.backward()
    assert empty_loss.item() == 0. and not logits.grad.any()
    image = _mask(truth[0],False,forward_up=True)
    assert ((image == (255,0,255)).all(axis=-1)).sum() == 2


def config():
    return {'width':64,'height':64,'lidar':False,'sensor_use':{'front_cam':True},
        'model':{'encoder':'ResNet18','decoder':None,'bev_decoder':'UNet','pretrained':False},
        'model_conf':{'BEV':True,'Seg':False,'Depth':False,'od':False}}


def dataset_setup(root,monkeypatch):
    pytest.importorskip('torch');cv2=pytest.importorskip('cv2')
    import sdv.training.modeling.data as data
    sensor={'sensors':{'front_cam':{'width':64,'height':64,'fov':90,'x':1.,'y':0.,'z':2.,'pitch':0.,'roll':0.,'yaw':0.}}}
    monkeypatch.setattr(data,'read_sensor_config',lambda:sensor)
    monkeypatch.setenv('SDV_TRAIN_DATA',str(root))
    (root/'scene/front_cam_rgb').mkdir(parents=True,exist_ok=True)
    import yaml
    (root/'scene/sensor_config.yaml').write_text(yaml.safe_dump(sensor))
    cv2.imwrite(str(root/'scene/front_cam_rgb/1.png'),np.zeros((64,64,3),np.uint8))
    return data


def test_actual_dataset_aux_and_model_backward(tmp_path,monkeypatch):
    torch=pytest.importorskip('torch');torch.set_num_threads(1)
    from sdv.training.modeling.model import models
    from sdv.training.modeling.loss import SimpleLoss
    scene,arrays,metadata=fixture_frame(tmp_path/'data')
    arrays['occupancy_validity'][:,250:]=0;save_arrays(scene,arrays)
    data=dataset_setup(tmp_path/'data',monkeypatch)
    dataset=data.bev_Dataset(True,config())
    sample=dataset[0]
    assert dataset.bev_schema==3 and len(sample)==18
    aux=dataset.get_bev_aux(0)
    assert aux['elevation_m'].dtype==torch.float32 and aux['validity'].dtype==torch.bool
    assert torch.all(aux['elevation_m']==7.)
    model=models(config(),pretrained=False).eval()
    output=model(sample[0][None],sample[13][None],sample[12][None],sample[11][None],
        sample[9][None],sample[8][None],None,None)
    assert all(x.shape==(1,1,200,200) for x in output[:4])
    loss=sum(SimpleLoss(2.13)(output[i],sample[i+2][None]) for i in range(4))
    assert torch.isfinite(loss)
    loss.backward()
    assert model.bevencode.das_head.weight.grad is not None
    # A unit feature at physical x=10,y=5 appears at model row125,col150.
    geometry=torch.tensor([10.,5.,0.]).reshape(1,1,1,1,1,3)
    grid=model.voxel_pooling(geometry,torch.ones(1,1,1,1,1,1))
    assert grid.shape==(1,1,200,200) and grid[0,0,125,150]==1
    from sdv.training.predictions import PredictionRecorder
    def collate(samples):
        return tuple(None if items[0] is None else
                     torch.empty(len(items),1,0,5) if isinstance(items[0],list) else torch.stack(items)
                     for items in zip(*samples))
    recorder=PredictionRecorder(tmp_path/'output',tmp_path,dataset)
    first=recorder.capture(model,dataset,collate,epoch=1)
    second=recorder.capture(model,dataset,collate,epoch=2)
    assert first['count']==second['count']==4
    assert first['samples']==second['samples']
    entries=json.loads((tmp_path/'output/predictions/index.json').read_text())['entries']
    assert {e['epoch'] for e in entries}=={1,2}
    assert all(e['bev_ground_truth']['schema_version']==3 for e in entries)
    from PIL import Image
    vehicle=next(e for e in entries if e['task']=='Vehicle')
    predicted=np.asarray(Image.open(tmp_path/'output'/vehicle['pred']))
    ground=np.asarray(Image.open(tmp_path/'output'/vehicle['gt']))
    np.testing.assert_array_equal((predicted==(255,0,255)).all(axis=-1),(ground==(255,0,255)).all(axis=-1))


def test_legacy_labels_and_calibration_remain_unchanged_and_mixing_rejected(tmp_path,monkeypatch):
    torch=pytest.importorskip('torch');cv2=pytest.importorskip('cv2')
    data=dataset_setup(tmp_path,monkeypatch)
    scene=tmp_path/'scene'
    mask=np.zeros((500,500),np.uint8);mask[170:190,200:220]=255
    for folder in ('das','lane','vehicle_mask','walker_mask'):
        (scene/folder).mkdir();cv2.imwrite(str(scene/folder/'1.png'),mask)
    dataset=data.bev_Dataset(True,config())
    assert dataset.bev_schema==1 and dataset.get_bev_aux(0) is None
    for label in dataset.get_bev(0):
        np.testing.assert_array_equal(label.numpy()[0],mask[150:350,150:350]/255.)
    np.testing.assert_array_equal(dataset.get_cam_para(0)[2].numpy()[0],np.eye(3))
    fixture_frame(tmp_path,stem='2')
    cv2.imwrite(str(scene/'front_cam_rgb/2.png'),np.zeros((64,64,3),np.uint8))
    with pytest.raises(ValueError,match='Mixed legacy'):
        data.bev_Dataset(True,config())


def test_generator_archive_round_trip_to_training_labels(tmp_path):
    pytest.importorskip('cv2')
    from sdv.bev.cache import GlobalRasterCache
    from sdv.bev.generator import HeightAwareBEVGenerator
    from sdv.bev.io import save_frame
    surfaces=[]
    for height,road_id in ((0.,1),(6.,2)):
        surfaces.append({'id':'road'+str(road_id),'road_id':road_id,'section_id':0,'lane_id':-1,
            'polygon_xyz':[[-20,-5,height],[20,-5,height],[20,5,height],[-20,5,height]],
            'center_xyz':[0,0,height],'neighbors':[],'kind':'driving','is_junction':False})
    cache=GlobalRasterCache.build(tmp_path/'cache','ControlledStacked','<OpenDRIVE />',surfaces)
    generator=HeightAwareBEVGenerator(cache)
    def actor(identity,x,height,road):
        return {'actor_id':identity,'object_class':'vehicle','x':x,'y':1.,'z':height+.3,'yaw':0.,
            'ground_z_m':height,'road_id':road,'lane_id':-1,'section_id':0,
            'bbox_min_z_m':height+.05,'bbox_max_z_m':height+2.05,'object_height_m':2.,
            'footprint_world_xyz':[[x-2,0,height],[x+2,0,height],[x+2,2,height],[x-2,2,height]]}
    ego=actor(1,0.,0.,1)
    generated=generator.generate(ego,[ego,actor(2,5.,0.,1),actor(3,5.,6.,2)],frame_id=100,timestamp=5.)
    scene,_,_=fixture_frame(tmp_path/'data')
    save_frame(scene/'bev_gt/1.npz',generated['arrays'])
    info=json.loads((scene/'info/1.json').read_text())
    info['bev_gt']=generated['metadata']
    (scene/'info/1.json').write_text(json.dumps(info))
    labels=model_ground_truth(scene,'1')
    validate_training_layers(labels['metadata'],['das','lane','vehicle','walker'])
    assert np.any(labels['vehicle']==1) and not np.any(labels['actor_id']==3)
    assert np.all(labels['object_ground_z_m'][labels['actor_id']==2]==0.)
    assert np.all(labels['object_height_m'][labels['actor_id']==2]==2.)
    assert not np.any(labels['vehicle'][labels['actor_id']==1]==1)  # Ego is a separate channel.


def web_scene(root, width=200, height=180, mpp=.25, anchor=(.4,.6)):
    from PIL import Image
    import yaml
    scene,arrays,metadata=fixture_frame(root,width=width,height=height,mpp=mpp,anchor=anchor)
    (scene/'front_cam_rgb').mkdir(exist_ok=True)
    Image.fromarray(np.zeros((64,64,3),np.uint8)).save(scene/'front_cam_rgb/1.png')
    sensor={'sensors':{'front_cam':{'width':64,'height':64,'fov':90,'x':1.,'y':0.,'z':2.,'pitch':0.,'roll':0.,'yaw':0.}}}
    (scene/'sensor_config.yaml').write_text(yaml.safe_dump(sensor))
    return scene


def test_web_preflight_accepts_numeric_only_variable_geometry_and_no_torch(tmp_path):
    from studio.dataset_tools import validate_scene
    scene=web_scene(tmp_path)
    checked=validate_scene(scene,config())
    assert checked['bev_schema']==3 and checked['label_format']=='height_aware_numeric_v3'
    assert {'bev_gt','info','front_cam_rgb'}<=set(checked['modalities'])
    assert all(value>0 for value in checked['valid_bev_pixels'].values())
    info=json.loads((scene/'info/1.json').read_text())
    del info['sdv_metadata']['synchronization']['sensors']['front_cam_rgb']['world_transform']
    (scene/'info/1.json').write_text(json.dumps(info))
    with pytest.raises(ValueError,match='actual RGB world calibration'):
        validate_scene(scene,config())


def test_preparation_preserves_numeric_labels_info_and_intrinsics(tmp_path):
    from studio.dataset_tools import prepare_dataset,validate_training_inputs
    first=web_scene(tmp_path/'collections/train')
    second=web_scene(tmp_path/'collections/val')
    class SceneCatalog:
        root=tmp_path
        def scene_path(self,identifier): return {'train':first,'val':second}[identifier]
    result=prepare_dataset(SceneCatalog(),'numeric',['train'],['val'],config())
    target=Path(result['path'])
    assert len(list(target.rglob('bev_gt/*.npz')))==2
    assert len(list(target.rglob('info/*.json')))==2
    assert len(list(target.rglob('sensor_config.yaml')))==3
    np.testing.assert_equal(load_bev_ground_truth(first,'1')[0]['elevation_m'],
        load_bev_ground_truth(next((target/'train').iterdir()),'1')[0]['elevation_m'])
    assert validate_training_inputs(tmp_path,result['settings'],config())==[]


def test_preflight_rejects_zero_valid_bev_targets(tmp_path):
    from studio.dataset_tools import validate_scene
    scene=web_scene(tmp_path)
    arrays,metadata=load_bev_ground_truth(scene,'1')
    arrays['occupancy_validity'][:]=0
    save_arrays(scene,arrays)
    with pytest.raises(ValueError,match='유효 학습 픽셀'):
        validate_scene(scene,config())


def test_checkpoint_signature_versions_new_contract_and_preserves_legacy(tmp_path):
    from sdv.training.checkpointing import experiment_signature,BEV_GT_CONTRACT_VERSION
    train=tmp_path/'train';val=tmp_path/'val';train.mkdir();val.mkdir()
    sensor=tmp_path/'sensor.yaml';sensor.write_text('sensors: {}')
    before=experiment_signature(config(),sensor,train,val)
    assert experiment_signature(config(),sensor,train,val,bev_schema=1)==before
    new=experiment_signature(config(),sensor,train,val,bev_schema=3)
    assert new.pop('bev_ground_truth_contract')==BEV_GT_CONTRACT_VERSION
    assert new==before
