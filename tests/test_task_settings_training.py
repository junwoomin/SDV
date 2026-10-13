"""Dynamic label/head/loss contracts with explicit small test fixtures."""
import copy
import os
from pathlib import Path
import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torchvision')
from sdv.data_contract import normalize_task_settings, segmentation_spec, SEMANTIC_COLORS
from sdv.training.modeling.model import models
from sdv.training.modeling.loss import Dice_BEC_Loss, silog_loss, FocalLoss
from sdv.training.task_metrics import EpochMetrics, SegmentationMetrics
from sdv.training.metric_registry import metric_registry
from sdv.training.predictions import _seg, _depth


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def config():
    return {'width': 128, 'height': 128, 'lidar': False,
        'sensor_use': {'front_cam': True},
        'model': {'encoder':'ResNet18', 'decoder':'FPN', 'bev_decoder':None, 'pretrained':False},
        'model_conf': {'BEV':False,'Seg':True,'Depth':True,'od':True},
        'task_settings': {'segmentation': {'mode':'carla','class_ids':[1,12,14], 'excluded':'ignore'},
                          'detection': {'class_ids':[1]}, 'depth': {'max_m':80.}}}


@pytest.mark.parametrize('excluded,channels', [('ignore',3),('background',4)])
def test_dynamic_heads_losses_backward_and_metrics(excluded, channels):
    cfg = config()
    cfg['task_settings']['segmentation']['excluded'] = excluded
    model = models(cfg, pretrained=False).train()
    image = torch.randn(2,1,3,128,128)
    matrix = torch.eye(3)[None,None].repeat(2,1,1,1)
    translation = torch.zeros(2,1,3)
    outputs = model(image,translation,matrix,matrix,translation,matrix,None,None)
    assert outputs[4].shape == (2,1,channels,128,128)
    assert outputs[6].shape[-1] == 1
    labels = torch.zeros_like(outputs[4]); labels[:,:,0] = 1
    labels[:,:,:,:,64:] = 0  # ignore pixels must not generate gradients
    seg_loss = Dice_BEC_Loss()(outputs[4],labels)
    outputs[4].retain_grad()
    depth_gt = torch.full_like(outputs[5], 25.)
    mask = torch.ones_like(depth_gt, dtype=torch.bool)
    anchors = model.anchors(image.flatten(0,1))
    annotation = torch.tensor([[[30.,30.,80.,80.,0.]], [[30.,30.,80.,80.,0.]]])
    focal = FocalLoss()
    cls, reg, _, _ = focal(outputs[6],outputs[7],anchors,annotation)
    loss = seg_loss + silog_loss(.85)(outputs[5],depth_gt,mask) + cls.mean() + reg.mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.count_nonzero(outputs[4].grad[:,:,:,:,:64]) > 0
    assert torch.count_nonzero(outputs[4].grad[:,:,:,:,64:]) == 0
    assert model.seg_output[-1].weight.grad is not None
    metrics = EpochMetrics(cfg['model_conf'], cfg['task_settings'])
    batch={'seg_GT':labels,'dept_GT':depth_gt,'depth_mask':mask,'annotations':annotation[:,None],'imgs':image}
    metrics.update(outputs,batch,anchors)
    values = metrics.compute('val')
    assert {'val_seg_miou','val_seg_mean_dice','val_seg_dice_class_0','val_depth_rmse','val_od_ap50'} <= values.keys()
    registry = metric_registry(cfg)
    assert registry['val_seg_iou_class_0']['class_name'] == ('Road' if excluded=='ignore' else 'Background / excluded')
    assert registry['val_od_ap50_class_0']['class_name'] == 'Pedestrian'
    assert registry['val_seg_miou']['maximum'] == 1.
    assert registry['val_depth_abs_rel']['maximum'] is None
    assert registry['val_loss']['maximum'] is None


def test_ignored_segmentation_and_invalid_depth_produce_zero_finite_gradients():
    pred = torch.zeros(1,1,3,2,2,requires_grad=True)
    loss = Dice_BEC_Loss()(pred,torch.zeros_like(pred))
    loss.backward()
    assert loss.item() == 0 and torch.count_nonzero(pred.grad) == 0
    depth = torch.ones(1,1,1,2,2,requires_grad=True)
    loss = silog_loss(.85)(depth,torch.zeros_like(depth),torch.zeros_like(depth,dtype=torch.bool))
    loss.backward()
    assert loss.item() == 0 and torch.isfinite(depth.grad).all()


def test_metrics_and_png_use_the_same_named_class_order_and_ignore():
    settings = normalize_task_settings({'segmentation':{'mode':'carla','class_ids':[1,14],'excluded':'ignore'}})
    labels = torch.tensor([[[1.,0.],[0.,0.]], [[0.,1.],[0.,0.]]])
    rgb = _seg(labels,segmentation_spec(settings),ground_truth=True)
    np.testing.assert_array_equal(rgb[0,0], SEMANTIC_COLORS[1])
    np.testing.assert_array_equal(rgb[0,1], SEMANTIC_COLORS[14])
    np.testing.assert_array_equal(rgb[1,0], [255,0,255])
    m = SegmentationMetrics(2); m.update(labels[None],labels[None])
    assert m.compute()['seg_miou'] == m.compute()['seg_dice_class_0'] == 1.


def test_depth_visual_range_does_not_change_label_tensor():
    value = torch.tensor([[[1.,40.],[80.,100.]]]); original = value.clone()
    setting = {'visual_min_m':0.,'visual_max_m':80.,'colormap':'gray'}
    rendered = _depth(value,setting,torch.tensor([[[True,True],[True,False]]]))
    assert torch.equal(value,original)
    assert rendered[0,1,0] == 128
    np.testing.assert_array_equal(rendered[1,1], [255,0,255])


def test_selected_bev_metrics_do_not_report_unselected_layers():
    settings={'bev':{'layers':['lane']}}
    metric=EpochMetrics({'BEV':True},settings)
    truth=torch.ones(1,1,2,2); pred=torch.ones_like(truth)
    metric.update({'DAS':pred,'Lane':pred,'Vehicle':pred,'walker':pred},
                  {'das_GT':truth,'lane_GT':truth,'vehicle_GT':truth,'walker_GT':truth})
    values=metric.compute()
    assert values['bev_lane_iou'] == values['bev_miou'] == 1.
    assert 'bev_das_iou' not in values


@pytest.mark.parametrize('decoder', ['ResNet','EfficientNet','UNet'])
@pytest.mark.parametrize('cameras', [1,2])
def test_bev_decoder_one_and_two_cameras_forward_backward(decoder,cameras):
    cfg=config();cfg['model_conf']={'BEV':True,'Seg':False,'Depth':False,'od':False}
    cfg['model']['bev_decoder']=decoder
    cfg['sensor_use']={'front_cam':True,'left_cam':cameras==2}
    model=models(cfg,pretrained=False).train()
    image=torch.randn(2,cameras,3,128,128,requires_grad=True)
    matrix=torch.eye(3)[None,None].repeat(2,cameras,1,1)
    intrinsic=matrix.clone();intrinsic[:,:,0,0]=64;intrinsic[:,:,1,1]=64
    intrinsic[:,:,0,2]=64;intrinsic[:,:,1,2]=64
    translation=torch.zeros(2,cameras,3)
    outputs=model(image,translation,matrix,intrinsic,translation,matrix,None,None)
    assert all(value.shape==(2,1,200,200) for value in outputs[:4])
    assert all(torch.isfinite(value).all() for value in outputs[:4])
    sum(value.square().mean() for value in outputs[:4]).backward()
    assert torch.isfinite(image.grad).all() and torch.count_nonzero(image.grad)>0


def test_loader_prefers_raw_native_labels_and_remaps_od_by_frame(tmp_path,monkeypatch):
    import cv2
    import sdv.training.modeling.data as data
    cfg=config();cfg.update(width=64,height=64)
    cfg['task_settings']['segmentation']['class_ids']=[1,14]
    cfg['task_settings']['depth'].update(min_m=10.,max_m=20.,out_of_range='clip')
    sensor={'sensors':{'front_cam':{'width':64,'height':64,'fov':90,'x':1.,'y':0.,'z':2.,'pitch':0.,'roll':0.,'yaw':0.}}}
    monkeypatch.setattr(data,'read_sensor_config',lambda:sensor)
    monkeypatch.setenv('SDV_TRAIN_DATA',str(tmp_path))
    scene=tmp_path/'session'/'Town01'/'scene'
    for folder in ('front_cam_rgb','front_cam_seg','front_cam_seg_ids','front_cam_depth','front_cam_depth_m','front_cam_bbs'):
        (scene/folder).mkdir(parents=True)
    for stem in ('2','10'):
        cv2.imwrite(str(scene/'front_cam_rgb'/(stem+'.png')),np.zeros((64,64,3),np.uint8))
        cv2.imwrite(str(scene/'front_cam_seg'/(stem+'.png')),np.zeros((64,64,3),np.uint8))
        ids=np.full((64,64),1,np.uint8);ids[:,32:]=14
        cv2.imwrite(str(scene/'front_cam_seg_ids'/(stem+'.png')),ids)
        cv2.imwrite(str(scene/'front_cam_depth'/(stem+'.png')),np.full((64,64),255,np.uint8))
        meters=np.full((64,64),1.,np.float32);meters[:,32:]=100.;meters[0,0]=np.nan
        np.save(scene/'front_cam_depth_m'/(stem+'.npy'),meters)
        np.save(scene/'front_cam_bbs'/(stem+'.npy'),np.array([[1,2,10,20,0],[4,5,15,30,1]],np.float32))
    dataset=data.bev_Dataset(True,cfg)
    assert [Path(path).stem for path in dataset.image_front]==['2','10']
    batch=dataset[0]
    assert batch[14].shape==(1,2,64,64) and batch[14][0,0,1,1]==1 and batch[14][0,1,1,40]==1
    assert batch[15][0,0,1,1]==10 and batch[15][0,0,1,40]==20
    assert not batch[16][0,0,0,0]
    assert batch[17][0].shape==(1,6)
    assert batch[17][0][0,5]==0  # original Pedestrian ID1 -> selected output index0
    assert batch[17][0][0,1:5].tolist()==[4.,5.,15.,30.]
    (scene/'front_cam_bbs'/'2.npy').unlink()
    with pytest.raises(ValueError,match='Missing aligned front_cam_bbs'):
        dataset[0]
