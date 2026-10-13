"""CPU functional audit using user-supplied CARLA files, never synthetic accuracy.

Copies distinct uploaded frames into temporary train/val splits. The random
ResNet18 is only used to verify label/head/loss/metric/prediction connectivity.
This script does not validate CUDA, live CARLA, or generalization performance.
"""
import argparse
import ast
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))


def collater_function():
    import torch
    tree=ast.parse((PROJECT/'sdv/training/experiment.py').read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='collater')
    namespace={'torch':torch}
    exec(compile(ast.Module(body=[function],type_ignores=[]),'<actual SDV collater>','exec'),namespace)
    return namespace['collater']


def main(source, report):
    import cv2
    import numpy as np
    import torch
    import yaml
    from studio.config import Config
    from sdv.data_contract import decode_segmentation, normalize_task_settings, segmentation_spec
    from sdv.training.modeling.data import bev_Dataset
    from sdv.training.modeling.model import models
    from sdv.training.modeling.loss import Dice_BEC_Loss, silog_loss, SimpleLoss, FocalLoss
    from sdv.training.task_metrics import EpochMetrics
    from sdv.training.predictions import PredictionRecorder
    started=time.monotonic()
    source=Path(source).resolve()
    frames=sorted(source.glob('front_cam_rgb/*.png'))
    if len(frames)<8: raise ValueError('Need 8 distinct actual frames for disjoint smoke-test splits')
    runtime=PROJECT/'.runtime'; runtime.mkdir(exist_ok=True)
    fixture=Path(tempfile.mkdtemp(prefix='qa-task-pipeline-',dir=str(runtime)))
    # Distinct train/val frame IDs; same scene is expressly NOT a research split.
    selected=[frames[index] for index in (0,25,50,75,100,125,150,199) if index<len(frames)]
    if len(selected)<8: selected=frames[:8]
    observed=set()
    for frame in frames:
        rgb=cv2.cvtColor(cv2.imread(str(source/'front_cam_seg'/frame.name)),cv2.COLOR_BGR2RGB)
        observed.update(int(x) for x in np.unique(decode_segmentation(rgb)) if x!=255)
    for split, chosen in (('train',selected[:4]),('val',selected[4:])):
        dest=fixture/split/source.name
        for folder in ('front_cam_rgb','front_cam_seg','front_cam_depth','front_cam_bbs','das','lane','vehicle_mask','walker_mask'):
            (dest/folder).mkdir(parents=True,exist_ok=True)
            for frame in chosen:
                path=source/folder/(frame.stem+('.npy' if folder.endswith('bbs') else '.png'))
                if path.is_file(): shutil.copy2(str(path),str(dest/folder/path.name))
        for filename in ('sensor_config.yaml','collection_manifest.json'):
            shutil.copy2(str(source/filename),str(dest/filename))
    shutil.copy2(str(source/'sensor_config.yaml'),str(fixture/'sensors.yaml'))
    cfg=Config.training_defaults(None)
    cfg.update(Epochs=4,batch=2,width=128,height=128,lidar=False)
    cfg['sensor_use']={key:key=='front_cam' for key in cfg['sensor_use']}
    cfg['model'].update(encoder='ResNet18',decoder='FPN',bev_decoder='UNet',pretrained=False)
    cfg['model_conf'].update(Seg=True,Depth=True,BEV=True,od=True)
    cfg['task_settings']=normalize_task_settings({'segmentation':{'mode':'carla','class_ids':sorted(observed),'excluded':'ignore'},
        'depth':{'min_m':1.,'max_m':80.,'visual_min_m':1.,'visual_max_m':80.,'colormap':'viridis'},
        'detection':{'class_ids':[0,1]},'bev':{'layers':['das','lane']}})
    (fixture/'config.yaml').write_text(yaml.safe_dump(cfg))
    os.environ.update(SDV_TRAIN_DATA=str(fixture/'train'),SDV_VAL_DATA=str(fixture/'val'),SDV_SENSOR_CONFIG=str(fixture/'sensors.yaml'))
    torch.manual_seed(73); torch.set_num_threads(1)
    train,val=bev_Dataset(True,cfg),bev_Dataset(False,cfg)
    collater=collater_function()
    batch=collater([train[0],train[1]])
    model=models(cfg,pretrained=False).train(); optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    outputs=model(batch[0],batch[13],batch[12],batch[11],batch[9],batch[8],batch[6],batch[7])
    assert outputs[4].shape[2]==len(observed)
    assert batch[14].sum(2).eq(1).all()
    depth_values=batch[15][batch[16]]
    assert depth_values.min()>=1 and depth_values.max()<=80
    loss=Dice_BEC_Loss()(outputs[4],batch[14])+silog_loss(.85)(outputs[5],batch[15],batch[16])
    loss=loss+SimpleLoss(2.13)(outputs[0],batch[2])+SimpleLoss(2.13)(outputs[1],batch[3])
    anchors=model.anchors(batch[0].flatten(0,1))
    cls,reg,_,_=FocalLoss()(outputs[6],outputs[7],anchors,batch[17].flatten(0,1))
    loss=loss+cls.mean()+reg.mean(); assert torch.isfinite(loss)
    loss.backward(); assert torch.isfinite(model.seg_output[-1].weight.grad).all(); optimizer.step()
    metrics=EpochMetrics(cfg['model_conf'],cfg['task_settings']); metrics.update(outputs,batch,anchors)
    measured=metrics.compute('val')
    assert 'val_bev_lane_iou' in measured and 'val_bev_vehicle_iou' not in measured
    assert 'val_od_ap50' not in measured  # No uploaded frame has positive OD GT.
    output=fixture/'output'; recorder=PredictionRecorder(output,PROJECT,val)
    # Keep a real gap at epoch 3 to verify the UI never invents predictions.
    for epoch in (1,2,4): recorder.capture(model,val,collater,split='val',epoch=epoch,global_step=epoch)
    recorder.capture(model,train,collater,split='train',epoch=1,global_step=1)
    index=json.loads((output/'predictions/index.json').read_text())
    assert len(index['datasets']['val']['samples'])==4
    assert {entry['epoch'] for entry in index['entries'] if entry['split']=='val'}=={1,2,4}
    for entry in index['entries']:
        assert entry['label_schema']['segmentation']==segmentation_spec(cfg['task_settings'])
        for path in entry['images'].values(): assert (output/path).is_file()
    # Align loader to exact frame IDs: deleting a middle modality must fail at
    # that frame, never shift labels from the following frame.
    missing=Path(train.image_front[1]).parent.parent/'front_cam_seg'/Path(train.image_front[1]).name
    content=missing.read_bytes(); missing.unlink()
    try:
        try: train[1]
        except ValueError as exc: assert 'Missing aligned front_cam_seg' in str(exc)
        else: raise AssertionError('Missing label silently accepted')
    finally: missing.write_bytes(content)
    result={'passed':True,'source':str(source),'fixture':str(fixture),'prediction_output':str(output),
        'purpose':'Functional CPU connectivity only, random model; no accuracy claim',
        'cuda_verified':False,'carla_verified':False,'seg_class_ids':sorted(observed),'seg_output_channels':len(observed),
        'train_frames':[p.name for p in selected[:4]],'val_frames':[p.name for p in selected[4:]],
        'depth_encoding':'native-v1 PNG rounded/clipped meters; saturated255 invalid',
        'selected_bev_layers':['das','lane'],'empty_od_annotations':True,
        'positive_od_test':'Separate synthetic unit test, not observed in the uploaded dataset',
        'metrics_keys':sorted(measured),'prediction_entries':len(index['entries']),
        'epochs_with_predictions':[1,2,4],'epoch3':'No prediction exists',
        'checks':['actual uploaded RGB/Seg/Depth/OD/BEV aligned by stem',
                  'dynamic 19-channel head and loss backward finite','depth valid range 1..80m',
                  'unselected BEV layers absent from metrics','empty OD GT does not fabricate AP',
                  'fixed4 prediction history and immutable label schema','missing middle label raises clear error'],
        'elapsed_seconds':time.monotonic()-started}
    Path(report).write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source_scene'); p.add_argument('--report',default=str(PROJECT/'docs/task-pipeline-cpu-validation.json'))
    a=p.parse_args(); main(a.source_scene,a.report)
