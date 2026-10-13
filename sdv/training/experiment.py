import argparse
import json
from sdv.runtime.processes import prepare,ROOT
prepare()
from sdv.config.validation import read_train_config,training_blockers
_parser=argparse.ArgumentParser()
_parser.add_argument('--config',default='data/train_config.yaml')
_args=_parser.parse_args()
train_config=read_train_config(_args.config)
_missing=training_blockers(train_config)
if _missing:
    raise SystemExit('학습 준비 필요: '+', '.join(_missing))
# Existing encoder constructors load cached pretrained weights only.
import torch.hub
def _no_model_download(*args,**kwargs):
    raise RuntimeError('사전학습 가중치가 로컬 캐시에 없습니다. 웹 학습 화면의 준비 상태 검사로 가중치를 준비한 뒤 다시 시작하세요.')
torch.hub.download_url_to_file=_no_model_download
import yaml
import sys
import os
import time
import math
from sdv.telemetry import configured_path, emit

from sdv.training.modeling.data import bev_Dataset
from sdv.training.modeling.model import models
import torch
from torch.utils.data import DataLoader
import logging
import torch.nn.functional as F
import cv2
import numpy as np
from sdv.training.checkpointing import (PAUSE_EXIT_CODE, experiment_signature, load_resume,
    pause_requested, restore_rng, save_resume)
from sdv.training.task_metrics import EpochMetrics
from sdv.data_contract import normalize_task_settings, segmentation_spec
from sdv.training.metric_registry import metric_registry
task_settings = normalize_task_settings(train_config.get('task_settings'))
bev_layers = set(task_settings['bev']['layers'])
from sdv.training.predictions import PredictionRecorder
import pandas as pd
import matplotlib.pyplot as plt
import shutil
from datetime import datetime
import glob
from sdv.training.modeling.loss import *
from torchvision.ops import nms

class_names = ['vehicles','pedestrians']
def compute_abs_rel(pred, gt, mask):
    if not bool(mask.any()):
        return 0.0
    pred = pred[mask]
    gt = gt[mask]

    abs_rel = torch.mean(torch.abs(pred - gt) / gt)
    return abs_rel.item()


def get_iou(preds, binimgs):
    with torch.no_grad():
        valid = torch.isfinite(binimgs) & torch.isfinite(preds)
        pred = (preds > 0) & valid
        tgt = (binimgs > .5) & valid
        intersect = (pred & tgt).sum().float().item()
        union = (pred | tgt).sum().float().item()
    return intersect, union, intersect / union if union > 0 else (1.0 if bool(valid.any()) else 0.0)
CLASS_COLORS = {item['output_index']: tuple(item['color']) for item in segmentation_spec(task_settings)}

def mask_to_color(mask):
    """
    (H, W) 형태의 클래스 인덱스를 RGB 이미지로 변환.
    """
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_idx, color in CLASS_COLORS.items():
        color_mask[mask == class_idx] = color
    return color_mask

def load_input_and_gt(load_path=None):
    if val_set.bev_schema in (3,4):
        # New numeric GT previews are written by PredictionRecorder, with real
        # validity/coordinates. The old optional test.pth is unrelated data.
        return None
    selected = load_path or configured_path("SDV_TEST_TENSOR", "img/test.pth")
    if os.path.isfile(selected):
        load_data = torch.load(selected, weights_only=True)
    else:
        # This tensor is used only for GT preview PNGs. The first real validation
        # frame is an equivalent source and avoids requiring an unrelated fixture.
        masks = val_set.get_bev(0)
        if any(mask is None for mask in masks): return None
        load_data = {name: value.unsqueeze(0) for name, value in
                     zip(('das_GT', 'lane_GT', 'vehicle_GT', 'walker_GT'), masks)}
    return {k: v.cuda() for k, v in load_data.items()}
def plot_metric(train_values, val_values, ylabel, title, filename):
    plt.figure()
    plt.plot(epochs, train_values, 'b-', label=f'Train {ylabel}')
    plt.plot(epochs, val_values, 'r-', label=f'Val {ylabel}')
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_dirs}/{filename}.png', dpi=500)
    plt.close()
def collater(data):
    imgs, imgs_ori,das_GT, lane_GT, vehicle_GT, walker_GT, \
    lidar_data, lidar_mask, post_rots, post_trans, \
    extrinsic, intrinsic, rotation, translation, \
    seg_GT, dept_GT, depth_mask, annots_batch = zip(*data)

    imgs = torch.stack(imgs)
    imgs_ori = torch.stack(imgs_ori)
    translation = torch.stack(translation)
    rotation = torch.stack(rotation)
    extrinsic = torch.stack(extrinsic)
    intrinsic = torch.stack(intrinsic)
    post_trans = torch.stack(post_trans)
    post_rots = torch.stack(post_rots)
    def stack_optional(values, label, dtype=None):
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise ValueError(f"{label}: 같은 배치에 누락된 표본이 있습니다.")
        return torch.stack([torch.as_tensor(value, dtype=dtype or torch.float32) for value in values])

    das_GT = stack_optional(das_GT, "DAS")
    lane_GT = stack_optional(lane_GT, "Lane")
    vehicle_GT = stack_optional(vehicle_GT, "Vehicle")
    walker_GT = stack_optional(walker_GT, "Walker")
    dept_GT = stack_optional(dept_GT, "Depth")
    depth_mask = stack_optional(depth_mask, "Depth mask", dtype=torch.bool)
    seg_GT = stack_optional(seg_GT, "Segmentation")
    lidar_data = stack_optional(lidar_data, "LiDAR")
    lidar_mask = stack_optional(lidar_mask, "LiDAR mask")

    B, N = imgs.shape[:2]
    max_num_annots = max((cam.shape[0] for annots in annots_batch for cam in annots), default=0)
    padded_annots = torch.ones((B, N, max_num_annots, 5)) * -1
    for b, camera_annotations in enumerate(annots_batch):
        if camera_annotations and len(camera_annotations) != N:
            raise ValueError("카메라 수와 객체 검출 annotation 수가 다릅니다.")
        for n, cam_annots in enumerate(camera_annotations):
            # resize_bbs historically prepends an image index. The loss expects
            # x1,y1,x2,y2,class, so strip only that documented sixth column.
            if cam_annots.shape[-1] == 6:
                cam_annots = cam_annots[:, 1:]
            elif cam_annots.shape[-1] != 5:
                raise ValueError("검출 annotation은 5개 또는 6개 열이어야 합니다.")
            num = cam_annots.shape[0]
            if num:
                padded_annots[b, n, :num, :] = cam_annots


    return (
        imgs, imgs_ori, das_GT, lane_GT, vehicle_GT, walker_GT, lidar_data,
        lidar_mask, post_rots, post_trans, extrinsic, intrinsic,
        rotation, translation, seg_GT, dept_GT, depth_mask,
        padded_annots
    )



def visualize_depth(img, max_depth=100.0):
    img = np.clip(img, 0, max_depth)
    img = (img / max_depth * 255).astype(np.uint8)
    img_color = cv2.applyColorMap(img, cv2.COLORMAP_PLASMA)
    return img_color
now = datetime.now()
formatted = now.strftime("%y-%m-%d-%H")
base_dir = str(configured_path("SDV_TRAIN_OUTPUT", f"Training_result/{formatted}"))
save_dirs = base_dir

resume_path = os.environ.get('SDV_TRAIN_RESUME')
if resume_path and not os.path.isfile(resume_path):
    raise ValueError('재개 체크포인트가 없습니다: ' + resume_path)
suffix = 1
if resume_path:
    if not os.path.isdir(save_dirs):
        raise ValueError('재개할 실험의 결과 경로가 없습니다.')
elif os.environ.get("SDV_TRAIN_OUTPUT"):
    if os.path.isdir(save_dirs) and os.listdir(save_dirs):
        raise ValueError("학습 결과 경로가 비어 있지 않습니다. 새 실험 경로를 지정하세요.")
else:
    while os.path.exists(save_dirs):
        save_dirs = f"{base_dir}-{suffix}"
        suffix += 1
os.makedirs(save_dirs, exist_ok=True)
epoch_records = []
telemetry_path = os.path.join(save_dirs, "events.jsonl")


# The selected queue configuration was validated before importing models.
if not resume_path:
    shutil.copy2(_args.config, f'{save_dirs}/train_config.yaml')
    with open(os.path.join(save_dirs, 'metric_registry.json'), 'w', encoding='utf-8') as registry_file:
        json.dump(metric_registry(train_config), registry_file, ensure_ascii=False, indent=2)
    shutil.copy2(configured_path('SDV_SENSOR_CONFIG', 'data/sensor_config.yaml'), f'{save_dirs}/sensor_config.yaml')


Epochs=train_config["Epochs"]

batch=train_config["batch"]

model=models(train_config)
if not torch.cuda.is_available():
    raise RuntimeError("현재 SDV 연구 모델은 CUDA GPU가 필요합니다. CUDA가 지원되는 Python 환경을 선택하세요.")
model.cuda()

optimizer_name=next(k for k,v in train_config['opt_checkboxes'].items() if v)
optimizer_class={'Adam':torch.optim.Adam,'AdamW':torch.optim.AdamW,'SGD':torch.optim.SGD}[optimizer_name]
opt = optimizer_class(model.parameters(), lr=train_config['LR'], weight_decay=1e-7)

step_ratio = 5 / 20
new_total_epochs = 50
step_size = max(1, int(step_ratio * Epochs))

sched = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=0.4)



train_set = bev_Dataset(True,train_config)
val_set = bev_Dataset(False,train_config)
if train_config['model_conf'].get('BEV') and train_set.bev_schema != val_set.bev_schema:
    raise ValueError('Train/validation BEV schema mismatch: legacy/schema-3/schema-4 data contracts require separate experiments')
if train_set.bev_schema in (3,4) and not resume_path:
    with open(os.path.join(save_dirs, 'bev_training_contract.json'), 'w', encoding='utf-8') as stream:
        json.dump({'schema_version':train_set.bev_schema, 'source':'provided_global_raster_and_simulator_actors' if train_set.bev_schema == 4 else 'simulator_ground_truth',
            'model_grid':{'x':[-20.,20.,.2],'y':[-20.,20.,.2]},
            'tensor_axes':'row=ego Y+, column=ego X+',
            'preview_axes':'forward-up/right-right',
            'camera_basis':'same-frame optical -> sensor world transform -> yaw-leveled ego',
            'loss_and_metrics':{'das':'validity','lane':'validity & lane_validity',
                'vehicle':'occupancy_validity','walker':'occupancy_validity'},
            'invalid_target':'NaN masked before BCE and metric counts',
            'height_auxiliary':'get_bev_aux; GT only, not model input',
            'model_heads':['das','lane','vehicle','walker']}, stream, indent=2)

train_loader = DataLoader(train_set, batch_size=batch,collate_fn=collater, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=batch, collate_fn=collater,shuffle=False, num_workers=0)
if not len(train_loader) or not len(val_loader):
    raise ValueError("학습/검증 데이터가 비어 있습니다. front_cam_rgb 표본과 분할 경로를 확인하세요.")
emit('training_preparing', phase='dataset_signature', output=save_dirs,
     message='재개 검증을 위한 실제 데이터셋 내용을 확인합니다.', _mirror_path=telemetry_path)
signature = experiment_signature(train_config,
    configured_path('SDV_SENSOR_CONFIG', 'data/sensor_config.yaml'), train_set.data_root, val_set.data_root,
    bev_schema=train_set.bev_schema)
resume_state = load_resume(resume_path, signature, save_dirs, model, opt, sched) if resume_path else None
start_epoch = resume_state['completed_epoch'] if resume_state else 0
if start_epoch >= Epochs:
    raise ValueError('모든 Epoch가 이미 완료된 체크포인트입니다.')
if resume_state:
    epoch_records = list(resume_state['epoch_records'])
emit("training_started", epochs=Epochs, steps_per_epoch=len(train_loader),
     train_samples=len(train_set), val_samples=len(val_set), output=save_dirs,
     resumed=bool(resume_state), start_epoch=start_epoch + 1, completed_epochs=start_epoch,
     global_step=resume_state['global_step'] if resume_state else 0,
     _mirror_path=telemetry_path)

from sdv.training.losses import build_losses
from sdv.training.loss_config import normalize_losses
loss_configuration=normalize_losses(train_config.get('losses'))
loss_modules=build_losses(loss_configuration).cuda()
loss_Simple,loss_dice,loss_silog,focal_Loss=(loss_modules[k] for k in ('BEV','Seg','Depth','od'))
with open(os.path.join(save_dirs,'loss_configuration.json'),'w',encoding='utf-8') as stream:json.dump(loss_configuration,stream,indent=2)
emit('training_loss_configuration',losses=loss_configuration,_mirror_path=telemetry_path)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    filename=os.path.join(f"{save_dirs}/results.log"),
    filemode='a' if resume_path else 'w',
    format='%(asctime)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(sys.stdout))


test_data = load_input_and_gt()

if test_data is not None:
    das_GT=test_data['das_GT']
    lane_GT=test_data['lane_GT']
    vehicle_GT=test_data['vehicle_GT']
    walker_GT=test_data['walker_GT']
    for i in range(walker_GT.shape[0]):
        save_dir = f"data/GT/{i}"
        os.makedirs(save_dir, exist_ok=True)
        das_gt_img = (das_GT[i, 0].cpu().numpy() * 255).astype(np.uint8)
        lane_gt_img = (lane_GT[i, 0].cpu().numpy() * 255).astype(np.uint8)
        vehicle_gt_img = (vehicle_GT[i, 0].cpu().numpy() * 255).astype(np.uint8)
        walker_gt_img = (walker_GT[i, 0].cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(save_dir, "DAS_gt.png"), das_gt_img)
        cv2.imwrite(os.path.join(save_dir, "Lane_gt.png"), lane_gt_img)
        cv2.imwrite(os.path.join(save_dir, "Vehicle_gt.png"), vehicle_gt_img)
        cv2.imwrite(os.path.join(save_dir, "Walker_gt.png"), walker_gt_img)

        top = np.hstack((das_gt_img, lane_gt_img))
        bottom = np.hstack((vehicle_gt_img, walker_gt_img))
        combined = np.vstack((top, bottom))
        cv2.imwrite(os.path.join(save_dir, "GT_combined.png"), combined)


        h, w = das_gt_img.shape
        merged_img = np.zeros((h, w, 3), dtype=np.uint8)

        merged_img[das_gt_img > 128] =     (60, 60,60)
        merged_img[lane_gt_img > 128] =    (0, 255, 0)
        merged_img[vehicle_gt_img > 128] = (255, 0, 0)
        merged_img[walker_gt_img > 128] =  (0, 0, 255)
        rect_w, rect_h = 12, 25
        center_x, center_y = w // 2, h // 2
        top_left = (center_x - rect_w // 2, center_y - rect_h // 2)
        bottom_right = (center_x + rect_w // 2, center_y + rect_h // 2)

        cv2.rectangle(merged_img, top_left, bottom_right, (255, 255, 255), thickness=-1)

        cv2.imwrite(os.path.join(save_dir, "GT.png"), merged_img)

counter=0

train_losses = []

train_bev_losses = []
train_DAS_losses = []
train_Lane_losses = []
train_Vehicle_losses = []
train_Walker_losses = []

train_seg_losses = []
train_depth_losses = []
train_OD_losses=[]

train_vehicle_ious = []
train_walker_ious = []
train_lane_ious = []
train_das_ious = []
train_m_bev_ious = []

train_abs_rel = []
train_seg_iou = []
train_OD_iou=[]
val_losses = []

val_bev_losses = []
val_DAS_losses = []
val_Lane_losses = []
val_Vehicle_losses = []
val_Walker_losses = []
val_OD_losses=[]
val_seg_losses = []
val_depth_losses = []

val_vehicle_ious = []
val_walker_ious = []
val_lane_ious = []
val_das_ious = []
val_m_bev_ious = []

val_abs_rel = []
val_seg_iou = []
val_OD_iou = []


best_val_miou = -float('inf')
best_val_loss = float('inf')

od_save=True
last_idx = len(train_loader) - 1
HISTORY_NAMES = (
    'train_losses', 'train_bev_losses', 'train_DAS_losses', 'train_Lane_losses',
    'train_Vehicle_losses', 'train_Walker_losses', 'train_seg_losses', 'train_depth_losses',
    'train_OD_losses', 'train_vehicle_ious', 'train_walker_ious', 'train_lane_ious',
    'train_das_ious', 'train_m_bev_ious', 'train_abs_rel', 'train_seg_iou', 'train_OD_iou',
    'val_losses', 'val_bev_losses', 'val_DAS_losses', 'val_Lane_losses', 'val_Vehicle_losses',
    'val_Walker_losses', 'val_OD_losses', 'val_seg_losses', 'val_depth_losses',
    'val_vehicle_ious', 'val_walker_ious', 'val_lane_ious', 'val_das_ious',
    'val_m_bev_ious', 'val_abs_rel', 'val_seg_iou', 'val_OD_iou')
if resume_state:
    for history_name in HISTORY_NAMES:
        restored = resume_state['histories'].get(history_name)
        if not isinstance(restored, list) or len(restored) != start_epoch:
            raise ValueError('재개할 학습 이력이 완전하지 않습니다: ' + history_name)
        globals()[history_name] = restored
    counter = resume_state['global_step']
    best_val_loss = resume_state['best_val_loss']

monitor = train_config.get('monitor', {})
train_preview_every = int(monitor.get('train_preview_every', 100))
val_preview_every = int(monitor.get('val_preview_every', 1))
prediction_recorder = PredictionRecorder(save_dirs, ROOT, val_set, resume_epoch=start_epoch if resume_state else None, resume_global_step=counter if resume_state else None)
resume_checkpoint = os.path.join(save_dirs, 'resume_checkpoint.pth')
checkpoint = os.path.join(save_dirs, 'best_loss_model.pth') if train_config.get('save_checkboxes', {}).get('Best_Loss', True) else None
last_path = os.path.join(save_dirs, 'last_model.pth') if train_config.get('save_checkboxes', {}).get('Last_Epoch', True) else None
if resume_state:
    # Model initialization and preview metadata may consume RNG. Restore only
    # immediately before the next DataLoader iterator is constructed.
    restore_rng(resume_state['rng'])
    partial_csv = os.path.join(save_dirs, 'train_metrics.csv')
    pd.DataFrame(epoch_records).to_csv(partial_csv + '.tmp', index=False)
    os.replace(partial_csv + '.tmp', partial_csv)
    emit('training_resumed', epoch=start_epoch + 1, completed_epochs=start_epoch,
         global_step=counter, resume_checkpoint=resume_checkpoint, _mirror_path=telemetry_path)
for epoch in range(start_epoch, Epochs):
    epoch_started = time.monotonic()
    train_task_metrics = EpochMetrics(train_config['model_conf'], task_settings)
    val_task_metrics = EpochMetrics(train_config['model_conf'], task_settings)
    epoch_train_loss = 0

    epoch_train_bev_loss = 0
    epoch_train_DAS_loss = 0
    epoch_train_OD_loss=0
    epoch_train_Lane_loss = 0
    epoch_train_Vehicle_loss = 0
    epoch_train_walker_loss = 0

    epoch_train_seg_loss = 0
    epoch_train_depth_loss = 0


    epoch_train_vehicle_iou = 0
    epoch_train_walker_iou = 0
    epoch_train_lane_iou = 0
    epoch_train_das_iou = 0

    epoch_train_seg_iou = 0
    epoch_train_abs_rel = 0
    epoch_train_OD_iou=0
    num_train_batches = 0
    model.train()
    for batchi, (imgs,imgs_ori,das_GT, lane_GT, vehicle_GT, walker_GT,lidar_data,
                 lidar_mask,  post_rots,post_trans,extrinsic, intrinsic,
                 rotation,translation,
                 seg_GT,dept_GT,depth_mask,
                 annotations

                 ) in enumerate(train_loader):
        torch.cuda.empty_cache()

        opt.zero_grad()
        loss=0
        loss_bev=0

        loss_seg=0
        loss_depth=0
        loss_od=0

        loss_DAS=0
        loss_Lane=0
        loss_Vehicle=0
        loss_walker=0



        DAS,Lane,Vehicle,walker,seg,depth,classification, regression =  model(imgs, translation, rotation, intrinsic,
                                                post_trans, post_rots,
                                                lidar_data, lidar_mask,
                                                )
        metric_anchors = model.anchors(imgs.flatten(0, 1).to(classification.device)) if train_config['model_conf']['od'] else None
        train_task_metrics.update(outputs=(DAS, Lane, Vehicle, walker, seg, depth, classification, regression),
            batch={'das_GT': das_GT, 'lane_GT': lane_GT, 'vehicle_GT': vehicle_GT,
                   'walker_GT': walker_GT, 'seg_GT': seg_GT, 'dept_GT': dept_GT,
                   'depth_mask': depth_mask, 'annotations': annotations, 'imgs': imgs},
            anchors=metric_anchors)


        if train_config['model_conf']['BEV']:
            das_GT = das_GT.cuda().float()
            lane_GT = lane_GT.cuda().float()
            vehicle_GT = vehicle_GT.cuda().float()
            walker_GT = walker_GT.cuda().float()

            loss_DAS = loss_Simple(DAS, das_GT) if 'das' in bev_layers else DAS.sum() * 0.0
            loss_Lane = loss_Simple(Lane, lane_GT) if 'lane' in bev_layers else Lane.sum() * 0.0
            loss_Vehicle = loss_Simple(Vehicle, vehicle_GT) if 'vehicle' in bev_layers else Vehicle.sum() * 0.0
            loss_walker = loss_Simple(walker, walker_GT) if 'walker' in bev_layers else walker.sum() * 0.0
            loss_bev = loss_walker+loss_DAS+loss_Lane+loss_Vehicle
            _, _, DAS_iou = get_iou(DAS, das_GT) if 'das' in bev_layers else (0, 0, 0.0)
            _, _, Lane_iou = get_iou(Lane, lane_GT) if 'lane' in bev_layers else (0, 0, 0.0)
            _, _, Vehicle_iou = get_iou(Vehicle, vehicle_GT) if 'vehicle' in bev_layers else (0, 0, 0.0)
            _, _, walker_iou = get_iou(walker, walker_GT) if 'walker' in bev_layers else (0, 0, 0.0)
            epoch_train_vehicle_iou += Vehicle_iou
            epoch_train_walker_iou += walker_iou
            epoch_train_lane_iou += Lane_iou
            epoch_train_das_iou += DAS_iou
        else:
            DAS_iou=None
            Lane_iou=None
            walker_iou=None
            Vehicle_iou=None



        if train_config['model_conf']['Seg']:
            seg_GT = seg_GT.cuda().float()
            loss_seg = loss_dice(seg, seg_GT)
            _, _, seg_iou = get_iou(seg, seg_GT)
            epoch_train_seg_iou += seg_iou
        else:
            seg_iou=None

        if train_config['model_conf']['Depth']:
            dept_GT = dept_GT.cuda().float()
            depth_mask=depth_mask.cuda()
            loss_depth = loss_silog(depth, dept_GT,depth_mask)
            abs_rel=compute_abs_rel(depth, dept_GT,depth_mask)
            epoch_train_abs_rel += abs_rel
        else:
            abs_rel=None

        if train_config['model_conf']['od']:
            B,N,M,ON=annotations.shape
            annotations=annotations.cuda().view(B*N,M,ON)
            B,N,C,W,H=imgs.shape
            imgs=imgs.view(B*N, C, W,H)
            anchors=metric_anchors
            loss_od_class,loss_od_reg,miou,out_predict = focal_Loss(classification, regression, anchors, annotations)
            loss_od=loss_od_class.mean()+loss_od_reg.mean()
            epoch_train_OD_iou+=miou
        else:
            miou=None

        loss=loss_bev+loss_depth+loss_seg+loss_od
        epoch_train_loss += loss.item()

        epoch_train_bev_loss     += loss_bev.item()     if isinstance(loss_bev, torch.Tensor) else 0.0
        epoch_train_depth_loss   += loss_depth.item()   if isinstance(loss_depth, torch.Tensor) else 0.0
        epoch_train_seg_loss     += loss_seg.item()     if isinstance(loss_seg, torch.Tensor) else 0.0
        epoch_train_walker_loss  += loss_walker.item()  if isinstance(loss_walker, torch.Tensor) else 0.0
        epoch_train_Vehicle_loss += loss_Vehicle.item() if isinstance(loss_Vehicle, torch.Tensor) else 0.0
        epoch_train_Lane_loss    += loss_Lane.item()    if isinstance(loss_Lane, torch.Tensor) else 0.0
        epoch_train_DAS_loss     += loss_DAS.item()     if isinstance(loss_DAS, torch.Tensor) else 0.0
        epoch_train_OD_loss     += loss_od.item()     if isinstance(loss_od, torch.Tensor) else 0.0


        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        opt.step()
        counter += 1
        save_idx = counter


        num_train_batches += 1
        if __import__('sdv.training.preview_config',fromlist=['should_capture']).should_capture(monitor,batchi + 1,counter):
            preview = prediction_recorder.capture(model, train_set, collater,
                split='train', epoch=epoch + 1, global_step=counter, epoch_step=batchi + 1, steps_per_epoch=len(train_loader))
            emit('training_preview', _mirror_path=telemetry_path, output=save_dirs, **preview)
        if batchi == 0 or counter % 10 == 0 or batchi == last_idx:
            emit("training_step", epoch=epoch + 1, epochs=Epochs, step=batchi + 1,
                 steps=len(train_loader), global_step=counter, loss=loss.item(),
                 progress=(epoch + .8 * (batchi + 1) / len(train_loader)) / Epochs,
                 metrics={"train_batch_loss": loss.item()}, _mirror_path=telemetry_path)
        if counter % 20 == 0:

            log_msg = f"TRAIN[{epoch}] [{counter}]  [{batchi:>4d}/{last_idx}]    " \
                    f"Loss: {loss.item():>7.3f}    "

            if loss_bev != 0:
                log_msg += f'loss_bev {loss_bev:>7.3f}    '
            if loss_DAS != 0:
                log_msg += f'loss_DAS {loss_DAS:>7.3f}    '
            if loss_Lane != 0:
                log_msg += f'loss_Lane {loss_Lane:>7.3f}    '
            if loss_Vehicle != 0:
                log_msg += f'loss_Vehicle {loss_Vehicle:>7.3f}    '
            if loss_walker != 0:
                log_msg += f'loss_walker {loss_walker:>7.3f}    '
            if loss_seg != 0:
                log_msg += f'loss_Seg {loss_seg:>7.3f}    '
            if loss_depth != 0:
                log_msg += f'loss_Depth {loss_depth:>7.3f}    '
            if loss_od != 0:
                log_msg += f'loss_OD_C {loss_od.mean():>7.3f}    '
            if abs_rel is not None:
                log_msg += f'abs_rel {abs_rel:>7.4f}    '
            if seg_iou is not None and task_settings['segmentation']['mode'] == 'legacy':
                log_msg += f'seg_iou {seg_iou:>7.3f}    '
            if Vehicle_iou is not None and 'vehicle' in bev_layers:
                log_msg += f'vehicle_iou {Vehicle_iou:>7.3f}    '
            if walker_iou is not None and 'walker' in bev_layers:
                log_msg += f'walker_iou {walker_iou:>7.3f}    '
            if Lane_iou is not None and 'lane' in bev_layers:
                log_msg += f'Lane_iou {Lane_iou:>7.3f}    '
            if DAS_iou is not None and 'das' in bev_layers:
                log_msg += f'DAS_iou {DAS_iou:>7.3f}    '
            if miou is not None:
                log_msg += f'miou {miou.cpu().item():>7.3f}    '
            if all(v is not None for v in [Vehicle_iou, walker_iou, Lane_iou, DAS_iou]):
                bev_m_iou = (Vehicle_iou + Lane_iou + DAS_iou + walker_iou) / 4
                log_msg += f'bev_m_iou {bev_m_iou:>7.3f}    '
            logger.info(log_msg)



    avg_train_loss = epoch_train_loss / num_train_batches

    avg_train_bev_loss = epoch_train_bev_loss / num_train_batches
    avg_train_depth_loss = epoch_train_depth_loss / num_train_batches
    avg_train_seg_loss = epoch_train_seg_loss / num_train_batches

    avg_train_walker_loss = epoch_train_walker_loss / num_train_batches
    avg_train_Vehicle_loss = epoch_train_Vehicle_loss / num_train_batches
    avg_train_Lane_loss = epoch_train_Lane_loss / num_train_batches
    avg_train_DAS_loss = epoch_train_DAS_loss / num_train_batches
    avg_train_OD_loss = epoch_train_OD_loss / num_train_batches

    avg_train_vehicle_iou = epoch_train_vehicle_iou / num_train_batches
    avg_train_walker_iou = epoch_train_walker_iou / num_train_batches
    avg_train_lane_iou = epoch_train_lane_iou / num_train_batches
    avg_train_das_iou = epoch_train_das_iou / num_train_batches

    avg_train_abs_rel = epoch_train_abs_rel / num_train_batches
    avg_train_seg_iou = epoch_train_seg_iou / num_train_batches
    avg_train_OD_iou = float(epoch_train_OD_iou / num_train_batches)

    avg_train_miou = sum(globals()['avg_train_' + layer + '_iou'] for layer in bev_layers) / max(1, len(bev_layers))
    if train_set.bev_schema in (3,4):
        actual_bev_metrics = train_task_metrics.compute()
        for layer in ('das', 'lane', 'vehicle', 'walker'):
            globals()['avg_train_' + layer + '_iou'] = actual_bev_metrics.get('bev_' + layer + '_iou', float('nan'))
        avg_train_miou = actual_bev_metrics.get('bev_miou', float('nan'))

    train_losses.append(avg_train_loss)

    train_bev_losses.append(avg_train_bev_loss)
    train_depth_losses.append(avg_train_depth_loss)
    train_seg_losses.append(avg_train_seg_loss)

    train_Walker_losses.append(avg_train_walker_loss)
    train_Vehicle_losses.append(avg_train_Vehicle_loss)
    train_Lane_losses.append(avg_train_Lane_loss)
    train_DAS_losses.append(avg_train_DAS_loss)
    train_OD_losses.append(avg_train_OD_loss)

    train_vehicle_ious.append(avg_train_vehicle_iou)
    train_walker_ious.append(avg_train_walker_iou)
    train_lane_ious.append(avg_train_lane_iou)
    train_das_ious.append(avg_train_das_iou)
    train_m_bev_ious.append(avg_train_miou)

    train_abs_rel.append(avg_train_abs_rel)
    train_seg_iou.append(avg_train_seg_iou)
    train_OD_iou.append(avg_train_OD_iou)


    logger.info("VAL")
    model.eval()
    epoch_val_loss = 0

    epoch_val_bev_loss = 0
    epoch_val_DAS_loss = 0
    epoch_val_OD_loss=0
    epoch_val_Lane_loss = 0
    epoch_val_Vehicle_loss = 0
    epoch_val_Walker_loss = 0

    epoch_val_seg_loss = 0
    epoch_val_depth_loss = 0
    epoch_val_abs_rel=0
    num_val_batches=0
    epoch_val_vehicle_iou = 0
    epoch_val_walker_iou = 0
    epoch_val_lane_iou = 0
    epoch_val_das_iou = 0
    epoch_val_OD=0
    epoch_val_seg_iou = 0
    with torch.no_grad():
        for batchi, (imgs,imgs_ori,das_GT, lane_GT, vehicle_GT, walker_GT,lidar_data,
                 lidar_mask,  post_rots,post_trans,extrinsic, intrinsic,
                 rotation,translation,
                 seg_GT,dept_GT,depth_mask,
                 annotations

                 ) in enumerate(val_loader):
            loss=0
            loss_bev=0

            loss_seg=0
            loss_depth=0
            loss_od=0

            loss_DAS=0
            loss_Lane=0
            loss_Vehicle=0
            loss_walker=0

            DAS,Lane,Vehicle,walker,seg,depth,classification, regression =  model(imgs.cuda(), translation.cuda(), rotation.cuda(), intrinsic.cuda(),
                                                    post_trans.cuda(), post_rots.cuda(),
                                                    lidar_data.cuda() if lidar_data is not None else None,
                                                    lidar_mask.cuda() if lidar_mask is not None else None,
                                                    )
            metric_anchors = model.anchors(imgs.flatten(0, 1).to(classification.device)) if train_config['model_conf']['od'] else None
            val_task_metrics.update(outputs=(DAS, Lane, Vehicle, walker, seg, depth, classification, regression),
                batch={'das_GT': das_GT, 'lane_GT': lane_GT, 'vehicle_GT': vehicle_GT,
                       'walker_GT': walker_GT, 'seg_GT': seg_GT, 'dept_GT': dept_GT,
                       'depth_mask': depth_mask, 'annotations': annotations, 'imgs': imgs},
                anchors=metric_anchors)


            if train_config['model_conf']['BEV']:
                das_GT = das_GT.cuda().float()
                lane_GT = lane_GT.cuda().float()
                vehicle_GT = vehicle_GT.cuda().float()
                walker_GT = walker_GT.cuda().float()

                loss_DAS = loss_Simple(DAS, das_GT) if 'das' in bev_layers else DAS.sum() * 0.0
                loss_Lane = loss_Simple(Lane, lane_GT) if 'lane' in bev_layers else Lane.sum() * 0.0
                loss_Vehicle = loss_Simple(Vehicle, vehicle_GT) if 'vehicle' in bev_layers else Vehicle.sum() * 0.0
                loss_walker = loss_Simple(walker, walker_GT) if 'walker' in bev_layers else walker.sum() * 0.0
                loss_bev = loss_walker+loss_DAS+loss_Lane+loss_Vehicle
                _, _, DAS_iou = get_iou(DAS, das_GT) if 'das' in bev_layers else (0, 0, 0.0)
                _, _, Lane_iou = get_iou(Lane, lane_GT) if 'lane' in bev_layers else (0, 0, 0.0)
                _, _, Vehicle_iou = get_iou(Vehicle, vehicle_GT) if 'vehicle' in bev_layers else (0, 0, 0.0)
                _, _, walker_iou = get_iou(walker, walker_GT) if 'walker' in bev_layers else (0, 0, 0.0)
                epoch_val_vehicle_iou += Vehicle_iou
                epoch_val_walker_iou += walker_iou
                epoch_val_lane_iou += Lane_iou
                epoch_val_das_iou += DAS_iou
            else:
                DAS_iou=0
                Lane_iou=0
                walker_iou=0
                Vehicle_iou=0


            if train_config['model_conf']['Seg']:
                seg_GT = seg_GT.cuda().float()
                loss_seg = loss_dice(seg, seg_GT)
                _, _, seg_iou = get_iou(seg, seg_GT)
                epoch_val_seg_iou += seg_iou
            else:
                seg_iou=0


            if train_config['model_conf']['Depth']:
                dept_GT = dept_GT.cuda().float()
                depth_mask=depth_mask.cuda()
                loss_depth = loss_silog(depth, dept_GT,depth_mask)
                abs_rel=compute_abs_rel(depth, dept_GT,depth_mask)
                epoch_val_abs_rel += abs_rel
            else:
                abs_rel=0


            if train_config['model_conf']['od']:
                B,N,M,ON=annotations.shape
                annotations=annotations.cuda().view(B*N,M,ON)
                B,N,C,W,H=imgs.shape
                imgs=imgs.view(B*N, C, W,H)
                anchors=metric_anchors
                loss_od_class,loss_od_reg,miou,out_predict = focal_Loss(classification, regression, anchors, annotations)
                loss_od=loss_od_class.mean()+loss_od_reg.mean()

                epoch_val_OD+=miou
            else:
                miou=0

            loss=loss_bev+loss_depth+loss_seg+loss_od


            epoch_val_loss += loss.item()
            epoch_val_bev_loss     += loss_bev.item()     if isinstance(loss_bev, torch.Tensor) else 0.0
            epoch_val_depth_loss   += loss_depth.item()   if isinstance(loss_depth, torch.Tensor) else 0.0
            epoch_val_seg_loss     += loss_seg.item()     if isinstance(loss_seg, torch.Tensor) else 0.0
            epoch_val_Walker_loss  += loss_walker.item()  if isinstance(loss_walker, torch.Tensor) else 0.0
            epoch_val_Vehicle_loss += loss_Vehicle.item() if isinstance(loss_Vehicle, torch.Tensor) else 0.0
            epoch_val_Lane_loss    += loss_Lane.item()    if isinstance(loss_Lane, torch.Tensor) else 0.0
            epoch_val_DAS_loss     += loss_DAS.item()     if isinstance(loss_DAS, torch.Tensor) else 0.0
            epoch_val_OD_loss     += loss_od.item()     if isinstance(loss_od, torch.Tensor) else 0.0


            num_val_batches += 1
            if batchi == 0 or (batchi + 1) % 10 == 0 or batchi + 1 == len(val_loader):
                emit('training_validation', epoch=epoch + 1, epochs=Epochs,
                     step=batchi + 1, steps=len(val_loader), global_step=counter,
                     progress=(epoch + .8 + .2 * (batchi + 1) / len(val_loader)) / Epochs,
                     _mirror_path=telemetry_path)

    avg_val_loss = epoch_val_loss / num_val_batches

    avg_val_bev_loss = epoch_val_bev_loss / num_val_batches
    avg_val_depth_loss = epoch_val_depth_loss / num_val_batches
    avg_val_seg_loss = epoch_val_seg_loss / num_val_batches

    avg_val_Walker_loss = epoch_val_Walker_loss / num_val_batches
    avg_val_Vehicle_loss = epoch_val_Vehicle_loss / num_val_batches
    avg_val_Lane_loss = epoch_val_Lane_loss / num_val_batches
    avg_val_DAS_loss = epoch_val_DAS_loss / num_val_batches

    avg_val_OD_loss = epoch_val_OD_loss / num_val_batches

    avg_val_vehicle_iou = epoch_val_vehicle_iou / num_val_batches
    avg_val_walker_iou = epoch_val_walker_iou / num_val_batches
    avg_val_lane_iou = epoch_val_lane_iou / num_val_batches
    avg_val_das_iou = epoch_val_das_iou / num_val_batches

    avg_val_seg_iou = epoch_val_seg_iou / num_val_batches
    avg_val_abs_rel = epoch_val_abs_rel / num_val_batches

    avg_val_OD_iou = float(epoch_val_OD / num_val_batches)

    avg_val_bev_miou = sum(globals()['avg_val_' + layer + '_iou'] for layer in bev_layers) / max(1, len(bev_layers))
    if val_set.bev_schema in (3,4):
        actual_bev_metrics = val_task_metrics.compute()
        for layer in ('das', 'lane', 'vehicle', 'walker'):
            globals()['avg_val_' + layer + '_iou'] = actual_bev_metrics.get('bev_' + layer + '_iou', float('nan'))
        avg_val_bev_miou = actual_bev_metrics.get('bev_miou', float('nan'))
    if not math.isfinite(avg_train_loss) or not math.isfinite(avg_val_loss):
        raise RuntimeError('학습/검증 Loss가 유한하지 않습니다. 입력 데이터와 Depth 유효 마스크를 확인하세요.')


    log_msg = f"VAL[{epoch}]  Loss: {avg_val_loss:>7.3f}    "
    if loss_bev != 0:
        log_msg += f'loss_bev {loss_bev:>7.3f}    '
    if avg_val_DAS_loss != 0:
        log_msg += f'loss_DAS {avg_val_DAS_loss:>7.3f}    '
    if avg_val_Lane_loss != 0:
        log_msg += f'loss_Lane {avg_val_Lane_loss:>7.3f}    '
    if avg_val_Vehicle_loss != 0:
        log_msg += f'loss_Vehicle {avg_val_Vehicle_loss:>7.3f}    '
    if avg_val_Walker_loss != 0:
        log_msg += f'loss_walker {avg_val_Walker_loss:>7.3f}    '
    if loss_seg != 0:
        log_msg += f'loss_Seg {loss_seg:>7.3f}    '
    if loss_depth != 0:
        log_msg += f'loss_Depth {loss_depth:>7.3f}    '
    if loss_od != 0:
        log_msg += f'loss_OD {loss_od:>7.3f}    '

    if avg_val_abs_rel != 0:
        log_msg += f'abs_rel {avg_val_abs_rel:>7.4f}    '
    if avg_val_seg_iou != 0 and task_settings['segmentation']['mode'] == 'legacy':
        log_msg += f'seg_iou {avg_val_seg_iou:>7.3f}    '
    if avg_val_vehicle_iou != 0 and 'vehicle' in bev_layers:
        log_msg += f'vehicle_iou {avg_val_vehicle_iou:>7.3f}    '
    if avg_val_walker_iou != 0 and 'walker' in bev_layers:
        log_msg += f'walker_iou {avg_val_walker_iou:>7.3f}    '
    if avg_val_lane_iou != 0 and 'lane' in bev_layers:
        log_msg += f'Lane_iou {avg_val_lane_iou:>7.3f}    '
    if avg_val_das_iou != 0 and 'das' in bev_layers:
        log_msg += f'DAS_iou {avg_val_das_iou:>7.3f}    '
    if avg_val_OD_iou != 0:
        log_msg += f'OD_iou {avg_val_OD_iou:>7.3f}    '
    if all(v is not None for v in [Vehicle_iou, walker_iou, Lane_iou, DAS_iou]):
        log_msg += f'bev_m_iou {avg_val_bev_miou:>7.3f}    '
    logger.info(log_msg)


    val_losses.append(avg_val_loss)

    val_bev_losses.append(avg_val_bev_loss)
    val_depth_losses.append(avg_val_depth_loss)
    val_seg_losses.append(avg_val_seg_loss)

    val_Walker_losses.append(avg_val_Walker_loss)
    val_Vehicle_losses.append(avg_val_Vehicle_loss)
    val_Lane_losses.append(avg_val_Lane_loss)
    val_DAS_losses.append(avg_val_DAS_loss)

    val_OD_losses.append(avg_val_OD_loss)

    val_vehicle_ious.append(avg_val_vehicle_iou)
    val_walker_ious.append(avg_val_walker_iou)
    val_lane_ious.append(avg_val_lane_iou)
    val_das_ious.append(avg_val_das_iou)
    val_m_bev_ious.append(avg_val_bev_miou)

    val_abs_rel.append(avg_val_abs_rel)
    val_seg_iou.append(avg_val_seg_iou)
    val_OD_iou.append(avg_val_OD_iou)


    # Preserve the checkpoint interface, but honor "best" rather than silently
    # replacing it with a worse final epoch. Atomic replacement avoids partial downloads.
    save_options = train_config.get('save_checkboxes', {})
    last_path = None
    if save_options.get('Last_Epoch', True):
        last_path = f'{save_dirs}/last_model.pth'
        torch.save(model.state_dict(), last_path + '.tmp')
        os.replace(last_path + '.tmp', last_path)
    checkpoint = f'{save_dirs}/best_loss_model.pth' if save_options.get('Best_Loss', True) else None
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        if checkpoint is not None:
            torch.save(model.state_dict(), checkpoint + '.tmp')
            os.replace(checkpoint + '.tmp', checkpoint)

    measured = {"train_loss": avg_train_loss, "val_loss": avg_val_loss}
    if train_config['model_conf']['BEV']:
        for layer in sorted(bev_layers):
            for split in ('train', 'val'):
                value = locals()['avg_' + split + '_' + layer + '_iou']
                if math.isfinite(value):
                    measured[split + '_' + layer + '_iou'] = value
        measured.update(train_bev_loss=avg_train_bev_loss, val_bev_loss=avg_val_bev_loss)
    if train_config['model_conf']['Depth']:
        measured.update(train_depth_loss=avg_train_depth_loss, val_depth_loss=avg_val_depth_loss,
                        train_abs_rel=avg_train_abs_rel, val_abs_rel=avg_val_abs_rel)
    if train_config['model_conf']['Seg']:
        measured.update(train_seg_loss=avg_train_seg_loss, val_seg_loss=avg_val_seg_loss)
        if task_settings['segmentation']['mode'] == 'legacy':
            measured.update(train_seg_iou=avg_train_seg_iou, val_seg_iou=avg_val_seg_iou)
    if train_config['model_conf']['od']:
        measured.update(train_OD_loss=avg_train_OD_loss, val_OD_loss=avg_val_OD_loss,
                        train_OD_iou=avg_train_OD_iou, val_OD_iou=avg_val_OD_iou)
    # Task accumulators compute standard metrics across the whole epoch;
    # the training losses and optimization themselves are unchanged.
    measured.update(train_task_metrics.compute(prefix='train'))
    measured.update(val_task_metrics.compute(prefix='val'))
    if (epoch + 1) % val_preview_every == 0:
        preview = prediction_recorder.capture(model, val_set, collater,
            split='val', epoch=epoch + 1, global_step=counter, epoch_step=len(train_loader), steps_per_epoch=len(train_loader))
        emit('training_preview', _mirror_path=telemetry_path, output=save_dirs, **preview)
    epoch_duration_sec = time.monotonic() - epoch_started
    measured['learning_rate'] = opt.param_groups[0]['lr']
    epoch_records.append({"epoch": epoch + 1, "epoch_duration_sec": epoch_duration_sec, **measured})
    partial_csv = f'{save_dirs}/train_metrics.csv'
    pd.DataFrame(epoch_records).to_csv(partial_csv + '.tmp', index=False)
    os.replace(partial_csv + '.tmp', partial_csv)
    sched.step()
    save_resume(resume_checkpoint, model, opt, sched, completed_epoch=epoch + 1,
        global_step=counter, best_val_loss=best_val_loss,
        histories={name: globals()[name] for name in HISTORY_NAMES},
        epoch_records=epoch_records, signature=signature, output=save_dirs, config=train_config)
    # Include real checkpoint serialization time in the UI's active-time ETA.
    epoch_duration_sec = time.monotonic() - epoch_started
    epoch_records[-1]['epoch_duration_sec'] = epoch_duration_sec
    pd.DataFrame(epoch_records).to_csv(partial_csv + '.tmp', index=False)
    os.replace(partial_csv + '.tmp', partial_csv)
    emit("training_epoch", epoch=epoch + 1, epochs=Epochs,
         progress=(epoch + 1) / Epochs, metrics=measured, checkpoint=checkpoint,
         last_checkpoint=last_path, output=save_dirs, epoch_duration_sec=epoch_duration_sec,
         duration_seconds=epoch_duration_sec, global_step=counter, _mirror_path=telemetry_path)
    emit('training_checkpoint', epoch=epoch + 1, next_epoch=epoch + 2,
         global_step=counter, resume_checkpoint=resume_checkpoint, _mirror_path=telemetry_path)
    if epoch + 1 < Epochs and pause_requested():
        emit('training_paused', epoch=epoch + 1, next_epoch=epoch + 2,
             global_step=counter, resume_checkpoint=resume_checkpoint, output=save_dirs,
             _mirror_path=telemetry_path)
        raise SystemExit(PAUSE_EXIT_CODE)


epochs = range(1, Epochs + 1)

# 1. Loss 그래프
plot_metric(train_losses, val_losses, 'Loss', 'Loss over Epochs', 'loss')

if train_config['model_conf']['BEV']:
    for layer, title in (('das', 'DAS'), ('lane', 'Lane'), ('vehicle', 'Vehicle'), ('walker', 'Walker')):
        if layer not in bev_layers: continue
        plot_metric(globals()['train_' + layer + '_ious'], globals()['val_' + layer + '_ious'],
                    title + ' IoU', title + ' IoU over Epochs', layer + '_iou')
        plot_metric(globals()['train_' + title + '_losses'], globals()['val_' + title + '_losses'],
                    title + ' Loss', title + ' Loss over Epochs', layer + '_loss')
    plot_metric(train_m_bev_ious, val_m_bev_ious, 'Mean IoU', 'Mean IoU over Epochs', 'mean_iou')

if train_config['model_conf']['Depth']:
    plot_metric(train_depth_losses, val_depth_losses, 'Depth Loss', 'Depth Loss over Epochs', 'depth_loss')
    plot_metric(train_abs_rel, val_abs_rel, 'Abs Rel', 'Abs Rel over Epochs', 'abs_rel')
if train_config['model_conf']['Seg']:
    plot_metric(train_seg_losses, val_seg_losses, 'Segmentation Loss', 'Segmentation Loss over Epochs', 'seg_loss')
    plot_metric([row.get('train_seg_miou', float('nan')) for row in epoch_records],
                [row.get('val_seg_miou', float('nan')) for row in epoch_records], 'mIoU', 'Segmentation mIoU over Epochs', 'seg_miou')
if train_config['model_conf']['od']:
    plot_metric(train_OD_iou, val_OD_iou, 'Object Detection IoU', 'Object Detection IoU over Epochs', 'od_iou')
    plot_metric(train_OD_losses, val_OD_losses, 'Object Detection Loss', 'Object Detection Loss over Epochs', 'OD_loss')

# CSV 저장
results = pd.DataFrame({
    'epoch': list(epochs),
    'train_loss': train_losses,
    'val_loss': val_losses,
    'train_vehicle_iou': train_vehicle_ious,
    'val_vehicle_iou': val_vehicle_ious,
    'train_walker_iou': train_walker_ious,
    'val_walker_iou': val_walker_ious,
    'train_lane_iou': train_lane_ious,
    'val_lane_iou': val_lane_ious,
    'train_das_iou': train_das_ious,
    'val_das_iou': val_das_ious,
    'train_m_bev_iou': train_m_bev_ious,
    'val_m_bev_iou': val_m_bev_ious,
    'train_OD_iou': train_OD_iou,
    'val_OD_iou': val_OD_iou,
    'train_abs_rel': train_abs_rel,
    'val_abs_rel': val_abs_rel,
    'train_seg_iou': train_seg_iou,
    'val_seg_iou': val_seg_iou,

    'train_depth_loss': train_depth_losses,
    'val_depth_loss': val_depth_losses,
    'train_seg_loss': train_seg_losses,
    'val_seg_loss': val_seg_losses,
    'train_DAS_loss': train_DAS_losses,
    'val_DAS_loss': val_DAS_losses,
    'train_Walker_loss': train_Walker_losses,
    'val_Walker_loss': val_Walker_losses,
    'train_Vehicle_loss': train_Vehicle_losses,
    'val_Vehicle_loss': val_Vehicle_losses,
    'train_Lane_loss': train_Lane_losses,
    'val_Lane_loss': val_Lane_losses,
    'train_OD_losses': train_OD_losses,
    'val_OD_losses': val_OD_losses,

})
# Keep the full historical schema for downstream legacy tooling; the web reads
# active-task epoch metrics only from the streaming CSV/events.
results.to_csv(f'{save_dirs}/legacy_train_metrics.csv', index=False)
emit('training_completed', epochs=Epochs, progress=1.0, output=save_dirs,
     checkpoint=checkpoint, last_checkpoint=last_path, _mirror_path=telemetry_path)
