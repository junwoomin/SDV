import yaml
import sys
import os

from train_model.data import bev_Dataset
from train_model.model import od_models
import torch
from torch.utils.data import DataLoader
import logging
import torch.nn.functional as F
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
from datetime import datetime
import glob
from train_model.loss import *
from utils import *

class_names = ['vehicles','pedestrians']


now = datetime.now()
formatted = now.strftime("%y-%m-%d-%H")
base_dir = f"Training_result/{formatted}"
save_dirs = base_dir

suffix = 1
while os.path.exists(save_dirs):
    save_dirs = f"{base_dir}-{suffix}"
    suffix += 1

os.makedirs(save_dirs)


with open('data/train_config.yaml', 'r') as f:
    train_config = yaml.safe_load(f)


Epochs=20

batch=8

model=od_models(train_config)
model.cuda()

opt = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-7)

step_ratio = 5 / 20
new_total_epochs = 50
step_size = max(1, int(step_ratio * Epochs))

sched = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=0.4)
color_list = standard_to_bgr(STANDARD_COLORS)


train_set = bev_Dataset(True,train_config)
val_set = bev_Dataset(False,train_config)

train_loader = DataLoader(train_set, batch_size=batch,collate_fn=collater, shuffle=True, num_workers=8)
val_loader = DataLoader(val_set, batch_size=batch, collate_fn=collater,shuffle=False, num_workers=8)

loss_Simple = SimpleLoss(2.13).cuda()
loss_dice = Dice_BEC_Loss().cuda()
loss_silog = silog_loss(0.85).cuda()
focal_Loss=get_loss().cuda()
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    filename=os.path.join(f"{save_dirs}/results.log"),
    filemode='w',
    format='%(asctime)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(sys.stdout)) 


test_data = load_input_and_gt()


das_GT=test_data['das_GT']
lane_GT=test_data['lane_GT']
vehicle_GT=test_data['vehicle_GT']
walker_GT=test_data['walker_GT']
for i in range(walker_GT[0].shape[0]):
    save_dir = f"data/GT/{i}"
    os.makedirs(save_dir, exist_ok=True)
    das_gt_img = (das_GT[i][0].cpu().numpy() * 255).astype(np.uint8)
    lane_gt_img = (lane_GT[i][0].cpu().numpy() * 255).astype(np.uint8)
    vehicle_gt_img = (vehicle_GT[i][0].cpu().numpy() * 255).astype(np.uint8)
    walker_gt_img = (walker_GT[i][0].cpu().numpy() * 255).astype(np.uint8)
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

    gt_mask   = torch.argmax(test_data['seg_GT'][i][0], dim=0).cpu().numpy().astype(np.uint8)   
    gt_color = mask_to_color(gt_mask)
    cv2.imwrite(os.path.join(save_dir, "seg_GT.png"), cv2.cvtColor(gt_color, cv2.COLOR_RGB2BGR))

    gt_depth = test_data['dept_GT'][i][0].cpu().numpy()
    cv2.imwrite(os.path.join(save_dir, "Depth_GT.png"), visualize_depth(gt_depth[0]))

    img_ori = test_data['imgs_ori'][i][0].permute(1, 2, 0).detach().cpu().numpy()
    img_ori = (img_ori * 255).astype(np.uint8)

    img_np = np.ascontiguousarray(img_ori)  
    gt_boxes = test_data['annotations'][i][0][:, :4].detach().cpu().numpy()
    gt_labels = test_data['annotations'][i][0][:, 4].detach().cpu().numpy().astype(int)

    for gt_box, gt_class in zip(gt_boxes, gt_labels):
        if gt_class == -1:
            continue
        x1, y1, x2, y2 = map(int, gt_box)
        label = class_names[gt_class] if gt_class < len(class_names) else 'Unknown'
        color = (255, 0, 0)  # 빨간색
        cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_np, f'{label} (GT)', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.imwrite(os.path.join(save_dir, "OD_GT.png"), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

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
if train_config['model_conf']['Seg']:
    os.makedirs(f"{save_dirs}/save_samples_seg/pred", exist_ok=True)
    os.makedirs(f"{save_dirs}/save_samples_seg/gt", exist_ok=True)
if train_config['model_conf']['Depth']:
    os.makedirs(f"{save_dirs}/save_samples_depth/pred", exist_ok=True)
    os.makedirs(f"{save_dirs}/save_samples_depth/gt", exist_ok=True)
if train_config['model_conf']['od']:
    os.makedirs(f"{save_dirs}/save_samples_od/pred", exist_ok=True)
    os.makedirs(f"{save_dirs}/save_samples_od/gt", exist_ok=True)

od_save=True
last_idx = len(train_loader) - 1
for epoch in range(Epochs):
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



        det_out,seg,depth =  model(imgs, translation, rotation, intrinsic,
                                                post_trans, post_rots,
                                                lidar_data, lidar_mask,
                                                )

        if train_config['model_conf']['BEV']:
            das_GT = das_GT.cuda().float()
            lane_GT = lane_GT.cuda().float()
            vehicle_GT = vehicle_GT.cuda().float()
            walker_GT = walker_GT.cuda().float()

            loss_DAS = loss_Simple(DAS, das_GT)
            loss_Lane = loss_Simple(Lane, lane_GT)
            loss_Vehicle = loss_Simple(Vehicle, vehicle_GT)
            loss_walker = loss_Simple(walker, walker_GT)
            loss_bev = loss_walker+loss_DAS+loss_Lane+loss_Vehicle
            _, _, DAS_iou = get_iou(DAS, das_GT)
            _, _, Lane_iou = get_iou(Lane, lane_GT)
            _, _, Vehicle_iou = get_iou(Vehicle, vehicle_GT)
            _, _, walker_iou = get_iou(walker, walker_GT)
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
            
            loss_od = focal_Loss(det_out[1], annotations.cuda(), model)
            miou=0
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
        with torch.no_grad():

            if counter % 20 == 0:

                if train_config['model_conf']['od']:

                    img_ori = imgs_ori[0][0].permute(1, 2, 0).detach().cpu().numpy()
                    img_ori = (img_ori * 255).astype(np.uint8)
                    img_np = np.ascontiguousarray(img_ori)  

                    det_pred = non_max_suppression(det_out[0], conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False)
                    det=det_pred[0].round()

                    if len(det):
                        for *xyxy,conf,cls in reversed(det):
                            label_det_pred = f'{class_names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, img_np , label=label_det_pred, color=[0,255,0], line_thickness=2)
                    
                    cv2.imwrite(f'{save_dirs}/save_samples_od/pred/od_{counter}.png', cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

                    img_np = np.ascontiguousarray(img_ori)  

                    for bbs in annotations:

                        if bbs[0] == 0:
                            gt_class = int(bbs[1].detach().cpu().numpy())
                            cx, cy, w, h = bbs[2:6].detach().cpu().numpy()

                            x1 = int(cx - w / 2)
                            y1 = int(cy - h / 2)
                            x2 = int(cx + w / 2)
                            y2 = int(cy + h / 2)

                            
                            label = class_names[gt_class] if gt_class < len(class_names) else 'Unknown'
                            color = (255, 0, 0)  # 빨간색
                            cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(img_np, f'{label} (GT)', (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                            
                    cv2.imwrite(f'{save_dirs}/save_samples_od/gt/od_{counter}.png', cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

                
                if train_config['model_conf']['Seg']:

                    pred_logits = seg[0][0]     
                    gt_onehot   = seg_GT[0][0]   

                    pred_mask = torch.argmax(pred_logits, dim=0).cpu().numpy().astype(np.uint8) 
                    gt_mask   = torch.argmax(gt_onehot, dim=0).cpu().numpy().astype(np.uint8)   

                    pred_color = mask_to_color(pred_mask)
                    gt_color = mask_to_color(gt_mask)

                    cv2.imwrite(f'{save_dirs}/save_samples_seg/pred/seg_{counter}.png', cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(f'{save_dirs}/save_samples_seg/gt/seg_{counter}.png', cv2.cvtColor(gt_color, cv2.COLOR_RGB2BGR))

                if train_config['model_conf']['Depth']:
                    pred_depth = depth[0][0].detach().cpu().numpy()
                    gt_depth = dept_GT[0][0].cpu().numpy()


                    pred_vis = visualize_depth(pred_depth[0])
                    gt_vis = visualize_depth(gt_depth[0])

                    cv2.imwrite(os.path.join(f'{save_dirs}/save_samples_depth/pred/depth_{counter}.png'), pred_vis)
                    cv2.imwrite(os.path.join(f'{save_dirs}/save_samples_depth/gt/depth_{counter}.png'), gt_vis)



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
                if seg_iou is not None:
                    log_msg += f'seg_iou {seg_iou:>7.3f}    '
                if Vehicle_iou is not None:
                    log_msg += f'vehicle_iou {Vehicle_iou:>7.3f}    '
                if walker_iou is not None:
                    log_msg += f'walker_iou {walker_iou:>7.3f}    '
                if Lane_iou is not None:
                    log_msg += f'Lane_iou {Lane_iou:>7.3f}    '
                if DAS_iou is not None:
                    log_msg += f'DAS_iou {DAS_iou:>7.3f}    '
                # if miou is not None:
                #     log_msg += f'miou {miou.cpu().item():>7.3f}    '

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
    avg_train_OD_iou = epoch_train_OD_iou / num_train_batches

    avg_train_miou = (avg_train_vehicle_iou + avg_train_walker_iou + avg_train_lane_iou + avg_train_das_iou) / 4

    train_losses.append(avg_train_loss)

    train_bev_losses.append(avg_train_bev_loss)
    train_depth_losses.append(avg_train_depth_loss)
    train_seg_losses.append(avg_train_seg_loss)

    train_Walker_losses.append(avg_train_walker_loss)
    train_Vehicle_losses.append(avg_train_Vehicle_loss)
    train_Lane_losses.append(avg_train_Lane_loss)
    train_DAS_losses.append(avg_train_DAS_loss)
    train_OD_losses.append(avg_train_DAS_loss)

    train_vehicle_ious.append(avg_train_vehicle_iou)
    train_walker_ious.append(avg_train_walker_iou)
    train_lane_ious.append(avg_train_lane_iou)
    train_das_ious.append(avg_train_das_iou)
    train_m_bev_ious.append(avg_train_miou)

    train_abs_rel.append(avg_train_abs_rel)
    train_seg_iou.append(avg_train_seg_iou)
    train_OD_iou.append(avg_train_OD_iou)


#     logger.info("VAL")
#     model.eval()
#     epoch_val_loss = 0

#     epoch_val_bev_loss = 0
#     epoch_val_DAS_loss = 0
#     epoch_val_OD_loss=0
#     epoch_val_Lane_loss = 0
#     epoch_val_Vehicle_loss = 0
#     epoch_val_Walker_loss = 0
    
#     epoch_val_seg_loss = 0
#     epoch_val_depth_loss = 0
#     epoch_val_abs_rel=0
#     num_val_batches=0
#     epoch_val_vehicle_iou = 0
#     epoch_val_walker_iou = 0
#     epoch_val_lane_iou = 0
#     epoch_val_das_iou = 0
#     epoch_val_OD=0
#     epoch_val_seg_iou = 0
#     with torch.no_grad():
#         for batchi, (imgs,imgs_ori,das_GT, lane_GT, vehicle_GT, walker_GT,lidar_data,
#                  lidar_mask,  post_rots,post_trans,extrinsic, intrinsic, 
#                  rotation,translation,
#                  seg_GT,dept_GT,depth_mask,
#                  annotations
                 
#                  ) in enumerate(val_loader):
#             loss=0
#             loss_bev=0

#             loss_seg=0
#             loss_depth=0
#             loss_od=0

#             loss_DAS=0
#             loss_Lane=0
#             loss_Vehicle=0
#             loss_walker=0

#             DAS,Lane,Vehicle,walker,seg,depth,classification, regression =  model(imgs, translation, rotation, intrinsic,
#                                                 post_trans, post_rots,
#                                                 lidar_data, lidar_mask,
#                                                 )

#             if train_config['model_conf']['BEV']:

#                 das_GT = das_GT.cuda().float()
#                 lane_GT = lane_GT.cuda().float()
#                 vehicle_GT = vehicle_GT.cuda().float()
#                 walker_GT = walker_GT.cuda().float()

#                 loss_DAS = loss_Simple(DAS, das_GT)
#                 loss_Lane = loss_Simple(Lane, lane_GT)
#                 loss_Vehicle = loss_Simple(Vehicle, vehicle_GT)
#                 loss_walker = loss_Simple(walker, walker_GT)
#                 loss_bev = loss_walker+loss_DAS+loss_Lane+loss_Vehicle
#                 _, _, DAS_iou = get_iou(DAS, das_GT)
#                 _, _, Lane_iou = get_iou(Lane, lane_GT)
#                 _, _, Vehicle_iou = get_iou(Vehicle, vehicle_GT)
#                 _, _, walker_iou = get_iou(walker, walker_GT)
#                 epoch_val_vehicle_iou += Vehicle_iou
#                 epoch_val_walker_iou += walker_iou
#                 epoch_val_lane_iou += Lane_iou
#                 epoch_val_das_iou += DAS_iou
#             else:
#                 DAS_iou=0
#                 Lane_iou=0
#                 walker_iou=0
#                 Vehicle_iou=0


#             if train_config['model_conf']['Seg']:
#                 seg_GT = seg_GT.cuda().float()
#                 loss_seg = loss_dice(seg, seg_GT)
#                 _, _, seg_iou = get_iou(seg, seg_GT)
#                 epoch_val_seg_iou += seg_iou
#             else:
#                 seg_iou=0


#             if train_config['model_conf']['Depth']:
#                 dept_GT = dept_GT.cuda().float()
#                 depth_mask=depth_mask.cuda()
#                 loss_depth = loss_silog(depth, dept_GT,depth_mask)
#                 abs_rel=compute_abs_rel(depth, dept_GT,depth_mask)
#                 epoch_val_abs_rel += abs_rel
#             else:
#                 abs_rel=0


#             if train_config['model_conf']['od']:
#                 B,N,M,ON=annotations.shape
#                 annotations=annotations.cuda().view(B*N,M,ON)
#                 B,N,C,W,H=imgs.shape
#                 imgs=imgs.view(B*N, C, W,H)
#                 anchors=model.anchors(imgs.cuda())
#                 loss_od_class,loss_od_reg,miou,out_predict = focal_Loss(classification, regression, anchors, annotations)
#                 loss_od=loss_od_class.mean()+loss_od_reg.mean()

#                 epoch_val_OD+=miou.cpu().item()
#             else:
#                 miou=0

#             loss=loss_bev+loss_depth+loss_seg+loss_od


#             epoch_val_loss += loss.item()
#             epoch_val_bev_loss     += loss_bev.item()     if isinstance(loss_bev, torch.Tensor) else 0.0
#             epoch_val_depth_loss   += loss_depth.item()   if isinstance(loss_depth, torch.Tensor) else 0.0
#             epoch_val_seg_loss     += loss_seg.item()     if isinstance(loss_seg, torch.Tensor) else 0.0
#             epoch_val_Walker_loss  += loss_walker.item()  if isinstance(loss_walker, torch.Tensor) else 0.0
#             epoch_val_Vehicle_loss += loss_Vehicle.item() if isinstance(loss_Vehicle, torch.Tensor) else 0.0
#             epoch_val_Lane_loss    += loss_Lane.item()    if isinstance(loss_Lane, torch.Tensor) else 0.0
#             epoch_val_DAS_loss     += loss_DAS.item()     if isinstance(loss_DAS, torch.Tensor) else 0.0
#             epoch_val_OD_loss     += loss_od.item()     if isinstance(loss_od, torch.Tensor) else 0.0


#             num_val_batches += 1

#     avg_val_loss = epoch_val_loss / num_val_batches

#     avg_val_bev_loss = epoch_val_bev_loss / num_val_batches
#     avg_val_depth_loss = epoch_val_depth_loss / num_val_batches
#     avg_val_seg_loss = epoch_val_seg_loss / num_val_batches

#     avg_val_Walker_loss = epoch_val_Walker_loss / num_val_batches
#     avg_val_Vehicle_loss = epoch_val_Vehicle_loss / num_val_batches
#     avg_val_Lane_loss = epoch_val_Lane_loss / num_val_batches
#     avg_val_DAS_loss = epoch_val_DAS_loss / num_val_batches

#     avg_val_OD_loss = epoch_val_OD_loss / num_val_batches

#     avg_val_vehicle_iou = epoch_val_vehicle_iou / num_val_batches
#     avg_val_walker_iou = epoch_val_walker_iou / num_val_batches
#     avg_val_lane_iou = epoch_val_lane_iou / num_val_batches
#     avg_val_das_iou = epoch_val_das_iou / num_val_batches

#     avg_val_seg_iou = epoch_val_seg_iou / num_val_batches
#     avg_val_abs_rel = epoch_val_abs_rel / num_val_batches

#     avg_val_OD_iou = epoch_val_OD / num_val_batches

#     avg_val_bev_miou = (avg_val_vehicle_iou + avg_val_walker_iou + avg_val_lane_iou + avg_val_das_iou) / 4


#     log_msg = f"VAL[{epoch}]  Loss: {loss.item():>7.3f}    "
#     if loss_bev != 0:
#         log_msg += f'loss_bev {loss_bev:>7.3f}    '
#     if avg_val_DAS_loss != 0:
#         log_msg += f'loss_DAS {avg_val_DAS_loss:>7.3f}    '
#     if avg_val_Lane_loss != 0:
#         log_msg += f'loss_Lane {avg_val_Lane_loss:>7.3f}    '
#     if avg_val_Vehicle_loss != 0:
#         log_msg += f'loss_Vehicle {avg_val_Vehicle_loss:>7.3f}    '
#     if avg_val_Walker_loss != 0:
#         log_msg += f'loss_walker {avg_val_Walker_loss:>7.3f}    '  
#     if loss_seg != 0:
#         log_msg += f'loss_Seg {loss_seg:>7.3f}    '  
#     if loss_depth != 0:
#         log_msg += f'loss_Depth {loss_depth:>7.3f}    ' 
#     if loss_od != 0:
#         log_msg += f'loss_OD {loss_od:>7.3f}    ' 
            
#     if avg_val_abs_rel != 0:
#         log_msg += f'abs_rel {avg_val_abs_rel:>7.4f}    '
#     if avg_val_seg_iou != 0:
#         log_msg += f'seg_iou {avg_val_seg_iou:>7.3f}    '
#     if avg_val_vehicle_iou != 0:
#         log_msg += f'vehicle_iou {avg_val_vehicle_iou:>7.3f}    '
#     if avg_val_walker_iou != 0:
#         log_msg += f'walker_iou {avg_val_walker_iou:>7.3f}    '
#     if avg_val_lane_iou != 0:
#         log_msg += f'Lane_iou {avg_val_lane_iou:>7.3f}    '
#     if avg_val_das_iou != 0:
#         log_msg += f'DAS_iou {avg_val_das_iou:>7.3f}    '
#     if avg_val_OD_iou != 0:
#         log_msg += f'OD_iou {avg_val_OD_iou:>7.3f}    '
#     if all(v is not None for v in [Vehicle_iou, walker_iou, Lane_iou, DAS_iou]):
#         log_msg += f'bev_m_iou {avg_val_bev_miou:>7.3f}    '
#     logger.info(log_msg)


#     val_losses.append(avg_val_loss)

#     val_bev_losses.append(avg_val_bev_loss)
#     val_depth_losses.append(avg_val_depth_loss)
#     val_seg_losses.append(avg_val_seg_loss)

#     val_Walker_losses.append(avg_val_Walker_loss)
#     val_Vehicle_losses.append(avg_val_Vehicle_loss)
#     val_Lane_losses.append(avg_val_Lane_loss)
#     val_DAS_losses.append(avg_val_DAS_loss)

#     val_DAS_losses.append(avg_val_OD_loss)

#     val_vehicle_ious.append(avg_val_vehicle_iou)
#     val_walker_ious.append(avg_val_walker_iou)
#     val_lane_ious.append(avg_val_lane_iou)
#     val_das_ious.append(avg_val_das_iou)
#     val_m_bev_ious.append(avg_val_bev_miou)

#     val_abs_rel.append(avg_val_abs_rel)
#     val_seg_iou.append(avg_val_seg_iou)
#     val_OD_iou.append(avg_val_OD_iou)


#     best_val_loss = avg_val_loss
#     torch.save(model.state_dict(), f'{save_dirs}/best_loss_model.pth')

#     sched.step()


# epochs = range(1, Epochs + 1)

# # 1. Loss 그래프
# plot_metric(train_losses, val_losses, 'Loss', 'Loss over Epochs', 'loss')

# if train_config['model_conf']['BEV']:

#     plot_metric(train_vehicle_ious, val_vehicle_ious, 'Vehicle IoU', 'Vehicle IoU over Epochs', 'vehicle_iou')
#     plot_metric(train_walker_ious, val_walker_ious, 'Walker IoU', 'Walker IoU over Epochs', 'walker_iou')
#     plot_metric(train_lane_ious, val_lane_ious, 'Lane IoU', 'Lane IoU over Epochs', 'lane_iou')
#     plot_metric(train_das_ious, val_das_ious, 'DAS IoU', 'DAS IoU over Epochs', 'das_iou')
#     plot_metric(train_m_bev_ious, val_m_bev_ious, 'Mean IoU', 'Mean IoU over Epochs', 'mean_iou')

#     plot_metric(train_DAS_losses, val_DAS_losses, 'DAS Loss', 'DAS Loss over Epochs', 'das_loss')
#     plot_metric(train_Walker_losses, val_Walker_losses, 'Walker Loss', 'Walker Loss over Epochs', 'walker_loss')
#     plot_metric(train_Vehicle_losses, val_Vehicle_losses, 'Vehicle Loss', 'Vehicle Loss over Epochs', 'vehicle_loss')
#     plot_metric(train_Lane_losses, val_Lane_losses, 'Lane Loss', 'Lane Loss over Epochs', 'lane_loss')

# if train_config['model_conf']['Depth']:
#     plot_metric(train_depth_losses, val_depth_losses, 'Depth Loss', 'Depth Loss over Epochs', 'depth_loss')
#     plot_metric(train_abs_rel, val_abs_rel, 'Abs Rel', 'Abs Rel over Epochs', 'abs_rel')
# if train_config['model_conf']['Seg']:
#     plot_metric(train_seg_losses, val_seg_losses, 'Segmentation Loss', 'Segmentation Loss over Epochs', 'seg_loss')
#     plot_metric(train_seg_iou, val_seg_iou, 'Segmentation IoU', 'Segmentation IoU over Epochs', 'seg_iou')
# if train_config['model_conf']['od']:
#     plot_metric(train_OD_iou, val_OD_iou, 'Object Detection IoU', 'Object Detection IoU over Epochs', 'seg_iou')
#     plot_metric(train_OD_losses, val_OD_losses, 'Object Detection Loss', 'Object Detection Loss over Epochs', 'OD_loss')

# # CSV 저장
# results = pd.DataFrame({
#     'epoch': list(epochs),
#     'train_loss': train_losses,
#     'val_loss': val_losses,
#     'train_vehicle_iou': train_vehicle_ious,
#     'val_vehicle_iou': val_vehicle_ious,
#     'train_walker_iou': train_walker_ious,
#     'val_walker_iou': val_walker_ious,
#     'train_lane_iou': train_lane_ious,
#     'val_lane_iou': val_lane_ious,
#     'train_das_iou': train_das_ious,
#     'val_das_iou': val_das_ious,
#     'train_m_bev_iou': train_m_bev_ious,
#     'val_m_bev_iou': val_m_bev_ious,
#     'train_OD_iou': train_OD_iou,
#     'val_OD_iou': val_OD_iou,
#     'train_abs_rel': train_abs_rel,
#     'val_abs_rel': val_abs_rel,
#     'train_seg_iou': train_seg_iou,
#     'val_seg_iou': val_seg_iou,

#     'train_depth_loss': train_depth_losses,
#     'val_depth_loss': val_depth_losses,
#     'train_seg_loss': train_seg_losses,
#     'val_seg_loss': val_seg_losses,
#     'train_DAS_loss': train_DAS_losses,
#     'val_DAS_loss': val_DAS_losses,
#     'train_Walker_loss': train_Walker_losses,
#     'val_Walker_loss': val_Walker_losses,
#     'train_Vehicle_loss': train_Vehicle_losses,
#     'val_Vehicle_loss': val_Vehicle_losses,
#     'train_Lane_loss': train_Lane_losses,
#     'val_Lane_loss': val_Lane_losses,
#     'train_OD_losses': train_OD_losses,
#     'val_OD_losses': val_OD_losses,
    
# })
# results.to_csv(f'{save_dirs}/train_metrics.csv', index=False)

