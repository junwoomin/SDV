import pygame
import numpy as np
import psutil
import time
import random
from PIL import Image ,ImageDraw
import glob
import os 
import cv2
import re
import os
import yaml
import subprocess
from pathlib import Path
import weakref
import copy

import numpy as np
import cv2
import carla  
import glob
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import math
from torchvision.ops.boxes import batched_nms
import torch
import torchvision
import webcolors

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.0001 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

    
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def compute_abs_rel(pred, gt, mask):
    pred = pred[mask]
    gt = gt[mask]

    abs_rel = torch.mean(torch.abs(pred - gt) / gt)
    return abs_rel.item()
def standard_to_bgr(list_color_name):
    standard = []
    for i in range(len(list_color_name) - 36):  # -36 used to match the len(obj_list)
        standard.append(from_colorname_to_bgr(list_color_name[i]))
    return standard
STANDARD_COLORS = [
    'LawnGreen', 'Chartreuse', 'Aqua', 'Beige', 'Azure', 'BlanchedAlmond', 'Bisque',
    'Aquamarine', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'AliceBlue', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

def from_colorname_to_bgr(color):
    rgb_color = webcolors.name_to_rgb(color)
    result = (rgb_color.blue, rgb_color.green, rgb_color.red)
    return result

def get_iou(preds, binimgs):
    with torch.no_grad():
        pred = preds > 0
        tgt = binimgs.bool()
        intersect = (pred & tgt).sum().float().item()
        union = (pred | tgt).sum().float().item()
    return intersect, union, intersect / union if union > 0 else 1.0
CLASS_COLORS = {
    0: (0, 255, 255),       
    1: (255, 0, 0),       
    2: (0, 255, 0),       
    3: (0, 0, 255),       
    4: (255, 255, 0),     
    5: (0, 0, 0), 
}

def get_index_label(label, obj_list):
    index = int(obj_list.index(label))
    return index

def mask_to_color(mask):
    """
    (H, W) 형태의 클래스 인덱스를 RGB 이미지로 변환.
    """
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_idx, color in CLASS_COLORS.items():
        color_mask[mask == class_idx] = color
    return color_mask

def load_input_and_gt(load_path="img/test.pt"):
    load_data = torch.load(load_path)
    return {
        k: v.cuda() if isinstance(v, torch.Tensor) else v
        for k, v in load_data.items()
    }
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
def xyxy_to_cxcywh(tensor):
    x1, y1, x2, y2 = tensor[:, 0], tensor[:, 1], tensor[:, 2], tensor[:, 3]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=1)


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
    if das_GT is not None:
        das_GT = torch.stack(das_GT)
        lane_GT = torch.stack(lane_GT)    
        vehicle_GT = torch.stack(vehicle_GT)
        walker_GT = torch.stack(walker_GT)    

    if das_GT is not None:
        dept_GT = torch.stack(dept_GT)  
        depth_mask = torch.stack(depth_mask)
    if seg_GT is not None:
        seg_GT = torch.stack(seg_GT)   


    if lidar_data is None:
        lidar_data = [torch.from_numpy(ld).float() if isinstance(ld, np.ndarray) else ld for ld in lidar_data]
        lidar_data = torch.stack(lidar_data)

        lidar_mask = [torch.from_numpy(lm).float() if isinstance(lm, np.ndarray) else lm for lm in lidar_mask]
        lidar_mask = torch.stack(lidar_mask)

    B = len(annots_batch)
    N = len(annots_batch[0])

    label_det=[]
    idx=0
    for b in range(B):
        for n in range(N):
            
            cam_annots = annots_batch[b][n]
            num = cam_annots.shape[0]
            if num > 0:
                bbox = xyxy_to_cxcywh(cam_annots[:, 1:5])  # (num_obj, 4)
                class_id = cam_annots[:, 5].unsqueeze(1)   # (num_obj, 1)
                cam_index = cam_annots[:, 0].unsqueeze(1)  # (num_obj, 1)
                batch_index = (cam_index + idx)

                
                det = torch.cat((batch_index, class_id, bbox), dim=1)  # (num_obj, 6)
                label_det.append(det)

                idx=idx+1



    if len(label_det):
        all_det = torch.cat(label_det, dim=0)  # shape: (total_obj, 6)

    return (
        imgs, imgs_ori, das_GT, lane_GT, vehicle_GT, walker_GT, lidar_data,
        lidar_mask, post_rots, post_trans, extrinsic, intrinsic,
        rotation, translation, seg_GT, dept_GT, depth_mask,
        all_det
    )


def visualize_depth(img, max_depth=100.0):
    img = np.clip(img, 0, max_depth)
    img = (img / max_depth * 255).astype(np.uint8)
    img_color = cv2.applyColorMap(img, cv2.COLORMAP_PLASMA)
    return img_color

def box_iou(box1, box2):

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1]) #(x2-x1)*(y2-y1)

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y
def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, labels=()):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output
def safe_imread(path):
    return cv2.imread(path) if os.path.isfile(path) else None

def fill_score_list(df,metrics,score_list, mode):
    for metric, (label, m) in metrics.items():
        if m != mode:
            continue
        if 'loss' in metric:
            idx = df[metric].idxmin()
            val_str = f"{df[metric].min():.4f}"
        else:
            idx = df[metric].idxmax()
            val_str = f"{df[metric].max() * 100:.2f}%" 
        epoch = int(df.loc[idx, 'epoch'])

        for i in range(3, len(score_list), 3):
            if score_list[i] == label:
                score_list[i+1] = val_str
                score_list[i+2] = str(epoch)
                break

def format_with_commas(number):
    number = float(number) if '.' in str(number) else int(number)
    return f"{number:,}"

SEMANTIC_COLORS = {
    0: (0, 0, 0),           # Unlabeled
    1: (128, 64, 128),      # Roads
    2: (244, 35, 232),      # SideWalks
    3: (70, 70, 70),        # Building
    4: (102, 102, 156),     # Wall
    5: (190, 153, 153),     # Fence
    6: (153, 153, 153),     # Pole
    7: (250, 170, 30),      # TrafficLight
    8: (220, 220, 0),       # TrafficSign
    9: (107, 142, 35),      # Vegetation
    10: (152, 251, 152),    # Terrain
    11: (70, 130, 180),     # Sky
    12: (220, 20, 60),      # Pedestrian
    13: (255, 0, 0),        # Rider
    14: (0, 0, 142),        # Car
    15: (0, 0, 70),         # Truck
    16: (0, 60, 100),       # Bus
    17: (0, 80, 100),       # Train
    18: (0, 0, 230),        # Motorcycle
    19: (119, 11, 32),      # Bicycle
    20: (110, 190, 160),    # Static
    21: (170, 120, 50),     # Dynamic
    22: (55, 90, 80),       # Other
    23: (45, 60, 150),      # Water
    24: (157, 234, 50),     # RoadLine
    25: (81, 0, 81),        # Ground
    26: (150, 100, 100),    # Bridge
    27: (230, 150, 140),    # RailTrack
    28: (180, 165, 180),    # GuardRail
}

def launch_carla_server(carla_root: Path, port: int) -> subprocess.Popen:
    """백그라운드에서 CARLA 서버를 실행하고 PID 반환"""
    server_bin = carla_root / "CarlaUE4.sh"
    cmd = [
        str(server_bin),
        "-quality_level=Low",
        "-benchmark",
        "-fps=" + str(15),
        "-RenderOffScreen",

        f"-world-port={port}",
    ]
    return subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )


def convert_semantics_to_color(semantics_img):

    colored_semantics = np.zeros((semantics_img.shape[0], semantics_img.shape[1], 3), dtype=np.uint8)

    for class_id, color in SEMANTIC_COLORS.items():
        mask = semantics_img == class_id
        colored_semantics[mask] = color
    colored_semantics=cv2.cvtColor(colored_semantics,cv2.COLOR_RGB2BGR)
    return colored_semantics


def get_depth(data):
    data = data.astype(np.float32)

    normalized = np.dot(data, [65536.0, 256.0, 1.0]) 
    normalized /=  (256 * 256 * 256 - 1)
    in_meters = 1000 * normalized

    return in_meters



WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY  = (180, 180, 180)
BLUE  = (0, 120, 255)
RED   = (255, 60, 60)



def pad_or_trim_to_np(x, shape, pad_val=0):
    shape = np.asarray(shape)
    pad = shape - np.minimum(np.shape(x), shape)
    zeros = np.zeros_like(pad)
    x = np.pad(x, np.stack([zeros, pad], axis=1), constant_values=pad_val)
    return x[:shape[0], :shape[1]]
def toggle_check(check_1, check_2):
    if check_1:
        return False
    elif not check_2:
        return True
    return check_1
def load_and_convert_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found at: {path}")
    if path.endswith(".png"):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return pygame.surfarray.make_surface(img.swapaxes(1, 0))

def load_model_test_input_images(paths):

    model_test_input_img = [None] * 9  # assuming index max is 7
    for idx, path in paths.items():
        model_test_input_img[idx] = load_and_convert_image(path)
    
    return model_test_input_img
def load_image_to_surface(path, convert_rgb=True):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found at: {path}")
    if convert_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return pygame.surfarray.make_surface(img.swapaxes(1, 0))


def postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold, iou_threshold):
    transformed_anchors = regressBoxes(anchors, regression)
    transformed_anchors = clipBoxes(transformed_anchors, x)
    scores = torch.max(classification, dim=2, keepdim=True)[0]
    scores_over_thresh = (scores > threshold)[:, :, 0]
    out = []
    for i in range(x.shape[0]):
        if scores_over_thresh[i].sum() == 0:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
            })
            continue

        classification_per = classification[i, scores_over_thresh[i, :], ...].permute(1, 0)
        transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...]
        scores_per = scores[i, scores_over_thresh[i, :], ...]
        scores_, classes_ = classification_per.max(dim=0)
        anchors_nms_idx = batched_nms(transformed_anchors_per, scores_per[:, 0], classes_, iou_threshold=iou_threshold)

        if anchors_nms_idx.shape[0] != 0:
            classes_ = classes_[anchors_nms_idx]
            scores_ = scores_[anchors_nms_idx]
            boxes_ = transformed_anchors_per[anchors_nms_idx, :]

            out.append({
                'rois': boxes_.cpu().numpy(),
                'class_ids': classes_.cpu().numpy(),
                'scores': scores_.cpu().numpy(),
            })
        else:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
            })

    return out
def load_model_test_output_images(paths):


    model_test_output_img = [None] * 21
    for idx, path in paths.items():
        convert_rgb = path.endswith('.png')
        model_test_output_img[idx] = load_image_to_surface(path, convert_rgb=convert_rgb)
    
    return model_test_output_img

def lidar_to_topview(points, res=0.2, x_range=(-50, 50), y_range=(-50, 50), z_range=(-2, 2), image_size=(500, 500)):
    x, y, _ = points[:, 0], points[:, 1], points[:, 2]


    x_rot = y
    y_rot = x


    x_img = ((x_rot - x_range[0]) / res).astype(np.int32)
    y_img = ((y_rot - y_range[0]) / res).astype(np.int32)


    x_img = np.clip(x_img, 0, image_size[0] - 1)
    y_img = np.clip(y_img, 0, image_size[1] - 1)


    topview_img = np.zeros(image_size, dtype=np.uint8)
    topview_img[image_size[1] - 1 - y_img, x_img] = 255  # 상하 반전 포함


    topview_img = np.stack([topview_img] * 3, axis=2)


    surface = pygame.surfarray.make_surface(topview_img.swapaxes(1, 0))
    return surface
def lidar_to_topview2(points, res=0.2, x_range=(-50, 50), y_range=(-50, 50), z_range=(-2, 2), image_size=(500, 500)):
    
    x, y, _ = points[:, 0], points[:, 1], points[:, 2]


    x_rot = y
    y_rot = x


    x_img = ((x_rot - x_range[0]) / res).astype(np.int32)
    y_img = ((y_rot - y_range[0]) / res).astype(np.int32)


    x_img = np.clip(x_img, 0, image_size[0] - 1)
    y_img = np.clip(y_img, 0, image_size[1] - 1)


    topview_img = np.zeros(image_size, dtype=np.uint8)
    topview_img[image_size[1] - 1 - y_img, x_img] = 255  # 상하 반전 포함


    topview_img = np.stack([topview_img] * 3, axis=2)

    return topview_img


def toggle_button_color(button_color):
    if button_color != GRAY:
        if button_color == RED:
            return BLUE
        elif button_color == BLUE:
            return RED
    return button_color

def boxing(rects, max_x_sum=20,max_y_sum=20):

    min_x = min(rect.x for rect in rects)
    min_y = min(rect.y for rect in rects)
    max_x = max(rect.x + rect.width for rect in rects)
    max_y = max(rect.y + rect.height for rect in rects)
    return pygame.Rect(min_x - 10, min_y - 10, (max_x - min_x) + max_x_sum, (max_y - min_y) + max_y_sum)
    


def checkbox_rect(row, col, top_left=(50, 80), cell_w=250, cell_h=60):  # cell_h increased for more spacing
    x = top_left[0] + col * cell_w
    y = top_left[1] + row * cell_h
    return pygame.Rect(x, y, 24, 24)

def draw_centered_text(screen, text, font, button_rect, color):
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect(center=button_rect.center)
    screen.blit(text_surface, text_rect)
def kill_existing_carla_processes():
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = proc.info.get("cmdline")
            if cmdline is None:
                continue

            if "CarlaUE4" in proc.info["name"] or any("CarlaUE4.sh" in c for c in cmdline):
                proc.kill()
        except Exception as e:
            print(f"[WARN] 프로세스 종료 실패: {e}")
    
    # 종료 대기
    for _ in range(30):  # 최대 30초
        still_running = False
        for proc in psutil.process_iter(["name", "cmdline"]):
            cmdline = proc.info.get("cmdline")
            if cmdline is None:
                continue

            if "CarlaUE4" in proc.info["name"] or any("CarlaUE4.sh" in c for c in cmdline):
                still_running = True
                break
        if not still_running:
            break
        time.sleep(1)


def format_number(n):
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    elif n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.2f}K"
    else:
        return str(n)


def get_random_ports():
    PORT = random.randint(1024, 30000)
    
    # TM_PORT는 PORT와 다르게 설정되도록 보장
    while True:
        TM_PORT = random.randint(1024, 30000)
        if TM_PORT != PORT:
            break

    return PORT, TM_PORT

def load_pil_image(path_list, index):
    if len(path_list) > 0:
        
        return Image.open(path_list[index])
    return None

def draw_bounding_boxes_on_pil(image, bbs_path, colors):
    if image is not None and bbs_path:
        draw = ImageDraw.Draw(image)
        data = np.load(bbs_path, allow_pickle=True).item()
        data.pop('traffic_light', None)
        for label, boxes in data.items():
            for box in boxes:
                draw.rectangle(box, outline=colors.get(label, 'white'), width=3)
    return image


def assign_paths_by_direction(base_paths, prefix):
    result = {
        f'{prefix}_front': [],
        f'{prefix}_left': [],
        f'{prefix}_right': [],
        f'{prefix}_rear': [],
        f'{prefix}_rear_left': [],
        f'{prefix}_rear_right': []
    }

    direction_order = ['front', 'rear_right', 'rear_left', 'rear', 'left', 'right']

    for path in base_paths:
        for ref_dir in direction_order:
            if ref_dir in path:
                lists = sorted(glob.glob(os.path.join(path, '*')))
                if not lists:
                    continue
                latest_file = lists[-1]

                if ref_dir == 'front':
                    result[f'{prefix}_front'] = [latest_file]
                    result[f'{prefix}_rear_right'] = [latest_file.replace(f'front_cam_{prefix}', f'rear_right_cam_{prefix}')]
                    result[f'{prefix}_rear_left'] = [latest_file.replace(f'front_cam_{prefix}', f'rear_left_cam_{prefix}')]
                    result[f'{prefix}_rear'] = [latest_file.replace(f'front_cam_{prefix}', f'rear_cam_{prefix}')]
                    result[f'{prefix}_left'] = [latest_file.replace(f'front_cam_{prefix}', f'left_cam_{prefix}')]
                    result[f'{prefix}_right'] = [latest_file.replace(f'front_cam_{prefix}', f'right_cam_{prefix}')]
                elif ref_dir == 'rear_right':
                    result[f'{prefix}_rear_right'] = [latest_file]
                    result[f'{prefix}_front'] = [latest_file.replace(f'rear_right_cam_{prefix}', f'front_cam_{prefix}')]
                    result[f'{prefix}_rear_left'] = [latest_file.replace(f'rear_right_cam_{prefix}', f'rear_left_cam_{prefix}')]
                    result[f'{prefix}_rear'] = [latest_file.replace(f'rear_right_cam_{prefix}', f'rear_cam_{prefix}')]
                    result[f'{prefix}_left'] = [latest_file.replace(f'rear_right_cam_{prefix}', f'left_cam_{prefix}')]
                    result[f'{prefix}_right'] = [latest_file.replace(f'rear_right_cam_{prefix}', f'right_cam_{prefix}')]
                elif ref_dir == 'rear_left':
                    result[f'{prefix}_rear_left'] = [latest_file]
                    result[f'{prefix}_front'] = [latest_file.replace(f'rear_left_cam_{prefix}', f'front_cam_{prefix}')]
                    result[f'{prefix}_rear_right'] = [latest_file.replace(f'rear_left_cam_{prefix}', f'rear_right_cam_{prefix}')]
                    result[f'{prefix}_rear'] = [latest_file.replace(f'rear_left_cam_{prefix}', f'rear_cam_{prefix}')]
                    result[f'{prefix}_left'] = [latest_file.replace(f'rear_left_cam_{prefix}', f'left_cam_{prefix}')]
                    result[f'{prefix}_right'] = [latest_file.replace(f'rear_left_cam_{prefix}', f'right_cam_{prefix}')]
                elif ref_dir == 'rear':
                    result[f'{prefix}_rear'] = [latest_file]
                    result[f'{prefix}_front'] = [latest_file.replace(f'rear_cam_{prefix}', f'front_cam_{prefix}')]
                    result[f'{prefix}_rear_right'] = [latest_file.replace(f'rear_cam_{prefix}', f'rear_right_cam_{prefix}')]
                    result[f'{prefix}_rear_left'] = [latest_file.replace(f'rear_cam_{prefix}', f'rear_left_cam_{prefix}')]
                    result[f'{prefix}_left'] = [latest_file.replace(f'rear_cam_{prefix}', f'left_cam_{prefix}')]
                    result[f'{prefix}_right'] = [latest_file.replace(f'rear_cam_{prefix}', f'right_cam_{prefix}')]
                elif ref_dir == 'left':
                    result[f'{prefix}_left'] = [latest_file]
                    result[f'{prefix}_front'] = [latest_file.replace(f'left_cam_{prefix}', f'front_cam_{prefix}')]
                    result[f'{prefix}_rear'] = [latest_file.replace(f'left_cam_{prefix}', f'rear_cam_{prefix}')]
                    result[f'{prefix}_rear_right'] = [latest_file.replace(f'left_cam_{prefix}', f'rear_right_cam_{prefix}')]
                    result[f'{prefix}_rear_left'] = [latest_file.replace(f'left_cam_{prefix}', f'rear_left_cam_{prefix}')]
                    result[f'{prefix}_right'] = [latest_file.replace(f'left_cam_{prefix}', f'right_cam_{prefix}')]
                elif ref_dir == 'right':
                    result[f'{prefix}_right'] = [latest_file]
                    result[f'{prefix}_front'] = [latest_file.replace(f'right_cam_{prefix}', f'front_cam_{prefix}')]
                    result[f'{prefix}_rear'] = [latest_file.replace(f'right_cam_{prefix}', f'rear_cam_{prefix}')]
                    result[f'{prefix}_rear_right'] = [latest_file.replace(f'right_cam_{prefix}', f'rear_right_cam_{prefix}')]
                    result[f'{prefix}_rear_left'] = [latest_file.replace(f'right_cam_{prefix}', f'rear_left_cam_{prefix}')]
                    result[f'{prefix}_left'] = [latest_file.replace(f'right_cam_{prefix}', f'left_cam_{prefix}')]

                return result

    return result
def get_last_log_values(log_path):
    with open(log_path, 'r') as f:
        lines = f.readlines()
    if not lines:
        return None

    last_line = lines[-1].strip()

    if last_line.strip().endswith("VAL"):
        return "VAL"
    elif last_line.strip().endswith("END"):
        return 'END'
    elif last_line == '':
        return None
    # "TRAIN"이 어디서 시작하는지 찾기
    train_idx = last_line.find("TRAIN")
    if train_idx == -1:
        return None

    train_line = last_line[train_idx:]

    match = re.search(
        r'TRAIN\[(\d+)\]\s+\[(\d+)\]\s+\[\s*(\d+)/(\d+)\]\s+Loss:\s+([\d\.]+)\s+vehicle_iou\s+([\d\.]+)\s+walker_iou\s+([\d\.]+)\s+Lane_iou\s+([\d\.]+)\s+DAS_iou\s+([\d\.]+)\s+m_iou\s+([\d\.]+)',
        train_line
    )

    if match:
        epoch = int(match.group(1))
        counter = int(match.group(2))
        batchi = int(match.group(3))
        last_idx = int(match.group(4))
        loss = float(match.group(5))
        vehicle_iou = float(match.group(6))
        walker_iou = float(match.group(7))
        lane_iou = float(match.group(8))
        das_iou = float(match.group(9))
        m_iou = float(match.group(10))

        return {
            'epoch': epoch,
            'counter': counter,
            'batchi': batchi,
            'last_idx': last_idx,
            'loss': loss,
            'vehicle_iou': vehicle_iou,
            'walker_iou': walker_iou,
            'lane_iou': lane_iou,
            'das_iou': das_iou,
            'm_iou': m_iou
        }
    else:
        return None

def draw_hoverable_button(screen, rect, color, text, mouse_pos, font, hover_font):

    hovered = rect.collidepoint(mouse_pos)
    if hovered:
        bigger_rect = pygame.Rect(rect.x - 3, rect.y - 3, rect.width + 7, rect.height + 7)
        pygame.draw.rect(screen, color, bigger_rect)
        draw_centered_text(screen, text, hover_font, bigger_rect, BLACK)
    else:
        pygame.draw.rect(screen, color, rect)
        draw_centered_text(screen, text, font, rect, BLACK)

def draw_dynamic_button(screen, rect, base_color, hover_color, text, font, hover_font, mouse_pos, radius=6):
    hovered = rect.collidepoint(mouse_pos)
    if hovered:
        bigger_rect = pygame.Rect(rect.x - 5, rect.y - 5, rect.width + 10, rect.height + 10)
        pygame.draw.rect(screen, hover_color, bigger_rect, border_radius=radius + 2)
        draw_centered_text(screen, text, hover_font, bigger_rect, WHITE)
    else:
        pygame.draw.rect(screen, base_color, rect, border_radius=radius)
        draw_centered_text(screen, text, font, rect, WHITE)


class World:
    def __init__(self, client, conf):

        self.client = client
        self.world = client.get_world()
        self.map = self.world.get_map()
        self.player = None
        self._actor_filter = 'vehicle.*'
        self.cam_sensor=None
        self.restart(conf)

    def restart(self, conf):
        self._destroy_sensors_only()

        self.sensors = []

        # 2. 차량이 없으면 최초 1회만 생성
        if self.player is None:
            vehicle_bp = self.world.get_blueprint_library().find("vehicle.tesla.model3")
            if vehicle_bp.has_attribute("role_name"):
                vehicle_bp.set_attribute("role_name", "hero")
            color = vehicle_bp.get_attribute("color").recommended_values[0]
            vehicle_bp.set_attribute("color", color)

            while self.player is None:
                spawn_points = self.map.get_spawn_points()
                spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
                self.player = self.world.try_spawn_actor(vehicle_bp, spawn_point)

            self.modify_vehicle_physics(self.player)
            self._vehicle = self.player

        self.world.tick()

        # 3. 센서 재설정
        for name in list(conf.keys()):
            if name == 'lidar':
                continue
            rgb_sensor = CameraSensor_RGB(self.player, conf[name])
            setattr(self, f"{name}_rgb", rgb_sensor)
            self.sensors.append(rgb_sensor.sensor)

        if 'lidar' in conf:
            self.lidar_sensor = LidarSensor(self.player, conf['lidar'])
            self.sensors.append(self.lidar_sensor.sensor)

    def _destroy_sensors_only(self):
        if hasattr(self, "sensors"):
            for sensor in self.sensors:
                if sensor is not None:
                    try:
                        sensor.destroy()
                    except:
                        pass
        self.sensors = []

    def modify_vehicle_physics(self, actor):
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass

    def destroy(self):
        actors = []

        if self.player is not None:
            actors.append(self.player)

        if hasattr(self, "sensors"):
            actors.extend(self.sensors)

        for actor in actors:
            if actor is not None:
                try:
                    actor.destroy()
                except:
                    pass

        self.sensors = []
        self.player = None

class CameraSensor_RGB:
    def __init__(self, parent_actor,conf):
        self.sensor = None
        self._parent = parent_actor
        world = self._parent.get_world()

        bp = world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(conf['width']))
        bp.set_attribute('image_size_y', str(conf['height']))
        bp.set_attribute('fov', str(conf['fov']))

        self.sensor = world.spawn_actor(bp, carla.Transform(
        carla.Location(x=conf['x'], y=conf['y'], z=conf['z']),                                              
        carla.Rotation(pitch=conf['pitch'], roll=conf['roll'], yaw=conf['yaw'])
        ), attach_to=self._parent)

        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: CameraSensor_RGB._on_image(weak_self, image))
        self.img=None
    @staticmethod
    def _on_image(weak_self, image):
        self = weak_self()
        if not self:
            return

        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        self.img = array

class LidarSensor:
    def __init__(self, parent_actor,conf):
        self.sensor = None
        self._parent = parent_actor
        world = self._parent.get_world()

        bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')

        bp.set_attribute('channels', '64')
        bp.set_attribute('range', '100')
        bp.set_attribute('points_per_second', '1200000')
        bp.set_attribute('rotation_frequency', '60')

        self.sensor = world.spawn_actor(bp, carla.Transform(
        carla.Location(x=conf['x'], y=conf['y'], z=conf['z']),                                              
        carla.Rotation(pitch=conf['pitch'], roll=conf['roll'], yaw=conf['yaw'])
        ), attach_to=self._parent)


        weak_self = weakref.ref(self)
        self.sensor.listen(lambda point_cloud: LidarSensor._on_point_cloud(weak_self, point_cloud))
        self.point_cloud=None

    @staticmethod
    def _on_point_cloud(weak_self, lidar_data):
        self = weak_self()
        if not self:
            return
        points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
        points = copy.deepcopy(points)
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        self.point_cloud=points


