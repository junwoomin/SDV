"""Unmodified isolated LRS reference functions for numeric parity tests.
Source: junwoomin/LRS commit 6eaf6c38450efc540f5fb1c86ed2980dd96a59bc.
MIT License, Copyright (c) 2026 woomin jun. Full license: sdv/driving/vendor/roach/LICENSE.
Imports are restricted to avoid simulator/GUI dependencies; functions unchanged.
"""
import math
from typing import Tuple
import numpy as np
import cv2

def make_local_bev_ego_aligned(
    global_mask: np.ndarray,
    center_px: Tuple[float, float],
    ego_yaw_rad: float,
    out_size: int = 200,
    out_mpp: float = 0.20,
    pixels_per_meter: float = 5.0,
    fill: int = 0,
    move_px: int = 0,
) -> np.ndarray:
    u0 = out_size / 2.0
    v0 = out_size / 2.0 + move_px
    k  = out_mpp * pixels_per_meter
    c  = math.cos(ego_yaw_rad)
    s  = math.sin(ego_yaw_rad)

    M = np.array([
        [-s * k, -c * k, center_px[0] + k * (s * u0 + c * v0)],
        [ c * k, -s * k, center_px[1] + k * (-c * u0 + s * v0)],
    ], dtype=np.float32)

    # float32 height mask도 처리 가능하도록 flags 분기
    if global_mask.dtype == np.float32:
        flags = cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP
        border_val = float(fill)
    else:
        flags = cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP
        border_val = int(fill)

    return cv2.warpAffine(
        global_mask, M, (out_size, out_size),
        flags=flags,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_val,
    )

def _obb_pts(cx, cy, hl, hw, theta):
    """OBB 꼭짓점 4개(int32) 반환."""
    c = math.cos(theta)
    s = math.sin(theta)
    pts = np.array([
        [ hl,  hw], [ hl, -hw],
        [-hl, -hw], [-hl,  hw],
    ], dtype=np.float32)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    pts = pts @ R.T
    pts[:, 0] += cx
    pts[:, 1] += cy
    return np.round(pts).astype(np.int32)

def world_to_pixel(x, y, OFFSET_X, OFFSET_Y, PPM):
    px = (x - OFFSET_X) * PPM
    py = (y - OFFSET_Y) * PPM
    return int(px), int(py)

def carla_rot_to_mat(carla_rotation):
    roll  = np.deg2rad(carla_rotation.roll)
    pitch = np.deg2rad(carla_rotation.pitch)
    yaw   = np.deg2rad(carla_rotation.yaw)

    yaw_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])
    pitch_matrix = np.array([
        [np.cos(pitch), 0, -np.sin(pitch)],
        [0, 1, 0],
        [np.sin(pitch), 0,  np.cos(pitch)]
    ])
    roll_matrix = np.array([
        [1, 0, 0],
        [0,  np.cos(roll), np.sin(roll)],
        [0, -np.sin(roll), np.cos(roll)]
    ])
    return yaw_matrix.dot(pitch_matrix).dot(roll_matrix)

def global_pixel_to_local_bev(px, py, ego_cx, ego_cy, ego_yaw,
                               out_size=200, out_mpp=0.20, pixels_per_meter=5.0,
                               move_px=0):
    u0 = out_size / 2.0
    v0 = out_size / 2.0 + move_px

    k = out_mpp * pixels_per_meter

    c = math.cos(ego_yaw)
    s = math.sin(ego_yaw)

    dx = px - ego_cx
    dy = py - ego_cy

    u = u0 + (-s * dx + c * dy) / k
    v = v0 + (-c * dx - s * dy) / k

    return int(round(u)), int(round(v))


def _render_path_local_bev(
    self,
    ref_path:     list,
    ref_path_idx: int,
    ego_pixel_x:  float,
    ego_pixel_y:  float,
    ego_yaw_rad:  float,
) -> np.ndarray:
    path_local_bev = np.zeros((self.OUT_SIZE, self.OUT_SIZE), dtype=np.uint8)
    start_idx = max(0, ref_path_idx - 8)
    end_idx   = min(len(ref_path), ref_path_idx + 50)
    local_pts = []

    for pt in ref_path[start_idx:end_idx]:
        global_px, global_py = world_to_pixel(
            pt[0], pt[1],
            self.OFFSET_X, self.OFFSET_Y, self.PPM,
        )
        local_u, local_v = global_pixel_to_local_bev(
            global_px, global_py,
            ego_pixel_x, ego_pixel_y, ego_yaw_rad,
            out_size=self.OUT_SIZE, out_mpp=1.0 / self.PPM,
            pixels_per_meter=self.PPM, move_px=self.move_px,
        )
        if 0 <= local_u < self.OUT_SIZE and 0 <= local_v < self.OUT_SIZE:
            local_pts.append([local_u, local_v])

    if len(local_pts) >= 2:
        cv2.polylines(
            path_local_bev,
            [np.array(local_pts, dtype=np.int32).reshape((-1, 1, 2))],
            isClosed=False, color=255, thickness=12,
        )
    elif len(local_pts) == 1:
        cv2.circle(path_local_bev, tuple(local_pts[0]), 6, 255, -1)

    return path_local_bev
