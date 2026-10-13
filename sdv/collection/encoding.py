"""CARLA semantic palette and packed depth decoding."""
import numpy as np
import cv2

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


def semantic_visibility_ratio(region, object_class):
    """CARLA 0.9.15 IDs, not the incompatible pre-0.9.14 tags 10/4."""
    if region.size == 0:
        return 0.0
    if object_class == 'Vehicle':
        visible = np.isin(region, (14, 15, 16, 17, 18, 19))
    elif object_class == 'Pedestrian':
        visible = region == 12
    else:
        raise ValueError('Unsupported OD class: {}'.format(object_class))
    return float(np.count_nonzero(visible)) / region.size


def depth_outputs(raw_bgr):
    """Exact metric label plus explicitly lossy backward-compatible preview."""
    metres = get_depth(raw_bgr).astype(np.float32)
    # Existing tools expect this filename/dtype; new loaders use depth_m NPY.
    preview = np.clip(np.rint(metres), 0, 255).astype(np.uint8)
    return metres, preview
