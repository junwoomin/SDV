import glob
import re
import torch
from torch.utils.data import Dataset
import torchvision
import cv2
import numpy as np
from pyquaternion import Quaternion
import os
import yaml
from pathlib import Path
from sdv.telemetry import configured_path
from sdv.config.validation import read_sensor_config
from sdv.data_contract import (normalize_task_settings, segmentation_spec,
    decode_segmentation, remap_segmentation, decode_depth, depth_encoding_for_scene)
from sdv.bev.training import (frame_bev_schema, model_ground_truth,
    camera_optical_to_bev_frame, validate_training_layers)
from scipy.spatial.transform import Rotation as R

def pad_or_trim_to_np(x, shape, pad_val=0):
  shape = np.asarray(shape)
  pad = shape - np.minimum(np.shape(x), shape)
  zeros = np.zeros_like(pad)
  x = np.pad(x, np.stack([zeros, pad], axis=1), constant_values=pad_val)
  return x[:shape[0], :shape[1]]

normalize_img = torchvision.transforms.Compose((
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
))



ToTensor = torchvision.transforms.Compose((
    torchvision.transforms.ToTensor(),

))


def img_transform(img, resize, resize_dims):
    post_rot2 = torch.eye(2)
    post_tran2 = torch.zeros(2)

    img = cv2.resize(img, resize_dims)

    rot_resize = torch.Tensor([[resize[0], 0],
                               [0, resize[1]]])
    post_rot2 = rot_resize @ post_rot2
    post_tran2 = rot_resize @ post_tran2

    post_tran = torch.zeros(3)
    post_rot = torch.eye(3)
    post_tran[:2] = post_tran2
    post_rot[:2, :2] = post_rot2
    return img, post_rot, post_tran

def assign_paths_by_direction(base_paths, prefix):
    result = {
        f'{prefix}_front': [],
        f'{prefix}_left': [],
        f'{prefix}_right': [],
        f'{prefix}_rear': [],
        f'{prefix}_rear_left': [],
        f'{prefix}_rear_right': []
    }

    for path in base_paths:
        key = None
        if 'rear_right' in path:
            key = f'{prefix}_rear_right'
        elif 'rear_left' in path:
            key = f'{prefix}_rear_left'
        elif 'rear' in path:
            key = f'{prefix}_rear'
        elif 'front' in path:
            key = f'{prefix}_front'
        elif 'left' in path:
            key = f'{prefix}_left'
        elif 'right' in path:
            key = f'{prefix}_right'

        if key:
            result[key] = sorted(glob.glob(os.path.join(path, '*')))

    return result


class bev_Dataset(Dataset):
	def __init__(self,train,train_config):
		self.data_root = configured_path('SDV_TRAIN_DATA' if train else 'SDV_VAL_DATA', 'data/train' if train else 'data/val')
		if train:
			self.image_front = (sorted(glob.glob(str(self.data_root / '*/front_cam_rgb/*'))))
			self.image_left = (sorted(glob.glob(str(self.data_root / '*/left_cam_rgb/*'))))
			self.image_right = (sorted(glob.glob(str(self.data_root / '*/right_cam_rgb/*'))))

			self.image_rear = (sorted(glob.glob(str(self.data_root / '*/rear_cam_rgb/*'))))
			self.image_rear_left = (sorted(glob.glob(str(self.data_root / '*/rear_left_cam_rgb/*'))))
			self.image_rear_right = (sorted(glob.glob(str(self.data_root / '*/rear_right_cam_rgb/*'))))

			self.bbs_front = (sorted(glob.glob(str(self.data_root / '*/front_cam_bbs/*'))))
			self.bbs_left = (sorted(glob.glob(str(self.data_root / '*/left_cam_bbs/*'))))
			self.bbs_right = (sorted(glob.glob(str(self.data_root / '*/right_cam_bbs/*'))))

			self.bbs_rear = (sorted(glob.glob(str(self.data_root / '*/rear_cam_bbs/*'))))
			self.bbs_rear_left = (sorted(glob.glob(str(self.data_root / '*/rear_left_cam_bbs/*'))))
			self.bbs_rear_right = (sorted(glob.glob(str(self.data_root / '*/rear_right_cam_bbs/*'))))

			self.seg_front = (sorted(glob.glob(str(self.data_root / '*/front_cam_seg/*'))))
			self.seg_left = (sorted(glob.glob(str(self.data_root / '*/left_cam_seg/*'))))
			self.seg_right = (sorted(glob.glob(str(self.data_root / '*/right_cam_seg/*'))))

			self.seg_rear = (sorted(glob.glob(str(self.data_root / '*/rear_cam_seg/*'))))
			self.seg_rear_left = (sorted(glob.glob(str(self.data_root / '*/rear_left_cam_seg/*'))))
			self.seg_rear_right = (sorted(glob.glob(str(self.data_root / '*/rear_right_cam_seg/*'))))

			self.depth_front = (sorted(glob.glob(str(self.data_root / '*/front_cam_depth/*'))))
			self.depth_left = (sorted(glob.glob(str(self.data_root / '*/left_cam_depth/*'))))
			self.depth_right = (sorted(glob.glob(str(self.data_root / '*/right_cam_depth/*'))))

			self.depth_rear = (sorted(glob.glob(str(self.data_root / '*/rear_cam_depth/*'))))
			self.depth_rear_left = (sorted(glob.glob(str(self.data_root / '*/rear_left_cam_depth/*'))))
			self.depth_rear_right = (sorted(glob.glob(str(self.data_root / '*/rear_right_cam_depth/*'))))


			self.bev = (sorted(glob.glob(str(self.data_root / '*/bev/*'))))

			self.lidar_paths = (sorted(glob.glob(str(self.data_root / '*/lidar/*'))))

		else:
			self.image_front = (sorted(glob.glob(str(self.data_root / '*/front_cam_rgb/*'))))
			self.image_left = (sorted(glob.glob(str(self.data_root / '*/left_cam_rgb/*'))))
			self.image_right = (sorted(glob.glob(str(self.data_root / '*/right_cam_rgb/*'))))

			self.image_rear = (sorted(glob.glob(str(self.data_root / '*/rear_cam_rgb/*'))))
			self.image_rear_left = (sorted(glob.glob(str(self.data_root / '*/rear_left_cam_rgb/*'))))
			self.image_rear_right = (sorted(glob.glob(str(self.data_root / '*/rear_right_cam_rgb/*'))))

			self.bev = (sorted(glob.glob(str(self.data_root / '*/bev/*'))))

			self.lidar_paths = (sorted(glob.glob(str(self.data_root / '*/lidar/*'))))

			self.bbs_front = (sorted(glob.glob(str(self.data_root / '*/front_cam_bbs/*'))))
			self.bbs_left = (sorted(glob.glob(str(self.data_root / '*/left_cam_bbs/*'))))
			self.bbs_right = (sorted(glob.glob(str(self.data_root / '*/right_cam_bbs/*'))))

			self.bbs_rear = (sorted(glob.glob(str(self.data_root / '*/rear_cam_bbs/*'))))
			self.bbs_rear_left = (sorted(glob.glob(str(self.data_root / '*/rear_left_cam_bbs/*'))))
			self.bbs_rear_right = (sorted(glob.glob(str(self.data_root / '*/rear_right_cam_bbs/*'))))

			self.seg_front = (sorted(glob.glob(str(self.data_root / '*/front_cam_seg/*'))))
			self.seg_left = (sorted(glob.glob(str(self.data_root / '*/left_cam_seg/*'))))
			self.seg_right = (sorted(glob.glob(str(self.data_root / '*/right_cam_seg/*'))))

			self.seg_rear = (sorted(glob.glob(str(self.data_root / '*/rear_cam_seg/*'))))
			self.seg_rear_left = (sorted(glob.glob(str(self.data_root / '*/rear_left_cam_seg/*'))))
			self.seg_rear_right = (sorted(glob.glob(str(self.data_root / '*/rear_right_cam_seg/*'))))

			self.depth_front = (sorted(glob.glob(str(self.data_root / '*/front_cam_depth/*'))))
			self.depth_left = (sorted(glob.glob(str(self.data_root / '*/left_cam_depth/*'))))
			self.depth_right = (sorted(glob.glob(str(self.data_root / '*/right_cam_depth/*'))))

			self.depth_rear = (sorted(glob.glob(str(self.data_root / '*/rear_cam_depth/*'))))
			self.depth_rear_left = (sorted(glob.glob(str(self.data_root / '*/rear_left_cam_depth/*'))))
			self.depth_rear_right = (sorted(glob.glob(str(self.data_root / '*/rear_right_cam_depth/*'))))


		# RGB frames anchor every modality by sequence + stem. Independently sorted
		# lists silently paired later labels with earlier frames when one was missing.
		self.image_front = sorted((str(path) for path in self.data_root.rglob('front_cam_rgb/*')
			if path.is_file() and path.suffix.lower() in ('.png', '.jpg', '.jpeg')),
			key=lambda value: (str(Path(value).parent), tuple((0, int(part)) if part.isdigit() else (1, part)
				for part in re.split(r'(\d+)', Path(value).stem))))
		self.task_settings = normalize_task_settings(train_config.get('task_settings'))
		self.segmentation_classes = segmentation_spec(self.task_settings)
		self.colors_bev = [(0, 0, 192),(0, 0, 128)] # Lane, DAS
		self.Lane, self.DAS = (0, 255, 0), (0, 0, 255)
		cfg = read_sensor_config()

		self.sensor_config = cfg['sensors']

		self.sensor_keys = [key for key, value in train_config['sensor_use'].items() if value and key != 'lidar']


		order = [
			'left_cam',
			'front_cam',
			'right_cam',
			'rear_left_cam',
			'rear_cam',
			'rear_right_cam',
		]

		self.sensor_keys = sorted(self.sensor_keys, key=lambda x: order.index(x))

		self.sensor_data = {
			'width': int(cfg['sensors'][self.sensor_keys[0]]['width']),
			'height': int(cfg['sensors'][self.sensor_keys[0]]['height']),

			'fov': int(cfg['sensors'][self.sensor_keys[0]]['fov']),
		}
		self.train_config=train_config
		# Never combine two different calibration/label coordinate contracts in
		# one experiment. A missing declared NPZ is an error, not a PNG fallback.
		self.bev_schema = None
		if train_config['model_conf'].get('BEV'):
			schemas = {frame_bev_schema(Path(path).parent.parent, Path(path).stem) or 1
				for path in self.image_front}
			if len(schemas) > 1:
				raise ValueError('Mixed legacy or numeric BEV schema datasets: use separate experiments for their different calibration contracts')
			self.bev_schema = next(iter(schemas), None)
		self.W=int(train_config['width'])
		self.H=int(train_config['height'])
		self.CARLA_SEG_COLOR = {
			# 'Buildings': [128, 64, 128],
			# 'Fences': [70, 130, 180],
			'Pedestrians': [102, 102, 156],
			# 'Poles': [190, 153, 153],
			'RoadLines': [153, 153, 153],
			'Roads': [250, 170, 30],
			'Sidewalks': [220, 220, 0],
			# 'Vegetation': [107, 142, 35],
			'Vehicles': [152, 251, 152],
			# 'Walls': [244, 35, 232],
			# 'TrafficSigns': [220, 20, 60],
			# "TrafficLight": [0, 0, 230],
			# "Sky": [255, 0, 0],
			'Other': [110, 190, 160],
		}

	def __len__(self):
		return len(self.image_front)

	def get_cam_para(self, idx=None):
		sensors = self.sensor_config
		if idx is not None:
			path = self._scene(idx) / 'sensor_config.yaml'
			if self.bev_schema in (3,4) and not path.is_file():
				raise ValueError('Numeric BEV requires the recorded sensor_config.yaml: ' + str(path))
			if path.is_file():
				with path.open(encoding='utf-8') as stream:
					sensors = yaml.safe_load(stream)['sensors']
		def get_cam_to_ego(dof):
			# Convert degree → radian
			pitch, roll, yaw = np.radians(dof[3]), np.radians(dof[4]), np.radians(dof[5])

			rotation_matrix = R.from_euler('xyz', [pitch, roll, yaw], degrees=False).as_matrix()
			translation = np.array(dof[:3])[:, None]  # (3, 1)

			cam_to_ego = np.vstack([
				np.hstack((rotation_matrix, translation)),
				np.array([0, 0, 0, 1])
			])
			return cam_to_ego, rotation_matrix, translation

		extrinsic_list = []
		rotation_list = []
		translation_list = []
		intrinsic_list=[]
		for cam in self.sensor_keys:
			cam_dof = [
				sensors[cam]['x'],
				sensors[cam]['y'],
				sensors[cam]['z'],
				sensors[cam]['pitch'],
				sensors[cam]['roll'],
				sensors[cam]['yaw'],
			]
			fov = sensors[cam]['fov']
			w = sensors[cam]['width']
			h = sensors[cam]['height']
			f = w / (2 * np.tan(fov * np.pi / 360))
			Cu = w / 2
			Cv = h / 2
			K = torch.tensor([
				[f, 0, Cu],
				[0, f, Cv],
				[0, 0, 1]
			], dtype=torch.float32)
			intrinsic_list.append(K.unsqueeze(0))

			intrinsic = torch.cat(intrinsic_list, dim=0)
			if self.bev_schema in (3,4):
				if idx is None:
					raise ValueError('Numeric BEV camera calibration requires a frame index')
				cam_to_ego, rot, tran = camera_optical_to_bev_frame(self._scene(idx), Path(self.image_front[idx]).stem, cam)
			else:
				cam_to_ego, rot, tran = get_cam_to_ego(cam_dof)
			extrinsic_list.append(torch.from_numpy(cam_to_ego).float().unsqueeze(0))
			rotation_list.append(torch.from_numpy(rot).float().unsqueeze(0))
			translation_list.append(torch.from_numpy(tran).float().unsqueeze(0))

		extrinsic = torch.cat(extrinsic_list, dim=0)
		rotation = torch.cat(rotation_list, dim=0)
		translation = torch.cat(translation_list, dim=0).squeeze(-1)



		return extrinsic, intrinsic, rotation, translation

	def sample_augmentation(self):

		resize = (self.W / self.sensor_data['width'], self.H / self.sensor_data['height'])
		resize_dims = (self.W, self.H)
		return resize, resize_dims

	def get_img(self, idx):
		resize, resize_dims = self.sample_augmentation()

		img_paths = [self._aligned(idx, camera + '_rgb') for camera in self.sensor_keys]

		imgs = []
		imgs_ori=[]
		post_rots = []
		post_trans = []

		for path in img_paths:


			img = cv2.imread(str(path))
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

			resize = (self.W / img.shape[1], self.H / img.shape[0])
			img, post_rot, post_tran = img_transform(img, resize, resize_dims)

			imgs_ori.append(ToTensor(img))

			img = normalize_img(img)

			imgs.append(img)
			post_rots.append(post_rot)
			post_trans.append(post_tran)

		imgs = torch.stack(imgs)
		imgs_ori = torch.stack(imgs_ori)
		post_rots = torch.stack(post_rots)
		post_trans = torch.stack(post_trans)


		return imgs,imgs_ori, post_rots, post_trans



	def _scene(self, idx):
		return Path(self.image_front[idx]).parent.parent

	def _aligned(self, idx, folder, suffixes=('.png', '.jpg', '.jpeg'), required=True):
		frame = Path(self.image_front[idx])
		for suffix in suffixes:
			path = frame.parent.parent / folder / (frame.stem + suffix)
			if path.is_file():
				return path
		if required:
			raise ValueError('Missing aligned {}: {}/{}'.format(folder, frame.parent.parent.name, frame.stem))
		return None

	def get_seg(self, idx):
		if not self.train_config['model_conf'].get('Seg'):
			return None
		masks = []
		for camera in self.sensor_keys:
			path = (self._aligned(idx, camera + '_seg_ids', ('.png',), required=False)
				if self.task_settings['segmentation']['mode'] == 'carla' else None)
			path = path or self._aligned(idx, camera + '_seg')
			image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
			if image is None:
				raise ValueError('Unreadable segmentation: ' + str(path))
			if image.ndim == 3:
				image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			if self.task_settings['segmentation']['mode'] == 'legacy':
				# Retain original output-channel meaning for existing checkpoints;
				# always resize categorical labels with nearest-neighbour.
				image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
				mask = np.full((self.H, self.W), 5, dtype=np.uint8)
				for index, color in enumerate(self.CARLA_SEG_COLOR.values()):
					mask[np.all(image == np.asarray(color), axis=-1)] = index
			else:
				ids = decode_segmentation(image)
				ids = cv2.resize(ids, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
				mask = remap_segmentation(ids, self.task_settings)
			# All-zero vectors represent ignored pixels throughout loss/metrics/render.
			binary = np.stack([(mask == index) for index in range(len(self.segmentation_classes))])
			masks.append(torch.from_numpy(binary.astype(np.float32)))
		return torch.stack(masks)

	def get_depth(self, idx):
		if not self.train_config['model_conf'].get('Depth'):
			return None, None
		settings = self.task_settings['depth']
		images, masks = [], []
		for camera in self.sensor_keys:
			path = self._aligned(idx, camera + '_depth_m', ('.npy',), required=False)
			path = path or self._aligned(idx, camera + '_depth', ('.npy', '.png'))
			image = np.load(str(path), allow_pickle=False) if path.suffix == '.npy' else cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
			encoding = ('meters_f32' if path.suffix == '.npy' else
				depth_encoding_for_scene(self._scene(idx), settings['encoding'], path))
			meters, valid = decode_depth(image, encoding)
			# Nearest avoids mixing valid/invalid values and preserves measured ranges.
			meters = cv2.resize(meters, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
			valid = cv2.resize(valid.astype(np.uint8), (self.W, self.H), interpolation=cv2.INTER_NEAREST).astype(bool)
			if settings['out_of_range'] == 'clip':
				meters[valid] = np.clip(meters[valid], settings['min_m'], settings['max_m'])
			else:
				valid &= (meters >= settings['min_m']) & (meters <= settings['max_m'])
			images.append(torch.from_numpy(meters.astype(np.float32)[None]))
			masks.append(torch.from_numpy(valid[None]))
		return torch.stack(images), torch.stack(masks)

	def get_bev(self, idx):
		if not self.train_config['model_conf'].get('BEV'):
			return None, None, None, None
		frame = Path(self.image_front[idx])
		scene = frame.parent.parent
		selected = set(self.task_settings['bev']['layers'])
		if self.bev_schema in (3,4):
			data = model_ground_truth(scene, frame.stem)
			validate_training_layers(data['metadata'], selected)
			return tuple(torch.from_numpy(data[name]) if name in selected else torch.zeros(1, 200, 200)
				for name in ('das', 'lane', 'vehicle', 'walker'))
		paths = {'das': 'das', 'lane': 'lane', 'vehicle': 'vehicle_mask', 'walker': 'walker_mask'}
		palette_path = self._aligned(idx, 'bev', required=False)
		palette = None
		result = []
		for name in ('das', 'lane', 'vehicle', 'walker'):
			if name not in selected:
				result.append(torch.zeros(1, 200, 200))
				continue
			path = self._aligned(idx, paths[name], required=False)
			if path is not None:
				mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
			elif palette_path is not None and name in ('das', 'lane'):
				if palette is None:
					stored = cv2.imread(str(palette_path))
					if stored is None:
						raise ValueError('Unreadable BEV fallback: ' + str(palette_path))
					palette = cv2.cvtColor(stored, cv2.COLOR_BGR2RGB)
				color = np.asarray((0, 0, 128 if name == 'das' else 192), dtype=np.uint8)
				mask = (np.all(palette == color, axis=-1).astype(np.uint8) * 255)
			else:
				raise ValueError('Missing selected BEV {}: {}/{}'.format(name, scene.name, frame.stem))
			if mask is None or mask.shape != (500, 500):
				raise ValueError('BEV {} must use the native 500x500, 0.2 m/px grid: {}'.format(name, path))
			mask = (mask[150:350, 150:350] == 255).astype(np.float32)
			result.append(torch.from_numpy(mask[None]))
		return tuple(result)

	def get_bev_aux(self, idx):
		"""Explicit simulator-GT tensors; never added to the model input tuple.

		Legacy captures have no recoverable road/actor heights and return None.
		Use validity with every height tensor; NaN means unknown, never zero.
		"""
		if self.bev_schema not in (3,4):
			return None
		frame = Path(self.image_front[idx])
		data = model_ground_truth(frame.parent.parent, frame.stem)
		return {name: (torch.from_numpy(value) if isinstance(value, np.ndarray) else value)
			for name, value in data.items() if name not in ('das', 'lane', 'vehicle', 'walker')}


	def get_lidar(self,idx):

		if not self.train_config.get('lidar'):
			return None,None
		lidar_path = self._aligned(idx, 'lidar', ('.npy',))

		lidar_data = np.load(lidar_path, allow_pickle=False)

		points = lidar_data[:, :3]

		num_points = points.shape[0]
		lidar_data = pad_or_trim_to_np(points, [81920, 5]).astype('float32')
		lidar_mask = np.ones(81920).astype('float32')
		lidar_mask[num_points:] *= 0.0

		return lidar_data, lidar_mask

	def resize_bbs(self, bbs, resize, idx=None):
		if bbs.numel() == 0:
			return bbs  # empty tensor

		x_scale, y_scale = resize

		# 좌표 스케일 적용
		bbs[:, 0] *= x_scale  # x1
		bbs[:, 2] *= x_scale  # x2
		bbs[:, 1] *= y_scale  # y1
		bbs[:, 3] *= y_scale  # y2

		idx_tensor = torch.full((bbs.shape[0], 1), 0, dtype=bbs.dtype, device=bbs.device)
		bbs = torch.cat((idx_tensor, bbs), dim=1)  # shape: (N, 6)

		return bbs

	def get_bbs_stack(self, idx):
		if not self.train_config['model_conf'].get('od'):
			return []
		bbs_list = []
		selected = self.task_settings['detection']['class_ids']
		for camera in self.sensor_keys:
			path = self._aligned(idx, camera + '_bbs', ('.npy',))
			data = np.load(str(path), allow_pickle=False)
			if data.size:
				if data.ndim != 2 or data.shape[1] != 5 or not np.isfinite(data).all():
					raise ValueError('Invalid N x 5 xyxy/class annotation: ' + str(path))
				data = data[np.isin(data[:, 4], selected)].copy()
				for index, class_id in enumerate(selected):
					data[data[:, 4] == class_id, 4] = -(index + 1)
				data[:, 4] = -data[:, 4] - 1
				rgb = cv2.imread(str(self._aligned(idx, camera + '_rgb')))
				resize = (self.W / rgb.shape[1], self.H / rgb.shape[0])
				tensor = self.resize_bbs(torch.from_numpy(data).float(), resize)
			else:
				tensor = torch.empty((0, 6), dtype=torch.float32)
			bbs_list.append(tensor)
		return bbs_list

	def __getitem__(self, idx):

		imgs, imgs_ori,post_rots,post_trans=self.get_img(idx)

		seg = self.get_seg(idx)
		depth,depth_mask = self.get_depth(idx)
		das, lane, vehicle, walker= self.get_bev(idx)

		lidar_data, lidar_mask=self.get_lidar(idx)
		extrinsic, intrinsic, rotation,translation = self.get_cam_para(idx)
		annots=self.get_bbs_stack(idx)



		return (imgs,imgs_ori,das, lane, vehicle, walker,lidar_data, lidar_mask,
		  post_rots,post_trans, extrinsic, intrinsic,
		  rotation,translation,seg,depth,depth_mask,annots)



# class_names = {
#     0:"vehicles",
#     1:"pedestrians",
#     2:"bicycles",
#     3:"trucks",

# }
