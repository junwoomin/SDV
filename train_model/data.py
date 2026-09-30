import glob
import torch
from torch.utils.data import Dataset
import torchvision
import cv2
import numpy as np
from pyquaternion import Quaternion
import os
import yaml
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
		if train:
			self.image_front = (sorted(glob.glob('data/train/*/front_cam_rgb/*')))
			self.image_left = (sorted(glob.glob('data/train/*/left_cam_rgb/*')))
			self.image_right = (sorted(glob.glob('data/train/*/right_cam_rgb/*')))

			self.image_rear = (sorted(glob.glob('data/train/*/rear_cam_rgb/*')))
			self.image_rear_left = (sorted(glob.glob('data/train/*/rear_left_cam_rgb/*')))
			self.image_rear_right = (sorted(glob.glob('data/train/*/rear_right_cam_rgb/*')))

			self.bbs_front = (sorted(glob.glob('data/train/*/front_cam_bbs/*')))
			self.bbs_left = (sorted(glob.glob('data/train/*/left_cam_bbs/*')))
			self.bbs_right = (sorted(glob.glob('data/train/*/right_cam_bbs/*')))

			self.bbs_rear = (sorted(glob.glob('data/train/*/rear_cam_bbs/*')))
			self.bbs_rear_left = (sorted(glob.glob('data/train/*/rear_left_cam_bbs/*')))
			self.bbs_rear_right = (sorted(glob.glob('data/train/*/rear_right_cam_bbs/*')))

			self.seg_front = (sorted(glob.glob('data/train/*/front_cam_seg/*')))
			self.seg_left = (sorted(glob.glob('data/train/*/left_cam_seg/*')))
			self.seg_right = (sorted(glob.glob('data/train/*/right_cam_seg/*')))

			self.seg_rear = (sorted(glob.glob('data/train/*/rear_cam_seg/*')))
			self.seg_rear_left = (sorted(glob.glob('data/train/*/rear_left_cam_seg/*')))
			self.seg_rear_right = (sorted(glob.glob('data/train/*/rear_right_cam_seg/*')))

			self.depth_front = (sorted(glob.glob('data/train/*/front_cam_depth/*')))
			self.depth_left = (sorted(glob.glob('data/train/*/left_cam_depth/*')))
			self.depth_right = (sorted(glob.glob('data/train/*/right_cam_depth/*')))

			self.depth_rear = (sorted(glob.glob('data/train/*/rear_cam_depth/*')))
			self.depth_rear_left = (sorted(glob.glob('data/train/*/rear_left_cam_depth/*')))
			self.depth_rear_right = (sorted(glob.glob('data/train/*/rear_right_cam_depth/*')))


			self.bev = (sorted(glob.glob('data/train/*/bev/*')))

			self.lidar_paths = (sorted(glob.glob('data/train/*/lidar/*')))
			
		else:
			self.image_front = (sorted(glob.glob('data/val/*/front_cam_rgb/*')))
			self.image_left = (sorted(glob.glob('data/val/*/left_cam_rgb/*')))
			self.image_right = (sorted(glob.glob('data/val/*/right_cam_rgb/*')))

			self.image_rear = (sorted(glob.glob('data/val/*/rear_cam_rgb/*')))
			self.image_rear_left = (sorted(glob.glob('data/val/*/rear_left_cam_rgb/*')))
			self.image_rear_right = (sorted(glob.glob('data/val/*/rear_right_cam_rgb/*')))

			self.bev = (sorted(glob.glob('data/val/*/bev/*')))

			self.lidar_paths = (sorted(glob.glob('data/val/*/lidar/*')))

			self.bbs_front = (sorted(glob.glob('data/val/*/front_cam_bbs/*')))
			self.bbs_left = (sorted(glob.glob('data/val/*/left_cam_bbs/*')))
			self.bbs_right = (sorted(glob.glob('data/val/*/right_cam_bbs/*')))

			self.bbs_rear = (sorted(glob.glob('data/val/*/rear_cam_bbs/*')))
			self.bbs_rear_left = (sorted(glob.glob('data/val/*/rear_left_cam_bbs/*')))
			self.bbs_rear_right = (sorted(glob.glob('data/val/*/rear_right_cam_bbs/*')))

			self.seg_front = (sorted(glob.glob('data/val/*/front_cam_seg/*')))
			self.seg_left = (sorted(glob.glob('data/val/*/left_cam_seg/*')))
			self.seg_right = (sorted(glob.glob('data/val/*/right_cam_seg/*')))

			self.seg_rear = (sorted(glob.glob('data/val/*/rear_cam_seg/*')))
			self.seg_rear_left = (sorted(glob.glob('data/val/*/rear_left_cam_seg/*')))
			self.seg_rear_right = (sorted(glob.glob('data/val/*/rear_right_cam_seg/*')))

			self.depth_front = (sorted(glob.glob('data/val/*/front_cam_depth/*')))
			self.depth_left = (sorted(glob.glob('data/val/*/left_cam_depth/*')))
			self.depth_right = (sorted(glob.glob('data/val/*/right_cam_depth/*')))

			self.depth_rear = (sorted(glob.glob('data/val/*/rear_cam_depth/*')))
			self.depth_rear_left = (sorted(glob.glob('data/val/*/rear_left_cam_depth/*')))
			self.depth_rear_right = (sorted(glob.glob('data/val/*/rear_right_cam_depth/*')))


		self.colors_bev = [(0, 0, 192),(0, 0, 128)] # Lane, DAS
		self.Lane, self.DAS = (0, 255, 0), (0, 0, 255)
		with open('data/sensor_config.yaml', 'r') as f:
			cfg = yaml.safe_load(f)

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

	def get_cam_para(self):
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
				self.sensor_config[cam]['x'], 
				self.sensor_config[cam]['y'],
				self.sensor_config[cam]['z'],
				self.sensor_config[cam]['pitch'],
				self.sensor_config[cam]['roll'],
				self.sensor_config[cam]['yaw'],
			]
			fov = self.sensor_config[cam]['fov']
			w = self.sensor_data['width']
			h = self.sensor_data['height']
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

		img_paths=[]
		if len(self.image_left) > idx and self.train_config['sensor_use']['left_cam']:
			img_paths.append(self.image_left[idx])
		if len(self.image_front) > idx and self.train_config['sensor_use']['front_cam']:
			img_paths.append(self.image_front[idx])
		if len(self.image_right) > idx and self.train_config['sensor_use']['right_cam']:
			img_paths.append(self.image_right[idx])
		if len(self.image_rear_left) > idx and self.train_config['sensor_use']['rear_left_cam']:
			img_paths.append(self.image_rear_left[idx])
		if len(self.image_rear) > idx and self.train_config['sensor_use']['rear_cam']:
			img_paths.append(self.image_rear[idx])
		if len(self.image_rear_right) > idx and self.train_config['sensor_use']['rear_right_cam']:
			img_paths.append(self.image_rear_right[idx])


		imgs = []
		imgs_ori=[]
		post_rots = []
		post_trans = []

		for path in img_paths:


			img = cv2.imread(path)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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



	def get_seg(self, idx):
		resize, resize_dims = self.sample_augmentation()

		paths = []
		if len(self.seg_left) > idx and self.train_config['sensor_use']['left_cam']: 
			paths.append(self.seg_left[idx])
		if len(self.seg_front) > idx and self.train_config['sensor_use']['front_cam']: 
			paths.append(self.seg_front[idx])
		if len(self.seg_right) > idx and self.train_config['sensor_use']['right_cam']: 
			paths.append(self.seg_right[idx])
		if len(self.seg_rear_left) > idx and self.train_config['sensor_use']['rear_left_cam']: 
			paths.append(self.seg_rear_left[idx])
		if len(self.seg_rear) > idx and self.train_config['sensor_use']['rear_cam']: 
			paths.append(self.seg_rear[idx])
		if len(self.seg_rear_right) > idx and self.train_config['sensor_use']['rear_right_cam']: 
			paths.append(self.seg_rear_right[idx])

		if not paths:
			return None

		color_map = self.CARLA_SEG_COLOR
		color_to_index = {tuple(v): i for i, (k, v) in enumerate(color_map.items())}
		num_classes = len(color_map)

		masks = []
		for path in paths:
			img = cv2.imread(path)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			img, _, _ = img_transform(img, resize, resize_dims)  # 이미지 증강
			h, w, _ = img.shape

			img_flat = img.reshape(-1, 3)
			mask_flat = np.array([color_to_index.get(tuple(pixel), color_to_index[tuple(color_map['Other'])]) for pixel in img_flat])
			mask = mask_flat.reshape(h, w)

			binary_masks = np.zeros((num_classes, h, w), dtype=np.uint8)
			for class_idx in range(num_classes):
				binary_masks[class_idx] = (mask == class_idx).astype(np.uint8)

			binary_masks = torch.tensor(binary_masks, dtype=torch.float32)
			masks.append(binary_masks)

		masks = torch.stack(masks)

		return masks

	
	def get_depth(self, idx):
		resize, resize_dims = self.sample_augmentation()


		paths=[]
		if len(self.depth_left) > idx and self.train_config['sensor_use']['left_cam']:
			paths.append(self.depth_left[idx])
		if len(self.depth_front) > idx and self.train_config['sensor_use']['front_cam']:
			paths.append(self.depth_front[idx])
		if len(self.depth_right) > idx and self.train_config['sensor_use']['right_cam']:
			paths.append(self.depth_right[idx])
		if len(self.depth_rear_left) > idx and self.train_config['sensor_use']['rear_left_cam']:
			paths.append(self.depth_rear_left[idx])
		if len(self.depth_rear) > idx and self.train_config['sensor_use']['rear_cam']:
			paths.append(self.depth_rear[idx])
		if len(self.depth_rear_right) > idx and self.train_config['sensor_use']['rear_right_cam']:
			paths.append(self.depth_rear_right[idx])


		if not paths:  
			return None, None  

		imgs = []
		masks = []

		for path in paths:
			img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # uint8로 읽힘
			img, _, _ = img_transform(img, resize, resize_dims)  # 이미지 증강
			img = img.astype(np.float32) * 100.0 / 255.0

			mask = img > 0.0  # 유효한 depth만 True
			img_tensor = ToTensor(img)  # (H, W) → Tensor (1, H, W) 또는 (H, W)
			mask_tensor = ToTensor(mask.astype(np.uint8)).bool()

			imgs.append(img_tensor)
			masks.append(mask_tensor)

		depth_tensor = torch.stack(imgs)     # (N, H, W)
		mask_tensor = torch.stack(masks)     # (N, H, W)

		return depth_tensor, mask_tensor


	
	def get_bev(self, idx):
		if not len(self.bev):
			return None,None,None,None
		bev_path = self.bev[idx]

		top = cv2.imread(bev_path)
		top = cv2.cvtColor(top, cv2.COLOR_BGR2RGB)

		das_mask = cv2.inRange(top, np.array((0, 0, 128), dtype="uint8"), np.array((0, 0, 128), dtype="uint8"))
		lane_mask = cv2.inRange(top, np.array((0, 0, 192), dtype="uint8"), np.array((0, 0, 192), dtype="uint8"))

		vehicle_mask = cv2.imread(bev_path.replace('bev', 'vehicle_mask'), cv2.IMREAD_GRAYSCALE)
		walker_mask = cv2.imread(bev_path.replace('bev', 'walker_mask'), cv2.IMREAD_GRAYSCALE)

		vehicle_mask = (vehicle_mask == 255).astype('uint8') * 255
		walker_mask = (walker_mask == 255).astype('uint8') * 255

		das_mask = das_mask[150:350, 150:350]
		lane_mask = lane_mask[150:350, 150:350]
		vehicle_mask = vehicle_mask[150:350, 150:350]
		walker_mask = walker_mask[150:350, 150:350]

		das_mask = ToTensor(das_mask)
		lane_mask = ToTensor(lane_mask)
		vehicle_mask = ToTensor(vehicle_mask)
		walker_mask = ToTensor(walker_mask)

		return das_mask, lane_mask, vehicle_mask, walker_mask



	def get_lidar(self,idx):

		if len(self.lidar_paths) <= 0:
			return None,None
		lidar_path = self.lidar_paths[idx]

		lidar_data = np.load(lidar_path, allow_pickle=True)  

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
		bbs_list = []

		resize, _ = self.sample_augmentation()


		paths=[]
		if len(self.bbs_left) > idx and self.train_config['sensor_use']['left_cam']:
			paths.append(self.bbs_left[idx])
		if len(self.bbs_front) > idx and self.train_config['sensor_use']['front_cam']:
			paths.append(self.bbs_front[idx])
		if len(self.bbs_right) > idx and self.train_config['sensor_use']['right_cam']:
			paths.append(self.bbs_right[idx])
		if len(self.bbs_rear_left) > idx and self.train_config['sensor_use']['rear_left_cam']:
			paths.append(self.bbs_rear_left[idx])
		if len(self.bbs_rear) > idx and self.train_config['sensor_use']['rear_cam']:
			paths.append(self.bbs_rear[idx])
		if len(self.bbs_rear_right) > idx and self.train_config['sensor_use']['rear_right_cam']:
			paths.append(self.bbs_rear_right[idx])

		for lst in paths:
			data = np.load(lst)

			if data.size > 0:
				tensor = torch.from_numpy(data).float()
				tensor = self.resize_bbs(tensor, resize) 
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
		extrinsic, intrinsic, rotation,translation = self.get_cam_para()
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