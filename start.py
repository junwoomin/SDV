import subprocess
import sys
import re

def install_and_import(package, import_name=None):
    try:
        if import_name:
            __import__(import_name)
        else:
            # Remove version info for import
            base_package = package.split("==")[0].split("=")[0]
            __import__(base_package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        if import_name:
            globals()[import_name] = __import__(import_name)
        else:
            base_package = package.split("==")[0].split("=")[0]
            globals()[base_package] = __import__(base_package)
packages = [
    ("pygame==2.0.0",),
    ("py_trees==0.6.5",),
    ("pyyaml==6.0.2", "yaml"),
    ("Pillow==10.4.0", "PIL"),
    ("psutil==6.0.0",),
    ("numpy==1.22.4",),
    ("dictor==0.1.12",),
    ("requests==2.32.3",),
    ("opencv-python==4.5.2.54", "cv2"), 
    ("tabulate==0.9.0",),
    ("carla==0.9.15",),
    ("six==1.16.0",),
    ("shapely==2.0.7",),
    ("ephem==4.2",),
    ("omegaconf==2.3.0",),
    ("gym==0.26.2",),
    ("h5py==3.11.0",),
    ("networkx==3.1",),
    ("hydra-core==1.3.2", "hydra"),
    ("pyquaternion==0.9.9",),
    ("efficientnet_pytorch==0.7.1",),
    ("pandas==2.0.2",),
    ("webcolors",),
]



for item in packages:
    if len(item) == 1:
        install_and_import(item[0])
    else:
        install_and_import(item[0], item[1])

def is_package_installed(package_name):
    result = subprocess.run(
        ["dpkg", "-s", package_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    return result.returncode == 0

def install_if_missing(package_list):
    for package in package_list:
        if not is_package_installed(package):
            subprocess.run(["sudo", "apt", "install", "-y", package], check=True)

packages = ["fonts-nanum", "libomp5", "curl"]

install_if_missing(packages)

def get_cuda_version():
    try:
        output = subprocess.check_output(["nvcc", "-V"], encoding='utf-8')
        match = re.search(r'release (\d+\.\d+)', output)
        if match:
            return match.group(1)  # '11.8'만 추출
        return None
    except Exception:
        return None


import threading
import time
import os
import pygame
import yaml
import urllib.request
import cv2
import glob
from pathlib import Path
import numpy as np
from utils import *
import copy 
import warnings
import shutil
import pandas as pd
import math
from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings("ignore", category=UserWarning)

class_names = {
    0:"vehicles",
    1:"pedestrians",
    2:"bicycles",
    3:"trucks",

}

def install_carla():
    global install_progress, install_message, installing, state

    installing = True
    try:
        carla_url = "https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz"
        maps_url  = "https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_0.9.15.tar.gz"

        install_message = "CARLA 기본 패키지 다운로드 중..."
        urllib.request.urlretrieve(carla_url, "CARLA_0.9.15.tar.gz", reporthook=download_hook(0.0, 0.4))

        install_message = "추가 맵 다운로드 중..."
        urllib.request.urlretrieve(maps_url, "AdditionalMaps_0.9.15.tar.gz", reporthook=download_hook(0.4, 0.7))

        install_message = "압축 해제 중..."
        install_progress = 0.75
        subprocess.run(["tar", "-xf", "CARLA_0.9.15.tar.gz"], check=True)
        install_progress = 0.85
        subprocess.run(["tar", "-xf", "AdditionalMaps_0.9.15.tar.gz"], check=True)
        install_progress = 0.95

        os.remove("CARLA_0.9.15.tar.gz")
        os.remove("AdditionalMaps_0.9.15.tar.gz")

        install_progress = 1.0
        install_message = "설치 완료!"
        os.chdir("..")
        time.sleep(1)
        state = Town_sensor_Select

    except Exception as e:
        install_message = f"설치 실패: {str(e)}"
    installing = False
def download_hook(start_ratio, end_ratio):
    def hook(block_num, block_size, total_size):
        global install_progress
        if total_size > 0:
            progress = min(block_num * block_size / total_size, 1.0)
            install_progress = start_ratio + progress * (end_ratio - start_ratio)
    return hook

def draw_checkbox(rect, checked, label, enabled=True):
    color = BLACK if enabled else GRAY
    pygame.draw.rect(screen, color, rect, 2)
    if checked:
        pygame.draw.line(screen, color, (rect.x + 4, rect.y + rect.height // 2), (rect.x + rect.width // 2, rect.y + rect.height - 4), 3)
        pygame.draw.line(screen, color, (rect.x + rect.width // 2, rect.y + rect.height - 4), (rect.x + rect.width - 4, rect.y + 4), 3)
    label_surf = font.render(label, True, color)
    screen.blit(label_surf, (rect.right + 10, rect.y - 2))
def draw_input_box(rect, value, active):
    pygame.draw.rect(screen, WHITE, rect)
    pygame.draw.rect(screen, BLUE if active else BLACK, rect, 2)
    txt = str(value)
    txt_surf = smallfont.render(txt, True, BLACK)
    screen.blit(txt_surf, (rect.x + 6, rect.y + 4))
    


new = False                          # 새 게임 여부
dragging = False                     # 드래그 상태
installing = False                   # 설치 진행 여부
install_progress = 0                 # 설치 진행률
install_message = "설치를 준비 중입니다..."  # 설치 메시지
current_dir = os.getcwd()            # 현재 작업 디렉토리
running = True                       # 게임 실행 상태
state = -1                           # 현재 상태

# 상수
SAVE_PATH = 'data/train/'            # 데이터 저장 경로
WINDOW_WIDTH, WINDOW_HEIGHT = 1400, 1000  # 창 크기

# 이미지 자산

car_top = pygame.image.load(f"{current_dir}/img/car_top.jpg")  # 차량 상단 이미지
car_side = pygame.image.load(f"{current_dir}/img/car_side.jpg")  # 차량 측면 이미지
CAR_TOP_W = 400                      # 차량 상단 이미지 너비
CAR_TOP_H = int(car_top.get_height() * CAR_TOP_W / car_top.get_width())  # 차량 상단 이미지 높이
CAR_SIDE_W = 400                     # 차량 측면 이미지 너비
CAR_SIDE_H = int(car_side.get_height() * CAR_SIDE_W / car_side.get_width())  # 차량 측면 이미지 높이
car_top = pygame.transform.scale(car_top, (CAR_TOP_W, CAR_TOP_H))  # 크기 조정된 상단 이미지
car_side = pygame.transform.scale(car_side, (CAR_SIDE_W, CAR_SIDE_H))  # 크기 조정된 측면 이미지
car_top_rect = car_top.get_rect(topleft=(50, 150))  # 상단 이미지 위치
car_side_rect = car_side.get_rect(topleft=(670, 200))  # 측면 이미지 위치


# 색상 정의
WHITE = (255, 255, 255)              # 흰색
BLACK = (0, 0, 0)                    # 검정색
GRAY = (180, 180, 180)               # 회색
BLUE = (0, 120, 255)                 # 파란색
RED = (255, 60, 60)                  # 빨간색
colors = {2: (255, 0, 0), 3: (255, 255, 0), 4: (0, 255, 0)}  # 객체 색상 매핑
colors_bev = [(0, 0, 192), (0, 0, 128)]  # BEV용 색상 (Lane, DAS)
Y, gray_ = (255, 255, 0), (80, 80, 80)  # 노란색, 회색

# 경로 리스트 (카메라 뷰 및 데이터 유형별)
left_rgb_paths = []                  # 좌측 RGB 경로
front_rgb_paths = []                 # 전면 RGB 경로
right_rgb_paths = []                 # 우측 RGB 경로
rear_rgb_paths = []                  # 후면 RGB 경로
rear_right_rgb_paths = []            # 후우측 RGB 경로
rear_left_rgb_paths = []             # 후좌측 RGB 경로

left_seg_paths = []                  # 좌측 세그먼테이션 경로
front_seg_paths = []                 # 전면 세그먼테이션 경로
right_seg_paths = []                 # 우측 세그먼테이션 경로
rear_seg_paths = []                  # 후면 세그먼테이션 경로
rear_right_seg_paths = []            # 후우측 세그먼테이션 경로
rear_left_seg_paths = []             # 후좌측 세그먼테이션 경로

left_depth_paths = []                # 좌측 깊이 경로
front_depth_paths = []               # 전면 깊이 경로
right_depth_paths = []               # 우측 깊이 경로
rear_depth_paths = []                # 후면 깊이 경로
rear_right_depth_paths = []          # 후우측 깊이 경로
rear_left_depth_paths = []           # 후좌측 깊이 경로

left_bbs_paths = []               # 좌측 2D 경계 상자 경로
front_bbs_paths = []              # 전면 2D 경계 상자 경로
right_bbs_paths = []              # 우측 2D 경계 상자 경로
rear_bbs_paths = []               # 후면 2D 경계 상자 경로
rear_right_bbs_paths = []         # 후우측 2D 경계 상자 경로
rear_left_bbs_paths = []          # 후좌측 2D 경계 상자 경로

# 입력 변수
town_input = "1"                     # 타운 입력
image_num_input = "1"                # 이미지 번호 입력
data_town_num = '100000'             # 데이터 타운 번호
input_img_width = '224'              # 입력 이미지 너비
input_img_height = '448'             # 입력 이미지 높이
train_modal_select_input = None      # 학습 모달 선택 입력
selected_mode = "RGB"                # 선택된 모드
OD_mode=False
train_input_mode = 'RGB'             # 학습 입력 모드
metric_vis_mode = 'BEV'              # 메트릭 시각화 모드
metric_vis_img_mode = 'graph'        # 메트릭 시각화 이미지 모드
input_Epochs = '20'                  # 에포크 수
input_batch = '4'                    # 배치 크기
input_LR = '0.001'                   # 학습률
hyperparameter_input = None          # 하이퍼파라미터 입력
town = 'None'                        # 선택된 타운
data_gan_num = 0                     # GAN 데이터 번호
count_data_gan = 0                   # GAN 데이터 카운트
epoch=0
# 버튼 색상 상태
bev_modal_color = RED                # BEV 모달 색상
seg_modal_color = RED                # 세그먼테이션 모달 색상
depth_modal_color = RED              # 깊이 모달 색상
OD_modal_color = GRAY                # 객체 탐지 모달 색상
Front_cam_color = GRAY               # 전면 카메라 색상
Left_cam_color = GRAY                # 좌측 카메라 색상
Right_cam_color = GRAY               # 우측 카메라 색상
Rear_cam_color = GRAY                # 후면 카메라 색상
Rear_right_cam_color = GRAY          # 후우측 카메라 색상
Rear_left_cam_color = GRAY           # 후좌측 카메라 색상
lidar_color = GRAY                   # 라이다 색상

# 플래그
not_next = True                      # 다음으로 진행하지 않음 플래그
check_resnet_encoder = False         # ResNet 인코더 체크
check_efficientnet_encoder = False   # EfficientNet 인코더 체크
check_fpn_decoder = False         # ResNet 디코더 체크
check_bifpn_decoder = False   # EfficientNet 디코더 체크
check_resnet_bev_decoder = False     # ResNet BEV 디코더 체크
check_efficientnet_bev_decoder = False  # EfficientNet BEV 디코더 체크
carla_data_gan_code = False          # CARLA GAN 코드 실행 플래그
train_code = False                   # 학습 코드 실행 플래그
max_idx = False                   

# 상태 상수
MENU = -1                            # 메뉴 상태
CARAL_INSTALL = -2                   # CARLA 설치 상태
Town_sensor_Select = 0               # 타운 및 센서 선택 상태
model_test = 1
sensor_position = 2                  # 센서 위치 상태
data_gan = 3                         # 데이터 GAN 상태

hyperparameter = 4                   # 하이퍼파라미터 상태
train_modal_select = 5               # 학습 모달 선택 상태
model_test2 = 6                       # 모델 테스트 상태
Train_loop = 7                       # 모델 테스트 상태

Train = 8                            # 학습 상태

metric = 9                           # 메트릭 상태
metric_vis = 10                       # 메트릭 시각화 상태
re_Train = 11                        # 재학습 상태
predict = 12                         # 예측 상태

# 버튼 사각형
loop_button = pygame.Rect(WINDOW_WIDTH - 600, WINDOW_HEIGHT - 80, 120, 50)  # 뒤로가기 버튼
back_button = pygame.Rect(WINDOW_WIDTH - 160, WINDOW_HEIGHT - 80, 120, 50)  # 뒤로가기 버튼
confirm_button = pygame.Rect(WINDOW_WIDTH - 300, WINDOW_HEIGHT - 80, 120, 50)  # 확인 버튼
home_button = pygame.Rect(WINDOW_WIDTH - 440, WINDOW_HEIGHT - 80, 120, 50)  # 홈 버튼
home2_button = pygame.Rect(WINDOW_WIDTH - 160, WINDOW_HEIGHT - 80, 120, 50)  # 홈 버튼 2
end_button = pygame.Rect(WINDOW_WIDTH - 160, WINDOW_HEIGHT - 80, 120, 50)  # 종료 버튼

set_button = pygame.Rect(WINDOW_WIDTH - 600, WINDOW_HEIGHT - 80, 120, 50)  # 홈 버튼

# 메뉴 버튼
data_gan_button = pygame.Rect(WINDOW_WIDTH - 1200, WINDOW_HEIGHT - 800, 200, 60)  # GAN 데이터 버튼
training_button = pygame.Rect(WINDOW_WIDTH - 1200, WINDOW_HEIGHT - 650, 200, 60)  # 학습 버튼
metric_button = pygame.Rect(WINDOW_WIDTH - 1200, WINDOW_HEIGHT - 500, 200, 60)  # 메트릭 버튼
predict_button = pygame.Rect(WINDOW_WIDTH - 1200, WINDOW_HEIGHT - 350, 200, 60)  # 예측 버튼

# 시각화 버튼
RGB_button = pygame.Rect(WINDOW_WIDTH - 1300, WINDOW_HEIGHT - 200, 150, 60)  # RGB 버튼
SEG_button = pygame.Rect(WINDOW_WIDTH - 1100, WINDOW_HEIGHT - 200, 150, 60)  # 세그먼테이션 버튼
Depth_button = pygame.Rect(WINDOW_WIDTH - 900, WINDOW_HEIGHT - 200, 150, 60)  # 깊이 버튼
OD_button = pygame.Rect(WINDOW_WIDTH - 700, WINDOW_HEIGHT - 200, 150, 60)  # 깊이 버튼
metric_vis_BEV_button = pygame.Rect(WINDOW_WIDTH - 1350, WINDOW_HEIGHT - 250, 100, 60)  # BEV 시각화 버튼
metric_vis_SEG_button = pygame.Rect(WINDOW_WIDTH - 1235, WINDOW_HEIGHT - 250, 100, 60)  # 세그먼테이션 시각화 버튼
metric_vis_Depth_button = pygame.Rect(WINDOW_WIDTH - 1120, WINDOW_HEIGHT - 250, 100, 60)  # 깊이 시각화 버튼

metric_vis_graph_button = pygame.Rect(WINDOW_WIDTH - 1350, WINDOW_HEIGHT - 150, 100, 60)  # 그래프 시각화 버튼
metric_vis_img_button = pygame.Rect(WINDOW_WIDTH - 1235, WINDOW_HEIGHT - 150, 100, 60)  # 이미지 시각화 버튼
metric_vis_score_button = pygame.Rect(WINDOW_WIDTH - 1120, WINDOW_HEIGHT - 150, 100, 60)  # 이미지 시각화 버튼


metric_R_button = pygame.Rect(WINDOW_WIDTH - 150, (WINDOW_HEIGHT / 2) - 75, 150, 150)  # 오른쪽 메트릭 버튼
metric_L_button = pygame.Rect(0, (WINDOW_HEIGHT / 2) - 75, 150, 150)  # 왼쪽 메트릭 버튼

# 기타 사각형
town_rect = pygame.Rect(WINDOW_WIDTH - 1350, WINDOW_HEIGHT - 950, 130, 40)  # 타운 입력 사각형
Cam_all = checkbox_rect(-1, 0, top_left=(580, 640))  # 모든 카메라 체크박스
od_all = pygame.Rect(1060, 580, 24, 24)  # 모든 깊이 체크박스
Depth_all = pygame.Rect(890, 580, 24, 24)  # 모든 깊이 체크박스
seg_all = pygame.Rect(750, 580, 24, 24)  # 모든 세그먼테이션 체크박스
Town_all = checkbox_rect(-1, 0, top_left=(80, 640))  # 모든 타운 체크박스
data_num_rect = pygame.Rect(100, 700, 280, 40)  # 데이터 번호 입력 사각형
data_num_all_rect = pygame.Rect(100, 800, 280, 40)  # 모든 데이터 번호 입력 사각형

# 학습 모델 설정 사각형
width_rect = pygame.Rect(WINDOW_WIDTH - 1300, WINDOW_HEIGHT - 600, 180, 40)  # 너비 입력 사각형
height_rect = pygame.Rect(WINDOW_WIDTH - 1300, WINDOW_HEIGHT - 540, 180, 40)  # 높이 입력 사각형
RGB_modal_text = pygame.Rect(WINDOW_WIDTH - 1240, WINDOW_HEIGHT - 650, 80, 60)  # RGB 모달 텍스트
train_modal_text = pygame.Rect(WINDOW_WIDTH - 1280, WINDOW_HEIGHT - 700, 100, 50)  # 학습 모달 텍스트
Front_cam_rect = pygame.Rect(WINDOW_WIDTH - 1350, WINDOW_HEIGHT - 400, 180, 50)  # 전면 카메라 사각형
Left_cam_rect = pygame.Rect(WINDOW_WIDTH - 1350, WINDOW_HEIGHT - 340, 180, 50)  # 좌측 카메라 사각형
Right_cam_rect = pygame.Rect(WINDOW_WIDTH - 1350, WINDOW_HEIGHT - 280, 180, 50)  # 우측 카메라 사각형
Rear_cam_rect = pygame.Rect(WINDOW_WIDTH - 1150, WINDOW_HEIGHT - 400, 180, 50)  # 후면 카메라 사각형
Rear_right_cam_rect = pygame.Rect(WINDOW_WIDTH - 1150, WINDOW_HEIGHT - 340, 215, 50)  # 후우측 카메라 사각형
Rear_left_cam_rect = pygame.Rect(WINDOW_WIDTH - 1150, WINDOW_HEIGHT - 280, 215, 50)  # 후좌측 카메라 사각형
lidar_rect = pygame.Rect(WINDOW_WIDTH - 950, WINDOW_HEIGHT - 400, 120, 50)  # 라이다 사각형
LR_rect = pygame.Rect(WINDOW_WIDTH - 1300, WINDOW_HEIGHT - 700, 180, 40)  # 학습률 입력 사각형
batch_rect = pygame.Rect(WINDOW_WIDTH - 1100, WINDOW_HEIGHT - 700, 180, 40)  # 배치 입력 사각형
Epochs_rect = pygame.Rect(WINDOW_WIDTH - 900, WINDOW_HEIGHT - 700, 180, 40)  # 에포크 입력 사각형

# 출력 설정 사각형
output_modal_text = pygame.Rect(WINDOW_WIDTH - 300, WINDOW_HEIGHT - 890, 150, 50)  # 출력 모달 텍스트
output_box = pygame.Rect(WINDOW_WIDTH - 350, WINDOW_HEIGHT - 910, 250, 600)  # 출력 박스
third_height = output_box.height // 3  # 출력 박스 높이의 1/3
bev_button_modal = pygame.Rect(output_box.left + 75, output_box.top + 80, 100, 60)  # BEV 버튼
seg_button_modal = pygame.Rect(output_box.left + 10, output_box.top + third_height + 10, 100, 60)  # 세그먼테이션 버튼
depth_button_modal = pygame.Rect(output_box.left + 140, output_box.top + third_height + 10, 100, 60)  # 깊이 버튼
OD_button_modal = pygame.Rect(output_box.left + 75, output_box.top + 2 * third_height + 10, 100, 60)  # 객체 탐지 버튼

# 체크박스 사각형
rect_resnet_encoder = checkbox_rect(-1, 0, top_left=(WINDOW_WIDTH - 1000, WINDOW_HEIGHT - 550))  # ResNet 인코더 체크박스
rect_efficent_encoder = checkbox_rect(-1, 0, top_left=(WINDOW_WIDTH - 1000, WINDOW_HEIGHT - 500))  # EfficientNet 인코더 체크박스
rect_resnet_decoder = checkbox_rect(-1, 0, top_left=(WINDOW_WIDTH - 600, WINDOW_HEIGHT - 550))  # ResNet 디코더 체크박스
rect_efficent_decoder = checkbox_rect(-1, 0, top_left=(WINDOW_WIDTH - 600, WINDOW_HEIGHT - 500))  # EfficientNet 디코더 체크박스
rect_resnet_bev_decoder = checkbox_rect(-1, 0, top_left=(WINDOW_WIDTH - 600, WINDOW_HEIGHT - 800))  # ResNet BEV 디코더 체크박스
rect_efficent_bev_decoder = checkbox_rect(-1, 0, top_left=(WINDOW_WIDTH - 600, WINDOW_HEIGHT - 750))  # EfficientNet BEV 디코더 체크박스

# 추가 버튼
R_button = pygame.Rect(WINDOW_WIDTH / 2, WINDOW_HEIGHT - 930, 150, 60)  # 오른쪽 버튼
L_button = pygame.Rect(WINDOW_WIDTH / 2 - 150, WINDOW_HEIGHT - 930, 150, 60)  # 왼쪽 버튼

# 센서 설정
SENSOR_CONFIG_rect_start_point = [50, 650]  # 센서 설정 시작 지점

# 스크롤바 설정
scrollbar_rect = pygame.Rect(WINDOW_WIDTH - 1300, WINDOW_HEIGHT - 70, 1100, 10)  # 스크롤바
handle_width = 20
scroll_x = 0
max_scroll_value = 1
handle_x = scrollbar_rect.x + int(scroll_x / max_scroll_value * (scrollbar_rect.width - handle_width))
handle_rect = pygame.Rect(handle_x, scrollbar_rect.y - 5, handle_width, 20)  # 스크롤바 핸들
scrollbar_rect_vis = pygame.Rect(WINDOW_WIDTH - 900, WINDOW_HEIGHT - 125, 800, 10)  # 시각화 스크롤바
handle_width_vis = 20
scroll_x_vis = 1
max_scroll_value_vis = 100
handle_x_vis = scrollbar_rect_vis.x + int(scroll_x_vis / max_scroll_value_vis * (scrollbar_rect_vis.width - handle_width_vis))
handle_rect_vis = pygame.Rect(handle_x, scrollbar_rect_vis.y - 5, handle_width_vis, 20)  # 시각화 스크롤바 핸들

# 모델 테스트 입력 이미지
model_test_input_img = [None] * 9
model_test_input_img_text = pygame.Rect(WINDOW_WIDTH - 1190, WINDOW_HEIGHT - 750, 100, 50)
box_w, box_h = 150, 100
last_row_w, last_row_h = 150, 150
start_x, start_y = WINDOW_WIDTH - 1350, WINDOW_HEIGHT - 700
gap_x, gap_y = 10, 20
model_test_input_img_boxes = []
model_test_input_img_none = []
for i in range(9):
    row = i // 3
    col = i % 3
    w, h = (last_row_w, last_row_h) if row == 2 else (box_w, box_h)
    x = start_x + col * (box_w + gap_x)
    y = start_y + row * (box_h + gap_y)
    rect = pygame.Rect(x, y, w, h)
    model_test_input_img_boxes.append(rect)
    text_pos = (x + w / 2 - 33, y + h / 2 - 20)
    model_test_input_img_none.append(text_pos)

# 모델 테스트 출력 이미지
model_test_output_img = [None] * 21
model_test_output_img_text = pygame.Rect(WINDOW_WIDTH - 290, WINDOW_HEIGHT - 820, 100, 50)
box_w, box_h = 150, 100
last_row_w, last_row_h = 100, 100
start_x, start_y = WINDOW_WIDTH - 500, WINDOW_HEIGHT - 780
gap_x, gap_y = 10, 20
model_test_output_img_boxes = []
model_test_output_img_none = []
for i in range(15):
    row = i // 3
    col = i % 3
    w, h = (last_row_w, last_row_h) if row == 6 else (box_w, box_h)
    offset = (box_w - w) // 2 if row == 6 else 0
    x = start_x + col * (box_w + gap_x) + offset
    y = start_y + row * (box_h + gap_y)
    rect = pygame.Rect(x, y, w, h)
    model_test_output_img_boxes.append(rect)
    text_pos = (x + w / 2 - 32, y + h / 2 - 20)
    model_test_output_img_none.append(text_pos)

# 데이터 GAN 입력 이미지
model_data_gan_input_img = [None] * 8
model_data_gan_input_img_text = pygame.Rect(WINDOW_WIDTH - 1190, WINDOW_HEIGHT - 750, 100, 50)
box_w, box_h = 300, 250
start_x, start_y = WINDOW_WIDTH - 1300, WINDOW_HEIGHT - 800
gap_x, gap_y = 10, 20
model_data_gan_input_img_boxes = []
model_data_gan_input_img_none = []
for i in range(8):
    row = i // 4
    col = i % 4
    x = start_x + col * (box_w + gap_x)
    y = start_y + row * (box_h + gap_y)
    rect = pygame.Rect(x, y, box_w, box_h)
    model_data_gan_input_img_boxes.append(rect)
    text_pos = (x + box_w / 2 - 40, y + box_h / 2 - 20)
    model_data_gan_input_img_none.append(text_pos)

# 학습 입력 이미지
model_train_input_img = [None] * 8
box_w, box_h = 300, 300
start_x, start_y = WINDOW_WIDTH - 1300, WINDOW_HEIGHT - 750
gap_x, gap_y = 10, 10
model_train_input_img_boxes = []
for i in range(8):
    row = i // 4
    col = i % 4
    x = start_x + col * (box_w + gap_x)
    y = start_y + row * (box_h + gap_y)
    rect = pygame.Rect(x, y, box_w, box_h)
    model_train_input_img_boxes.append(rect)

# 메트릭 시각화 입력 이미지
metric_vis_input_img = [None] * 8
box_w, box_h = 250, 250
start_x, start_y = WINDOW_WIDTH - 1200, WINDOW_HEIGHT - 800
gap_x, gap_y = 10, 10
metric_vis_input_img_boxes = []
for i in range(8):
    row = i // 4
    col = i % 4
    x = start_x + col * (box_w + gap_x)
    y = start_y + row * (box_h + gap_y)
    rect = pygame.Rect(x, y, box_w, box_h)
    metric_vis_input_img_boxes.append(rect)

metric_vis_text = []
num_cols = 3
num_rows = 7
box_w, box_h = 160, 70
gap_x, gap_y = 0, 0
start_x, start_y = WINDOW_WIDTH - 1200, WINDOW_HEIGHT - 780
for i in range(num_cols * num_rows):
    row = i // num_cols
    col = i % num_cols
    x = start_x + col * (box_w + gap_x)
    y = start_y + row * (box_h + gap_y)
    rect = pygame.Rect(x, y, box_w, box_h)
    metric_vis_text.append(rect)

metric_vis_text2 = []
gap_x, gap_y = 0, 0
start_x, start_y = WINDOW_WIDTH - 670, WINDOW_HEIGHT - 780
for i in range(num_cols * num_rows):
    row = i // num_cols
    col = i % num_cols
    x = start_x + col * (box_w + gap_x)
    y = start_y + row * (box_h + gap_y)
    rect = pygame.Rect(x, y, box_w, box_h)
    metric_vis_text2.append(rect)

box_w, box_h = 550, 350
sensor_img = None
sensor_img_rect = pygame.Rect(WINDOW_WIDTH - 1350, WINDOW_HEIGHT - 400, box_w, box_h)



metric_vis_img = pygame.Rect(300, 100, 800, 600)  # 메트릭 시각화 이미지

# Pygame 초기화
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("CARLA Config UI")
clock = pygame.time.Clock()

# 폰트 설정
font_small = pygame.font.Font("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", 17)
font = pygame.font.Font("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", 28)
font_2 = pygame.font.Font("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", 30)

font_big = pygame.font.Font("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", 38)
smallfont = pygame.font.SysFont("arial", 22)

# 타운 및 센서 리스트
towns = ["Town01", "Town02", "Town03", "Town04", "Town05", "Town06", "Town07", "Town10"]
main_towns = [t for t in towns if "_val" not in t]
sensors = ["front_cam","front_cam_pv", "left_cam", "right_cam", "rear_cam", "rear_right_cam", "rear_left_cam", "lidar"]
options = ["seg", "depth",'od']
output_options = ["segmentation", "depth", "2d_Object_Detection", "3d_Object_Detection", 'Bird-Eye-View']

# 기본 센서 설정
configs = {
    "front_cam_pv": {
        "x": 1.7,
        "y": 0.0,
        "z": 1.54,
        "seg": False,
        "depth": False,
        "od": False,
        "width": 1280,
        "height": 720,
        "fov": 100.0,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0
    },
    "front_cam": {
        "x": 1.7,
        "y": 0.0,
        "z": 1.54,
        "seg": False,
        "depth": False,
        "od": False,
        "width": 400,
        "height": 300,
        "fov": 64.6,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0
    },
    "left_cam": {
        "x": 1.53,
        "y": -0.78,
        "z": 1.52,
        "seg": False,
        "depth": False,
        "od": False,
        "width": 400,
        "height": 300,
        "fov": 64.6,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": -45.0
    },
    "right_cam": {
        "x": 1.53,
        "y": 0.78,
        "z": 1.52,
        "seg": False,
        "depth": False,
        "od": False,
        "width": 400,
        "height": 300,
        "fov": 64.6,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 45.0
    },
    "rear_cam": {
        "x": -1.0,
        "y": 0.0,
        "z": 1.54,
        "seg": False,
        "depth": False,
        "od": False,
        "width": 400,
        "height": 300,
        "fov": 64.6,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 180.0
    },
    "rear_right_cam": {
        "x": -0.78,
        "y": 0.78,
        "z": 1.51,
        "seg": False,
        "depth": False,
        "od": False,
        "width": 400,
        "height": 300,
        "fov": 64.6,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 135.0
    },
    "rear_left_cam": {
        "x": -0.78,
        "y": -0.78,
        "z": 1.51,
        "seg": False,
        "depth": False,
        "od": False,
        "width": 400,
        "height": 300,
        "fov": 64.6,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": -135.0
    },
    "lidar": {
        "x": 0.0,
        "y": 0.0,
        "z": 1.73,
        "seg": False,
        "depth": False,
        "od": False,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0
    }
}
configs_ori = configs

# 옵티마이저 및 저장 옵션
opt = ["Adam", "AdamW", "SGD"]
save = ["Best_Loss", "Best_iou", "Last_Epoch"]
save_other = ["DeepX"]

# 선택 및 UI 관련 변수
confirm_delete = False               # 삭제 확인 플래그
confirm_delete2=False
setting = False               
setting2 = False               

delete_target = None                 # 삭제 대상
delete_name = ""                     # 삭제 이름
clickable_buttons = []               # 클릭 가능한 버튼 리스트
Training_result=[]
selected_towns = []                  # 선택된 타운
selected_sensors = {}                # 선택된 센서
delete_buttons = []                  # 삭제 버튼 리스트
delete_buttons2=[]
text_input = ""                      # 텍스트 입력
active_input = None                  # 활성 입력
current_sensor_idx = 0               # 현재 센서 인덱스
buttons_per_page = 9                 # 페이지당 버튼 수
button_width, button_height = 350, 220  # 버튼 크기

# 중앙 사각형
center_x = WINDOW_WIDTH // 2
center_y = WINDOW_HEIGHT // 2 
box_w, box_h = 1250, 760
box_x = center_x - box_w // 2
box_y = (center_y - box_h // 2 ) + 20
central_rect = pygame.Rect(box_x, box_y, box_w, box_h)  # 중앙 사각형
re_button = pygame.Rect(center_x - 150, center_y - 150, 300, 300)  # 재설정 버튼
gap_x, gap_y = 45, 20                # 버튼 간격



button_gap = 20                      # 버튼 간격
page = 0                             # 현재 페이지
total_pages = 0                      # 총 페이지 수
buttons_per_row = 3                  # 행당 버튼 수
buttons_per_col = 3                  # 열당 버튼 수
vis_start_x, vis_start_y = 750, 750  # 시각화 버튼 시작 위치
vis_buttons_row, vis_buttons_col = 2, 3  # 시각화 버튼 행/열
selected_btn = 0                     # 선택된 버튼
vis_button_width, vis_button_height = 170, 60  # 시각화 버튼 크기
vis_gap_x, vis_gap_y = 20, 20       # 시각화 버튼 간격
vis_bev_buttons = []
Training_loop=[]
for i in range(6):
    row = i // vis_buttons_col
    col = i % vis_buttons_col
    btn_x = vis_start_x + col * (vis_button_width + vis_gap_x)
    btn_y = vis_start_y + row * (vis_button_height + vis_gap_y)
    btn_rect = pygame.Rect(btn_x, btn_y, vis_button_width, vis_button_height)
    vis_bev_buttons.append(btn_rect)
selected_idx_metric = 0              # 선택된 메트릭 인덱스
vis_bev = ['Loss', 'mIoU', 'Das_IoU', 'Walker_IoU', 'Vehicle_IoU', 'Lane_IoU']  # BEV 시각화 옵션

# 체크박스 상태
town_checkboxes = {t: False for t in towns}  # 타운 체크박스

sensor_checkboxes = {s: False for s in sensors}  # 센서 체크박스
option_checkboxes = {s: {o: False for o in options} for s in sensors}  # 옵션 체크박스

save_other_checkboxes = {s: False for s in save_other}  # 기타 저장 체크박스
save_checkboxes = {s: False for s in save}  # 저장 체크박스
opt_checkboxes = {s: False for s in opt}  # 옵티마이저 체크박스

# 선택된 결과 및 설정
selected_result = None               # 선택된 결과
selected_train_config = None         # 선택된 학습 설정
selected_sensor_config = None        # 선택된 센서 설정

BEV_metrics = {
    'train_loss': ('Loss', 'Train'),
    'val_loss': ('Loss', 'Validation'),
    'train_miou': ('mIoU', 'Train'),
    'val_miou': ('mIoU', 'Validation'),
    'train_vehicle_iou': ('Vehicle_IoU', 'Train'),
    'val_vehicle_iou': ('Vehicle_IoU', 'Validation'),
    'train_walker_iou': ('Walker_IoU', 'Train'),
    'val_walker_iou': ('Walker_IoU', 'Validation'),
    'train_lane_iou': ('Lane_IoU', 'Train'),
    'val_lane_iou': ('Lane_IoU', 'Validation'),
    'train_das_iou': ('Das_IoU', 'Train'),
    'val_das_iou': ('Das_IoU', 'Validation'),
}


CAR_LENGTH, CAR_HEIGHT = 6.3, 3.7  # 차량 실제 크기 (미터)
scale_x = CAR_LENGTH / CAR_TOP_W     # X축 스케일
scale_y = CAR_LENGTH / CAR_TOP_H     # Y축 스케일
scale_z = CAR_HEIGHT / CAR_SIDE_H    # Z축 스케일

carla_bool=False
Path('./data/').mkdir(parents=True, exist_ok=True)
kill_existing_carla_processes()


while running:
    screen.fill(WHITE)
    mouse_pos = pygame.mouse.get_pos()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if state == MENU:
                if data_gan_button.collidepoint(mouse_pos):
                    os.makedirs("carla", exist_ok=True)
                    os.chdir("carla")

                    if os.path.exists("CarlaUE4.sh"):
                        os.chdir("..")
                        state = Town_sensor_Select
                    else:
                        state = CARAL_INSTALL
                elif training_button.collidepoint(mouse_pos):
                    state = hyperparameter
                elif metric_button.collidepoint(mouse_pos):
                    total_pages = 0
                    state = metric
                elif predict_button.collidepoint(mouse_pos):
                     state = predict
            if state!= CARAL_INSTALL:
                if end_button.collidepoint(mouse_pos):
                    if state in (MENU, data_gan, Train):
                        pygame.quit()
                        os._exit(0)
                if home_button.collidepoint(mouse_pos):
                    if state not in (predict, re_Train,MENU, data_gan, Train,metric):
                        state = MENU
                if home2_button.collidepoint(mouse_pos):
                    if state in (predict, re_Train,metric):# 홈 
                        state = MENU
                if back_button.collidepoint(mouse_pos):

                    if state == sensor_position:
                        if current_sensor_idx==0:
                            state -= 1
                        else:
                            current_sensor_idx -= 1
                    elif state == metric_vis:
                        clickable_buttons=[]
                        Training_result=[]
                        selected_result = None
                        selected_train_config = None
                        selected_sensor_config = None
                        state = metric
                    else:
                        if state in (predict,hyperparameter,Town_sensor_Select):
                            state=MENU
                        elif state not in (predict, re_Train,MENU, data_gan, Train,metric,Train_loop):
                            state-=1

                if confirm_button.collidepoint(mouse_pos):
                    if state == Town_sensor_Select:
                        selected_towns = [t for t in main_towns if town_checkboxes[t]]
                        selected_sensors = {s: option_checkboxes[s] for s in sensors if sensor_checkboxes[s]}
                        if selected_towns and selected_sensors:
                            sensor_cfg = {}
                            for s in selected_sensors:
                                cfg = configs[s].copy()
                                if s != "lidar":
                                    cfg.update(selected_sensors[s])
                                sensor_cfg[s] = cfg
                            selected_towns = [t for t in towns if town_checkboxes.get(t)]

                            with open('data/sensor_config.yaml', 'w') as f:
                                yaml.dump({"towns": selected_towns, "sensors": sensor_cfg,'data_num':data_town_num}, f)

                            configs=sensor_cfg

                        depth_use = any(option_checkboxes[s]["depth"] for s in cam_sensors if sensor_checkboxes[s])
                        seg_use = any(option_checkboxes[s]["seg"] for s in cam_sensors if sensor_checkboxes[s])
                        ob_use = any(option_checkboxes[s]["od"] for s in cam_sensors if sensor_checkboxes[s])
                        bev_use = sum(sensor_checkboxes[s] for s in cam_sensors) >= 3

                        if not depth_use:
                            depth_modal_color=GRAY
                        else:
                            depth_modal_color=RED

                        if not seg_use:
                            seg_modal_color=GRAY
                        else:
                            seg_modal_color=RED    
                        if not ob_use:
                            OD_modal_color=GRAY
                        else:
                            OD_modal_color=RED

                        if not bev_use:
                            bev_modal_color=GRAY
                        else:
                            bev_modal_color=RED 
                        if selected_towns and selected_sensors:
                            configs_ori = configs
                            state = state + 1

                    elif state == model_test:

                        state = state+1

                        current_sensor_idx = 0
                        if not carla_bool:
                            PORT, TM_PORT = get_random_ports()

                            carla_root = Path.cwd() / "carla"
                            os.environ["CARLA_ROOT"]   = str(carla_root)
                            os.environ["CARLA_SERVER"] = str(carla_root / "CarlaUE4.sh")
                            
                            server_proc = launch_carla_server(carla_root, PORT)


                            time.sleep(4)
                            client = carla.Client("localhost", PORT)
                            client.set_timeout(60.0)
                            ego_world=World(client,configs)
                            carla_bool=True

                    elif state == sensor_position:
                        current_sensor_idx += 1
                        sensor_img=None
                        if current_sensor_idx >= len(selected_sensors):
                            with open('data/sensor_config.yaml', 'w') as f:
                                yaml.dump({"towns": selected_towns,
                                           "sensors": configs, 
                                           'data_num':data_town_num,
                                           },f)

                            kill_existing_carla_processes()
                            sensor_config=configs
                            state+=1

                            carla_data_gan_code = subprocess.Popen(["python", "carla_run.py"],)

                    elif state == hyperparameter:
                        save_selected = any(save_checkboxes.values())
                        opt_selected = any(opt_checkboxes.values())


                        if save_selected and opt_selected:
                            state = state + 1

                    elif state == train_modal_select:
                        train_use = False
                        bev_train_use = False
                        seg=False
                        depth=False
                        train_encoder=False
                        bev_ok=False

                        out_ok=False
                        
                        if bev_modal_color==BLUE or seg_modal_color==BLUE or OD_modal_color==BLUE:
                            out_ok=True

                        if bev_modal_color==BLUE:
                            bev_train_use=True
                            if check_resnet_bev_decoder or check_efficientnet_bev_decoder:
                                bev_ok= True
                                
                        else:
                            bev_ok= True

                        if seg_modal_color==BLUE or depth_modal_color==BLUE or OD_modal_color==BLUE:
                            if check_fpn_decoder or check_bifpn_decoder:
                                train_use= True
                                if seg_modal_color==BLUE:
                                    seg=True
                                if depth_modal_color==BLUE:
                                    depth=True  
                                if OD_modal_color==BLUE:
                                    od=True 
                        else:
                            train_use=True

                        if check_resnet_encoder or check_efficientnet_encoder:
                            train_encoder= True
                            

                        if lidar_color==BLUE:
                            lidar_use=True
                        else:
                            lidar_use=False

                        if train_use and bev_ok and train_encoder and out_ok:
                            sensor_status = {
                                'front_cam': Front_cam_color == BLUE,
                                'left_cam': Left_cam_color == BLUE,
                                'right_cam': Right_cam_color == BLUE,
                                'rear_cam': Rear_cam_color == BLUE,
                                'rear_left_cam': Rear_left_cam_color == BLUE,
                                'rear_right_cam': Rear_right_cam_color == BLUE,
                                'lidar': lidar_color == BLUE,
                            }
                            model_conf = {}
                            if check_resnet_encoder:
                                model_conf["encoder"] = "ResNet"
                            elif check_efficientnet_encoder:
                                model_conf["encoder"] = "EfficientNet"
                            else:
                                model_conf["encoder"] = "None"

                            if check_fpn_decoder:
                                model_conf["decoder"] = "FPN"
                            elif check_bifpn_decoder:
                                model_conf["decoder"] = "Bi-FPN"
                            else:
                                model_conf["decoder"] = "None"

                            if check_resnet_bev_decoder:
                                model_conf["bev_decoder"] = "ResNet"
                            elif check_efficientnet_bev_decoder:
                                model_conf["bev_decoder"] = "EfficientNet"
                            else:
                                model_conf["bev_decoder"] = "None"
        
                            config={
                                'BEV': bev_train_use,
                                'Seg': seg,
                                'Depth': depth,
                                'od':od
                            }
                            
                            
                            state = state+1
                            

                    elif state == model_test2:
                        total_pages = 0
                        Training_loop.append({
                                        'model_conf':config,
                                        'lidar':lidar_use,
                                        'width':input_img_width,
                                        'height':input_img_height,
                                        'opt_checkboxes':opt_checkboxes,
                                        'save_checkboxes':save_checkboxes,
                                        'save_other_checkboxes':save_other_checkboxes,
                                        'sensor_use':sensor_status,
                                        'Epochs':input_Epochs,
                                        'batch':input_batch,
                                        'LR':input_LR,
                                        'model':model_conf,

                                        })
                        state = state + 1

                    elif state == Train_loop:
                        if len(Training_loop)>0:
                            Path('./data/loop').mkdir(parents=True, exist_ok=True)
                            per_town_samples: int = len(Training_loop)

                            digit_width = max(2, len(str(per_town_samples)) + 1)
                            
                            for idx,(conf) in enumerate(Training_loop):
                                fname = f"{idx:0{digit_width}d}"
                                with open(f'data/loop/train_config_{fname}.yaml', 'w') as f:
                                    yaml.dump(conf,f)

                            state = state + 1
                            train_code = subprocess.Popen(["python", "train.py"],

                                                              )
                    elif state == metric_vis:
                        state = re_Train



            # 클릭 액션
            if state == Town_sensor_Select:
                if Town_all.collidepoint(mouse_pos):
                    new_val = not all(town_checkboxes[t] for t in main_towns)
                    for t in towns:  # Update all towns, including _val variants
                        town_checkboxes[t] = new_val
                for idx, t in enumerate(main_towns):
                    r = checkbox_rect(idx // 3, idx % 3,top_left=(80, 150),cell_w=150, cell_h=120)
                    if r.collidepoint(mouse_pos):
                        town_checkboxes[t] = not town_checkboxes[t]
                        val_town = f"{t}_val"
                        if val_town in town_checkboxes:
                            town_checkboxes[val_town] = town_checkboxes[t]
                if Cam_all.collidepoint(mouse_pos):
                    cam_sensors = [s for s in sensors if s]
                    new_val = not all(sensor_checkboxes[s] for s in cam_sensors)
                    for s in cam_sensors:
                        sensor_checkboxes[s] = new_val
                if seg_all.collidepoint(mouse_pos):
                    cam_sensors = [s for s in sensors if s != "lidar" and sensor_checkboxes[s]]
                    if cam_sensors:
                        new_val = not all(option_checkboxes[s]["seg"] for s in cam_sensors)
                        for s in cam_sensors:
                            option_checkboxes[s]["seg"] = new_val
                if Depth_all.collidepoint(mouse_pos):
                    cam_sensors = [s for s in sensors if s != "lidar" and sensor_checkboxes[s]]
                    if cam_sensors:
                        new_val = not all(option_checkboxes[s]["depth"] for s in cam_sensors)
                        for s in cam_sensors:
                            option_checkboxes[s]["depth"] = new_val

                if od_all.collidepoint(mouse_pos):
                    cam_sensors = [s for s in sensors if s != "lidar" and sensor_checkboxes[s]]
                    if cam_sensors:
                        new_val = not all(option_checkboxes[s]["od"] for s in cam_sensors)
                        for s in cam_sensors:
                            option_checkboxes[s]["od"] = new_val

                for idx, s in enumerate(sensors):
                    r = checkbox_rect(idx, 0, top_left=(580, 150), cell_w=230)
                    if r.collidepoint(mouse_pos):
                        sensor_checkboxes[s] = not sensor_checkboxes[s]
                    '''if s != "lidar":
                        for j, o in enumerate(options):
                            r_opt = checkbox_rect(idx, j + 1, top_left=(600, 150))
                            if r_opt.collidepoint(mouse_pos) and sensor_checkboxes[s]:
                                option_checkboxes[s][o] = not option_checkboxes[s][o]'''

                if data_num_rect.collidepoint(event.pos):
                        train_modal_select_input = "town_num"
                
            elif state == sensor_position:
                if set_button.collidepoint(mouse_pos):
                    setting=True


                sensor = list(selected_sensors.keys())[current_sensor_idx]
                if car_top_rect.collidepoint(mouse_pos):
                    px = mouse_pos[0] - car_top_rect.x
                    py = mouse_pos[1] - car_top_rect.y
                    configs[sensor]['x'] = round(px * scale_x - CAR_LENGTH / 2, 2)
                    configs[sensor]['y'] = round(py * scale_y - CAR_LENGTH / 2, 2)

                if car_side_rect.collidepoint(mouse_pos):
                    px = mouse_pos[0] - car_side_rect.x 
                    pz = mouse_pos[1] - car_side_rect.y
                
                    configs[sensor]['x'] = round(px * scale_x - CAR_LENGTH / 2, 2) 
                    configs[sensor]['z'] = round(CAR_HEIGHT - pz * scale_z, 2)


                exclude_keys = {'seg', 'depth', 'od'}
                params = [k for k in configs[sensor].keys() if k not in exclude_keys]

                for i, p in enumerate(params):
                    col = i % 3
                    row = i // 3
                    rect = pygame.Rect(780 + col * 200, 600 + row * 60, 150, 36)
                    if rect.collidepoint(mouse_pos):
                        active_input = (sensor, p)
                        text_input = ""

            elif state == data_gan:
                if RGB_button.collidepoint(mouse_pos):
                    selected_mode = 'RGB'
                elif SEG_button.collidepoint(mouse_pos):
                    selected_mode = 'SEG'
                elif Depth_button.collidepoint(mouse_pos):
                    selected_mode = 'Depth'
                elif OD_button.collidepoint(mouse_pos):
                    if OD_mode:
                        OD_mode = False
                    else:
                        OD_mode = True

            elif state == hyperparameter:
                hyperparameter_input=None
                for idx, t in enumerate(opt):
                    r = checkbox_rect(idx, 0, top_left=(800, 200), cell_h=50)
                    if r.collidepoint(mouse_pos):
                        for key in opt_checkboxes.keys():
                            opt_checkboxes[key] = False
                        opt_checkboxes[t] = True

                for idx, t in enumerate(save):
                    r = checkbox_rect(idx ,0 ,top_left=(1100, 200),cell_h=50)
                    if r.collidepoint(mouse_pos):
                        save_checkboxes[t] = not save_checkboxes[t]
                for idx, t in enumerate(save_other):
                    r = checkbox_rect(idx ,0,top_left=(1100, 500),cell_h=50)
                    if r.collidepoint(mouse_pos):
                        save_other_checkboxes[t] = not save_other_checkboxes[t]

                if Epochs_rect.collidepoint(event.pos):
                        hyperparameter_input = "Epochs"
                        text_input = ""
                if batch_rect.collidepoint(event.pos):
                        hyperparameter_input = "batch"
                        text_input = ""
                if LR_rect.collidepoint(event.pos):
                        hyperparameter_input = "LR"
                        text_input = ""

            elif state == train_modal_select:
                train_modal_select_input=None
                if width_rect.collidepoint(event.pos):
                    train_modal_select_input = "width"
                    text_input = ""
                elif height_rect.collidepoint(event.pos):
                    train_modal_select_input = "height"
                    text_input = ""

                elif bev_button_modal.collidepoint(event.pos):
                    bev_modal_color = toggle_button_color(bev_modal_color)
                elif seg_button_modal.collidepoint(event.pos):
                    seg_modal_color = toggle_button_color(seg_modal_color)
                elif depth_button_modal.collidepoint(event.pos):
                    depth_modal_color = toggle_button_color(depth_modal_color)
                elif OD_button_modal.collidepoint(event.pos):
                    OD_modal_color = toggle_button_color(OD_modal_color)


                elif Front_cam_rect.collidepoint(event.pos):
                    Front_cam_color = toggle_button_color(Front_cam_color)
                elif Left_cam_rect.collidepoint(event.pos):
                    Left_cam_color = toggle_button_color(Left_cam_color)
                elif Right_cam_rect.collidepoint(event.pos):
                    Right_cam_color = toggle_button_color(Right_cam_color)
                elif Rear_cam_rect.collidepoint(event.pos):
                    Rear_cam_color = toggle_button_color(Rear_cam_color)
                elif Rear_right_cam_rect.collidepoint(event.pos):
                    Rear_right_cam_color = toggle_button_color(Rear_right_cam_color)
                elif Rear_left_cam_rect.collidepoint(event.pos):
                    Rear_left_cam_color = toggle_button_color(Rear_left_cam_color)

                elif lidar_rect.collidepoint(event.pos):
                    lidar_color = toggle_button_color(lidar_color)
                elif rect_resnet_encoder.collidepoint(event.pos):
                    check_resnet_encoder=toggle_check(check_resnet_encoder,check_efficientnet_encoder)
                elif rect_efficent_encoder.collidepoint(event.pos):
                    check_efficientnet_encoder=toggle_check(check_efficientnet_encoder,check_resnet_encoder)
                elif rect_resnet_decoder.collidepoint(event.pos):
                    check_fpn_decoder=toggle_check(check_fpn_decoder,check_bifpn_decoder)
                elif rect_efficent_decoder.collidepoint(event.pos):
                    check_bifpn_decoder=toggle_check(check_bifpn_decoder,check_fpn_decoder)
                elif rect_resnet_bev_decoder.collidepoint(event.pos):
                    check_resnet_bev_decoder=toggle_check(check_resnet_bev_decoder,check_efficientnet_bev_decoder)
                elif rect_efficent_bev_decoder.collidepoint(event.pos):
                    check_efficientnet_bev_decoder=toggle_check(check_efficientnet_bev_decoder,check_resnet_bev_decoder)
          
            elif state == Train:
                if handle_rect.collidepoint(event.pos):
                    dragging = True
                    new=False
                elif scrollbar_rect.collidepoint(event.pos):
                    relative_x = max(min(event.pos[0] - scrollbar_rect.x, scrollbar_rect.width - handle_width), 0)
                    scroll_x = round(relative_x / (scrollbar_rect.width - handle_width) * max_scroll_value)
                    new=False

            elif state == metric:
                if R_button.collidepoint(event.pos):
                    if page < total_pages - 1:
                        page += 1
                elif L_button.collidepoint(event.pos):
                    if page > 0:
                        page -= 1
                if confirm_delete:
                    if yes_btn.collidepoint(event.pos):
                        shutil.rmtree(delete_target)
                        confirm_delete = False
                    elif no_btn.collidepoint(event.pos):
                        confirm_delete = False
                else:
                    for del_btn, path, name in delete_buttons:
                        if del_btn.collidepoint(event.pos):
                            confirm_delete = True
                            delete_target = path
                            delete_name = name
                            break
                for idx_metric, (rect) in enumerate(clickable_buttons):
                    if rect.collidepoint(event.pos):
                        selected_idx_metric=idx_metric
                        state = metric_vis

            elif state == metric_vis:
                if handle_rect_vis.collidepoint(event.pos):
                    dragging = True
                if RGB_button.collidepoint(mouse_pos):
                    if selected_train_config['model_conf']['BEV']:
                        metric_vis_mode = 'BEV'
                elif SEG_button.collidepoint(mouse_pos):
                    if selected_train_config['model_conf']['Seg']:
                        metric_vis_mode = 'Seg'
                elif Depth_button.collidepoint(mouse_pos):
                    if selected_train_config['model_conf']['Depth']:
                        metric_vis_mode = 'Depth'
                elif metric_vis_img_button.collidepoint(mouse_pos):
                    metric_vis_img_mode = 'img'
                elif metric_vis_graph_button.collidepoint(mouse_pos):
                    metric_vis_img_mode = 'graph'
                elif metric_vis_score_button.collidepoint(mouse_pos):
                    metric_vis_img_mode = 'score'

                elif metric_R_button.collidepoint(mouse_pos):
                    if selected_idx_metric < len(Training_result)-1:
                        selected_idx_metric+=1
                        selected_result=Training_result[selected_idx_metric]
                        scroll_x_vis2 = len(glob.glob(f'{selected_result}/predict/*')) -1
                        if scroll_x_vis>scroll_x_vis2:
                            scroll_x_vis=scroll_x_vis2
                        elif max_idx:
                            scroll_x_vis=scroll_x_vis2
                elif metric_L_button.collidepoint(mouse_pos):
                    if selected_idx_metric > 0:
                        selected_idx_metric-=1
                        selected_result=Training_result[selected_idx_metric]
                        scroll_x_vis2 = len(glob.glob(f'{selected_result}/predict/*')) -1
                        if scroll_x_vis>scroll_x_vis2:
                            scroll_x_vis=scroll_x_vis2
                        elif max_idx:
                            scroll_x_vis=scroll_x_vis2
                elif scrollbar_rect_vis.collidepoint(event.pos):
                    relative_x = max(min(event.pos[0] - scrollbar_rect_vis.x, scrollbar_rect_vis.width - handle_width_vis), 0)
                    scroll_x_vis = round(relative_x / (scrollbar_rect_vis.width - handle_width_vis) * max_scroll_value_vis)

                for i in range(6):
                    if vis_bev_buttons[i].collidepoint(mouse_pos):
                        selected_btn=i
                if handle_rect_vis.collidepoint(event.pos):
                    dragging = True
            elif state == re_Train:  
                if re_button.collidepoint(mouse_pos):
                    state = hyperparameter
                    save_other_checkboxes = {s: False for s in save_other}
                    save_checkboxes = {s: False for s in save}
                    opt_checkboxes = {s: False for s in opt}
                    input_Epochs='20'
                    input_batch='4'
                    input_LR='0.001'

            elif state == Train_loop:
                if loop_button.collidepoint(event.pos):
                    state = hyperparameter
                    input_Epochs = '20'                 
                    input_batch = '4'                   
                    input_LR = '0.001'                   
                    hyperparameter_input = None    
                if confirm_delete2:
                    if yes_btn.collidepoint(event.pos):
                        del Training_loop[delete_idx]
                        confirm_delete2 = False
                    elif no_btn.collidepoint(event.pos):
                        confirm_delete2 = False
                else:      
                    for del_btn, idx in delete_buttons2:
                        if del_btn.collidepoint(event.pos):
                            delete_idx=i
                            confirm_delete2 = True

                            break

        elif event.type == pygame.KEYDOWN:

            if state == Town_sensor_Select:
                if event.key == pygame.K_RETURN:
                    if train_modal_select_input == 'town_num':
                        try:
                            val = int(float(text_input))
                            data_town_num = val
                        except ValueError:
                            data_town_num=0
                        text_input=""
                        train_modal_select_input=""
                elif event.key == pygame.K_BACKSPACE:
                    text_input = text_input[:-1]
                else:
                    text_input += event.unicode
                        
            if state == train_modal_select:
                if event.key == pygame.K_RETURN:
                    if train_modal_select_input == "width":
                        try:
                            val = int(float(text_input))
                            input_img_width = val
                        except ValueError:
                            input_img_width=0
                    elif train_modal_select_input == "height":
                        try:
                            val = int(float(text_input))
                            input_img_height = val
                        except ValueError:
                            input_img_height=0
                    text_input=""
                    train_modal_select_input=""
                elif event.key == pygame.K_BACKSPACE:
                    text_input = text_input[:-1]
                else:
                    text_input += event.unicode

            elif state == sensor_position:
                if active_input:
                    if event.key == pygame.K_RETURN:
                        try:
                            val = float(text_input)
                            if active_input[1] in ['width', 'height', 'rotation_frequency', 'points_per_second']:
                                val = int(val)

                            configs[active_input[0]][active_input[1]] = val

                        except ValueError:
                            print("잘못된 숫자 입력")
                        active_input = None
                    elif event.key == pygame.K_BACKSPACE:
                        text_input = text_input[:-1]
                    else:
                        text_input += event.unicode

            elif state == hyperparameter:

                if event.key == pygame.K_RETURN:
                    if hyperparameter_input == "Epochs":
                        
                        try:
                            val = int(float(text_input))
                            input_Epochs = val
                        except ValueError:
                            input_Epochs=1
                    elif hyperparameter_input == "batch":
                        try:
                            val = int(float(text_input))
                            input_batch = val
                        except ValueError:
                            input_batch=1
                    elif hyperparameter_input == "LR":
                        try:
                            val = (float(text_input))
                            input_LR = val
                        except ValueError:
                            input_LR=0.01

                    text_input=""
                    hyperparameter_input=""
                elif event.key == pygame.K_BACKSPACE:
                    text_input = text_input[:-1]
                else:
                    text_input += event.unicode
        
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            dragging = False
            new=False
        elif event.type == pygame.MOUSEMOTION and dragging:
            new=False
            mouse_x = event.pos[0]
            if state == Train:
                relative_x = max(min(mouse_x - scrollbar_rect.x, scrollbar_rect.width - handle_width), 0)
                scroll_x = round(relative_x / (scrollbar_rect.width - handle_width) * max_scroll_value)
            elif state == metric_vis:
                relative_x = max(min(mouse_x - scrollbar_rect_vis.x, scrollbar_rect_vis.width - handle_width_vis), 0)
                scroll_x_vis = round(relative_x / (scrollbar_rect_vis.width - handle_width_vis) * max_scroll_value_vis) +1

    # --------------- 시각화 -----------------
    if state == CARAL_INSTALL:
        if not installing:
            threading.Thread(target=install_carla, daemon=True).start()

        screen.fill(WHITE)
        title = font.render("CARLA 설치 중...", True, BLACK)
        screen.blit(title, (50, 50))

        # 설치 메시지 출력
        msg = font.render(install_message, True, BLACK)
        screen.blit(msg, (50, 120))

        # 진행 바 그리기
        bar_w = 800
        bar_h = 30
        pygame.draw.rect(screen, GRAY, (50, 200, bar_w, bar_h))
        pygame.draw.rect(screen, BLUE, (50, 200, int(bar_w * install_progress), bar_h))

        # 퍼센트 출력
        percent_text = font.render(f"{install_progress * 100:.3f}%", True, BLACK)
        screen.blit(percent_text, (50 + bar_w + 20, 200))
        
    elif state == MENU:
        title = font.render("모드를 선택하세요", True, BLACK)
        screen.blit(title, (50, 50))

        buttons = [
            (data_gan_button, "데이터 생성"),
            (training_button, "학습"),
            (metric_button, "결과"),
            #(predict_button, "예측")
        ]

        for btn_rect, label in buttons:
            hovered = btn_rect.collidepoint(mouse_pos)
            if hovered:
                bigger_button = pygame.Rect(btn_rect.x - 5, btn_rect.y - 5, btn_rect.width + 10, btn_rect.height + 10)
                pygame.draw.rect(screen, BLUE, bigger_button, border_radius=8)
                draw_centered_text(screen, label, font_2, bigger_button, WHITE)
            else:
                pygame.draw.rect(screen, BLUE, btn_rect, border_radius=6)
                draw_centered_text(screen, label, font, btn_rect, WHITE)

    elif state == Town_sensor_Select:

        title = font.render("데이터 생성 할 Town 및 센서를 선택해주세요.", True, BLACK)
        screen.blit(title, (50, 20))

        title=font_small.render("Bird Eye View 모델 사용하실시 Cam All 권장 / Cam 마다 이미지 사이즈가 다르면 학습이 안되는게 있을 수 있습니다.", True, RED)
        screen.blit(title, (50, 70))

        cam_sensors = [s for s in sensors if s != "lidar"]
        all_checked = all(sensor_checkboxes[s] for s in cam_sensors)
        any_checked = any(sensor_checkboxes[s] for s in cam_sensors)

        draw_checkbox(Cam_all, all_checked, "Cam All")

        seg_all_enabled = any(sensor_checkboxes[s] for s in cam_sensors)
        if any_checked:
            seg_all_checked = all(option_checkboxes[s]["seg"] for s in cam_sensors if sensor_checkboxes[s])
        else:
            seg_all_checked=False
        draw_checkbox(seg_all,  seg_all_checked, "Seg All", enabled=seg_all_enabled)

        depth_all_enabled = any(sensor_checkboxes[s] for s in cam_sensors)
        if any_checked:
            depth_all_checked = all(option_checkboxes[s]["depth"] for s in cam_sensors if sensor_checkboxes[s])
        else:
            depth_all_checked=False
        draw_checkbox(Depth_all,  depth_all_checked, "Depth All", enabled=depth_all_enabled)

        ob_all_enabled = any(sensor_checkboxes[s] for s in cam_sensors)
        if any_checked:
            od_all_checked = all(option_checkboxes[s]["od"] for s in cam_sensors if sensor_checkboxes[s])
        else:
            od_all_checked=False
        draw_checkbox(od_all,  od_all_checked, "OD All", enabled=depth_all_enabled)



        for idx, s in enumerate(sensors):
            r = checkbox_rect(idx, 0, top_left=(580, 150))
            draw_checkbox(r, sensor_checkboxes[s], s)
            if s != "lidar":
                for j, o in enumerate(options):
                    r_opt = checkbox_rect(idx, j + 1, top_left=(580, 150), cell_w=230)
                    draw_checkbox(r_opt, option_checkboxes[s][o], o, enabled=sensor_checkboxes[s])

        draw_checkbox(Town_all, all(town_checkboxes[t] for t in main_towns), "All")
        for idx, t in enumerate(main_towns):
            r = checkbox_rect(idx // 3, idx % 3,top_left=(80, 150),cell_w=150, cell_h=120)
            draw_checkbox(r, town_checkboxes[t], t)



        count = sum(v for v in town_checkboxes.values())
        label_text = font_small.render("각 타운마다 데이터 양", True, BLACK)
        label_pos = (data_num_rect.x, data_num_rect.y - 25)
        screen.blit(label_text, label_pos)


        pygame.draw.rect(screen, (200,200,200) if train_modal_select_input == "town_num" else WHITE, data_num_rect)
        pygame.draw.rect(screen, BLACK, data_num_rect, 2)
        screen.blit(font.render(f"{format_with_commas(data_town_num)}", True, BLACK), (data_num_rect.x + 5, data_num_rect.y + 5))


        label_text = font_small.render(f"{format_number(int(data_town_num)*int(count))}", True, BLACK)
        label_pos = (data_num_all_rect.x+230, data_num_all_rect.y - 25)
        screen.blit(label_text, label_pos)

        label_text = font_small.render("데이터 총량", True, BLACK)
        label_pos = (data_num_all_rect.x, data_num_all_rect.y - 25)
        screen.blit(label_text, label_pos)
        pygame.draw.rect(screen, WHITE, data_num_all_rect)
        pygame.draw.rect(screen, BLACK, data_num_all_rect, 2)
        screen.blit(font.render(f"{format_with_commas(int(data_town_num)*int(count))}", True, BLACK), (data_num_all_rect.x + 5, data_num_all_rect.y + 5))
 
    elif state == model_test:
        title = font.render("생성되는 데이터", True, BLACK)
        screen.blit(title, (50, 50))

        title = font.render("입력 데이터", True, BLACK)
        screen.blit(title, model_test_input_img_text)
        rects = model_test_input_img_boxes + [model_test_input_img_text]
        [
            Front_cam_color,
            Left_cam_color,
            Right_cam_color,
            Rear_cam_color,
            Rear_right_cam_color,
            Rear_left_cam_color,
            lidar_color
        ]

        paths = {
            0: 'img/test_img/rgb_left/test.png',
            1: 'img/test_img/rgb_front/test.png',
            2: 'img/test_img/rgb_right/test.png',
            4: 'img/test_img/rgb_rear/test.png',
            3: 'img/test_img/rgb_rear_left/test.png',
            5: 'img/test_img/rgb_rear_right/test.png',
            7: 'img/test_img/top_view/test.jpg',
        }
        if not sensor_checkboxes['front_cam']:
            del paths[1]
        if not sensor_checkboxes['left_cam']:
            del paths[0]
        if not sensor_checkboxes['right_cam']:
            del paths[2]
        if not sensor_checkboxes['rear_cam']:
            del paths[4]
        if not sensor_checkboxes['rear_right_cam']:
            del paths[5]
        if not sensor_checkboxes['rear_left_cam']:
            del paths[3]
        if not sensor_checkboxes['lidar']:
            del paths[7]

        model_test_input_img = load_model_test_input_images(paths)

        pygame.draw.rect(screen, (0, 0, 0), boxing(rects,max_x_sum=20,max_y_sum=20), 3)
        for i, rect in enumerate(model_test_input_img_boxes):
            if i == 6 or i == 8:
                continue
            if model_test_input_img[i] is not None:
                img = pygame.transform.scale(model_test_input_img[i], (rect.width, rect.height))
                screen.blit(img, rect.topleft)
            else:
                pygame.draw.rect(screen, GRAY, rect, 2)
                title = font.render("Noen", True, BLACK)
                screen.blit(title, model_test_input_img_none[i])

        paths = {
        # Segmentation
        0: 'img/test_img/seg_left/test.png',
        1: 'img/test_img/seg_front/test.png',
        2: 'img/test_img/seg_right/test.png',
        3: 'img/test_img/seg_rear_left/test.png',
        4: 'img/test_img/seg_rear/test.png',
        5: 'img/test_img/seg_rear_right/test.png',

        # Depth
        6: 'img/test_img/depth_left/test.png',
        7: 'img/test_img/depth_front/test.png',
        8: 'img/test_img/depth_right/test.png',
        9: 'img/test_img/depth_rear_left/test.png',
        10: 'img/test_img/depth_rear/test.png',
        11: 'img/test_img/depth_rear_right/test.png',

        # BEV
        13: 'img/test_img/bev_re/test.jpg'
        }

        if not seg_all_checked:
            del paths[1]
            del paths[0]
            del paths[2]
            del paths[4]
            del paths[5]
            del paths[3]
        else:
            if not sensor_checkboxes['front_cam']:
                del paths[1]
            if not sensor_checkboxes['left_cam']:
                del paths[0]
            if not sensor_checkboxes['right_cam']:
                del paths[2]
            if not sensor_checkboxes['rear_cam']:
                del paths[4]
            if not sensor_checkboxes['rear_right_cam']:
                del paths[5]
            if not sensor_checkboxes['rear_left_cam']:
                del paths[3]

        if not depth_all_checked:
            del paths[7]
            del paths[6]
            del paths[8]
            del paths[10]
            del paths[11]
            del paths[9]
        else:
            if not sensor_checkboxes['front_cam']:
                del paths[7]
            if not sensor_checkboxes['left_cam']:
                del paths[6]
            if not sensor_checkboxes['right_cam']:
                del paths[8]
            if not sensor_checkboxes['rear_cam']:
                del paths[10]
            if not sensor_checkboxes['rear_right_cam']:
                del paths[11]
            if not sensor_checkboxes['rear_left_cam']:
                del paths[9]

        if not sum(sensor_checkboxes.values()) >=3:
            del paths[13]

            
        model_test_output_img = load_model_test_output_images(paths)


        title = font.render("결과", True, BLACK)
        screen.blit(title, model_test_output_img_text)
        rects = model_test_output_img_boxes + [model_test_output_img_text]

        pygame.draw.rect(screen, (0, 0, 0), boxing(rects,max_x_sum=20,max_y_sum=20), 3)
        for i, rect in enumerate(model_test_output_img_boxes):
            if i == 12 or i == 14:
                continue
            if model_test_output_img[i] is not None:
                img = pygame.transform.scale(model_test_output_img[i], (rect.width, rect.height))
                screen.blit(img, rect.topleft)
            else:
                pygame.draw.rect(screen, GRAY, rect, 2)
                title = font.render("Noen", True, BLACK)
                screen.blit(title, model_test_output_img_none[i])

    elif state == sensor_position:
        if setting2:
            ego_world.restart(configs)
            setting2=False
            setting=False
        title = font.render(f"{sensor} 위치 및 파라미터 설정 {current_sensor_idx+1}번째 센서 설정", True, BLACK)
        screen.blit(title, (50, 50))

        title = font_small.render("홈이나 1번째 센서 설정 뒤로 가면 설정값 초기화", True, BLUE)
        screen.blit(title, (50, 90))

        title = font_small.render("너무 많은 SET버튼 사용시 렉 유발/ 이미지의 포인트는 대략적인 위치입니다.", True, RED)
        screen.blit(title, (50, 120))

        hovered = set_button.collidepoint(mouse_pos)
        if hovered:
            bigger_button = pygame.Rect(set_button.x - 5, set_button.y - 5, set_button.width + 10, set_button.height + 10)
            pygame.draw.rect(screen, (255, 80, 80), bigger_button, border_radius=8)
            draw_centered_text(screen, "SET", font_2, bigger_button, WHITE)
        else:
            pygame.draw.rect(screen, RED, set_button, border_radius=6)
            draw_centered_text(screen, "SET", font, set_button, WHITE)

        pygame.draw.rect(screen, GRAY, sensor_img_rect, 2)

        if not 'lidar' in sensor:
            sensor_img = getattr(ego_world, f"{sensor}_rgb").img
        else:
            if ego_world.lidar_sensor.point_cloud is not None:
                sensor_img = lidar_to_topview2(ego_world.lidar_sensor.point_cloud[:, :4])
            else:
                sensor_img = None
        sensor = list(selected_sensors.keys())[current_sensor_idx]
        screen.blit(car_top, car_top_rect)
        screen.blit(car_side, car_side_rect)
        pygame.draw.rect(screen, BLACK, car_top_rect, 2)
        pygame.draw.rect(screen, BLACK, car_side_rect, 2)
        px = int((configs[sensor]['x'] + CAR_LENGTH / 2) / scale_x + car_top_rect.x)
        py = int((configs[sensor]['y'] + CAR_LENGTH / 2) / scale_y + car_top_rect.y)
        pz = int((CAR_HEIGHT - configs[sensor]['z']) / scale_z + car_side_rect.y)



        fov_length = 100
        fov_angle = 20 

        yaw_rad = -(math.radians(configs[sensor]['yaw']))
        yaw_deg = (configs[sensor]['yaw'] + 360) % 360     

        left_yaw = yaw_rad - math.radians(fov_angle)
        right_yaw = yaw_rad + math.radians(fov_angle)
        left_end_x = px + fov_length * math.cos(left_yaw)
        left_end_y = py - fov_length * math.sin(left_yaw)
        right_end_x = px + fov_length * math.cos(right_yaw)
        right_end_y = py - fov_length * math.sin(right_yaw)

        pygame.draw.circle(screen, RED, (px, py), 4)
        if not 'lidar' in sensor:
            pygame.draw.line(screen, RED, (px, py), (left_end_x, left_end_y), 2)
            pygame.draw.line(screen, RED, (px, py), (right_end_x, right_end_y), 2)

        pitch_rad = -(math.radians(configs[sensor]['pitch']))
        left_pitch = pitch_rad - math.radians(fov_angle)
        right_pitch = pitch_rad + math.radians(fov_angle)

        side_px = car_side_rect.x + px - 50 
        facing_back = 100 <= yaw_deg <= 260 

        signed_length_pitch = -fov_length if facing_back else fov_length

        left_end_x = side_px + signed_length_pitch * math.cos(left_pitch)
        left_end_z = pz + signed_length_pitch * math.sin(left_pitch)

        right_end_x = side_px + signed_length_pitch * math.cos(right_pitch)
        right_end_z = pz + signed_length_pitch * math.sin(right_pitch)

        pygame.draw.circle(screen, RED, (side_px, pz), 4)
        if not 'lidar' in sensor:
            pygame.draw.line(screen, RED, (side_px, pz), (left_end_x, left_end_z), 2)
            pygame.draw.line(screen, RED, (side_px, pz), (right_end_x, right_end_z), 2)




        exclude_keys = {'seg', 'depth', 'od'}
        params = [k for k in configs[sensor].keys() if k not in exclude_keys]

        for i, p in enumerate(params):
            col = i % 3
            row = i // 3
            rect = pygame.Rect(780 + col * 200, 600 + row * 60, 150, 36)
            draw_input_box(rect, configs[sensor][p], active_input == (sensor, p))
            label = smallfont.render(p, True, BLACK)
            screen.blit(label, (rect.x, rect.y - 22))

        if sensor_img is not None:
            sensor_img = sensor_img[:, :, ::-1]
            img = pygame.transform.scale(pygame.surfarray.make_surface(sensor_img.swapaxes(1, 0)), (sensor_img_rect.width, sensor_img_rect.height))
            screen.blit(img, sensor_img_rect.topleft)
        else:
            draw_centered_text(screen, "조금만 기다려주세요.", font, sensor_img_rect, RED)


    elif state == data_gan:
        title = font.render("데이터 생성", True, BLACK)
        screen.blit(title, (50, 50))

        bar_w = 1000
        pygame.draw.rect(screen, GRAY, (230, 50, bar_w, 40))
        progress = int(data_gan_num) / int(data_town_num)
        pygame.draw.rect(screen, BLUE, (230, 50, int(bar_w * progress), 40))


        draw_hoverable_button(screen, RGB_button, BLUE, "RGB", mouse_pos, font, font_2)        
        draw_hoverable_button(screen, SEG_button, BLUE, "SEG", mouse_pos, font, font_2)        
        draw_hoverable_button(screen, Depth_button, BLUE, "Depth", mouse_pos, font, font_2)        
        draw_hoverable_button(screen, OD_button, BLUE, "OD", mouse_pos, font, font_2)        

        title = font_small.render(f"{int(count_data_gan/(int(data_town_num)))}/{count} 타운 진행", True, BLACK)
        screen.blit(title, (1250, 55))

        Towns=glob.glob('data/Town*')
        Towns.sort()
        if Towns:
            count_data_gan=0
            for town in Towns:
                paths=glob.glob(f'{town}/*')
                data_gan_num=len(glob.glob(f'{paths[0]}/*'))
                count_data_gan+=data_gan_num
                if data_gan_num < int(data_town_num)-10:
                    break


        title = font_small.render(f"생성하고 있는 맵: {town.split('/')[-1]}", True, BLACK)
        screen.blit(title, (230, 100))

        title = font_small.render(f"생성 된 데이터 양: {format_with_commas(count_data_gan)}", True, BLACK)
        screen.blit(title, (500, 100))

        title = font_small.render(f"생성 해야되는 양: {format_with_commas(int(data_town_num)*int(count))}", True, BLACK)
        screen.blit(title, (800, 100))

        if int(count_data_gan/(int(data_town_num)))>=count:
            state =hyperparameter

        if town != 'None':
            test_paths=glob.glob(f'{town}/*')
            rgb_path = []
            seg_path = []
            depth_path = []
            bbss_path = []
            lidar_paths = []
            bev_paths = []

            for path in test_paths:
                if '_rgb' in path:
                    rgb_path.append(path)
                elif '_seg' in path:
                    seg_path.append(path)
                elif '_depth' in path:
                    depth_path.append(path)
                elif '_bbs' in path:
                    bbss_path.append(path)      
                elif 'lidar' in path:
                    lidar_paths = glob.glob(path + '/*')
                    lidar_paths.sort()
                elif 'bev' in path:
                    bev_paths = glob.glob(path + '/*')
                    bev_paths.sort()
            if len(rgb_path) > 0:
                rgb_results = assign_paths_by_direction(rgb_path, 'rgb')
                front_rgb_paths = rgb_results.get('rgb_front', [])
                left_rgb_paths = rgb_results.get('rgb_left', [])
                rear_right_rgb_paths = rgb_results.get('rgb_rear_right', [])
                rear_left_rgb_paths = rgb_results.get('rgb_rear_left', [])
                rear_rgb_paths = rgb_results.get('rgb_rear', [])
                right_rgb_paths = rgb_results.get('rgb_right', [])

            if len(seg_path) > 0:
                seg_results = assign_paths_by_direction(seg_path, 'seg')
                front_seg_paths = seg_results.get('seg_front', [])
                left_seg_paths = seg_results.get('seg_left', [])
                rear_right_seg_paths = seg_results.get('seg_rear_right', [])
                rear_left_seg_paths = seg_results.get('seg_rear_left', [])
                rear_seg_paths = seg_results.get('seg_rear', [])
                right_seg_paths = seg_results.get('seg_right', [])

            if len(depth_path) > 0:
                depth_results = assign_paths_by_direction(depth_path, 'depth')
                front_depth_paths = depth_results.get('depth_front', [])
                left_depth_paths = depth_results.get('depth_left', [])
                rear_right_depth_paths = depth_results.get('depth_rear_right', [])
                rear_left_depth_paths = depth_results.get('depth_rear_left', [])
                rear_depth_paths = depth_results.get('depth_rear', [])
                right_depth_paths = depth_results.get('depth_right', [])
            
            if len(bbss_path) > 0:
                bbs_results = assign_paths_by_direction(bbss_path, 'bbs')
                front_bbs_paths = bbs_results.get('bbs_front', [])
                left_bbs_paths = bbs_results.get('bbs_left', [])
                rear_right_bbs_paths = bbs_results.get('bbs_rear_right', [])
                rear_left_bbs_paths = bbs_results.get('bbs_rear_left', [])
                rear_bbs_paths = bbs_results.get('bbs_rear', [])
                right_bbs_paths = bbs_results.get('bbs_right', [])
            

            directions = ['left', 'front', 'right', 'rear', 'rear_right', 'rear_left']
            path_dict = {
                'RGB': [left_rgb_paths, front_rgb_paths, right_rgb_paths, rear_right_rgb_paths,rear_rgb_paths, rear_left_rgb_paths],
                'SEG': [left_seg_paths, front_seg_paths, right_seg_paths,  rear_right_seg_paths,rear_seg_paths, rear_left_seg_paths],
                'Depth': [left_depth_paths, front_depth_paths, right_depth_paths, rear_right_depth_paths, rear_depth_paths,  rear_left_depth_paths],
            }
            OD_BBS=[left_bbs_paths, front_bbs_paths, right_bbs_paths, rear_right_bbs_paths, rear_bbs_paths,  rear_left_bbs_paths]
            surface_indices = [0, 1, 2, 4, 5, 6]
            for i, direction in enumerate(directions):
                path_list = path_dict[selected_mode][i]
                bbs=None
                try:
                    if OD_mode:
                        bbs=np.load(OD_BBS[i][-1])
                except Exception as e:
                    continue
                try:
                    pil_img = load_pil_image(path_list, -1) 
                    if OD_mode:
                        if bbs is not None:
                            draw = ImageDraw.Draw(pil_img)
                            for box in bbs:
                                x1, y1, x2, y2, cls_id = box
                                label = class_names.get(int(cls_id), f"Class {int(cls_id)}")
                                draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=2)
                                draw.text((x1, y1 - 10), label, fill="red")

                except Exception as e:
                    continue

                if pil_img is not None:
                    try:
                        img_array = np.array(pil_img)
                        if img_array.ndim == 2:
                            img_array = np.stack((img_array,) * 3, axis=-1)
                        model_data_gan_input_img[surface_indices[i]] = pygame.surfarray.make_surface(img_array.swapaxes(0, 1))
                    except Exception as e:
                        continue
            if len(lidar_paths) > 0:
                try:
                    lidar_data = np.load(lidar_paths[-1], allow_pickle=True)
                    points = lidar_data[:, :4]
                    model_data_gan_input_img[7] = lidar_to_topview(points)
                except Exception as e:
                    continue
            if len(bev_paths) > 0:
                try:
                    top = cv2.imread(bev_paths[-1])
                    top = cv2.cvtColor(top, cv2.COLOR_BGR2RGB)
                    for color, new_color in zip(colors_bev, [Y, gray_]):
                        lower_bound = np.array(color, dtype="uint8")
                        upper_bound = np.array(color, dtype="uint8")
                        
                        mask = cv2.inRange(top, lower_bound, upper_bound)
                        
                        top[mask != 0] = new_color

                    vehicle_mask = cv2.imread(bev_paths[-1].replace('bev', 'vehicle_mask'), cv2.IMREAD_GRAYSCALE)
                    top[vehicle_mask == 255] = (0, 0, 255)
                    walker_mask = cv2.imread(bev_paths[-1].replace('bev', 'walker_mask'), cv2.IMREAD_GRAYSCALE)
                    top[walker_mask == 255] = (255, 0, 255)
                    walker_mask = cv2.imread(bev_paths[-1].replace('bev', 'light_topview'))
                    walker_mask = cv2.cvtColor(walker_mask, cv2.COLOR_BGR2RGB)

                    colors = {
                        2: (255, 0, 0),      # Red
                        3: (255, 255, 0),    # Yellow
                        4: (0, 255, 0),      # Green
                    }

                    for color in colors.values():
                        color_array = np.array(color, dtype=np.uint8)
                        mask = np.all(walker_mask == color_array, axis=-1)
                        top[mask] = color


                    top=top[150:350, 150:350]

                    center_y, center_x = top.shape[0] // 2, top.shape[1] // 2

                    rect_top_left = (center_x - 5, center_y - 10)
                    rect_bottom_right = (center_x + 5, center_y + 10)

                    cv2.rectangle(top, rect_top_left, rect_bottom_right, (255, 255, 255), thickness=-1)  # -1이면 채움

                    model_data_gan_input_img[3] = pygame.surfarray.make_surface(top.swapaxes(1, 0))
                except Exception as e:
                    continue

            for i, rect in enumerate(model_data_gan_input_img_boxes):
                pygame.draw.rect(screen, GRAY, rect, 2)
                if model_data_gan_input_img[i] is not None:
                    img = pygame.transform.scale(model_data_gan_input_img[i], (rect.width, rect.height))
                    screen.blit(img, rect.topleft)
                    label = font.render(f"{i+1}", True, RED)
                    screen.blit(label, (rect.x + 10, rect.y + 10))
                else:
                    title = font.render("Noen", True, BLACK)
                    screen.blit(title, model_data_gan_input_img_none[i])

        if int(count_data_gan)>=(int(data_town_num)*int(count)-5):
            state = hyperparameter

    elif state == hyperparameter:
        title = font.render("하이퍼파라미터 설정", True, BLACK)
        screen.blit(title, (50, 50))

        title = font.render("옵티마이저", True, BLACK)
        screen.blit(title, (800, 150))
        for idx, t in enumerate(opt):
            r = checkbox_rect(idx ,0 ,top_left=(800, 200),cell_h=50)
            draw_checkbox(r, opt_checkboxes[t], t)

        title = font.render("저장 옵션", True, BLACK)
        screen.blit(title, (1100, 150))
        for idx, t in enumerate(save):
            r = checkbox_rect(idx ,0,top_left=(1100, 200),cell_h=50)
            draw_checkbox(r, save_checkboxes[t], t)

        title = font.render("추가 저장 포멧", True, BLACK)
        screen.blit(title, (1100, 450))
        for idx, t in enumerate(save_other):
            r = checkbox_rect(idx ,0,top_left=(1100, 500),cell_h=50)
            draw_checkbox(r, save_other_checkboxes[t], t)


        label_text = font_small.render("Epochs", True, BLACK)
        label_pos = (Epochs_rect.x, Epochs_rect.y - 25)
        screen.blit(label_text, label_pos)
        pygame.draw.rect(screen, (200,200,200) if hyperparameter_input == "Epochs" else WHITE, Epochs_rect)
        pygame.draw.rect(screen, BLACK, Epochs_rect, 2)
        screen.blit(font.render(f"{input_Epochs}", True, BLACK), (Epochs_rect.x + 5, Epochs_rect.y + 5))

        label_text = font_small.render("배치", True, BLACK)
        label_pos = (batch_rect.x, batch_rect.y - 25)
        screen.blit(label_text, label_pos)
        pygame.draw.rect(screen, (200,200,200) if hyperparameter_input == "batch" else WHITE, batch_rect)
        pygame.draw.rect(screen, BLACK, batch_rect, 2)
        screen.blit(font.render(f"{input_batch}", True, BLACK), (batch_rect.x + 5, batch_rect.y + 5))

        label_text = font_small.render("학습률", True, BLACK)
        label_pos = (LR_rect.x, LR_rect.y - 25)
        screen.blit(label_text, label_pos)
        pygame.draw.rect(screen, (200,200,200) if hyperparameter_input == "LR" else WHITE, LR_rect)
        pygame.draw.rect(screen, BLACK, LR_rect, 2)
        screen.blit(font.render(f"{input_LR}", True, BLACK), (LR_rect.x + 5, LR_rect.y + 5))

    elif state == train_modal_select:

        title = font.render("학습 모델 설정", True, BLACK)
        screen.blit(title, (50, 50))
        title = font_small.render("회색 사용 불가, 파랑 사용, 빨간 사용 안함", True, BLUE)
        screen.blit(title, (50, 90))

        with open('data/sensor_config.yaml', 'r') as f:
            cfg = yaml.safe_load(f)
        sensor_config=cfg['sensors']

        key=list(sensor_config.keys())
        if Front_cam_color != RED:
            if 'front_cam' in key:
                Front_cam_color=BLUE
            else:
                Front_cam_color=GRAY
        if Left_cam_color != RED:
            if 'left_cam' in key:
                Left_cam_color=BLUE
            else:
                Left_cam_color=GRAY
        if lidar_color != RED:
            if 'lidar' in key:
                lidar_color=BLUE
            else:
                lidar_color=GRAY
        if Rear_cam_color != RED:
            if 'rear_cam' in key:
                Rear_cam_color=BLUE
            else:
                Rear_cam_color=GRAY
        if Rear_left_cam_color != RED:
            if 'rear_left_cam' in key:
                Rear_left_cam_color=BLUE
            else:
                Rear_left_cam_color=GRAY
        if Rear_right_cam_color != RED:
            if 'rear_right_cam' in key:
                Rear_right_cam_color=BLUE
            else:
                Rear_right_cam_color=GRAY
        if Right_cam_color != RED:
            if 'right_cam' in key:
                Right_cam_color=BLUE
            else:
                Right_cam_color=GRAY

        
        draw_hoverable_button(screen, Front_cam_rect, Front_cam_color, "Front_cam", mouse_pos, font, font_2)
        draw_hoverable_button(screen, Left_cam_rect, Left_cam_color, "Left_cam", mouse_pos, font, font_2)
        draw_hoverable_button(screen, Right_cam_rect, Right_cam_color, "Right_cam", mouse_pos, font, font_2)
        draw_hoverable_button(screen, Rear_cam_rect, Rear_cam_color, "Rear_cam", mouse_pos, font, font_2)
        draw_hoverable_button(screen, Rear_right_cam_rect, Rear_right_cam_color, "Rear_right_cam", mouse_pos, font, font_2)
        draw_hoverable_button(screen, Rear_left_cam_rect, Rear_left_cam_color, "Rear_left_cam", mouse_pos, font, font_2)
        draw_hoverable_button(screen, lidar_rect, lidar_color, "lidar", mouse_pos, font, font_2)



        if sum(color == RED for color in [
            Front_cam_color,
            Left_cam_color,
            Right_cam_color,
            Rear_cam_color,
            Rear_right_cam_color,
            Rear_left_cam_color,
            lidar_color
        ]) > 3:
            bev_modal_color=RED

        title = font.render("입력 데이터", True, BLACK)
        screen.blit(title, train_modal_text)



        title = font.render(f"{train_input_mode}", True, BLACK)
        text_rect = title.get_rect()
        text_rect.topleft = RGB_modal_text.topleft 
        pygame.draw.rect(screen, BLACK, text_rect.inflate(10, 10), 2)  
        screen.blit(title, text_rect)

        pygame.draw.rect(screen, (200,200,200) if train_modal_select_input == "width" else WHITE, width_rect)
        pygame.draw.rect(screen, BLACK, width_rect, 2)
        screen.blit(font.render(f"넓이: {input_img_width}", True, BLACK), (width_rect.x + 5, width_rect.y + 5))

        pygame.draw.rect(screen, (200,200,200) if train_modal_select_input == "height" else WHITE, height_rect)
        pygame.draw.rect(screen, BLACK, height_rect, 2)
        screen.blit(font.render(f"높이: {input_img_height}", True, BLACK), (height_rect.x + 5, height_rect.y + 5))

        rects = [width_rect, height_rect, train_modal_text]
        
        pygame.draw.rect(screen, (0, 0, 0), boxing(rects), 3)

        pygame.draw.rect(screen, BLACK, output_modal_text, 2)
        screen.blit(font.render("출력 데이터", True, BLACK), (output_modal_text.x + 5, output_modal_text.y + 5))


        pygame.draw.rect(screen, BLACK, output_box, 3)
        pygame.draw.line(screen, BLACK,
                        (output_box.left, output_box.top + third_height),
                        (output_box.right, output_box.top + third_height),
                        2)
        pygame.draw.line(screen, BLACK,
                        (output_box.left, output_box.top + 2 * third_height),
                        (output_box.right, output_box.top + 2 * third_height),
                        2)


        draw_hoverable_button(screen, bev_button_modal, bev_modal_color, "BEV", mouse_pos, font, font_2)
        draw_hoverable_button(screen, seg_button_modal, seg_modal_color, "SEG", mouse_pos, font, font_2)
        
        draw_hoverable_button(screen, OD_button_modal, OD_modal_color, "OD", mouse_pos, font, font_2)
        draw_hoverable_button(screen, depth_button_modal, depth_modal_color, "Depth", mouse_pos, font, font_2)


        screen.blit(font_small.render("Image Encoder", True, BLACK), (rect_resnet_encoder.x + 30, rect_resnet_encoder.y - 50))
        draw_checkbox(rect_resnet_encoder, check_resnet_encoder, "Resnet",not check_efficientnet_encoder)
        draw_checkbox(rect_efficent_encoder, check_efficientnet_encoder, "EfficientNet",not check_resnet_encoder)
        rects=[rect_resnet_encoder,rect_efficent_encoder]
        pygame.draw.rect(screen, (0, 0, 0), boxing(rects,max_x_sum=180,max_y_sum=30), 3)

        if depth_modal_color==BLUE or seg_modal_color==BLUE:
            screen.blit(font_small.render("Depth, Seg Decoder", True, BLACK), (rect_resnet_decoder.x + 10, rect_resnet_decoder.y - 50))
            draw_checkbox(rect_resnet_decoder, check_fpn_decoder, "FPN",not check_bifpn_decoder)
            draw_checkbox(rect_efficent_decoder, check_bifpn_decoder, "Bi-FPN",not check_fpn_decoder)
            rects=[rect_resnet_decoder,rect_efficent_decoder]
            pygame.draw.rect(screen, (0, 0, 0), boxing(rects,max_x_sum=180,max_y_sum=30), 3)


        if bev_modal_color==BLUE:
            screen.blit(font_small.render("BEV Decoder", True, BLACK), (rect_resnet_bev_decoder.x + 30, rect_resnet_bev_decoder.y - 50))
            draw_checkbox(rect_resnet_bev_decoder, check_resnet_bev_decoder, "Resnet",not check_efficientnet_bev_decoder)
            draw_checkbox(rect_efficent_bev_decoder, check_efficientnet_bev_decoder, "EfficientNet",not check_resnet_bev_decoder)
            rects=[rect_resnet_bev_decoder,rect_efficent_bev_decoder]
            pygame.draw.rect(screen, (0, 0, 0), boxing(rects,max_x_sum=180,max_y_sum=30), 3)



    elif state == model_test2:
        title = font.render("모델 예상도", True, BLACK)
        screen.blit(title, (50, 50))
        title=font_small.render("Cam 마다 Seg, depth 선택이 다르면 학습이 불가", True, RED)
        screen.blit(title, (50, 90))

        title = font.render("입력 데이터", True, BLACK)
        screen.blit(title, model_test_input_img_text)
        rects = model_test_input_img_boxes + [model_test_input_img_text]
        [
            Front_cam_color,
            Left_cam_color,
            Right_cam_color,
            Rear_cam_color,
            Rear_right_cam_color,
            Rear_left_cam_color,
            lidar_color
        ]


        paths = {
            0: 'img/test_img/rgb_left/test.png',
            1: 'img/test_img/rgb_front/test.png',
            2: 'img/test_img/rgb_right/test.png',
            4: 'img/test_img/rgb_rear/test.png',
            3: 'img/test_img/rgb_rear_left/test.png',
            5: 'img/test_img/rgb_rear_right/test.png',
            7: 'img/test_img/top_view/test.jpg',
        }
        if Front_cam_color !=BLUE:
            del paths[1]
        if Left_cam_color !=BLUE:
            del paths[0]
        if Right_cam_color !=BLUE:
            del paths[2]
        if Rear_cam_color !=BLUE:
            del paths[4]
        if Rear_right_cam_color !=BLUE:
            del paths[5]
        if Rear_left_cam_color !=BLUE:
            del paths[3]
        if lidar_color !=BLUE:
            del paths[7]

        model_test_input_img = load_model_test_input_images(paths)



        pygame.draw.rect(screen, (0, 0, 0), boxing(rects,max_x_sum=20,max_y_sum=20), 3)
        for i, rect in enumerate(model_test_input_img_boxes):
            if i == 6 or i == 8:
                continue
            if model_test_input_img[i] is not None:
                img = pygame.transform.scale(model_test_input_img[i], (rect.width, rect.height))
                screen.blit(img, rect.topleft)
            else:
                pygame.draw.rect(screen, GRAY, rect, 2)
                title = font.render("Noen", True, BLACK)
                screen.blit(title, model_test_input_img_none[i])


        paths = {
        # Segmentation
        0: 'img/test_img/seg_left/test.png',
        1: 'img/test_img/seg_front/test.png',
        2: 'img/test_img/seg_right/test.png',
        3: 'img/test_img/seg_rear_left/test.png',
        4: 'img/test_img/seg_rear/test.png',
        5: 'img/test_img/seg_rear_right/test.png',

        # Depth
        6: 'img/test_img/depth_left/test.png',
        7: 'img/test_img/depth_front/test.png',
        8: 'img/test_img/depth_right/test.png',
        9: 'img/test_img/depth_rear_left/test.png',
        10: 'img/test_img/depth_rear/test.png',
        11: 'img/test_img/depth_rear_right/test.png',

        # BEV
        13: 'img/test_img/bev_re/test.jpg'
        }
        all_seg_true = all(
            sensor_config[name]['seg']
            for name in sensor_config
            if name != 'lidar'
        )
        if seg_modal_color != BLUE or not all_seg_true:
            del paths[1]
            del paths[0]
            del paths[2]
            del paths[4]
            del paths[5]
            del paths[3]
        else:
            if Front_cam_color !=BLUE:
                del paths[1]
            if Left_cam_color !=BLUE:
                del paths[0]
            if Right_cam_color !=BLUE:
                del paths[2]
            if Rear_cam_color !=BLUE:
                del paths[4]
            if Rear_right_cam_color !=BLUE:
                del paths[5]
            if Rear_left_cam_color !=BLUE:
                del paths[3]
        all_depth_true = all(
            sensor_config[name]['depth']
            for name in sensor_config
            if name != 'lidar'
        )
        if depth_modal_color != BLUE or not all_depth_true:
            del paths[7]
            del paths[6]
            del paths[8]
            del paths[10]
            del paths[11]
            del paths[9]
        else:
            if Front_cam_color !=BLUE:
                del paths[7]
            if Left_cam_color !=BLUE:
                del paths[6]
            if Right_cam_color !=BLUE:
                del paths[8]
            if Rear_cam_color !=BLUE:
                del paths[10]
            if Rear_right_cam_color !=BLUE:
                del paths[11]
            if Rear_left_cam_color !=BLUE:
                del paths[9]

        if bev_modal_color !=BLUE:
            del paths[13]

            
        model_test_output_img = load_model_test_output_images(paths)


        title = font.render("결과", True, BLACK)
        screen.blit(title, model_test_output_img_text)
        rects = model_test_output_img_boxes + [model_test_output_img_text]

        pygame.draw.rect(screen, (0, 0, 0), boxing(rects,max_x_sum=20,max_y_sum=20), 3)
        for i, rect in enumerate(model_test_output_img_boxes):
            if i == 12 or i == 14:
                continue
            if model_test_output_img[i] is not None:
                img = pygame.transform.scale(model_test_output_img[i], (rect.width, rect.height))
                screen.blit(img, rect.topleft)
            else:
                pygame.draw.rect(screen, GRAY, rect, 2)
                title = font.render("Noen", True, BLACK)
                screen.blit(title, model_test_output_img_none[i])

    elif state == Train_loop:
        title = font.render("학습 대기열", True, BLACK)
        screen.blit(title, (50, 50))
        title=font_small.render("예약을 해두면 학습이 자동이 됩니다.", True, RED)
        screen.blit(title, (50, 90))
        if total_pages > 1:
            pygame.draw.rect(screen, BLUE, R_button, border_radius=6)
            if R_button.collidepoint(mouse_pos):
                screen.blit(font_2.render("==>", True, WHITE), 
                            font_2.render("==>", True, WHITE).get_rect(center=R_button.center))
            else:
                screen.blit(font.render("==>", True, WHITE), 
                            font.render("==>", True, WHITE).get_rect(center=R_button.center))

            pygame.draw.rect(screen, BLUE, L_button, border_radius=6)
            if L_button.collidepoint(mouse_pos):
                screen.blit(font_2.render("<==", True, WHITE), 
                            font_2.render("<==", True, WHITE).get_rect(center=L_button.center))
            else:
                screen.blit(font.render("<==", True, WHITE), 
                            font.render("<==", True, WHITE).get_rect(center=L_button.center))

            screen.blit(font.render(f'{page+1}/{total_pages}', True, BLACK), (R_button.x - 20, L_button.y - 50))

        pygame.draw.rect(screen, BLACK, central_rect, 3)


        total_pages = (len(Training_loop) - 1) // buttons_per_page + 1

        start_idx = page * buttons_per_page
        end_idx = min(start_idx + buttons_per_page, len(Training_loop))

        start_x = central_rect.x + 50
        start_y = central_rect.y + 30 
        delete_buttons2 = []

        for i, train_config in enumerate(Training_loop[start_idx:end_idx]):
            row = i // buttons_per_row
            col = i % buttons_per_row

            btn_x = start_x + col * (button_width + gap_x)
            btn_y = start_y + row * (button_height + gap_y) 
            btn_rect = pygame.Rect(btn_x, btn_y, button_width, button_height)
            name = f'{i}번 '
    
            pygame.draw.rect(screen, BLUE, btn_rect)
            current_rect = btn_rect

            with open('data/sensor_config.yaml', 'r') as f:
                sensor_config = yaml.safe_load(f)
            sensor_use = train_config['sensor_use']
            num_cam_sensors = sum(v for k, v in sensor_use.items() if k != 'lidar' and v)
            enabled_models = ' '.join([k for k, v in train_config['model_conf'].items() if v])
            town_nums_str = ', '.join([t.replace("Town", "") for t in sensor_config['towns']])

            screen.blit(font_small.render(f"Epochs: {train_config['Epochs']}, batch: {train_config['batch']}, LR: {train_config['LR']}", True, WHITE), (current_rect.x + 10, current_rect.y + 45))
            screen.blit(font_small.render(f'Use Task: {enabled_models}', True, WHITE), (current_rect.x + 10, current_rect.y + 75))
            screen.blit(font_small.render(f"UsedTown: {town_nums_str}", True, WHITE), (current_rect.x + 10, current_rect.y + 105))
            screen.blit(font_small.render(f'Use lidar: {train_config["lidar"]}, Use cam:{num_cam_sensors}, {train_config["width"]} X {train_config["height"]} ', True, WHITE), (current_rect.x + 10, current_rect.y + 135))
            screen.blit(font_small.render(f'Img-Encoder: {train_config["model"]["encoder"]}', True, WHITE), (current_rect.x + 10, current_rect.y + 165))
            screen.blit(font_small.render(f'Decoder: {train_config["model"]["decoder"]}, BEV:{train_config["model"]["bev_decoder"]}', True, WHITE), (current_rect.x + 10, current_rect.y + 195))


            screen.blit(font_small.render(f"Data: {format_number(sensor_config['data_num']*len(sensor_config['towns']))}", True, WHITE), (current_rect.x + 230, current_rect.y + 15))
            screen.blit(font.render(name, True, WHITE), (current_rect.x + 10, current_rect.y + 10))

            del_btn_rect = pygame.Rect(btn_rect.right - 50, btn_rect.bottom - 40, 40, 30)
            hovered = del_btn_rect.collidepoint(mouse_pos)

            if hovered:
                bigger_rect = pygame.Rect(del_btn_rect.x - 3, del_btn_rect.y - 3, del_btn_rect.width + 6, del_btn_rect.height + 6)
                pygame.draw.rect(screen, RED, bigger_rect, border_radius=6)
                screen.blit(font.render("X", True, WHITE), font.render("X", True, WHITE).get_rect(center=bigger_rect.center))
                delete_buttons2.append((bigger_rect, i))
            else:
                pygame.draw.rect(screen, RED, del_btn_rect, border_radius=4)
                screen.blit(font_small.render("X", True, WHITE), font_small.render("X", True, WHITE).get_rect(center=del_btn_rect.center))
                delete_buttons2.append((del_btn_rect, i))

            # 이름 표시
            screen.blit(font.render(name, True, WHITE), (btn_rect.x + 10, btn_rect.y + 10))


    elif state == Train:
        bar_w = 1000
        if os.path.isfile('data/results.log'):
            log=get_last_log_values('data/results.log')
            if log=="VAL":
                title = font.render("검증용 데이터셋으로 평가하는중", True, BLACK)
                screen.blit(title, (50, 50))
                title = font.render(f"Epoch: {epoch}/{input_Epochs}", True, BLACK)
                screen.blit(title, (150, 120))

                try:
                    max_scroll_value = len(glob.glob('data/predict/*'))
                    if new:
                        scroll_x=max_scroll_value
                    handle_x = scrollbar_rect.x + int(scroll_x / max_scroll_value * (scrollbar_rect.width - handle_width))
                    handle_rect = pygame.Rect(handle_x, scrollbar_rect.y - 5, handle_width, 20)

                    pygame.draw.rect(screen, (180, 180, 180), scrollbar_rect)
                    pygame.draw.rect(screen, (100, 100, 255), handle_rect)
                    scroll_value_text = font.render(f"{int(scroll_x)}", True, BLACK)
                    text_x = scrollbar_rect.centerx - scroll_value_text.get_width() // 2
                    screen.blit(scroll_value_text, (text_x, scrollbar_rect.y + 25))

                    if int(scroll_x)==max_scroll_value:
                        new=True
                        model_train_input_img[2] =cv2.imread(f'data/GT/0/GT.png')
                        model_train_input_img[3] =cv2.imread(f'data/GT/1/GT.png')
                        model_train_input_img[6] =cv2.imread(f'data/GT/2/GT.png')
                        model_train_input_img[7] =cv2.imread(f'data/GT/3/GT.png')

                        model_train_input_img[0] = safe_imread(f'data/predict/{log["counter"]}/0/predict.png')
                        model_train_input_img[1] = safe_imread(f'data/predict/{log["counter"]}/1/predict.png')
                        model_train_input_img[4] = safe_imread(f'data/predict/{log["counter"]}/2/predict.png')
                        model_train_input_img[5] = safe_imread(f'data/predict/{log["counter"]}/3/predict.png')


                        for i, rect in enumerate(model_train_input_img_boxes):
                            pygame.draw.rect(screen, GRAY, rect, 2)
                            if model_train_input_img[i] is not None:
                                img = pygame.transform.scale(pygame.surfarray.make_surface(model_train_input_img[i].swapaxes(1, 0)), (rect.width, rect.height))
                                screen.blit(img, rect.topleft)
                                label = font.render(f"{i+1}", True, RED)
                                screen.blit(label, (rect.x + 10, rect.y + 10))
                    else:
                        new=False
                        model_train_input_img[2] =cv2.imread(f'data/GT/0/GT.png')
                        model_train_input_img[3] =cv2.imread(f'data/GT/1/GT.png')
                        model_train_input_img[6] =cv2.imread(f'data/GT/2/GT.png')
                        model_train_input_img[7] =cv2.imread(f'data/GT/3/GT.png')
                        if scroll_x !=0:
                            model_train_input_img[0] = safe_imread(f'data/predict/{int(scroll_x)*20}/0/predict.png')
                            model_train_input_img[1] = safe_imread(f'data/predict/{int(scroll_x)*20}/1/predict.png')
                            model_train_input_img[4] = safe_imread(f'data/predict/{int(scroll_x)*20}/2/predict.png')
                            model_train_input_img[5] = safe_imread(f'data/predict/{int(scroll_x)*20}/3/predict.png')


                        for i, rect in enumerate(model_train_input_img_boxes):
                            pygame.draw.rect(screen, GRAY, rect, 2)
                            if model_train_input_img[i] is not None:
                                img = pygame.transform.scale(pygame.surfarray.make_surface(model_train_input_img[i].swapaxes(1, 0)), (rect.width, rect.height))
                                screen.blit(img, rect.topleft)
                                label = font.render(f"{i+1}", True, RED)
                                screen.blit(label, (rect.x + 10, rect.y + 10))
                except Exception as e:
                    continue
            elif log==None:
                title = font.render("학습", True, BLACK)
                screen.blit(title, (50, 50))
                title = font.render(f"Epoch: {epoch}/{input_Epochs}", True, BLACK)
                screen.blit(title, (150, 120))

                try:
                    max_scroll_value = len(glob.glob('data/predict/*'))
                    if new:
                        scroll_x=max_scroll_value
                    handle_x = scrollbar_rect.x + int(scroll_x / max_scroll_value * (scrollbar_rect.width - handle_width))
                    handle_rect = pygame.Rect(handle_x, scrollbar_rect.y - 5, handle_width, 20)

                    pygame.draw.rect(screen, (180, 180, 180), scrollbar_rect)
                    pygame.draw.rect(screen, (100, 100, 255), handle_rect)
                    scroll_value_text = font.render(f"{int(scroll_x)}", True, BLACK)
                    text_x = scrollbar_rect.centerx - scroll_value_text.get_width() // 2
                    screen.blit(scroll_value_text, (text_x, scrollbar_rect.y + 25))

                    if int(scroll_x)==max_scroll_value:
                        new=True
                        model_train_input_img[2] =cv2.imread(f'data/GT/0/GT.png')
                        model_train_input_img[3] =cv2.imread(f'data/GT/1/GT.png')
                        model_train_input_img[6] =cv2.imread(f'data/GT/2/GT.png')
                        model_train_input_img[7] =cv2.imread(f'data/GT/3/GT.png')
                        model_train_input_img[0] = safe_imread(f'data/predict/{log["counter"]}/0/predict.png')
                        model_train_input_img[1] = safe_imread(f'data/predict/{log["counter"]}/1/predict.png')
                        model_train_input_img[4] = safe_imread(f'data/predict/{log["counter"]}/2/predict.png')
                        model_train_input_img[5] = safe_imread(f'data/predict/{log["counter"]}/3/predict.png')


                        for i, rect in enumerate(model_train_input_img_boxes):
                            pygame.draw.rect(screen, GRAY, rect, 2)
                            if model_train_input_img[i] is not None:
                                img = pygame.transform.scale(pygame.surfarray.make_surface(model_train_input_img[i].swapaxes(1, 0)), (rect.width, rect.height))
                                screen.blit(img, rect.topleft)
                                label = font.render(f"{i+1}", True, RED)
                                screen.blit(label, (rect.x + 10, rect.y + 10))
                    else:
                        new=False
                        model_train_input_img[2] =cv2.imread(f'data/GT/0/GT.png')
                        model_train_input_img[3] =cv2.imread(f'data/GT/1/GT.png')
                        model_train_input_img[6] =cv2.imread(f'data/GT/2/GT.png')
                        model_train_input_img[7] =cv2.imread(f'data/GT/3/GT.png')
                        if scroll_x !=0:
                            model_train_input_img[0] = safe_imread(f'data/predict/{int(scroll_x)*20}/0/predict.png')
                            model_train_input_img[1] = safe_imread(f'data/predict/{int(scroll_x)*20}/1/predict.png')
                            model_train_input_img[4] = safe_imread(f'data/predict/{int(scroll_x)*20}/2/predict.png')
                            model_train_input_img[5] = safe_imread(f'data/predict/{int(scroll_x)*20}/3/predict.png')


                        for i, rect in enumerate(model_train_input_img_boxes):
                            pygame.draw.rect(screen, GRAY, rect, 2)
                            if model_train_input_img[i] is not None:
                                img = pygame.transform.scale(pygame.surfarray.make_surface(model_train_input_img[i].swapaxes(1, 0)), (rect.width, rect.height))
                                screen.blit(img, rect.topleft)
                                label = font.render(f"{i+1}", True, RED)
                                screen.blit(label, (rect.x + 10, rect.y + 10))
                except Exception as e:
                    continue
            elif log=='END':
                state = metric

            else:
                title = font.render("학습", True, BLACK)
                screen.blit(title, (50, 50))
                epoch=log['epoch']
                
                pygame.draw.rect(screen, GRAY, (150, 50, bar_w, 50))

                progress = int(log['batchi']) / int(log['last_idx'])
                pygame.draw.rect(screen, BLUE, (150, 50, int(bar_w * progress), 50))

                title = font.render(f"Epoch: {log['epoch']}/{input_Epochs}", True, BLACK)
                screen.blit(title, (150, 120))

                title = font.render(f"{log['batchi']} / {log['last_idx']}", True, BLACK)
                screen.blit(title, (1180, 50))

                title = font.render(f"Loss: {log['loss']}", True, BLACK)
                screen.blit(title, (550, 120))


                title = font.render(f"vehicle_iou : {log['vehicle_iou']}", True, BLACK)
                screen.blit(title, (150, 160))
                title = font.render(f"walker_iou : {log['walker_iou']}", True, BLACK)
                screen.blit(title, (420, 160))
                title = font.render(f"lane_iou : {log['lane_iou']}", True, BLACK)
                screen.blit(title, (660, 160))
                title = font.render(f"das_iou : {log['das_iou']}", True, BLACK)
                screen.blit(title, (880, 160))
                title = font.render(f"m_iou : {log['m_iou']}", True, BLACK)
                screen.blit(title, (1100, 160))

                try:
                    max_scroll_value = len(glob.glob('data/predict/*'))
                    if new:
                        scroll_x=max_scroll_value
                    handle_x = scrollbar_rect.x + int(scroll_x / max_scroll_value * (scrollbar_rect.width - handle_width))
                    handle_rect = pygame.Rect(handle_x, scrollbar_rect.y - 5, handle_width, 20)

                    pygame.draw.rect(screen, (180, 180, 180), scrollbar_rect)
                    pygame.draw.rect(screen, (100, 100, 255), handle_rect)
                    scroll_value_text = font.render(f"{int(scroll_x)}", True, BLACK)
                    text_x = scrollbar_rect.centerx - scroll_value_text.get_width() // 2
                    screen.blit(scroll_value_text, (text_x, scrollbar_rect.y + 25))

                    if int(scroll_x)==max_scroll_value:
                        new=True
                        model_train_input_img[2] =cv2.imread(f'data/GT/0/GT.png')
                        model_train_input_img[3] =cv2.imread(f'data/GT/1/GT.png')
                        model_train_input_img[6] =cv2.imread(f'data/GT/2/GT.png')
                        model_train_input_img[7] =cv2.imread(f'data/GT/3/GT.png')

                        model_train_input_img[0] = safe_imread(f'data/predict/{log["counter"]}/0/predict.png')
                        model_train_input_img[1] = safe_imread(f'data/predict/{log["counter"]}/1/predict.png')
                        model_train_input_img[4] = safe_imread(f'data/predict/{log["counter"]}/2/predict.png')
                        model_train_input_img[5] = safe_imread(f'data/predict/{log["counter"]}/3/predict.png')


                        for i, rect in enumerate(model_train_input_img_boxes):
                            pygame.draw.rect(screen, GRAY, rect, 2)
                            if model_train_input_img[i] is not None:
                                img = pygame.transform.scale(pygame.surfarray.make_surface(model_train_input_img[i].swapaxes(1, 0)), (rect.width, rect.height))
                                screen.blit(img, rect.topleft)
                                label = font.render(f"{i+1}", True, RED)
                                screen.blit(label, (rect.x + 10, rect.y + 10))
                    else:
                        new=False
                        model_train_input_img[2] =cv2.imread(f'data/GT/0/GT.png')
                        model_train_input_img[3] =cv2.imread(f'data/GT/1/GT.png')
                        model_train_input_img[6] =cv2.imread(f'data/GT/2/GT.png')
                        model_train_input_img[7] =cv2.imread(f'data/GT/3/GT.png')

                        model_train_input_img[0] = safe_imread(f'data/predict/{int(scroll_x)*20}/0/predict.png')
                        model_train_input_img[1] = safe_imread(f'data/predict/{int(scroll_x)*20}/1/predict.png')
                        model_train_input_img[4] = safe_imread(f'data/predict/{int(scroll_x)*20}/2/predict.png')
                        model_train_input_img[5] = safe_imread(f'data/predict/{int(scroll_x)*20}/3/predict.png')

                        for i, rect in enumerate(model_train_input_img_boxes):
                            pygame.draw.rect(screen, GRAY, rect, 2)
                            if model_train_input_img[i] is not None:
                                img = pygame.transform.scale(pygame.surfarray.make_surface(model_train_input_img[i].swapaxes(1, 0)), (rect.width, rect.height))
                                screen.blit(img, rect.topleft)
                                label = font.render(f"{i+1}", True, RED)
                                screen.blit(label, (rect.x + 10, rect.y + 10))
                except Exception as e:
                    continue
        else:
            title = font.render("학습 준비중 . . .", True, BLACK)
            title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
            screen.blit(title, title_rect)

    elif state == metric:

        title = font.render("성능표", True, BLACK)
        screen.blit(title, (50, 50))
        title=font_small.render("회색은 학습이 끝까지 진행을 하지못하여서 진행 불가", True, RED)
        screen.blit(title, (50, 90))

        if total_pages > 1:
            pygame.draw.rect(screen, BLUE, R_button, border_radius=6)
            if R_button.collidepoint(mouse_pos):
                screen.blit(font_2.render("==>", True, WHITE), 
                            font_2.render("==>", True, WHITE).get_rect(center=R_button.center))
            else:
                screen.blit(font.render("==>", True, WHITE), 
                            font.render("==>", True, WHITE).get_rect(center=R_button.center))

            pygame.draw.rect(screen, BLUE, L_button, border_radius=6)
            if L_button.collidepoint(mouse_pos):
                screen.blit(font_2.render("<==", True, WHITE), 
                            font_2.render("<==", True, WHITE).get_rect(center=L_button.center))
            else:
                screen.blit(font.render("<==", True, WHITE), 
                            font.render("<==", True, WHITE).get_rect(center=L_button.center))

            screen.blit(font.render(f'{page+1}/{total_pages}', True, BLACK), (R_button.x - 20, L_button.y - 50))

        pygame.draw.rect(screen, BLACK, central_rect, 3)


        Training_results = sorted(glob.glob('Training_result/*'))
        total_pages = (len(Training_results) - 1) // buttons_per_page + 1

        start_idx = page * buttons_per_page
        end_idx = min(start_idx + buttons_per_page, len(Training_results))

        start_x = central_rect.x + 50
        start_y = central_rect.y + 30 
        delete_buttons = []
        Training_result=[]
        clickable_buttons=[]
        for i, result in enumerate(Training_results):
            has_data = len(glob.glob(f'{result}/*.png')) > 1
            if has_data:
                Training_result.append(result)

        for i, result in enumerate(Training_results[start_idx:end_idx]):
            row = i // buttons_per_row
            col = i % buttons_per_row

            btn_x = start_x + col * (button_width + gap_x)
            btn_y = start_y + row * (button_height + gap_y) 
            btn_rect = pygame.Rect(btn_x, btn_y, button_width, button_height)
            name = os.path.basename(result)
        

            has_data = len(glob.glob(f'{result}/*.png')) > 1
            if has_data:
                hovered = btn_rect.collidepoint(mouse_pos)
                if hovered:
                    enlarged_rect = pygame.Rect(btn_rect.x - 5, btn_rect.y - 5, btn_rect.width + 10, btn_rect.height + 10)
                    pygame.draw.rect(screen, BLUE, enlarged_rect) 
                    current_rect = enlarged_rect
                else:
                    pygame.draw.rect(screen, BLUE, btn_rect)
                    current_rect = btn_rect

                with open(f'{result}/train_config.yaml', 'r') as f:
                    train_config = yaml.safe_load(f)

                with open(f'{result}/sensor_config.yaml', 'r') as f:
                    sensor_config = yaml.safe_load(f)
                # sensor_use = train_config['sensor_use']
                # num_cam_sensors = sum(v for k, v in sensor_use.items() if k != 'lidar' and v)
                num_cam_sensors=3
                enabled_models = ' '.join([k for k, v in train_config['model_conf'].items() if v])
                town_nums_str = ', '.join([t.replace("Town", "") for t in sensor_config['towns']])

                screen.blit(font_small.render(f"Epochs: {train_config['Epochs']}, batch: {train_config['batch']}, LR: {train_config['LR']}", True, WHITE), (current_rect.x + 10, current_rect.y + 45))
                screen.blit(font_small.render(f'Use Task: {enabled_models}', True, WHITE), (current_rect.x + 10, current_rect.y + 75))
                screen.blit(font_small.render(f"UsedTown: {town_nums_str}", True, WHITE), (current_rect.x + 10, current_rect.y + 105))
                screen.blit(font_small.render(f'Use lidar: {train_config["lidar"]}, Use cam:{num_cam_sensors}, {train_config["width"]} X {train_config["height"]} ', True, WHITE), (current_rect.x + 10, current_rect.y + 135))
                screen.blit(font_small.render(f'Img-Encoder: {train_config["model"]["encoder"]}', True, WHITE), (current_rect.x + 10, current_rect.y + 165))
                screen.blit(font_small.render(f'Decoder: {train_config["model"]["decoder"]}, BEV:{train_config["model"]["bev_decoder"]}', True, WHITE), (current_rect.x + 10, current_rect.y + 195))


                screen.blit(font_small.render(f"Data: {format_number(sensor_config['data_num']*len(sensor_config['towns']))}", True, WHITE), (current_rect.x + 230, current_rect.y + 15))
                screen.blit(font.render(name, True, WHITE), (current_rect.x + 10, current_rect.y + 10))
                clickable_buttons.append((current_rect))

            else:
                pygame.draw.rect(screen, GRAY, btn_rect)
                del_btn_rect = pygame.Rect(btn_rect.right - 50, btn_rect.bottom - 40, 40, 30)
                pygame.draw.rect(screen, RED, del_btn_rect)
                screen.blit(font_small.render("X", True, WHITE), (del_btn_rect.x + 14, del_btn_rect.y + 5))
                delete_buttons.append((del_btn_rect, result, name))
                screen.blit(font.render(name, True, WHITE), (btn_rect.x + 10, btn_rect.y + 10))

    elif state == metric_vis:
        selected_result=Training_result[selected_idx_metric]

        with open(f'{Training_result[selected_idx_metric]}/train_config.yaml', 'r') as f:
            selected_train_config = yaml.safe_load(f)

        with open(f'{Training_result[selected_idx_metric]}/sensor_config.yaml', 'r') as f:
            selected_sensor_config = yaml.safe_load(f)
        enabled_models = ' '.join([k for k, v in selected_train_config['model_conf'].items() if v])

        town_nums_str = ', '.join([t.replace("Town", "") for t in selected_sensor_config['towns']])
        title = font.render("성능표", True, BLACK)
        screen.blit(title, (50, 50))
        name = os.path.basename(selected_result)
                
        if metric_vis_img_mode == 'graph':
            base_colors = [BLUE, GRAY, GRAY]
        elif metric_vis_img_mode == 'img':
            base_colors = [GRAY, BLUE, GRAY]
        else:
            base_colors = [GRAY, GRAY, BLUE]

        buttons = [metric_vis_graph_button, metric_vis_img_button, metric_vis_score_button]
        labels = ["Graph", "Image", "Score"]

        for i, (btn, label) in enumerate(zip(buttons, labels)):
            color = base_colors[i]
            hovered = btn.collidepoint(mouse_pos)
            
            if hovered:
                bigger_rect = pygame.Rect(btn.x - 5, btn.y - 5, btn.width + 10, btn.height + 10)
                pygame.draw.rect(screen, color, bigger_rect, border_radius=8)
                screen.blit(font_2.render(label, True, BLACK), font_2.render(label, True, BLACK).get_rect(center=bigger_rect.center))
            else:
                pygame.draw.rect(screen, color, btn, border_radius=6)
                screen.blit(font.render(label, True, BLACK), font.render(label, True, BLACK).get_rect(center=btn.center))


        # --------- BEV, SEG, Depth ----------
        bev_active = selected_train_config['model_conf']['BEV']
        seg_active = selected_train_config['model_conf']['Seg']
        depth_active = selected_train_config['model_conf']['Depth']

        vis_buttons = [
            ("BEV", bev_active, metric_vis_BEV_button, metric_vis_mode == 'BEV'),
            ("SEG", seg_active, metric_vis_SEG_button, metric_vis_mode == 'Seg'),
            ("Depth", depth_active, metric_vis_Depth_button, metric_vis_mode == 'Depth'),
        ]

        for label, is_enabled, btn, is_selected in vis_buttons:
            if not is_enabled:
                pygame.draw.rect(screen, GRAY, btn, border_radius=6)
                screen.blit(font.render(label, True, BLACK), font.render(label, True, BLACK).get_rect(center=btn.center))
                continue

            color = BLUE if is_selected else BLACK
            hovered = btn.collidepoint(mouse_pos)
            
            if hovered:
                bigger_rect = pygame.Rect(btn.x - 5, btn.y - 5, btn.width + 10, btn.height + 10)
                pygame.draw.rect(screen, color, bigger_rect, border_radius=8)
                screen.blit(font_2.render(label, True, BLACK), font_2.render(label, True, BLACK).get_rect(center=bigger_rect.center))
            else:
                pygame.draw.rect(screen, color, btn, border_radius=6)
                screen.blit(font.render(label, True, BLACK), font.render(label, True, BLACK).get_rect(center=btn.center))


        if selected_idx_metric < len(Training_result) - 1:
            pygame.draw.rect(screen, BLUE, metric_R_button, border_radius=6)
            if metric_R_button.collidepoint(mouse_pos):
                screen.blit(font_2.render("==>", True, WHITE),
                            font_2.render("==>", True, WHITE).get_rect(center=metric_R_button.center))
            else:
                screen.blit(font.render("==>", True, WHITE),
                            font.render("==>", True, WHITE).get_rect(center=metric_R_button.center))
        else:
            pygame.draw.rect(screen, GRAY, metric_R_button, border_radius=6)
            screen.blit(font.render("==>", True, WHITE),
                        font.render("==>", True, WHITE).get_rect(center=metric_R_button.center))

        if selected_idx_metric > 0:
            pygame.draw.rect(screen, BLUE, metric_L_button, border_radius=6)
            if metric_L_button.collidepoint(mouse_pos):
                screen.blit(font_2.render("<==", True, WHITE),
                            font_2.render("<==", True, WHITE).get_rect(center=metric_L_button.center))
            else:
                screen.blit(font.render("<==", True, WHITE),
                            font.render("<==", True, WHITE).get_rect(center=metric_L_button.center))
        else:
            pygame.draw.rect(screen, GRAY, metric_L_button, border_radius=6)
            screen.blit(font.render("<==", True, WHITE),
                        font.render("<==", True, WHITE).get_rect(center=metric_L_button.center))



        if metric_vis_img_mode=='graph':

            if metric_vis_mode == 'BEV':
                for i in range(6):
                    rect = vis_bev_buttons[i]
                    text = vis_bev[i]
                    hovered = rect.collidepoint(mouse_pos)
                    selected = (i == selected_btn)

                    if hovered:
                        bigger_rect = pygame.Rect(rect.x - 5, rect.y - 5, rect.width + 10, rect.height + 10)
                        pygame.draw.rect(screen, BLUE if selected else WHITE, bigger_rect)
                        pygame.draw.rect(screen, BLACK, bigger_rect, 2)
                        draw_centered_text(screen, f"{text}", font_2, bigger_rect, BLACK)
                    else:
                        pygame.draw.rect(screen, BLUE if selected else WHITE, rect)
                        pygame.draw.rect(screen, BLACK, rect, 2)
                        draw_centered_text(screen, f"{text}", font, rect, BLACK)

                if selected_btn==0:
                    loss_img = cv2.imread(f'{selected_result}/loss.png')
                    img = cv2.cvtColor(loss_img, cv2.COLOR_BGR2RGB)
                elif selected_btn==1:
                    mean_iou_img = cv2.imread(f'{selected_result}/mean_iou.png')
                    img = cv2.cvtColor(mean_iou_img, cv2.COLOR_BGR2RGB)
                elif selected_btn==2:
                    das_iou_img = cv2.imread(f'{selected_result}/das_iou.png')
                    img = cv2.cvtColor(das_iou_img, cv2.COLOR_BGR2RGB)
                elif selected_btn==3:
                    walker_iou_img = cv2.imread(f'{selected_result}/walker_iou.png')
                    img = cv2.cvtColor(walker_iou_img, cv2.COLOR_BGR2RGB)
                elif selected_btn==4:
                    vehicle_iou_img = cv2.imread(f'{selected_result}/vehicle_iou.png')
                    img = cv2.cvtColor(vehicle_iou_img, cv2.COLOR_BGR2RGB)
                else:
                    lane_iou_img = cv2.imread(f'{selected_result}/lane_iou.png')
                    img = cv2.cvtColor(lane_iou_img, cv2.COLOR_BGR2RGB)


                img = pygame.transform.scale(pygame.surfarray.make_surface(img.swapaxes(1, 0)), (metric_vis_img.width, metric_vis_img.height))
                screen.blit(img, metric_vis_img.topleft)
        elif metric_vis_img_mode=='img':
            max_scroll_value_vis = len(glob.glob(f'{selected_result}/predict/*')) -1
            if max_scroll_value_vis<=scroll_x_vis:
                max_idx=True
            else:
                max_idx=False

            handle_x_vis = scrollbar_rect_vis.x + int(scroll_x_vis / max_scroll_value_vis * (scrollbar_rect_vis.width - handle_width_vis))
            handle_rect_vis = pygame.Rect(handle_x_vis, scrollbar_rect_vis.y - 5, handle_width_vis, 20)

            pygame.draw.rect(screen, (180, 180, 180), scrollbar_rect_vis)
            pygame.draw.rect(screen, (100, 100, 255), handle_rect_vis)
            scroll_value_text = font.render(f"{int(scroll_x_vis)}", True, BLACK)
            text_x = scrollbar_rect_vis.centerx - scroll_value_text.get_width() // 2
            screen.blit(scroll_value_text, (text_x, scrollbar_rect_vis.y - 35))

            metric_vis_input_img[2] =cv2.imread(f'{selected_result}/GT/0/GT.png')
            metric_vis_input_img[3] =cv2.imread(f'{selected_result}/GT/1/GT.png')
            metric_vis_input_img[6] =cv2.imread(f'{selected_result}/GT/2/GT.png')
            metric_vis_input_img[7] =cv2.imread(f'{selected_result}/GT/3/GT.png')

            metric_vis_input_img[0]=cv2.imread(f'{selected_result}/predict/{int(scroll_x_vis)*20}/0/predict.png')
            metric_vis_input_img[1]=cv2.imread(f'{selected_result}/predict/{int(scroll_x_vis)*20}/1/predict.png')
            metric_vis_input_img[4]=cv2.imread(f'{selected_result}/predict/{int(scroll_x_vis)*20}/2/predict.png')
            metric_vis_input_img[5]=cv2.imread(f'{selected_result}/predict/{int(scroll_x_vis)*20}/3/predict.png')

            for i, rect in enumerate(metric_vis_input_img_boxes):
                pygame.draw.rect(screen, GRAY, rect, 2)
                if metric_vis_input_img[i] is not None:
                    img = pygame.transform.scale(pygame.surfarray.make_surface(metric_vis_input_img[i].swapaxes(1, 0)), (rect.width, rect.height))
                    screen.blit(img, rect.topleft)
                    label = font.render(f"{i+1}", True, RED)
                    screen.blit(label, (rect.x + 10, rect.y + 10))
        else:
            Score_BEV = [
                'Train',     'Value','Best Epoch',
                'Loss',       '','',
                'mIoU',       '','',
                'Vehicle_IoU','','',
                'Walker_IoU', '','',
                'Lane_IoU',   '','',
                'Das_IoU',    '','',
            ]

            Score_BEV2 = [
                'Validation',     'Value','Best Epoch',
                'Loss',       '','',
                'mIoU',       '','',
                'Vehicle_IoU','','',
                'Walker_IoU', '','',
                'Lane_IoU',   '','',
                'Das_IoU',    '','',
            ]
            df = pd.read_csv(f'{selected_result}/train_metrics.csv')
            fill_score_list(df,BEV_metrics,Score_BEV, 'Train')
            fill_score_list(df,BEV_metrics,Score_BEV2, 'Validation')

            for i, rect in enumerate(metric_vis_text):
                pygame.draw.rect(screen, GRAY, rect, 2)
                draw_centered_text(screen, f"{Score_BEV[i]}", font, metric_vis_text[i], BLACK)

            for i, rect in enumerate(metric_vis_text2):
                pygame.draw.rect(screen, GRAY, rect, 2)
                draw_centered_text(screen, f"{Score_BEV2[i]}", font, metric_vis_text2[i], BLACK)
                
        #else:
        screen.blit(font_small.render(f"Epochs: {selected_train_config['Epochs']},    batch: {selected_train_config['batch']},   LR: {selected_train_config['LR']}   {selected_train_config['width']} X {selected_train_config['height']},   Use Task: {enabled_models},  Use lidar: {train_config['lidar']},   UsedTown: {town_nums_str}", True, BLACK),
                                     (270, 60))
        
        screen.blit(font_small.render(f'DataSize: {format_number(selected_sensor_config["data_num"]*len(selected_sensor_config["towns"]))},  Use cam:{num_cam_sensors},    Img-Encoder: {train_config["model"]["decoder"]}     Img-Decoder: {train_config["model"]["encoder"]},    BEV-Decoder:{train_config["model"]["bev_decoder"]}', True, BLACK),
                                     (270, 100))


        screen.blit(font_small.render(name, True, BLACK), (150, 60))


    elif state == re_Train:
        title = font.render("재학습", True, BLACK)
        screen.blit(title, (50, 50))

        text_surface = font.render("재학습", True, WHITE)
        text_rect = text_surface.get_rect(center=re_button.center)
        pygame.draw.rect(screen, BLUE, re_button)
        screen.blit(text_surface, text_rect)



    if setting:
        popup_rect = pygame.Rect(WINDOW_WIDTH // 2 - 200, WINDOW_HEIGHT // 2 - 100, 400, 200)
        pygame.draw.rect(screen, (50, 50, 50), popup_rect)
        pygame.draw.rect(screen, WHITE, popup_rect, 3)
        
        draw_centered_text(screen, "센서 세팅 중 . .", font_2, popup_rect, WHITE)

        setting2=True
        
        
    if confirm_delete:
        popup_rect = pygame.Rect(WINDOW_WIDTH // 2 - 200, WINDOW_HEIGHT // 2 - 100, 400, 200)
        pygame.draw.rect(screen, (50, 50, 50), popup_rect)
        pygame.draw.rect(screen, WHITE, popup_rect, 3)
        
        msg = font.render(f"Delete '{delete_name}'?", True, WHITE)
        screen.blit(msg, (popup_rect.x + 50, popup_rect.y + 40))

        yes_btn = pygame.Rect(popup_rect.x + 60, popup_rect.y + 120, 100, 40)
        no_btn = pygame.Rect(popup_rect.x + 240, popup_rect.y + 120, 100, 40)

        pygame.draw.rect(screen, RED, yes_btn)
        pygame.draw.rect(screen, BLUE, no_btn)

        screen.blit(font.render("Yes", True, WHITE), (yes_btn.x + 20, yes_btn.y + 5))
        screen.blit(font.render("No", True, WHITE), (no_btn.x + 25, no_btn.y + 5))

    if confirm_delete2:
        popup_rect = pygame.Rect(WINDOW_WIDTH // 2 - 200, WINDOW_HEIGHT // 2 - 100, 400, 200)
        pygame.draw.rect(screen, (50, 50, 50), popup_rect)
        pygame.draw.rect(screen, WHITE, popup_rect, 3)
        
        
        msg = font.render("삭제할까요?", True, WHITE)
        screen.blit(msg, (popup_rect.x + 120, popup_rect.y + 40))

        yes_btn = pygame.Rect(popup_rect.x + 60, popup_rect.y + 120, 100, 40)
        no_btn = pygame.Rect(popup_rect.x + 240, popup_rect.y + 120, 100, 40)

        pygame.draw.rect(screen, RED, yes_btn)
        pygame.draw.rect(screen, BLUE, no_btn)

        screen.blit(font.render("Yes", True, WHITE), (yes_btn.x + 20, yes_btn.y + 5))
        screen.blit(font.render("No", True, WHITE), (no_btn.x + 25, no_btn.y + 5))


    if state!= CARAL_INSTALL:
        if state in (MENU, data_gan, Train):
            draw_dynamic_button(screen, end_button, RED, (255, 80, 80), "종료", font, font_2, mouse_pos)
        elif state in (predict, re_Train,metric):
            draw_dynamic_button(screen, home2_button, GRAY, (180, 180, 180), "홈", font, font_2, mouse_pos)

        else:
            draw_dynamic_button(screen, confirm_button, BLUE, BLUE, "계속", font, font_2, mouse_pos)
            draw_dynamic_button(screen, home_button, GRAY, (180, 180, 180), "홈", font, font_2, mouse_pos)
            if state != Train_loop:
                draw_dynamic_button(screen, back_button, RED, (255, 80, 80), "뒤로", font, font_2, mouse_pos)

        if state == Train_loop:
            draw_dynamic_button(screen, loop_button, RED, (255, 80, 80), "추가", font, font_2, mouse_pos)




    pygame.display.flip()
    clock.tick(60)

pygame.quit()


