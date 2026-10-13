# 모델 구성과 확장

## 지원 선택값

| 구성 | 선택값 | 조건 |
|---|---|---|
| 이미지 인코더 | ResNet18 / ResNet34 / ResNet50 / ResNet101 | 기존 `ResNet` 이름은 ResNet34의 별칭 |
| 이미지 인코더 | EfficientNet | 기존 efficientnet-pytorch B0와 체크포인트 구조 유지 |
| 이미지 인코더 | EfficientNet_B0 / EfficientNet_B1 | torchvision 0.11 이상 |
| 이미지 인코더 | MobileNetV3_Small / MobileNetV3_Large | torchvision 0.9 이상 |
| 이미지 인코더 | ConvNeXt_Tiny | torchvision 0.12 이상 |
| 이미지 디코더 | FPN / Bi-FPN / UNet | 같은 C5/C4/C3 입력, P3~P7 다섯 특징맵 출력 |
| BEV 디코더 | ResNet / EfficientNet / UNet | BEV 선택 시 사용; EfficientNet은 efficientnet-pytorch 필요 |

웹의 `/api/training/models`는 연구 Python에서 실제로 확인한 종속성·constructor 유무로 선택 가능 여부와 이유를 제공합니다. 지원하지 않는 환경의 모델은 회색으로 표시합니다. Python 버전 지원과 설치된 PyTorch/torchvision 버전별 모델 지원은 별개입니다. 기존 CUDA 패키지를 실행기가 자동 교체하지 않습니다.

화면 순서는 **학습 데이터 → 입력 센서 → 출력 태스크 → 모델 구성 → 학습 파라미터**입니다. BEV를 선택하지 않으면 BEV 디코더는 None이며 BEV 계산을 건너뜁니다. 객체 탐지는 기존 P3~P7 앵커와 일치하도록 입력 너비·높이 모두 128의 배수여야 합니다. Seg/Depth에는 기존 32의 배수 조건을 유지합니다.

## 수정한 특징맵 계약

기존 ResNet은 layer1~4의 네 특징맵을 얕은 순서로 반환했지만 FPN은 깊은 순서의 세 특징맵을 요구했습니다. 이 불일치가 `too many values to unpack (expected 3)`의 원인이었습니다. 새 인코더는 모두 `(C5, C4, C3)`, 공간 stride `(32, 16, 8)`을 반환하며 각 채널 수를 registry에 선언합니다. lifted depth 특징은 기존처럼 stride16이며 BEV geometry, 깊이 구간, 좌표 범위, 네 출력 맵과 손실 정의를 유지합니다.

Bi-FPN의 bottom-up 마지막 fusion에서도 stride16과 stride32 텐서를 함께 더하던 오류를 수정했습니다. 대상 특징맵 크기에 맞춰 pool/interpolate하며, 기존 가중 fusion과 학습 파라미터는 유지합니다. 새 UNet은 해당 공통 특징맵 계약을 구현한 추가 선택지입니다.

`model.py`는 배치를 모델이 놓인 장치로 이동합니다. 기존 CUDA 학습 호출을 유지하면서 CPU 계약 검사를 실행할 수 있습니다. 실제 전체 학습 엔진은 CUDA를 요구합니다. ResNet34와 기존 EfficientNet의 모듈명·state_dict key/shape를 유지했고 원본 소스와 비교하여 strict load와 lifted 출력 수치가 일치하는지 검사했습니다.

## 가중치 준비

모든 인코더의 공식 가중치 URL·파일명·SHA256 접두사는 `sdv/training/model_registry.py`에 있습니다. 준비 상태 검사는 선택한 인코더 파일만 백그라운드에서 다운로드·검증하며, 상태 조회에서는 다운로드하지 않습니다. ResNet과 ResNet34는 캐시와 다운로드 잠금을 공유합니다.

연구 인코더는 torchvision 버전별 `pretrained=True`의 파일명 차이를 피하도록 먼저 무가중치 모델을 만들고 registry의 고정 V1 state_dict를 로드합니다. 기존 EfficientNet 파일은 기존 구현과 분리하여 유지합니다. 기본값은 사전학습 사용이며 코드/YAML의 명시적 `model.pretrained: false`는 임의 초기화 검사를 허용합니다. 웹 기본 실행은 사전학습 가중치를 준비합니다.

## 추가 방법

1. `model_registry.py`의 BACKBONES에 ID, provider, constructor, family, C5/C4/C3 채널·특징 index, 최소 버전, 공식 가중치 metadata를 등록합니다. 웹 선택 목록·설정 검증·선택 파일 준비는 이 registry를 공유합니다.
2. ResNet 계열은 layer4/3/2 adapter를, torchvision의 `features` 계열은 선언한 index를 사용하는 adapter를 재사용합니다. 출력 구조가 다른 새 계열에는 `encoder.py`의 `build_encoder`에 해당 adapter를 연결합니다. ID 등록만으로 다른 구조의 모델이 호환된다고 간주하지 않습니다.
3. 이미지/BEV 디코더를 추가할 때는 registry의 지원 목록과 `model.py`의 decoder factory를 함께 등록하고 기존 출력 채널·shape 계약을 구현합니다.
4. 실제 연구 환경에서 `python -m pytest tests/test_model_contract.py -q`로 forward/backward와 채널·stride·앵커·BEV 출력 검사를 실행합니다. 원본 수치 비교 검사는 `SDV_LEGACY_MODELING_ROOT`에 보존한 원본 modeling 경로를 지정하면 실행되며, 원본이 없으면 해당 두 검사는 명시적으로 skip합니다.

## 확인 범위

실제 CPU 환경 Python3.10 / torch2.2.2 / torchvision0.17.2에서 11 인코더 ID × 3 이미지 디코더 33조합, Seg/Depth/OD 통합, BEV 3종, 원본 체크포인트 호환 등 **46개 검사**를 통과했습니다. 공식 ResNet34·기존 EfficientNet-B0·MobileNetV3 Small 파일은 실제 로딩·forward까지 확인했습니다. 다른 신규 가중치는 임의 초기화 계약만 검사했으며, CUDA 전체 Epoch 학습·수렴·실제 데이터셋 품질·LiDAR/torch_scatter·BEV train-mode normalization은 미검증입니다.

상세 기록: 실제 모델 검사 (local-only artifact: `model-contract-validation.json`; not distributed / 로컬 검증 자료·미배포), 웹 회귀 (local-only artifact: `model-regression-validation.json`; not distributed / 로컬 검증 자료·미배포), 브라우저 검사 (local-only artifact: `model-ux-browser-validation.json`; not distributed / 로컬 검증 자료·미배포).

## 태스크별 클래스·거리 설정과 최신 CPU 검사

`task_settings`는 작업을 대기열에 추가할 때 설정 파일에 고정됩니다. Segmentation의 실제 CARLA ID 선택은 데이터 로더의 Label Mapping, 모델 출력 채널, BCE+Dice 손실의 Ignore Mask, 전체 Epoch의 혼동행렬, 클래스 이름과 색상에 함께 적용됩니다. 제외 클래스는 Background 또는 Ignore로 선택합니다. 설정이 없는 기존 작업은 기존 6채널 순서와 체크포인트를 유지하는 `legacy` 모드로 읽습니다. 이 모드의 과거 이름을 새 CARLA 0.9.15 클래스 이름으로 해석하면 안 됩니다.

Depth는 수집 형식에 명시된 미터 단위를 읽습니다. 원본 float32 `_depth_m/*.npy`가 우선이며, 과거 native-v1 PNG의 255는 포화되어 원래 거리를 복원할 수 없는 값으로 제외합니다. 학습 거리 Mask/Clip과 표시 거리·Color Map은 독립적입니다. Object Detection은 선택한 원본 객체 ID를 연속 출력 채널로 변환합니다. BEV는 기존 DAS/Lane/Vehicle/Walker 가운데 선택한 레이어만 손실과 성능 지표에 포함합니다.

실제 업로드된 자료의 서로 다른 Train 4개·Val 4개 Frame으로 **19채널 Seg + Depth 1~80m + 빈 OD Annotation + DAS/Lane BEV**의 CPU Forward/Backward를 확인했습니다. 같은 실제 연구 엔진을 테스트 프로세스에서만 CPU로 옮겨 2 Epoch 연속 실행과 Epoch 1 일시정지·재개를 비교했습니다. 최종 모델 텐서, 옵티마이저 학습 결과, Scheduler, 기록된 지표와 고정 Val 4개의 GT PNG가 일치했습니다. 업로드의 OD 파일 200개는 모두 빈 Annotation이므로 실제 Positive Box의 성능은 검증되지 않았습니다. 선택 ID와 Box 손실 연결은 별도의 명시적인 합성 단위 검사로 확인했습니다.

BEV의 ResNet/EfficientNet/UNet 디코더는 카메라 1개와 2개 각각에 대해 실제 CPU Forward/Backward가 통과했습니다. 시야 범위와 모델 정확도가 보장된다는 뜻은 아닙니다. 검사 기록은 `task-pipeline-cpu-validation.json`, `pipeline-pytest-cpu.xml`에 있습니다. 실제 CARLA 수집·CUDA 실행·연구용 분할에 대한 수렴과 정확도는 미검증입니다.

## 기존 BEV 기하 변환의 제한

원본 BEV 라벨은 500×500, 0.2m/px 격자이며, 기존 로더는 중앙 `[150:350,150:350]`의 200×200 영역을 사용합니다. 모델의 X/Y Voxel 범위는 실제 소스상 `[-20,+20]m`, 간격 0.2m입니다. Lifted Feature의 카메라 깊이 샘플은 2m 이상 50m 미만입니다. 저장 원본의 전체 ±50m 범위와 모델이 학습하는 중앙 ±20m 영역은 구분해야 합니다.

기존 `get_cam_to_ego`는 SciPy XYZ Euler에 `[pitch, roll, yaw]`를 넣고, 카메라 Optical 축(X 오른쪽·Y 아래·Z 전방)과 CARLA 센서 축(X 전방·Y 오른쪽·Z 위) 사이의 명시적인 축 변환 없이 기존 BEV 역투영에 연결합니다. 이 부분은 별도의 기하 정합 검증과 기존 체크포인트 이행 계획이 필요합니다. 이번 수정에서는 과거 체크포인트의 변환 수식을 조용히 바꾸지 않았고, 수집한 Scene의 실제 Calibration을 읽도록 연결했습니다. **현재 BEV 출력을 기하 보정이 검증된 Planning 입력이나 TCP 호환 데이터라고 주장하지 않습니다.** BEV 범위·해상도·Ego 중심은 이 검증이 끝나기 전까지 읽기 전용입니다.

### 이전 일시정지 실험의 재개 제한

기존 6채널 모델의 **Head·state_dict 구조 호환**과 과거 Label 처리의 완전한 재현은 서로 다릅니다. 이번 버전은 동일한 설정 파일이어도 Label 보간을 Nearest로 수정하고, native-v1 Depth의 실제 미터 단위를 복원하며, Frame 이름으로 모달리티를 정렬합니다. 따라서 재개 체크포인트 서명에 `data_pipeline_schema: 2`를 추가했습니다. 이 표식이 없거나 다른 체크포인트는 모델·Optimizer 상태를 적용하기 전에 재개를 차단합니다. 이전 일시정지 작업은 그 체크포인트를 만든 이전 SDV 코드로 재개하거나, 수정된 코드에서 새 실험으로 시작해야 합니다. 이전 체크포인트의 서명을 수동으로 바꾸어 재개하는 것은 검증된 이행 방법이 아닙니다.
