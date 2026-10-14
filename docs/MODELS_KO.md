# 모델·태스크·학습 설정

[사용 안내](USER_GUIDE_KO.md) · [데이터 파이프라인](PIPELINE_KO.md)

## 모델 선택

| 구성 | 선택값 |
| --- | --- |
| 이미지 Encoder | ResNet18 / ResNet34 / ResNet50 / ResNet101 |
| 이미지 Encoder | EfficientNet, EfficientNet_B0 / B1, MobileNetV3_Small / Large, ConvNeXt_Tiny |
| 이미지 Decoder | FPN / Bi-FPN / UNet |
| BEV Decoder | ResNet / EfficientNet / UNet |

선택 가능 여부는 연구 Python의 실제 PyTorch·torchvision·추가 의존성으로 결정합니다. `ResNet`은 ResNet34 별칭이며 `EfficientNet`은 efficientnet-pytorch 기반 선택입니다. BEV를 끄면 BEV Decoder와 계산을 건너뜁니다.

Encoder는 `(C5, C4, C3)`와 공간 stride `(32, 16, 8)`의 공통 특징맵 계약을 사용합니다. 객체 탐지 입력의 너비·높이는 128의 배수, Seg·Depth 입력은 32의 배수 조건을 사용합니다. 공식 사전학습 가중치 정보는 `sdv/training/model_registry.py`에 있으며 준비 검사에서 선택한 파일을 다운로드·검증합니다.

## 태스크와 손실

| 태스크 | 설정과 결과 |
| --- | --- |
| Segmentation | 실제 관측 클래스 선택, Background/Ignore 처리, 손실과 클래스별 IoU·Dice |
| Depth | 원본 거리 인코딩, 학습 거리·유효 마스크, RMSE·MAE·AbsRel 등 지표 |
| Object Detection | 실제 양성 박스의 클래스, 박스 손실과 AP·Precision·Recall |
| BEV | DAS·Lane·Vehicle·Pedestrian 레이어와 태스크별 손실·지표 |

Route·Traffic Light·Stop Line은 BEV 표시 레이어와 의미를 구분합니다. 선택 데이터에 필요한 라벨이 없으면 해당 태스크의 사용 조건을 화면에서 확인합니다. 깊이 학습 범위와 시각화 범위는 독립적이며 Invalid 값은 손실·지표에서 제외합니다.

## 학습 관리

Epoch, Batch, 학습률, Optimizer, 태스크별 양수 가중치와 예측 기록 주기를 설정합니다. 대기열에 넣은 작업은 당시 데이터·모델·보정 사본을 유지합니다. **일시정지**는 현재 Epoch를 마친 뒤 체크포인트를 저장하고, **재개**는 호환 상태에서 이어갑니다. **다시 시작**은 Epoch 1부터 새 실험을 만듭니다.

결과에서는 TRAIN·VAL을 구분해 지표, Best 값과 해당 Epoch를 확인합니다. Input / Ground truth / Prediction 탐색은 저장된 기록을 사용하며 실험 비교에는 같은 데이터셋 fingerprint가 필요합니다. 양성 박스가 없는 검증 분할은 탐지 성능을 판단하기에 충분하지 않습니다.

## BEV 기하와 체크포인트

기존 500×500, 0.2 m/pixel 라벨의 중앙 200×200 crop은 ±20 m 범위이며 전체 원본은 ±50 m입니다. 모델 projection grid와 카메라 optical/CARLA 축 처리는 별도 정합 검증이 필요합니다. 기록된 Calibration을 사용하되 BEV 예측을 Planning 입력으로 쓰기 전 이 계약을 검증해야 합니다.

새·이전 BEV 스키마와 데이터 처리 계약은 혼합하지 않습니다. 과거 체크포인트의 서명을 수동 변경해 재개하는 대신 호환 코드를 사용하거나 새 실험을 시작합니다. 프로토타입의 기능 동작과 모델 수렴·정확도는 각각 평가합니다.
