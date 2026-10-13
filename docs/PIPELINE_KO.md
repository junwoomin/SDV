# 시퀀스 분석·태스크 설정·Planning 데이터 파이프라인

이번 구현은 제공된 `collections.zip`의 실제 200개 Frame과 현재 수집·학습 코드를 비교한 결과에 기반합니다. 원본 자료, 연구 성능, 하드웨어 검증 여부를 구분합니다. 정량 감사는 [DATASET_AUDIT_KO.md](DATASET_AUDIT_KO.md), 새 수집 계약은 [COLLECTION_SCHEMA_V2_KO.md](COLLECTION_SCHEMA_V2_KO.md)에 있습니다.

## 실제 제공 자료

`collections/collection_13cd2a2077084429/Town01/scene_20261008_140246_0001/` 한 시퀀스입니다. Front RGB 400×300 PNG 200개를 기준으로 같은 basename의 `front_cam_seg`, `front_cam_depth`, `front_cam_bbs`, BEV 마스크 및 `info`가 대응합니다. 수집 스냅샷 `sensor_config.yaml`과 `collection_manifest.json`을 사용하며 현재 센서 설정으로 과거 보정을 대체하지 않습니다. 제공된 자료는 한 시퀀스이므로 운영 학습/검증 분할에는 별도로 수집한 독립 시퀀스가 필요합니다. CPU 기능 검사는 그 한 시퀀스의 명시적인 소수 Frame만 분리했으며, 연구 성능이나 누출 없는 검증 성적으로 해석하지 않습니다.

Segmentation에서 발견한 원본 ID는 **0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 14, 20, 21, 22, 23, 24, 26, 28**입니다. 이름·RGB 팔레트·클래스별 픽셀 수와 동일 샘플의 세 단계 이미지는 감사 문서에 있습니다. 기존 여섯 색의 의미가 CARLA 0.9.15와 어긋나 97.878%가 Other로 합쳐졌습니다. 선형 Resize로 만들어진 팔레트 밖 색상도 제거했습니다. 새 명시적 CARLA 설정에서는 정수 ID로 변환한 뒤 Nearest Resize하고 선택된 ID를 연속 Head 인덱스로 매핑합니다.

실제 `front_cam_bbs` 200개는 모두 `float32`, shape `(0,)`입니다. 수집 코드의 계약은 **N×5 `[x1,y1,x2,y2,class_id]`**, 원본 카메라 픽셀 좌표의 2D 박스입니다. 클래스 0 Vehicle, 1 Pedestrian이고 객체 ID·3D 박스는 이 파일에 없습니다. 빈 파일을 누락 파일과 구분해 `0 boxes`로 표시합니다. 기존 빈 Annotation을 RGB에서 임의로 복원하지 않습니다. 수집 코드의 구형 Semantic ID 필터와 카메라 투영행렬 재사용을 수정했으며, 새 CARLA 수집에서의 박스 품질은 미검증입니다.

기존 native-v1 Depth PNG는 이미 반올림된 미터 값입니다. 학습의 추가 `100/255` 배율을 제거했습니다. 255에 포화된 9,210,672/24,000,000 픽셀(38.378%)의 원래 거리는 알 수 없어 Invalid로 제외합니다. 새 수집은 float32 미터 NPY를 원본으로 저장하고 PNG는 미리보기로 유지합니다.

## 화면 사용

- **Simulation → 시퀀스 분석 / 저장 데이터 재생**: 세션과 카메라를 선택합니다. 같은 Frame의 RGB·Segmentation·Depth·OD·BEV 및 우측 Scene Information을 함께 조회합니다. 가로 Timeline의 클릭·드래그·이전/다음·좌우 키·재생/일시정지·속도를 사용합니다. 이미지 클릭으로 확대합니다. 저장 Frame 인덱스와 CARLA native Frame ID는 별개입니다.
- **라이브 수집**: 실제 저장 완료 Frame의 차량 정보를 갱신합니다. 라이브와 저장 재생을 구분하며 존재하는 BEV 레이어만 토글합니다. 시뮬레이터가 아직 저장하지 않은 이미지를 생성하지 않습니다.
- **Training → 고급 설정**: 기본 접힌 Segmentation/Depth/OD/BEV 설정에서 실제 선택 데이터의 클래스를 확인합니다. 새 초안은 관측된 CARLA ID를 제안합니다. 과거 실험에 `task_settings`가 없으면 이전 6채널 계약을 유지하며 기존 Checkpoint를 임의 변환하지 않습니다. 수정 이전의 일시정지 체크포인트는 데이터 처리 계약 버전이 달라 이 버전에서 재개를 거부합니다. 기존 코드로 재개하거나 새 실험을 만들어야 합니다. 올바른 CARLA 클래스 학습에는 새 실험을 만드세요.
- **Training → 모니터링**: Epoch Timeline에서 실제 기록과 고정 VAL 4개 예측을 조회합니다. TRAIN은 저장된 Step Timeline으로 탐색합니다. 예측이 저장되지 않은 Epoch는 데이터 없음입니다. 동일 데이터 내용의 VAL 표본 선택은 모델과 Epoch에 걸쳐 고정됩니다.
- Metric 표는 **Current / Best / Best Epoch / Maximum**을 표시합니다. Current는 선택 Epoch, Best는 실제 전체 기록의 방향별 최선입니다. Maximum은 이론적 상한이며 Loss·RMSE·MAE·AbsRel에는 None입니다. 표시만 소수점 셋째 자리이고 원래 값은 Tooltip, 로그와 파일에 유지합니다. 클래스별 IoU/Dice는 실제 실험 클래스 이름으로 접힌 상세 영역에 표시합니다.

## 태스크 설정의 실제 연결

| 태스크 | 적용 경로 | 제한 |
|---|---|---|
| Segmentation | 관측 ID 선택 → Loader remap → Head 채널 → Ignore/Background Loss → 혼동행렬·mIoU·mDice·정답/예측 팔레트 | 제외값은 명시적 Background 또는 Ignore 255. 원본 알 수 없는 색은 Ignore |
| Depth | 원본 단위 디코딩 → 유효거리 Mask/Clip → Loss/AbsRel·RMSE·MAE·δ1 → 별도 표시 범위/Color map | Invalid는 항상 제외. 학습 범위와 시각화 범위는 별개 |
| Detection | 관측 클래스 선택 → 원본 클래스/Head 인덱스 remap → Head/Loss/AP·Precision·Recall → 카메라 박스 Overlay | 제공 자료는 양성 Box 0개로 클래스 선택 불가. 긍정 박스 검사는 명시적 단위시험 자료 사용 |
| BEV | 실제 DAS/Lane/Vehicle/Pedestrian 레이어 선택 → 해당 Loss·Metric·예측 저장 | Route·Traffic Light·Stop Line은 표시 레이어. 새 임의 Semantic Head를 만들지 않음 |

BEV는 실제 forward/backward로 확인된 1개 이상의 카메라를 지원합니다. **BEV geometry는 아직 연구 검증이 필요합니다.** 기존 수집 BEV는 0.2m/px의 ±50m, 학습 라벨 crop과 모델 projection grid는 ±20m입니다. 기존 카메라 optical/CARLA 축 처리도 추가 감사 대상입니다. 보정을 검증했다고 표시하거나 변경 가능한 범위 옵션을 제공하지 않습니다. 과거 모델 동작을 유지하며 이 제한을 화면과 모델 문서에 표시합니다.

큐에 추가하는 순간 설정·데이터 경로·센서 보정을 실험별로 고정합니다. 실행 중 편집한 설정은 새 작업에만 적용됩니다. 기존 Epoch 경계 일시정지/재개, 순서 변경, 결과 보존 삭제 동작을 유지합니다.

## 저장·동기화·품질 검사

새 수집은 기존 파일에 필요한 Sidecar와 `info.sdv_metadata`를 추가합니다. 원본 필드를 중복 대체하지 않고 ego/road/navigation/traffic/environment/synchronization 및 출처를 저장합니다. RGB·라벨·LiDAR·IMU의 native Frame ID를 비교하며 불일치 Frame을 정상 자료로 저장하지 않습니다. 정확히 알 수 없는 Stop Line 거리·신호 가시성 등은 null입니다. CARLA API가 반환한 Ego 적용 신호와 단순 화면 가시성을 혼동하지 않습니다.

기존 `info`에는 속도·현재/다음 Control·위치/회전·IMU·Boolean 신호 위험·충돌·BEV 픽셀 waypoint가 있습니다. 기존 자료에 없는 native timestamp, Road/Lane ID, 신호 Enum 등은 **N/A**입니다. 설정 FPS로 실제 timestamp를 만들어 내지 않습니다.

품질 검사는 실제 전체 시퀀스의 Sensor 파일 정렬, 박스 좌표/클래스, Semantic 유효 ID, Depth Invalid/범위, 시간/Frame 누락, ego 이상값, Navigation/Traffic 누락을 Frame별 원인으로 반환합니다. 제공 자료에서는 첫 Frame IMU 가속도 norm 약 15,145.204m/s²도 경고합니다. 파일 basename 일치만으로 과거 센서 native Frame 동기화를 검증했다고 주장하지 않습니다.

## Planning 관측/Target 내보내기

선택한 한 시퀀스에서 JSON 내보내기를 제공합니다. 현재 관측 `inputs`와 이후 실제 `info`의 위치·제어에서 계산한 `targets.future_observations`를 분리하고 원본 이미지 경로와 Calibration을 참조합니다. 미래 ego 좌표는 원본 Frame의 전체 Pitch/Yaw/Roll 역변환으로 계산합니다. 세션 경계·누락·Timestamp 역전·Teleport를 넘어 연결하지 않습니다. Planner가 온라인에 제공한 계획 waypoint와 실제 미래 주행 Ground Truth는 별도입니다.

기존 데이터에는 시간이 없으므로 `offset_samples`만 제공하고 `offset_s=null`, 시간 연속성 미검증으로 표시합니다. 실제 TCP의 입력/Target 계약을 검증한 변환기는 아니며 **TCP 완전 호환을 주장하지 않습니다**. 향후 실제 TCP 코드에 맞춘 데이터 어댑터와 시계열 학습/검증 분할이 필요합니다.

## 실행 및 재현

```bash
chmod +x start.sh
./start.sh
# http://127.0.0.1:8765
```

업데이트된 웹 NumPy 의존성은 기존과 분리된 `.venv-web`에 자동 적용합니다. 기존 사용자 CARLA 자동 종료 동작은 유지됩니다. Python 3.7–3.10 웹 호환 검사와 CPU/브라우저 결과는 [VALIDATION_KO.md](VALIDATION_KO.md)를 확인하세요. 실제 CARLA 재수집, CUDA 장시간 학습, 성능 수렴, BEV 기하 보정, 실제 TCP 학습은 이번 환경에서 검증하지 않았습니다.
