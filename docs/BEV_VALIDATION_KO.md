# Height-aware BEV 최종 검증 결과

> 이전 v9 Patch 기반 구현의 검증 기록입니다. 현재 전체 PNG Crop·Rotation 방식의 결과는 [Global Map 통합 보고](GLOBAL_MAP_INTEGRATION_KO.md)를 확인하세요.

검증일: 2026-10-12. 제공된 SDV Studio v8을 기준으로 LRS/HBLR 소스를 확인하고 새 BEV 수집·저장·웹 조회·학습 경로를 연결했다. 이 보고서는 실제 자료 검증, 통제용 자동화 검사, 아직 실행하지 못한 환경을 구분한다.

## 최종 자동화 검사

| 환경 / 검사 | 통과 | 건너뜀 | 실패 |
|---|---:|---:|---:|
| Python 3.7.16 전체 `tests` | 565 | 20 | 0 |
| Python 3.12 웹 환경 전체 `tests` | 563 | 22 | 0 |
| Python 3.12 CPU PyTorch 모델·학습 관련 검사 | 152 | 0 | 0 |

앞의 두 행은 동일한 회귀 검사를 다른 환경에서 실행한 결과이며 서로 다른 테스트 수로 합산하지 않는다. 웹 환경은 의도적으로 연구용 Torch를 설치하지 않았다. 해당 의존성이 필요한 검사와 선택적 모듈 검사는 건너뛰었고, 모델·Prediction·Loss·Metric·Checkpoint는 별도 CPU Torch 환경에서 실행했다. Python 3.12 웹 환경에서는 CARLA native API 검사와 Python 3.7 전용 재현 검사도 건너뛰었다.

CPU 학습 검사는 실제 schema 3 샘플의 Dataset Loader, 모델 forward/backward, 유효 픽셀만 사용하는 Loss/Metric, 경사로 보정, Prediction 렌더링과 체크포인트 계약을 포함한다. GPU 학습 완료를 뜻하지 않는다. 웹 의존성의 httpx 및 기존 torchvision API 관련 deprecation 경고는 남아 있지만 실패는 없었다.

원본: `validation/bev-python37-tests.xml`, `bev-web-tests.xml`, `bev-training-tests.xml`.

```bash
# 웹/일반 회귀 검사: 각 Python 환경에 맞는 의존성 설치 후 실행
python -m pytest tests -q

# 연구용 CPU Torch 환경
python -m pytest tests/test_bev_training.py tests/test_model_contract.py \
  tests/test_task_metrics.py tests/test_predictions.py \
  tests/test_training_checkpointing.py tests/test_task_settings_training.py -q
```

프로젝트 전체 Python 파일의 Python 3.7 문법, `start.sh`의 Bash 문법, 변경된 JavaScript 문법을 검사했다. Python 3.7에서 BEV CLI와 실제 OpenDRIVE 캐시 생성도 실행했다. 이번 변경 전체를 Python 3.8/3.9/3.10에서 각각 실행한 것은 아니며, 이전 버전의 검증 기록과 구분한다.

## 실제 소스·지도 대조

LRS 원본 crop 함수를 직접 실행하여 Town01/04/05, 여러 Yaw와 실제 제공 raster를 비교한 **60건 모두 픽셀 차이 0**이었다. 이는 affine 재사용의 일치 검사이며 원본 PNG의 모든 Semantic 정답을 보증하는 것은 아니다.

CARLA 0.9.15의 실제 오프라인 `carla.Map`으로 제공 XODR을 읽었다. 최종 `carla_waypoint_ribbons_v8` 캐시는 2 m 간격으로 생성하고, 독립적인 1 m 간격 native Driving waypoint에서 높이 조회를 검사했다.

| Map | 조회 성공 / 검사 좌표 | Ground Z 최대 오차 | Ground Z P95 오차 |
|---|---:|---:|---:|
| Town01 | 6,461 / 6,461 | 0.000000 m | 0.000000 m |
| Town04 | 33,670 / 33,670 | 0.040908 m | 0.000200 m |
| Town05 | 24,212 / 24,212 | 0.014667 m | 0.000039 m |
| 합계 | **64,343 / 64,343** | — | — |

오차는 native waypoint Z와 연속 표면 보간 Z의 차이다. 카메라 정합 오차, raster 픽셀 양자화 오차 또는 도로 전체 면적의 정확도 점수가 아니다. 실제 지도 교차 높이 4곳에서도 상·하부 표면이 각각 선택되었다.

![Town04 동일 XY, 서로 다른 실제 도로 높이 비교](validation/bev-town04-overpass-reference-comparison.png)

Town04의 동일 XY에서 Z 0 m / 11 m로 LRS 원본, HBLR 연결 원본, 새 SDV Semantic 및 미터 높이를 비교했다. 위 그림과 숫자 NPZ는 실제 제공 지도·OpenDRIVE를 사용한 정적 출력이며 촬영된 RGB나 모델 Prediction이 아니다. 두 Ego Anchor의 높이 오차는 0 m다. 원본 간 Anchor, 정수화, Semantic 종류가 달라 세 구현 간 IoU를 정확도 점수로 만들지 않았다.

같은 XY에 배치한 통제용 Box 두 개는 아래 view에 ID 100만, 위 view에 ID 101만 포함되었다. 이 Actor 검사는 실제 Map 위의 합성 입력이며 실제 차량 수집 기록이 아니다. 그림과 정적 NPZ에는 이 Box를 넣지 않았다.

실제 지도 검증에서 발견한 시작점 누락, Road/Section 경계, 불연속 native 연결, 중앙 Shoulder 너머 반대 차로 문제를 수정했다. 확인할 수 없는 연결은 제외하고 좌표와 원인을 캐시 `build_report`에 기록한다. 상세 결과·재현 명령·원본 소스 해시는 [실제 지도 검증 보고](validation/BEV_REFERENCE_VALIDATION_KO.md)와 해당 JSON에 있다.

## 기존 데이터 및 학습 호환성

제공 `collections.zip`의 실제 **200 Frame / 800 BEV Mask**를 기존 Loader crop과 비교하여 픽셀 차이 **0**을 확인했다. DAS/Lane/Vehicle/Walker의 기존 18개 배치 항목과 4개 Head를 유지했다. 제공 데이터의 Vehicle/Walker mask는 모두 0이었으며, 이를 객체 검출 성능 검증으로 사용하지 않았다.

신규 schema 3에서는 물리 좌표 sampling, 독립 유효 mask, 같은 Frame의 실제 센서 보정을 사용한다. 선택하지 않은 Layer, 누락된 보정, 구형·신형 schema 혼합을 준비 검사에서 차단한다. 미확인 픽셀을 배경으로 학습하지 않는다. 웹 준비 검사가 Torch/CARLA를 import하지 않는 것도 확인했다.

높이 및 Actor 보조 tensor는 Ground Truth 경로로 제공한다. 기존 모델에 Height 예측 Head나 추가 입력 채널을 강제로 넣지 않았다. Planning export는 센서 입력, 시뮬레이터 GT, 실제 미래 관측 Target을 분리하며 TCP 전용 학습 호환성은 미검증이다.

원본: `validation/bev-training-legacy-validation.json`.

## 웹·오류 처리

Chromium 1920×1080, 1366×768, 390×844에서 가로 넘침 및 콘솔 오류가 없었다. KO/EN 설정 저장, BEV 수치 보기, Actor 선택, 키보드 Timeline, Legacy 높이 N/A, 같은 저장 완료 Frame의 라이브 정보 조회를 검사했다.

Timeline은 이미지 바로 위에 배치했다. 1366×768에서 Timeline 하단은 431 px, 이미지 하단은 763 px였고 다음 Frame 이동 후 페이지 스크롤은 0 px로 유지되었다. 1920×1080의 이미지 영역은 500 px 높이다.

실제 제공 데이터의 Frame 0/100/199에서 RGB·Segmentation·Depth·Object Detection·BEV 5종이 정상 렌더링되었다. 신규 숫자형 데이터 UI는 명시적인 통제용 fixture로 검사했으며 이를 실제 CARLA 수집 결과로 제시하지 않는다. 브라우저 측정 원본은 `bev-web-validation.json`이다.

손상된 NPZ와 잘못된 Geometry·Actor·Surface·Settings metadata는 경고/품질 오류 및 이미지 API 422로 처리했다. 서버 500 없이 다른 정상 Frame의 RGB는 계속 조회되었다. 독립 검토 회귀 검사도 전체 검사에 포함했다.

## 생성 성능

최종 캐시, 500×500, 0.2 m/pixel, warm cache에서 2회 준비 후 10회 측정했다. 실제 Town01 waypoint에 통제용 4.5×2 m Box를 배치했다. Intel Xeon Platinum 8573C 공유 CPU 환경, Python 3.7.16 / NumPy 1.21.6 / OpenCV 4.8.1이다.

| Actor 수 | 프레임 생성 중앙값 | P95 |
|---:|---:|---:|
| 6 | 71.054 ms | 81.017 ms |
| 20 | 85.118 ms | 95.055 ms |
| 100 | 158.522 ms | 279.116 ms |

CARLA 시뮬레이션, 센서 처리, 파일 저장, 웹 전송 시간은 포함하지 않는다. 실시간 수집 FPS를 보장하지 않으며, 높은 수집률이 필요한 경우 사용자 장비에서 전체 처리량을 측정해야 한다. ROI·캐시·제한된 그래프 탐색 최적화는 느린 기준 구현과 **25개 수치 채널에서 NaN을 포함하여 동일**했다. 성능을 위해 정답 채널을 근사하거나 제거하지 않았다.

## 미검증·미구현 범위

- 실제 GPU CARLA 서버 주행, RGB/Actor/BEV 동일 Frame 동기화, 경사로 연속 진입·탈출, 주변 Actor 누락률, 신호 전환 및 실제 Stop Line paint 정합은 미검증이다.
- Ground는 변환된 Box 바닥과 지도 표면을 이용한 추정이며 물리적 바퀴 접촉 측정이 아니다. 낙하·점프·collision mesh 차이는 실제 엔진에서 추가 검증해야 한다.
- Broken/복합 차선의 실제 dash 위상은 현재 API로 확정하지 못해 해당 지원 영역을 invalid로 처리한다. 정확한 paint-object/texture 대조는 후속 작업이다.
- Static Obstacle·건물 mesh·정확한 barrier 및 확인되지 않은 연결 표면은 완성된 GT로 제공하지 않는다. 모든 Town·모든 Sidewalk 면적의 완전성을 주장하지 않는다.
- 기존 데이터에 없는 높이·3D Actor 정보를 복원하지 않는다. 신규 Height GT가 필요한 실험은 새 형식으로 수집해야 한다.
- Height 예측 Head, GPU 학습, TCP 학습 코드의 실제 입력·Target 호환성은 이번 완료 범위에 포함하지 않는다.

설치·설정·좌표·스키마·수정 파일은 [통합 보고](BEV_INTEGRATION_KO.md), 원본 소스 분석은 [소스 보고](BEV_SOURCES_KO.md), 기준 ZIP 대비 파일 목록과 해시는 `BEV_CHANGED_FILES.json`을 참고한다.
