# 제공 Global Map 기반 BEV 수정 보고

2026-10-12. 이번 기본 생성 방식은 **첨부 `data.zip`의 Town 전체 PNG를 직접 읽어 캐시하고, Ego World Pose로 Crop·Rotation한 뒤 동적 Actor만 그리는 방식**이다. 이전 v9의 waypoint 표면 Patch 생성 경로는 현재 기본 수집에서 호출하지 않는다. 기존 데이터와 명시적인 이전 모드는 호환용으로 남겼다.

## 실행과 설정

```bash
cd sdv_studio
chmod +x start.sh
./start.sh
```

시뮬레이션 → **BEV 생성 고급 설정** → **Global PNG 직접 Crop**을 선택한다. 원본 지도는 패키지의 `assets/bev_maps/`에 포함되어 있어 별도 지도 생성 명령이 필요 없다. 이전 웹 기본값 `height_aware`는 저장 설정을 읽을 때 새 기본 모드로 전환한다. 이미 실행 중인 작업의 설정 사본이나 과거 데이터는 변경하지 않는다.

| 설정 | 의미 |
|---|---|
| 원본 지도 해상도 | 실제 `data5`의 5 PPM 또는 `data10`의 10 PPM 선택 |
| Local 출력 해상도 | 원본 유지 또는 Custom m/pixel. 원본 PPM과 독립적으로 기록 |
| 가로·세로 범위 | m 단위로 입력하면 정수 출력 pixel 수로 변환하고 실제 범위를 표시 |
| 출력 너비·높이 | 각 64~1024 px. 물리 범위는 각 200 m 이하 |
| Ego Anchor | 출력 가로·세로 크기의 0~1 비율. 기본 중앙 |
| Ego Yaw 회전 | 켜면 Ego 전방이 위쪽. 끄면 World +X가 위, +Y가 오른쪽 |
| Low/High 선택 | 존재하는 높이별 이미지 사용. 미터 단위 Height 영상과 다름 |

40×40 m에서 원본 해상도 유지 시 5 PPM은 200×200 px, 10 PPM은 400×400 px다. 브라우저에서 PPM을 바꿀 때 물리 범위를 유지하는 것을 검증했다. Custom 출력은 단일 Affine의 scale로 처리하며 Global PNG를 임의 크기로 바꿔 저장하지 않는다. 낮은 해상도로 줄이면 얇은 차선·정지선이 사라질 수 있으므로 UI와 Frame 경고에 표시한다.

없는 Town/PPM, 수정되거나 손상된 원본 파일은 수집 시작 전에 차단한다. 임의 waypoint 지도를 대신 만들어 진행하지 않는다. Town01/02/03/04/05/07/10HD만 제공되었으며 Town06 등의 지도는 없다.

원본 확인용 CLI:

```bash
python -m sdv.bev maps
python -m sdv.bev maps --root assets/bev_maps
```

예전 `python -m sdv.bev build`는 이전 OpenDRIVE 방식의 명시적 도구일 뿐, 새 수집에 필요한 단계가 아니다.

## 원본 조사와 PPM 확인

ZIP의 PNG/NPY **166개를 바이트 변경 없이** 포함했고, 7 Town × 2 해상도의 14개 조합을 catalog로 등록했다. 파일별 SHA-256·크기·offset·해상도·실제 Layer 파일을 기록했다. 전체 구조와 Town별 이미지 크기는 [원본 조사 보고](GLOBAL_MAP_SOURCE_AUDIT_KO.md)에 있다.

폴더명만 보고 5/10 PPM을 정하지 않았다. 원본 LRS/HBLR 코드의 좌표식, 같은 offset과 약 두 배의 canvas 크기, 실제 CARLA 0.9.15 Map의 Driving 좌표를 2.5/5/10/20 PPM으로 투영한 결과를 대조했다. 올바른 해상도에서 6개 Town은 검사한 중심점 전부가 DAS 내부에 들어갔다. Town05의 48개 누락은 해상도가 아니라 아래의 경사로 Layer 선택 문제였다.

Global 좌표는 `u=(x-offset_x)×PPM`, `v=(y-offset_y)×PPM`이다. 두 해상도의 물리 영역은 같은 좌표식으로 계산하며, 원본 canvas 끝단은 정수 반올림 때문에 최대 0.1 m 차이 날 수 있다. 서로 다른 원본 raster의 경계 pixel이 완전히 동일하다고 가정하지 않는다.

새 ZIP의 실제 픽셀은 이전에 제공된 LRS PNG와 다르다. 이번에는 반드시 새 첨부의 이미지를 사용한다. LRS의 검증된 `make_local_bev_ego_aligned` 역 Affine와 `cv2.warpAffine` 최근접 처리를 공통 좌표 함수로 재사용했다.

## 생성 흐름과 캐시

1. 실제 Town과 선택한 PPM의 catalog·파일 해시·offset을 확인한다.
2. 선택 Layer의 필요한 전체 PNG를 메모리에 로드한다. 현재 Town 인스턴스의 캐시는 768 MiB까지 사용한다.
3. 동일 Frame의 Ego `(x,y,z,yaw)`와 Ground 정보를 읽어 Low/High를 고른다.
4. 하나의 역 Affine로 전체 이미지에서 Local 영역을 최근접 추출한다. 별도 중간 Crop으로 회전 모서리를 자르지 않는다.
5. 같은 좌표 함수로 실제 3D Box footprint, 신호 상태 및 현재 Route를 추가한다.
6. 숫자 NPZ, 기존 호환 PNG, 동일 Frame JSON을 저장한다.

Town04 10 PPM의 전체 canvas는 10,294×9,321이다. 선택 채널만 디코딩하고 Town/PPM 인스턴스를 유지한다. 초기 로딩과 고가 Layer 첫 사용에는 메모리·디코딩 비용이 있지만 매 Frame마다 원본 PNG를 읽지 않는다. 검증에서는 warm cache 이후 PNG 읽기 **0회**였다. 메모리 사용량에는 캐시 외의 디코딩 임시 버퍼와 Local 배열도 추가된다.

설정은 작업 시작 시 고정한다. 실행 중 웹에서 다음 수집 설정을 바꾸더라도 해당 작업의 지도 경로·PPM이 변하지 않는 것을 실제 자식 프로세스로 검사했다.

## 높이 선택과 Actor

Town04/05에 실제 `_high` 이미지가 있다. 확인한 HBLR 연결 구현은 각각 5 m / 7 m 임계값을 사용하지만, 새 Town05 PNG의 Road37 진입 경사로 일부는 Z가 7 m보다 낮아도 high에만 존재한다.

새 정책은 **실제 CARLA Driving ground가 확인된 경우** 해당 World pixel의 Low/High DAS 지원을 함께 확인한다. 둘 중 하나에만 해당 도로가 있으면 그 이미지를 선택하고, 두 이미지가 겹치는 곳은 Ground Z 기반 원본 HBLR 기준을 적용한다. 선택 근거와 임계값 보완 여부를 Frame에 기록한다. 경사로 전체 Road ID를 무조건 high로 고정하지 않는다.

이 보완을 Sidewalk에 적용하면 고가도로 아래의 보행자를 위 도로로 잘못 올릴 수 있다. 실제 Town05 하부 보도 반례를 확인했고, 이 경우에는 exclusive DAS 보완을 적용하지 않는다. Ground 검증은 `project_to_road=False` waypoint와 변환된 Box 바닥면 중심의 Z 차이가 0.75 m 이내일 때만 인정한다. 이는 Map/Box에 근거한 추정이며 바퀴 접촉 센서 측정은 아니다.

차량·보행자의 3D Box 8개 꼭짓점, 크기·회전·Actor ID를 유지하고 수평 투영 Polygon으로 Occupancy를 만든다. Box의 상단 높이로 도로 Layer를 선택하지 않는다. Low/High가 있는 지도에서 Ground를 확인하지 못한 객체의 footprint는 Occupancy 정답에서 invalid로 처리한다. 단일 Layer만 제공된 Town의 수직 분리를 보장한다고 주장하지 않는다.

Traffic Light는 실제 CARLA 상태와 stop waypoint를 사용한다. 제공 `stop_line_BEV.png`는 상태를 알 수 없는 별도 `static_stopline` mask로 저장하고, 웹에서 흰색 원본 정지선으로 구분한다. 이 이미지 값으로 Red/Yellow/Green을 만들지 않는다. Route는 현재 실제 planner waypoint이며 미래 관측 Target과 구분한다.

## 저장 스키마와 학습

새 수집은 **schema 4**, 기존 숫자 BEV는 schema 3, 이전 PNG 데이터셋은 기존 방식대로 읽는다. `bev_gt/<frame>.npz`와 `info/<frame>.json`을 함께 사용한다.

| 항목 | 새 형식 |
|---|---|
| `drivable`, `lane`, `road_boundary` | 제공 PNG의 흑백 지원 영역 `>0`를 binary로 Crop |
| `validity` | 원본 PNG canvas 안쪽. 바깥 padding은 0 |
| `lane_validity`, `occupancy_validity` | 각각의 라벨 가용성·Ground 확인 여부 반영 |
| `elevation_m` | 제공 Height 영상이 없어 NaN |
| `elevation_validity` | 미터 단위 높이 유효성. 이번 원본에서는 전부 0 |
| 객체 높이 채널 | 실제 Box min/max Z, object height, 확인한 Ground Z를 별도 보존 |
| `surface_id` | Low/High 원본 이미지 인덱스. 연속 도로 표면 ID가 아님 |
| `world_to_bev_pixel` | 저장된 실제 World→Local 2×3 Affine |
| `global_map` | 실제 Town·PPM·선택 Layer·소스 경로·SHA·선택 근거 |
| `timings` | Lookup·좌표·Crop/Rotation·Actor·합성·전체 ms |

원본 PNG는 그대로 보관한다. 일부 정지선의 antialias grayscale을 semantic Class ID나 높이로 해석하지 않는다. 모델 Label은 지원 영역을 binary로 정의하며, 색상 이미지는 표시용이다. Class ID 구성은 기존 0~13을 유지한다.

기존 DAS/Lane/Vehicle/Walker **4개 Head 및 모델 입력 채널을 유지**했다. Local 출력의 PPM·Anchor·회전 상태가 달라도 실제 Affine를 이용해 기존 모델의 ±20 m, 0.2 m 격자로 조회한다. 회전을 끈 저장 데이터도 실제 Ego Yaw를 별도로 사용한다. 카메라 보정, Loss와 Metric의 invalid 제외, Prediction 표시를 함께 검증했다. Schema 혼합·보정 누락·존재하지 않는 static layer 선택은 차단한다.

Schema 4의 별도 체크포인트 처리 계약을 추가했으며, 기존 schema 3와 PNG 재개 계약은 유지한다. 높이는 보조 GT이고 온라인 입력 채널로 자동 추가하지 않는다. Planning export의 센서 입력/시뮬레이터 GT/미래 Target 분리도 유지한다.

## 실제 이미지와 검증 결과

![같은 World Pose와 40m 영역의 5/10 PPM 비교](validation/global-map-direct-crop-comparison.png)

Town04 상·하부와 Town05 경사로, 5/10 PPM, Yaw 0/90/180/270°에서 DAS/Lane/Boundary **72건 모두 원본 LRS Affine 결과와 픽셀 차이 0**이었다. 비교 그림은 각각 200×200 / 400×400 원본 해상도로 같은 40×40 m를 나타낸다. 실제 제공 지도와 native 좌표를 사용한 정적 출력이며 새 CARLA 주행 화면이나 모델 Prediction은 아니다.

수집 통합 검사는 `surface_patches`, `to_opendrive`, `generate_waypoints`, 이전 `build_cache`가 호출되면 실패하도록 구성하고 새 PNG 경로가 정상 동작함을 확인했다. 동일 Frame NPZ/JSON 저장, 지도 파일 누락 차단, Actor Box 높이·다른 Layer 제외, padding 및 Affine를 검사했다.

실제 Town01 원본으로 5/10 PPM × 출력 0.2/0.25 m/px × 회전 ON/OFF 8조합을 Generator→NPZ→학습 로더에 연결했다. 모든 정합성 검사에서 오류가 없었다. 다만 5 PPM을 0.25 m/px로 줄인 회전 OFF 사례는 원본 1px 차선이 생성 단계에서 사라졌다. 이를 임의 팽창시켜 수정하지 않고 해상도 손실로 기록했다. 실제 CPU 모델 forward/loss/backward도 검사했다.

Chromium에서 설정 저장·새로고침, 5→10 PPM의 물리 범위 유지, 없는 Town 비활성화, Height N/A, 한국어/영어, 수평 Timeline을 확인했다. 1366×768과 1920×1080에서 가로 넘침·콘솔 오류는 없었다. 제공 RGB/Pose와 원본 PNG를 이용한 오프라인 UI 검사이며 실시간 재수집은 아니다.

최종 자동화 결과는 다음과 같다. 앞의 두 행은 동일 회귀 검사를 서로 다른 환경에서 실행한 것으로 중복 합산하지 않는다. 건너뜀은 웹 환경에 설치하지 않은 Torch 등의 연구 의존성, Python 버전 전용 검사에 해당하며 통과로 계산하지 않았다.

| 환경 | 통과 | 건너뜀 | 실패 |
|---|---:|---:|---:|
| Python 3.7.16 전체 | 630 | 21 | 0 |
| Python 3.12 웹 환경 전체 | 628 | 23 | 0 |
| CPU Torch 모델·학습 회귀 | 152 | 0 | 0 |
| CPU Torch schema 4 학습 연결 | 15 | 0 | 0 |

전체 Python의 3.7 문법, JavaScript와 Bash 문법 및 Python 3.7 지도 catalog CLI도 확인했다. 이번 변경을 Python 3.8/3.9/3.10에서 각각 재실행한 것은 아니다. 최종 테스트 수와 실행 환경의 원본은 [자동화 검사 요약](validation/global-map-test-summary.json)에 있다. 세부 원본은 다음 파일에 기록했다.

- `validation/global-map-source-verification.json`: 전체 원본·PPM·좌표·72 Crop·성능·소스 해시
- `validation/global-map-real-training-validation.json`: 실제 PNG의 학습 8조합
- `global-map-web-validation.json`: 브라우저·설정 저장 검증
- `validation/global-map-*-tests.xml`: 실제 pytest 실행 결과

## 처리 시간

동일 Town01 좌표, 통제용 Actor 20개, 500×500·0.2 m/px, OpenCV 9 threads, 2회 준비 후 10회 측정했다. Python 3.7.16 / NumPy 1.21.6 / OpenCV 4.8.1의 공유 CPU 환경이다.

| 생성 방식 | 중앙값 | P95 |
|---|---:|---:|
| 이전 v9의 Patch 캐시 방식 | 75.615 ms | 79.900 ms |
| 제공 PNG 직접 Crop | **11.657 ms** | **14.747 ms** |

새 방식의 단계별 중앙값: Map lookup 0.040 ms, 좌표 0.085 ms, Crop/Rotation 3.707 ms, Actor 3.325 ms, 합성 4.071 ms. 각 단계의 중앙값 합은 전체 중앙값과 정확히 같을 필요가 없다. 두 방식의 정적 Label 정의가 다르므로 정확도 점수로 비교하지 않는다. 두 결과 모두 통제용 Actor 20개를 포함했다.

별도의 40×40 m 정적 출력 측정은 5 PPM 200×200에서 2.011~2.699 ms, 10 PPM 400×400에서 6.272~6.881 ms였다. 가장 큰 Town04 10 PPM의 첫 디코딩 프레임은 6.907초였으며, 이후 warm 중앙값은 6.272 ms였다. 첫 로딩과 매 Frame 비용을 구분해야 한다. 실제 시뮬레이션·센서 처리·파일 저장·웹 전송은 이 수치에 포함하지 않는다. 수집 FPS 보장이 아니다.

## 남은 제한

- 실제 CARLA 서버의 RGB/Actor/BEV 동기화, 연속 주행의 Layer 전환, 신호 전환, 전체 수집 FPS 및 GPU 학습은 미검증이다.
- 제공 PNG에 연속적인 미터 높이·Road/Lane ID raster가 없다. 해당 영상은 N/A이며 Actor의 native Ground Z와 혼동하지 않는다.
- Low/High의 겹치는 영역에는 원본 HBLR 임계값이 남는다. 모든 경사로·복잡한 다층 도로를 일반적으로 해결한 방식으로 주장하지 않는다.
- Town01/02의 lane_all·lane_center·lane_all_broken은 동일 이미지이고 lane_BEV는 비어 있다. 제공된 annotation을 그대로 사용하며 실제 painted lane으로 독립 검증되었다고 주장하지 않는다. Boundary 일부도 DAS의 Canny edge다.
- Crosswalk·stencil 원본은 보존·목록화했지만 별도 학습 Head를 추가하지 않았다. High에 없는 채널을 Low에서 복사하지 않는다.
- Custom 낮은 해상도는 얇은 Semantic 라벨을 잃을 수 있다. 의미 보존을 위한 임의 팽창이나 가짜 픽셀을 추가하지 않는다.

수정 핵심은 `sdv/bev/global_map.py`, `global_generator.py`, schema/io/training/quality, collection adapter/workflow/metadata, Studio 설정·작업·API·미리보기와 정적 UI다. **기준 v9 대비 전체 추가·수정 파일과 해시는 `GLOBAL_MAP_CHANGED_FILES.json`**에 기록했다. 이전 BEV 보고서와 비교 이미지는 역사 기록으로 유지하며 현재 방식의 검증 결과와 구분한다.
