# LRS / HBLR 소스와 BEV 생성 근거

분석일: 2026-10-12. 이 문서는 이번 구현에서 확인한 원본 코드, 제공 자료, 실제 지도 파일과 새로 개발한 기능을 구분한다. CARLA 서버의 실제 주행 검증 결과를 대신하지 않는다.

## 확인한 원본

| 자료 | 확인한 버전과 내용 |
|---|---|
| 제공 `LRS_V1.zip`, `LRS_V1(1).zip` | 자산 포함 / 코드 전용 두 버전. 제공 자산을 우선 분석했다. |
| [junwoomin/LRS](https://github.com/junwoomin/LRS/tree/fe9cbc2f8f38fa5f7808e6d161e645160958c6a2) | 커밋 `fe9cbc2f8f38fa5f7808e6d161e645160958c6a2`. `sim/Global_Dynamic_MapManager.py`, `sim/utility.py`, `sim/driving.py`, `sim/traffic.py`, `data_gan.py`, `loop.py`를 확인했다. 핵심 지도/좌표 모듈은 제공 ZIP과 일치한다. |
| [junwoomin/HBLR](https://github.com/junwoomin/HBLR/tree/7fcff7490fd97739d36d06d738aa9a3d422d6ad0) | 커밋 `7fcff7490fd97739d36d06d738aa9a3d422d6ad0`. 저장소 전체 트리를 확인했다. 문서, 비교 PNG와 LICENSE가 있고 실행 코드는 없다. |
| 제공 `HBLR(2).pptx` | 18개 슬라이드. 차선 종류, 존재 여부, 주행 가능 영역, 고가도로 중첩 비교. 17–18쪽 원본 이미지도 직접 확인했다. 좌표, Pose, 높이 수치, 재현용 지도 생성 코드는 포함하지 않는다. |
| [HBLR README가 연결한 AGILEQ-Training/data_gen](https://github.com/SungjinDavidLee/AGILEQ-Training/tree/f3eba609de1af9d07a1c3e37283afdc51e7fcf22/data_gen) | 커밋 `f3eba609de1af9d07a1c3e37283afdc51e7fcf22`. 실제 BEV 실행 코드는 `collector/bev.py`의 `BEVMap`. `runner.py`, `writer.py`, 센서/프레임 버퍼도 확인했다. |

HBLR의 문서에 있는 개선 예시만 보고 미공개 알고리즘을 추정하지 않았다. HBLR의 `docs/analysis.md`도 비교 이미지로 정확한 생성 알고리즘이나 전환 구간의 동작을 확정할 수 없다고 명시한다.

## 현재 기본 경로: 사용자가 제공한 `data.zip` 원본 PNG

후속 지시에 따라 현재 기본 모드 `global_raster`는 새로 제공된 `data.zip`의 `data5`와 `data10`을 직접 읽는다. CARLA waypoint로 정적 지도를 재생성하지 않는다. 기존 LRS ZIP의 동명 지도로 바꾸지도 않는다. 아래의 초기 자료 분석과 선택적 surface cache 설명은 원본 출처·호환 기능을 기록한 것이며, 현재 기본 수집 경로와 구분해야 한다.

`assets/bev_maps`에는 새 ZIP의 PNG/NPY 166개를 원본 byte 그대로 포함했다. Town01/02/03/04/05/07/10HD의 5 PPM 및 10 PPM, 총 14개 조합이다. `catalog.json`은 파일별 SHA-256, 크기, PNG 모드·해상도, 원본 world offset, 실제 파일 이름과 채널 대응을 기록한다. 폴더 이름만으로 해상도를 추측하지 않았다. LRS의 `PPM=5.0`과 HBLR 연결 코드의 `PIXELS_PER_METER=5.0`, 실제 CARLA 0.9.15 OpenDRIVE waypoint의 이미지 투영 결과로 좌표 등록을 확인했다. 정확한 실측은 `docs/validation/global-map-source-verification.json`과 별도 검증 결과를 따른다.

- `data5`: 0.2 m/pixel, `data10`: 0.1 m/pixel. 각 Town의 두 버전은 같은 World XY offset을 쓴다.
- 새 PNG의 `DAS_BEV`는 과거 LRS 자산과 크기·offset이 같아도 실제 pixel이 다르다. 따라서 과거 지도로 대체하지 않는다.
- Town04/05에는 high/low 정적 이미지가 있다. **새 ZIP에는 metric elevation 이미지가 없다.** Layer 이름을 float 높이 정답으로 변환하지 않는다.
- RGBA 이미지의 alpha는 255인 배경도 포함하므로 occupancy로 사용하지 않는다. 실제 RGB grayscale의 `> 0`만 binary support로 읽는다. 일부 Stop Line의 antialiasing 값은 신호 상태나 Semantic Class ID가 아니다.
- `all_lane_BEV`/`lane_all_BEV` 원본을 직접 사용한다. Solid/Broken의 union으로 재생성하면 일부 Town의 실제 pixel이 달라진다.
- Town01/02의 `lane_BEV`와 crosswalk는 원본부터 전부 0이다. `lane_all_BEV`, `lane_center_BEV`, `lane_all_broken_BEV`가 서로 동일하므로 이 자산의 차선 종류를 독립적인 실제 paint 정답으로 확정하지 않는다. 이 정보는 catalog 경고에 남긴다.
- high에 없는 Stop Line/Crosswalk 등은 없는 채널로 반환한다. low 이미지를 high로 복사해서 표시하지 않는다.

### 현재 Loader와 높이 선택 계약

`sdv/bev/global_map.py`의 `catalog(root)`는 전체 PNG를 decode하지 않고 목록·파일 무결성을 확인한다. `load_global_map(root, town, ppm)`는 선택한 Town/해상도만 검증하고, 각 채널은 실제 사용 시 한 번 decode한다. readonly 배열은 기본 768 MiB LRU에 저장한다. Town04 10 PPM은 한 uint8 plane만 약 96 MB이므로 원본 해상도와 필요한 채널 수에 따라 메모리를 사용한다. 모든 Town/해상도를 동시에 decode하지 않는다.

`select_layer(ego_pose, channels=...)`는 실제 원본 마스크, World Offset, m/pixel, 선택 근거, 제공되지 않은 채널, 경고를 반환한다. `layer_for_points`는 Actor/Route/Stop Line에도 같은 높이 선택 규칙을 적용한다.

1. CARLA에서 검증한 지면 waypoint Z가 있으면 사용한다. Actor bounding-box 상단 높이를 도로 높이로 사용하지 않는다.
2. HBLR 연결 코드의 기존 기준 Town04 5 m / Town05 7 m를 높이 분기의 근거로 보존한다.
3. **native ground가 확인된 Driving lane**에서 해당 World XY가 low/high 중 한쪽 DAS에만 존재하면 실제 지도 support를 우선한다. Town05 Road37의 Z≈6.095 m ramp는 high에만 존재하여 기존 7 m 분기로 low를 고르면 누락되는 사례다.
4. Sidewalk에는 이 override를 적용하지 않는다. 실제 Town05 Road9의 지면 0 m 보도 위로 high DAS만 겹치는 반례가 있어, “한쪽 이미지에만 있다”는 이유로 위층에 배치하면 잘못된다. `ground_verified=False`나 미확인 높이를 문자열 출처만으로 검증된 ground로 승격하지 않는다.
5. 두 이미지 모두 support가 있거나 둘 다 없으면 기존 높이 분기를 사용하고 선택 근거를 metadata에 남긴다. 실제 도로 연결 그래프를 제공하지 않는 PNG만으로 모든 다층 교차·경사로가 정확히 분류된다고 주장하지 않는다.

이미지 선택은 metric surface elevation의 복원을 의미하지 않는다. 알려진 지면 Z, 객체 Min/Max Z와 Height, low/high 소속은 각각 별도의 metadata다. 현재 PNG 경로는 실제 원본 raster를 crop/회전하며 정적 도로를 다시 그리지 않는다.

## 실제 HBLR 연결 코드

`collector/bev.py`는 사전 생성 PNG를 읽어서 Local BEV를 추출한다.

- 출력 200×200, 5 PPM, 0.2 m/pixel, 약 40×40 m.
- 중심 좌표는 `int((world_xy - world_offset) * 5)`로 먼저 정수화한다.
- 회전은 CARLA yaw + 90°. 출력 중심은 99.5 pixel이다.
- `sampling_grid()`에서 정수 최근접 좌표를 만들고 `sample()`에서 원본 이미지 값을 그대로 복사한다. 지도 밖은 0이다.
- `static_masks()`는 Town04에서 ego Z > 5 m, Town05에서 ego Z > 7 m이면 `das_full_high.png`, `lane_full_high.png`로 교체한다. 정지선과 전체 도로 이미지는 교체하지 않는다.
- 도로 연결 그래프, Road/Lane ID, 경사로 연속 표면, pixel별 elevation, 높이 유효성 정보는 저장하지 않는다. Offline Global Map 생성기도 이 모듈에는 없다.
- 차량은 Actor 중심·yaw와 bounding-box X/Y 크기를 이용한 2D 사각형이다. 최소 크기는 1 m이다. 보행자는 1.4×1.4 m 고정 사각형이다.
- Actor Z를 JSON footprint에 기록하지만 도로 Layer로 필터링하지 않는다. 3D bounding-box center/rotation, Ground Z, Min/Max Z, 실제 객체 높이는 분리하지 않는다.
- Route는 agent의 현재 계획 waypoint 최대 5개를 두께 16 pixel로 그린다. 이름이 `future_waypoints`여도 실제 미래에 관측한 차량 궤적과 같다고 해석하면 안 된다.
- `runner.py`는 world tick, 센서 frame, snapshot frame의 일치를 검사한다. `writer.py`는 이미지 저장 후 마지막에 JSON을 확정한다.

따라서 이 코드는 HBLR의 실제 기존 지도 활용·높이 분기·최근접 샘플링의 근거다. native Ground 검증, 실제 PNG support를 이용한 ramp 분기 보완, 3D Actor Ground/Height 분리는 이번에 추가한 기능이다. 별도 topology 기반 surface cache는 현재 기본 PNG 경로와 구분된다.

## 초기 LRS / AGILEQ 지도 자산 실측 (새 `data.zip`과 구분)

AGILEQ의 `data5` 파일 39개를 커밋에 고정하여 내려받았으며 각각 Git blob SHA 일치를 확인했다. 제공 LRS의 동명 파일과 38개가 byte 단위로 일치한다. 유일한 차이는 Town01 `lane_full.png`이다.

다음 크기는 실제 파일의 높이×너비다.

| Town | 도로/차선 크기 | Stop Line 크기 | 추가 high 도로/차선 |
|---|---:|---:|---|
| Town01 | 2163×2492 | 2163×2492 | 없음 |
| Town02 | 1505×1506 | 1505×1506 | 없음 |
| Town03 | 2584×2490 | **2584×2584** | 없음 |
| Town04 | 4660×5147 | 4660×5147 | 있음 |
| Town05 | 2584×2935 | 2584×2935 | 있음 |
| Town07 | 2366×1952 | **2366×2366** | 없음 |
| Town10HD | 1549×1622 | 1549×1622 | 없음 |

Town03/Town07의 Stop Line 캔버스가 다르다. 크기 차이만으로 실제 Stop Line 좌표 오차를 단정할 수는 없지만, 모든 파일의 범위가 같다고 가정하는 무검증 합성은 안전하지 않다.

실제 AGILEQ `read_map()`을 실행해 확인한 결과 대부분의 단일 채널은 0/255이지만 Town01 Lane은 73종 grayscale 값, Town03 Lane은 0/40/244/250/252/255를 가진다. Town01 AGILEQ Lane과 제공 LRS Lane은 140 pixel이 다르며, 중간 grayscale pixel은 각각 163개와 23개다. 이 PNG 값을 정수 Semantic Class ID로 그대로 사용하면 안 된다. 원본의 정확한 binarization 계약 없이 임의 threshold를 정답으로 확정하지 않는다.

## 기존 Height PNG의 신뢰 범위

LRS에는 Town04/Town05의 `height_low.png`, `height_high.png`가 있다. 실제 런타임은 `gray / 255 * 16`으로 m 값을 복원한다.

| Town | Low 실제 값 범위 | High 실제 값 범위 | 크기 |
|---|---:|---:|---:|
| Town04 | uint8 0–138 | uint8 0–191 | 4660×5147 |
| Town05 | uint8 0–107 | uint8 0–176 | 2584×2935 |

이 인코딩의 간격은 약 0.062745 m다. 다만 이는 코드에서 확인한 복원 규칙일 뿐, 실제 표면 높이에 대한 측정 오차를 뜻하지 않는다. 원본 생성 코드·타당성 mask·offset별 검증·연결 관계·수집 CARLA 지도 해시가 없어 metric Ground Truth로 인증할 수 없다. 0은 실제 0 m와 미기록 배경을 구별하지 못한다. 음수 높이와 16 m 초과 높이도 이 계약으로 표현할 수 없다.

높이 분기 임계값도 코드별로 다르다. LRS `driving.py`는 Town04 4 m / Town05 3 m, LRS `data_gan.py`는 3 m / 7 m, HBLR 연결 `collector/bev.py`는 5 m / 7 m다. 동일 XY에서 어느 이미지를 고를지 기준이 하나로 고정되어 있지 않다.

현재 기본 PNG 생성기는 이 과거 Height PNG를 실제 float32 높이 정답으로 승격하지 않으며, 새로 제공된 이미지에 존재하지 않는 높이를 만들지 않는다. 별도의 선택적 surface cache는 실제 지도 XYZ를 이용하지만 사용자 제공 PNG를 다시 그리는 기본 경로로 사용하지 않는다.

## 재사용 및 확장 경계

| 항목 | 실제 재사용 근거 | 이번 확장 |
|---|---|---|
| Local crop | LRS `make_local_bev_ego_aligned`, `global_pixel_to_local_bev`, nearest `warpAffine` | 직사각형 출력, 명시적 anchor, float 좌표 보존, validity 동반 |
| 동적 polygon | LRS OBB 회전·`fillPoly`/`fillConvexPoly` 구조 | 실제 CARLA 3D bbox, bounding-box 중심과 자세, ground/height 분리 |
| Global 정적 지도 재사용 | HBLR 연결 `BEVMap`과 새 제공 data5/data10 | 원본 PNG/NPY 보존, 해시 검증 catalog, 선택 채널 lazy cache |
| 높이 선택 | 기존 Town별 high/low 분기 | native Ground 검증 및 Driving lane의 실제 DAS support로 ramp 선택 보완, 추정 근거 기록 |
| 신호 상태 | 기존 정적 Stop Line 자산, LRS 신호 패치 캐시 구조 | 실제 CARLA state와 frame 동기화. LRS 합성 신호 주기를 실제 GT로 쓰지 않음 |

## 별도 Surface Cache 계약 (기본 PNG 경로 아님)

`sdv/bev/cache.py`의 `GlobalRasterCache`는 CARLA를 import하지 않는 공통 모듈이다.

- 입력: Road/Section/Lane ID, 순서 있는 `polygon_xyz`, 실제 연결 patch ID, 경계/차선 geometry, center XYZ.
- Layer ID는 고정 높이 구간이 아니라 표면 patch ID다. 경사로의 표면 높이는 삼각형 내부에서 연속 보간한다.
- Town 전체 크기×Layer 수 배열 대신 각 patch의 작은 사각 범위만 저장한다.
- 출력: `metadata.json`, `metadata.sha256`, `world_offset.npy`, `layers/<hashed_patch_id>.npz`.
- 채널: `road`, `drivable`, `lane`, `road_boundary`, `sidewalk`, `junction`, `validity`, `lane_validity`는 uint8 0/1. `height`는 float32 m, invalid는 NaN.
- `offset_world`는 각 patch 배열 `[0,0]`의 World XY다. `bounds_px`는 전체 raster에서 `[xmin,ymin,xmax_exclusive,ymax_exclusive]`다.
- Global raster에서는 World X가 u, World Y가 v로 증가한다. integer pixel 좌표가 World sampling 위치를 나타낸다.
- 캐시 식별자는 Town, OpenDRIVE SHA-256, resolution, generator version, builder options, 전체 geometry/topology에 의존한다. 파일 checksum을 확인하며 임시 폴더에서 완성 후 확정한다.
- `find()`로 실제 지도 해시·builder options가 일치하는 캐시만 조회한다. 같은 조건의 서로 다른 geometry 캐시가 두 개면 임의 선택하지 않는다.
- Local query는 공간 tile index를 사용한다. 배열 cache는 기본 4096 patch와 64 MiB를 동시에 제한한다. 작은 patch가 많은 교차로에서 개수 제한만으로 매 Frame 파일을 다시 읽는 현상을 줄인다.
- `lookup_ground()`는 XY 내부의 실제 삼각형 표면에서 Ground Z를 구한다. 지정한 Road/Lane은 제약이며, 다른 도로로 조용히 대체하지 않는다. 서로 다른 높이가 동일하게 근접하면 `None`을 반환한다.
- `graph_distances()`는 연결된 patch 중심의 3D 거리 합으로 계산한 최단 거리(m)다. 주행 허용 방향과 별개로 뒤쪽 도로도 관찰할 수 있도록 물리 연결을 양방향으로 조회한다. 방향 있는 원본 연결은 metadata의 `neighbors`에 남긴다.
- 연결별 거리는 로딩 시 한 번 계산하며, `max_distance`로 요청한 지역 밖의 탐색을 중단한다. `build_report`는 제외한 geometry의 원인·위치를 기록하며 캐시 identity와 checksum에 함께 포함된다.

## 선택적 Surface Cache에서 Lane 정답의 보수적 처리

CARLA Waypoint의 `Broken`/복합 marking 종류만으로 실제 dash 길이·간격·위상을 확정할 수 없다. 이를 실선으로 채우면 기존 HBLR이 지적한 잘못된 Lane GT를 다시 만든다.

현재 Cache는 확인된 단일 Solid geometry 또는 adapter가 명시적으로 실제 paint segment라고 제공한 geometry만 그린다. Broken/복합/unknown 경계는 `lane_validity`에서 해당 support band와 2 pixel 여유를 제외하고 `unsupported_markings`에 기록한다. 이 mask는 학습 Loss/Metric에도 전달되어야 한다. Boundary geometry와 실제 CARLA texture의 정합성은 별도의 실제 장면 검증이 필요하다. OpenDRIVE paint-object parser를 완성했다고 주장하지 않는다.

## 이 문서 작성 시 실행한 검증

`tests/test_bev_global_map.py`: **12개 통과**. 제공한 14개 Town/PPM 목록, 실제 PNG pixel 보존, alpha 배제, high 채널 누락 유지, 지면 없는 경우 경고, bounded lazy cache, 원본 byte 보존, 파일/SHA/manifest 손상 거부, 실제 Town05 Road37 ramp 및 Road9 하부 보도 반례를 5/10 PPM에서 확인했다. 이 수치는 loader 단위 테스트이며, 최종 전체 검증 결과는 별도 통합 테스트 보고를 따른다.

`tests/test_bev_cache.py`: **14개 통과**. 평면, 연속 경사, 동일 XY의 상·하층, 모호한 Ground Z 거부, ID 제약, 연결 거리와 역방향 가시성, 거리 제한, 캐시 재사용/무효화, 파일 손상, bounded array cache, 먼 표면 간 sparse 저장, 오목 polygon, 미확정 Broken marking 무시, geometry 제외 보고의 불변성을 검사했다. 이는 통제 geometry에 대한 코드 검증이다.

HBLR/LRS 제공 이미지와 source를 직접 열고 크기·dtype·값 분포·해시를 검사했다. 제공 slide에는 동일 Pose를 다시 실행할 수 있는 입력이 없어, 슬라이드의 `Ori/Our`와 새 생성물의 좌표 오차나 mIoU를 측정하지 않았다. 실제 CARLA 주행, 시각적 road paint 일치, Actor 누락률, 실시간 FPS는 이 문서의 소스 분석만으로 검증 완료라고 할 수 없다.

## 출처와 라이선스

LRS와 HBLR 저장소는 MIT이며 copyright `(c) 2026 woomin jun`이다. 재사용 코드의 고지는 `THIRD_PARTY_NOTICES`에 유지한다. 연결된 AGILEQ-Training tree에서는 저장소 전체에 적용되는 LICENSE를 찾지 못했다. `data_gen/THIRD_PARTY_NOTICES.txt`는 CARLA에서 가져온 navigation 코드의 MIT 출처를 명시한다. 따라서 custom AGILEQ 코드나 이미지 전체를 MIT라고 임의 분류하지 않는다.
