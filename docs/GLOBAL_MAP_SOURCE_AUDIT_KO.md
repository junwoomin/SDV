# 첨부 Global BEV 지도 원본 조사

이번 첨부 `data.zip`의 전체 PNG를 직접 읽고, 기존 LRS/HBLR 소스와 실제 CARLA 0.9.15 OpenDRIVE 좌표를 대조했다. 폴더 이름만으로 해상도를 정하지 않았다. 정적 지도를 waypoint에서 다시 그리는 방식은 이 자료의 사용 방식이 아니다. 제공한 전체 이미지를 Town·해상도별로 한 번 로드하고, 매 프레임 실제 pose에 맞춰 crop/rotation해야 한다.

## 실제 구성

- ZIP SHA-256: `8d890c75ee64e92b460667a286c0cae5cae6c8916e3ef8014ae921fd503f09e7`.
- 압축 크기 4,736,721 bytes, 총 183 entries, 파일 압축 해제 합계 15,557,345 bytes.
- `data5`와 `data10` 각각 Town01, Town02, Town03, Town04, Town05, Town07, Town10HD가 있다. 각 Town에 `world_offset.npy`가 있다.
- Town04/05에는 `DAS_BEV_high`, `all_lane_BEV_high`, `Boundary_BEV_high`, `lane_solid_BEV_high`, `lane_broken_BEV_high`, `lane_center_BEV_high`가 있다. 다른 Town에는 high 세트가 없다.
- **첨부 ZIP에는 미터 단위 Height, `height_low.png`, `height_high.png`, Road/Lane ID raster가 없다.** `high`라는 파일명은 별도 semantic 이미지 이름이며 픽셀 값이 높이를 뜻하지 않는다.
- 유일한 notebook `data10/Town04/Untitled-1.ipynb`는 `DAS_BEV.png`를 grayscale → 5×5 Gaussian blur → Canny(100,200) 처리하여 `Boundary_BEV.png`로 저장한다. 전체 지도 생성 스크립트나 PPM/높이 보정 설정은 포함하지 않는다.

## 해상도와 좌표 근거

| Town | 5 PPM 전체 크기 W×H | 10 PPM 전체 크기 W×H | World offset X, Y (m) |
|---|---:|---:|---|
| Town01 | 2492×2163 | 4984×4326 | -52.059906, -52.049961 |
| Town02 | 1506×1505 | 3012×3011 | -57.459728, 55.390747 |
| Town03 | 2490×2584 | 4980×5169 | -199.063843, -259.271240 |
| Town04 | 5147×4660 | 10294×9321 | -565.269043, -446.146118 |
| Town05 | 2935×2584 | 5871×5169 | -326.044525, -257.875092 |
| Town07 | 1952×2366 | 3904×4732 | -254.627014, -298.980927 |
| Town10HD | 1622×1549 | 3245×3099 | -164.595215, -118.729042 |

한 Town/해상도 세트 안의 모든 PNG는 가로·세로 크기가 일치한다. 두 해상도의 offset은 동일하며, 10 PPM 크기는 5 PPM의 두 배에서 최대 1 pixel 차이다. 끝단 반올림 때문에 실제 표시 범위가 최대 0.1 m 차이 날 수 있다.

근거는 세 가지다.

1. 첨부 data5의 크기와 offset이 기존 LRS `sim_using_data/data5`와 일치한다. 기존 LRS `loop.py`는 `ppm=5.0`, HBLR 연결 구현 `data_gen/collector/bev.py`는 `PIXELS_PER_METER=5.0`을 명시한다.
2. 실제 `carla.Map`으로 원래 Town의 `.xodr`를 읽고 5 m 간격 Driving waypoint를 2.5/5/10/20 PPM 후보로 투영했다. 첨부 data5는 5 PPM, data10은 10 PPM에서만 6개 Town의 모든 표본이 DAS 내부에 들어갔다. 다른 후보의 적중률은 최대 약 33.6%였다. Town05의 99.05% 결과는 아래의 layer 정책 불일치 때문이다.
3. 같은 해상도에서 기존 LRS DAS와 **픽셀 자체는 다르다**. data5에서 Town별 50,833~1,072,721 pixel 차이가 있었다. 기존 이미지를 새 첨부의 대체물로 사용하면 안 된다.

확인한 변환은 `global_col=(world_x-offset_x)×PPM`, `global_row=(world_y-offset_y)×PPM`이다. CARLA의 yaw는 degree이며, ego 정렬에서는 전방이 이미지 위쪽, 오른쪽이 이미지 오른쪽이다. Global PNG의 pixel과 Local 출력의 pixel 해상도는 별도 설정이다. 예를 들어 10 PPM 원본을 0.2 m/pixel Local 출력으로 조회할 수 있으나, 최종 Local이 10 PPM인 것은 아니다.

## 실제 PNG 값과 라벨 의미

일부 PNG는 L, 일부는 RGB/RGBA이다. 컬러 semantic palette가 아니라 RGB가 같은 흑백 mask이며, RGBA의 alpha는 255다. **alpha를 포함한 `any(pixel>0)`를 사용하면 배경까지 전부 양성이 된다.** RGB 또는 luminance만 사용해야 한다.

| 실제 파일 | 조사 결과와 처리 기준 |
|---|---|
| `DAS_BEV.png`, `_high` | Drivable mask. 원본 픽셀을 직접 crop한다. |
| `all_lane_BEV.png`, `_high` | 제공된 통합 차선 mask. 임의 재합성하지 않는다. |
| `lane_solid_BEV`, `lane_broken_BEV` | 별도 실선/점선 mask. 점선을 실제 PNG에서 읽는다. |
| `lane_center_BEV` | 별도 중심선 데이터. 도색 실선과 같은 의미로 단정하지 않는다. |
| `Boundary_BEV` / Town07의 `boundary_BEV` | 실제 파일명을 구분한다. Town04 notebook은 DAS의 영상 edge임을 보여준다. 모든 boundary가 독립 도로 경계 annotation이라는 증거는 없다. |
| `crosswalk_BEV`, `stencil_BEV`, `stencil_stop_BEV` | 별도 mask. `all_lane`에 자동 포함시키지 않는다. |
| `stop_line_BEV` | 정적 선 geometry. 일부 Town은 32~170개 grayscale 값으로 antialiasing되어 있다. 픽셀 값을 Red/Yellow/Green 신호 상태나 Class ID로 해석하지 않는다. 원본 intensity와 binary label 변환 규칙을 구분한다. |

Town03/04/10HD의 `all_lane`은 solid OR broken과 정확히 같다. Town05는 5 PPM에서 47 pixel, 10 PPM에서 167 pixel 차이가 있고, Town07은 각각 1/4 pixel 차이가 있다. 따라서 합집합 재생성 대신 제공한 all 파일을 사용해야 원본이 보존된다. Crosswalk와 stencil은 대부분 별도 영역이다.

Town01/02는 예외다. 두 해상도에서 `lane_BEV.png`와 `crosswalk_BEV.png`가 모두 0이다. `lane_all_BEV.png`, `lane_all_broken_BEV.png`, `lane_center_BEV.png`는 같은 픽셀이다. 파일 이름만 보고 서로 다른 세 가지 정답을 만들거나 없는 paint mask를 있다고 표시하면 안 된다. 빈 파일은 빈 파일로 유지하고, 중심선과 일치하는 제공 mask라는 사실을 표시한다.

## 높이 선택과 Town05 경사로

검토한 HBLR 연결 구현은 Town04 `z>5`, Town05 `z>7`이면 high DAS/lane 이미지를 선택한다. 원래 LRS 코드에는 Town04/05에 4/3 m 또는 3/7 m를 사용하는 서로 다른 정책도 있다. 일관된 범용 도로 높이 모델이 원본에 구현되어 있는 것은 아니다.

Town05에서 HBLR의 7 m 기준을 새 이미지에 그대로 적용하면 실제 5 m waypoint 5,053개 중 48개가 잘못 low를 선택한다. 모두 Road37 경사로이며 실제 Z는 4.051~6.906 m, 해당 pixel은 low DAS=0 / high DAS=255다. 예를 들어 Road37 lane -3, s=60 m의 실제 위치는 `(-52.895119, -207.519623, 6.094700)`이고 data5 pixel `(1365,251)`은 high에만 있다. PNG가 없거나 PPM이 잘못된 문제가 아니다.

Road37의 1 m 표본 4,890개를 추가로 검사했다.

| 실제 PNG 지지 | 표본 수 | Native 지면 Z 범위 | s 범위 |
|---|---:|---:|---:|
| low와 high 모두 있음 | 1,011 | 0~10.000077 m | 0~814 m |
| high만 있음 | 3,879 | 3.587246~10.026056 m | 58~749 m |

Road37 전체를 항상 high로 분류해서도 안 된다. 도로가 지상에서 고가로 이어진다. 또한 high mask만 존재한다고 해당 높이 도로에 소속된다고 단정할 수 없다. 실제 `(-240.845596,-99.490433,0)`은 Road9 lane -5 **하부 Sidewalk**이지만 low DAS=0/high DAS=255이다. Road3/24의 다른 하부 Sidewalk에서도 같은 반례를 확인했다.

따라서 layer 정책은 원본 기준·실제 ground 확인·mask support 사용 여부를 metadata에 구분해야 한다. 검증된 ramp에 대한 제한적 보완을 모든 Actor나 Sidewalk에 확대하면 안 된다. 전체 PNG를 다시 rasterize하지 않고, 실제 CARLA ground identity/height를 선택 보조 정보로 사용하는 것은 별도 문제다.

## 기존 Height PNG와의 관계

이전에 제공된 LRS에는 `sim_using_data/height_estimator/Town04|Town05/height_low.png`와 `height_high.png`가 있다. `loop.py`가 `pixel/255×16`으로 변환하는 것은 실제 코드에서 확인된다. 하지만 이 파일들은 **새 ZIP에 포함되지 않았고**, 원본 생성·유효 mask·미터 단위 정확도까지 확인된 것은 아니다. 새 RGB/DAS 값을 높이로 바꾸거나, 파일이 없는 상태에서 가상의 높이 채널을 생성하면 안 된다. 외부 height 자료를 연결한다면 출처·offset·PPM·인코딩·validity를 명시적으로 등록해야 한다.

## 필요한 manifest 항목

- source ZIP/hash, 파일 상대경로와 SHA-256, Town, 실제 W/H, PPM와 meters-per-pixel.
- `world_offset_xy_m`, global 축 방향, CARLA yaw 단위, 원본→Local affine 및 Local geometry.
- base/high별 실제 파일 목록, missing/empty 구분, grayscale/alpha 처리, antialias mask 변환 규칙.
- 실제 available semantic layer와 선택 정책의 provenance. high 파일이 없는 channel을 base에서 임의 복사하지 않는다.
- height source는 첨부 기준 `null`; native Actor ground Z와 dense image height를 구분한다.
- PNG마다 고정 label source를 기록하고, 실행 중에는 설정 snapshot과 원본 파일 hash를 고정한다.

전체 이미지 header·hash·histogram, PPM 후보별 실제 좌표 검사, Town05 경사로 표본은 [global-map-source-verification.json](validation/global-map-source-verification.json)에 저장했다. 실제 CARLA 서버의 RGB/Actor 동기화, 모든 PNG의 독립 GT 정확성은 이번 소스 조사에서 검증하지 않았다.

## 실제 production crop 및 실행 시간 검증

![동일 영역의 5/10 PPM 직접 crop](validation/global-map-direct-crop-comparison.png)

새 production `GlobalMapBEVGenerator`를 실제 첨부 PNG로 실행했다. Town04 하부·상부와 Town05 경사로의 실제 확인된 위치에서 yaw 0/90/180/270°를 각각 조회했다. 5 PPM은 200×200, 10 PPM은 400×400으로 **둘 다 같은 40×40 m 영역**이다. 원본 LRS `make_local_bev_ego_aligned` 함수를 같은 이미지·pose에 실행한 DAS/Lane/Boundary **72건에서 픽셀 차이 0**을 확인했다. 기록된 좌표 변환 행렬의 ego anchor 위치 오차는 최대 `3.411e-14 m`였다. Catalog의 14개 Town/해상도 variant도 원본 hash 검사를 통과했다.

두 해상도의 원본 PNG 자체는 독립적으로 그려진 edge/paint pixel 차이가 있으므로, 10 PPM 결과를 절반으로 줄여 5 PPM과 강제로 일치시키지 않았다. 위 그림에서도 점선과 중심선이 원본 그대로 보인다. 첨부에 없는 dense elevation은 모든 검사에서 NaN, `elevation_validity`는 0이었다. 100개 RGB/RGBA 이미지의 전체 RGB 채널도 서로 같으며 모든 alpha가 255임을 확인했다.

동일 Town01 pose, 동일한 실제 native waypoint 위 통제용 box 20개, 500×500 / 0.2 m/pixel 조건에서 기존 v8 sparse-patch 캐시와 새 직접 crop을 비교했다. 이전 캐시는 다시 만들지 않았고, 양쪽 모두 OpenCV 9 threads, warm-up 2회 후 10회 측정이다. 두 방식의 정적 라벨은 다르므로 **실행 시간 비교이며 정확도 동등성을 주장하지 않는다**.

| 동일 입력의 처리 방식 | 중앙값 | P95 | 포함된 통제용 Actor |
|---|---:|---:|---:|
| 이전 v8 sparse patch | 75.615 ms | 79.900 ms | 20 |
| 새 전체 PNG 직접 crop | 11.657 ms | 14.747 ms | 20 |

새 방식의 내부 phase 중앙값은 map lookup 0.040 ms, 좌표 계산 0.085 ms, crop/rotation 3.707 ms, Actor 처리 3.325 ms, 합성 4.071 ms다. 단계별 중앙값의 합은 전체 시간 중앙값과 일치하지 않을 수 있다.

Actor가 없는 40×40 m 출력에서는 5 PPM/200²가 2.011~2.699 ms, 10 PPM/400²가 6.272~6.881 ms였다. 모든 warm 측정에서 **PNG 파일 재읽기 0회**를 확인했다. 큰 Town04 10 PPM 이미지를 처음 decode하는 프레임은 약 6.907초가 걸렸으므로 이 최초 로드 시간을 warm 생성 시간과 분리했다. 이 수치는 공유 CPU 실행 환경의 측정값이며 CARLA 시뮬레이션, 센서 처리, 파일 저장, 웹 전송이 포함된 수집 FPS를 보장하지 않는다.

재현 명령:

```bash
python scripts/validate_global_maps.py --lrs-root /path/to/LRS --runs 10
```

기존 캐시가 있을 때만 `--legacy-cache /path/to/old/cache --legacy-xodr /path/to/Town01.xodr`를 추가하면 이전 방식의 runtime 비교도 수행한다. 원본 LRS 코드는 알려진 소스 hash와 일치하는지 확인한 뒤 비교 함수만 실행한다.
