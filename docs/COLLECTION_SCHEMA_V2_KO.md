# 수집 스키마 v2 및 검증 범위

기존 `Town/scene/{modality}/{frame}` 구조와 `info/*.json`의 기존 키를 유지한다. 새로 수집한 데이터는 `collection_manifest.json`의 `schema_version: 2`, `format: sdv-native-masks-v2`로 구분한다. 기존 데이터 파일은 수정하거나 추정 복원하지 않는다.

## 저장 포맷 수정

| 항목 | 기존 데이터에서 확인한 문제 | 신규 저장 |
|---|---|---|
| Segmentation | 색상 PNG만 저장하여 정수 클래스 ID를 직접 확인하기 어려움 | 기존 색상 PNG와 함께 `{camera}_seg_ids/{frame}.png`에 원본 Red 채널 uint8 ID 저장 |
| Depth | CARLA packed BGR를 미터 float로 디코딩한 뒤 PNG에 직접 써서 uint8 반올림·255m 포화 발생 | `{camera}_depth_m/{frame}.npy` float32 미터를 학습용 원본으로 저장. 기존 PNG는 호환용 반올림·포화 미터 미리보기로 유지 |
| Object Detection | 옛 semantic ID 차량 10/보행자 4로 가시성 필터; 현 데이터의 클래스 ID와 불일치. Truck/Bicycle 그룹 제외 | 차량 14~19, 보행자 12로 필터. 실제 Vehicle 그룹에 Truck/Bicycle 포함. 카메라별 해상도/FOV의 K 사용 |
| 센서 동기화 | 가장 최근 `.img`만 읽어 native frame 일치 여부 확인 불가 | 모든 필수 RGB/Seg/Depth/LiDAR/IMU packet의 native frame이 현재 world tick과 일치할 때만 저장 |

Depth 식은 `1000 * (R + 256*G + 65536*B) / 16777215` m이다. NPY가 있는 신규 샘플에서는 PNG를 metric label로 사용하면 안 된다. 이미 포화된 옛 Depth와 옛 빈 OD annotation에서 손실된 정보는 복구할 수 없다.

OD NPY는 `[N, 5]` float32이고 열 순서는 `x1, y1, x2, y2, class_id`이다. 좌표는 해당 카메라의 2D 픽셀이며 Vehicle=0, Pedestrian=1이다. 빈 배열도 `[0, 5]`로 저장한다. 기존 annotation에는 객체 ID가 없다. 3D 박스 평가 또는 신호등 객체 검출이 지원된다고 해석하지 않는다.

## 원본 Frame과 시간

`info/{stem}.json.sdv_metadata.synchronization`:

- `sequence_id`: Town/scene 상대 경로. 다른 Map의 같은 scene 번호와 구분
- `frame_id`: 저장 파일 stem. 수집 카운터이며 CARLA frame과 다름
- `carla_frame`: `World.tick()`과 `WorldSnapshot.frame`으로 확인한 원본 시뮬레이터 frame
- `timestamp_s`: WorldSnapshot의 시뮬레이션 경과 시간
- `sensors`: 센서별 원본 frame, timestamp_s, 측정 당시 world transform
- `aligned`: 필수 센서의 native frame 일치 여부

각 센서 callback은 이미지·frame·timestamp·측정 당시 world transform을 하나의 packet으로 게시한다. IMU도 고정한 packet에서 저장한다. 저장 시 packet 참조를 고정하여 이후 callback과 섞이지 않는다. 센서가 늦으면 최대 2초 대기 후 그 frame을 생략한다. 10회 연속 동기화 실패 시 명확한 오류로 수집을 종료하며 센서 tick과 CARLA 상태 확인을 안내한다. 센서 대기 직후와 최종 commit 직전에도 world frame을 확인한다. 다른 클라이언트가 추가 tick을 발생시켰다면 혼합 데이터를 commit하지 않고 오류로 종료한다. `info` JSON은 모든 이미지 및 annotation이 쓰인 다음 원자적으로 게시하는 commit marker이다. 중단 시 JSON 없는 부분 파일은 완성 샘플이 아니다.

LiDAR의 기존 `points * 2` 처리는 연구 입력 호환성을 위해 유지했다. Manifest에 해당 배율을 명시한다. native CARLA 좌표와 intensity로 복구할 때 모든 열을 2로 나누어야 한다.

## 추가 메타데이터

기존 `ego_vehicle_loc`, `ego_vehicle_rot`, `current_speed`, `current_control`, `next_control`, BEV 좌표의 `global_waypoints`, `bev_future_waypoints`, `imu`, collision 플래그는 유지한다. 추가 관측은 `sdv_metadata` 아래에 저장한다.

- **ego**: Actor의 velocity(m/s), acceleration(m/s²), angular velocity(deg/s), gear, reverse, hand brake. `applied_control`은 현재 이미지 frame 이후 제출된 실제 제어 명령이며 다음 물리 tick에 영향을 준다고 명시한다. 조향 noise를 더하기 전의 기존 `next_control`과 혼동하지 않는다.
- **road**: Waypoint의 road/section/lane ID, lane type/width, lane change, 좌우 marking type/color/width, junction 상태/ID, Vehicle의 speed limit(km/h).
- **navigation**: 실제 local planner의 target RoadOption, incoming RoadOption, target speed, target world waypoint, 현재 queue의 최대 30개 world waypoint. 실제 global route는 변경될 때 `routes/{sha256}.json`으로 한 번 저장하고 참조한다. 이 값들은 온라인 경로 계획이며 미래에 실제 주행한 위치 GT가 아니다.
- **traffic**: 실제 TrafficLight ID/state/world position, pole까지 유클리드 거리, stop waypoint, affected lane waypoint, OpenDRIVE ID, 최초 관측 frame, 매 simulator tick에서 최초 확인한 상태 변경 frame.
- **environment**: 실제 map name/weather, 반경 60m 이내 vehicle/pedestrian actor ID/type/위치/속도/거리.

신호등의 Ego 적용 여부는 `Vehicle.get_traffic_light()`가 반환한 actor ID로만 결정한다. 카메라에 보이는지 여부는 추정하지 않고 null이다. `get_stop_waypoints()`의 stop position은 저장하지만, 해당 정지선이 현재 경로에 적용된다고 추정하거나 임의의 stop-line distance를 만들지 않는다. 없거나 SDK가 반환하지 못하는 값은 null이다. 신호 적용 API 호출 실패는 `affecting_signal_lookup_available: false`, `affects_ego: null`로 기록하여 정상적으로 적용 신호가 없는 false와 구분한다.

기존 수집 루프는 충돌 시 차량과 scene을 재생성한다. 이를 바꾸지 않고 재생성 전에 `events/collision_{native_frame}.json`으로 실제 collision callback의 상대 actor와 충격량, frame/timestamp를 저장한다. 종단 tick 이미지가 항상 저장되는 것은 아니며 파일에 그 제한을 명시한다.

## 좌표와 Calibration

World/Ego는 CARLA의 왼손 좌표계, x 전방·y 우측·z 위쪽, 위치는 미터, 회전은 degree이다. 카메라 투영은 sensor-local `(x,y,z)`를 optical `(y,-z,x)`로 변환한다. `sensor_config.yaml`의 카메라별 width/height/FOV와 부착 위치·회전이 calibration의 원본이며, 매 frame 센서의 실제 world transform도 저장한다.

기존 BEV 수집기는 500×500, 5 pixel/m, Ego 중심 (250,250), 전후좌우 각 50m이다. Ego local `(x,y)`의 투영은 `u=250+5*y`, `v=250-5*x`이다. 이 수집 범위를 변경한 것처럼 표시하지 않는다.

## 검증

`tests/test_collection_metadata.py`는 실제 함수에 CARLA와 동일한 속성/메서드를 제공하는 통제된 객체를 넣어 검증한다. native 값 유지, null 처리, 교차 frame 거부, 원자적 저장, 신호등 적용/상태 변경, online route 참조, 실제 미터 NPY 보존, OD 현재 클래스 및 카메라별 투영을 확인한다. 이는 실제 CARLA 서버 또는 GPU에서의 실행 검증을 대신하지 않는다.

참고: CARLA 0.9.15 Python API (`Vehicle`, `Actor`, `TrafficLight`, `SensorData`) 및 Depth sensor 정의: https://carla.readthedocs.io/en/0.9.15/python_api/ , https://carla.readthedocs.io/en/0.9.15/ref_sensors/ . 0.9.15 문서 페이지의 Semantic ID 표는 옛 순서를 보여주므로 해당 표를 그대로 사용하지 않았다. 제공 데이터와 프로젝트의 실제 팔레트를 확인했고, 변경 이력을 명시한 최신 공식 Semantic sensor 표(0.9.13→0.9.14 변경)와 대조했다: https://carla.readthedocs.io/en/0.9.16/ref_sensors/ .
