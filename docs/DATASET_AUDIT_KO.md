# 실제 수집 데이터 진단 및 Label 계약 수정

검사 대상: 업로드된 `collections.zip`의 `collection_13cd2a2077084429/Town01/scene_20261008_140246_0001`. 임의 샘플을 만들지 않고 저장된 200개 Frame 전체를 분석했다. 원본 ZIP은 수정하지 않았다. 상세 수치와 Frame별 결과는 `dataset-audit.json`에 있다.

## 실제 구조

- `front_cam_rgb`, `front_cam_seg`, `front_cam_depth`: 각 200개 PNG, 400×300. RGB/Seg는 3채널, Depth는 uint8 단일채널.
- `front_cam_bbs`: 200개 float32 NPY. 저장 계약은 `[x1,y1,x2,y2,class_id]`, 2D 픽셀 좌표이며 Actor ID는 없다. **실제 파일 200개 모두 빈 `(0,)` 배열**이다.
- `das`, `lane`, `vehicle_mask`, `walker_mask`, `stopline`, `route`: 각 200개 500×500 마스크. `light_topview`는 RGB 500×500.
- `info`: 200개 JSON. 파일명 `00000_a`부터 `00199_a`까지 순서가 끊기지 않는다. 이것은 저장 샘플 번호이며 CARLA Frame ID나 Timestamp가 아니다.
- `collection_manifest.json`: `sdv-native-masks-v1`, Town01, FPS=10, save_every_ticks=2, weather=cycle. 단위·동기화 정보가 충분하지 않으므로 저장 주기만으로 실제 Timestamp를 확정하지 않는다.

## Segmentation 결론

**실제 클래스 매핑 오류와 Resize 오류가 함께 있었다.** 수집기는 현재 `sdv/collection/encoding.py`의 29색 팔레트로 저장한다. 기존 학습기는 오래된 6채널 색상표를 사용했다. 예를 들어 실제 Wall을 Pedestrians, Pole을 RoadLines, Traffic Light를 Roads, Traffic Sign을 Sidewalks, Terrain을 Vehicles로 학습하고 있었다. 나머지는 Other로 합쳤다. 사용자 의도에 따른 정상적인 클래스 통합으로 볼 수 없다.

전체 24,000,000픽셀 중 저장된 팔레트에서 해독되지 않는 색상은 **0개**였다. 실제 클래스는 19종이다.

| 실제 ID | 클래스 | 픽셀 수 | 비율 |
|---:|---|---:|---:|
| 0 | Unlabeled | 36,325 | 0.151% |
| 1 | Road | 6,720,452 | 28.002% |
| 2 | Sidewalk | 3,427,872 | 14.283% |
| 3 | Building | 2,504,874 | 10.437% |
| 4 | Wall | 202,779 | 0.845% |
| 5 | Fence | 101,195 | 0.422% |
| 6 | Pole | 231,379 | 0.964% |
| 7 | Traffic Light | 74,208 | 0.309% |
| 8 | Traffic Sign | 923 | 0.004% |
| 9 | Vegetation | 719,453 | 2.998% |
| 11 | Sky | 9,107,798 | 37.949% |
| 14 | Car | 4,070 | 0.017% |
| 20 | Static | 338,561 | 1.411% |
| 21 | Dynamic | 70,394 | 0.293% |
| 22 | Other | 273,349 | 1.139% |
| 23 | Water | 1,059 | 0.004% |
| 24 | Road Line | 23,750 | 0.099% |
| 26 | Bridge | 161,251 | 0.672% |
| 28 | Guard Rail | 308 | 0.001% |

기존 6채널 매핑은 원본 크기에서 23,490,711픽셀(**97.878%**)을 Other로 처리한다. BGR→RGB 변환 자체는 올바르다. 잘못된 것은 ID/팔레트 정의와 정확한 색상 매칭 전에 수행한 선형 보간이다.

동일한 `00000_a`를 실제 기존 `Dataset.get_seg()`와 기존 모니터 `_seg()`에 통과시켰다. 256×256에서 기존 Loader는 6채널 중 3개 채널만 남기고 Other 64,430픽셀(**98.312%**)을 만들었다. 기존 모니터는 이 잘못된 Loader 출력을 그대로 정확하게 색칠하고 있었다. 따라서 표시만 바꿔서는 수정되지 않는다.

선형 RGB Resize는 같은 샘플에서 팔레트에 없는 중간색 **2,848픽셀**을 만들었다. 수정 방식은 원본 RGB에서 정수 ID를 먼저 복원하고 Nearest Neighbor로 Resize한 다음 선택된 클래스를 출력 채널로 매핑한다. 같은 샘플의 Nearest 결과에서 잘못된 ID는 **0개**다. 원본 ID 0은 1픽셀뿐이므로 축소 후 사라지는 것은 정상적인 다운샘플링이며 매핑 오류와 구분한다.

![원본, 기존 Loader, 기존 모니터, 수정된 Semantic 비교](screenshots/audit-seg-before-after.png)

수정 후 **실제 200개 Frame 모두 새 Loader를 실행**했다. 19채널의 출력 ID를 원래 Semantic ID로 복원한 결과가 모든 Frame에서 원본 ID의 Nearest Resize 결과와 정확히 같았다. Ignore 픽셀은 0개였으며, 수정된 모니터 GT 색상도 정확히 일치했다. 선택하지 않은 클래스는 설정에 따라 Ignore(255, Loss/Metric 제외) 또는 Background로 명시적으로 매핑된다. 기존 저장 실험/Checkpoint에는 명시적 legacy 모드를 유지하여 출력 채널을 몰래 바꾸지 않는다. 신규 웹 실험은 실제 발견된 클래스에 기반한 수정된 CARLA 매핑을 쓴다.

## Depth 결론

CARLA BGRA의 B,G,R 바이트에 `[65536,256,1]`을 곱해 미터로 변환하는 수집기 계산은 맞다. 그러나 float 미터 배열을 PNG에 직접 저장하여 **정수 미터 uint8로 반올림·포화**시켰다. 이후 학습기가 이 값을 `100/255`로 다시 곱했으므로 거리 단위가 틀어졌다.

실제 저장 범위는 2~255, 전체 픽셀의 **38.378%**가 255로 포화되어 있다. 해당 값의 정확한 원래 거리와 1m 미만 소수 정밀도는 기존 파일에서 복구할 수 없다. 수정 Loader는 이 Native v1 데이터를 정수 미터로 읽고 포화값을 Invalid로 처리한다. 원본 첫 Frame의 65,536개 Resize 픽셀에서 실제 Loader 값과 미터 값, Mask가 일치함을 검사했다. 새 수집기는 별도 float32 미터 NPY를 보존한다. 원래 uint8=0~100m 규약을 쓰던 외부 Legacy 데이터는 명시적 인코딩/출처에 따라 호환 처리한다. 값의 크기만 보고 인코딩을 추측하지 않는다.

## Object Detection 결론

`front_cam_bbs`는 코드상 2D XYXY+Class 배열이며 지원 Class는 0=Vehicle, 1=Pedestrian이다. 3D Box와 Object ID는 저장되지 않았다. RGB와 동일 stem을 사용하는 설계지만 실제 CARLA 센서 Frame ID가 저장되지 않아 센서 단위 동기화를 사후 입증할 수 없다.

**제공된 200개 배열에는 Box가 하나도 없다.** 원본 Segmentation에는 Car 4,070픽셀이 존재하지만 이 사실만으로 정확한 Box/Actor ID를 생성할 수 없다. 기존 수집기의 가시성 필터는 현재 ID와 다른 Vehicle=10, Pedestrian=4를 사용하며, Truck/Bicycle 분리 목록을 최종 Box 생성에서 빠뜨렸다. 앞으로의 수집을 수정해야 한다. 기존 자료 미리보기는 실제 RGB와 ‘Annotation 없음’을 표시하며 가짜 Bounding Box를 추가하지 않는다. 양성 OD Box의 실제 CARLA 수집 결과 검증은 실행 환경이 없어 미검증이다.

## 기존 info에 있는 정보와 없는 정보

실제 저장값: `ego_vehicle_loc`(World 위치), `ego_vehicle_rot`(Pitch/Yaw/Roll), `current_speed`(km/h), `current_control`/`next_control`(Throttle/Steer/Brake), `current_waypoint`(World 위치), `imu`(가속도·각속도·Compass), `collision`, `vehicle_in_front`, `traffic_light`, `traffic_light_20`, `traffic_light_distance`, `global_waypoints`, `bev_future_waypoints`.

주의: `traffic_light`는 Red/Green 문자열이 아니라 기존 Agent의 ‘적색 신호가 Ego에 영향을 주는가’ Boolean이다. `traffic_light_distance`도 기존 코드에서 같은 Road·진행 방향의 첫 Trigger Waypoint까지 거리이며 정확한 Stop Line 거리로 표시해서는 안 된다. `global_waypoints`와 `bev_future_waypoints`는 이름과 달리 저장 직전에 BEV 픽셀 좌표로 변환된 값이다. World 좌표나 미래 실제 주행 GT로 재해석하면 안 된다.

미저장: 실제 Frame/Timestamp, 센서별 Frame/Timestamp, Gear, Velocity 벡터, Road/Lane ID·폭·Marking·Junction, RoadOption, Target Speed, Signal ID/Enum State/변화 Frame/정확한 Stop Line, 주변 Actor 목록, 실제 World Route, Calibration 변환 정의. 기존 데이터에서 해당 값은 N/A로 유지한다. Route와 미래 실제 Ego 궤적은 다른 개념이며, 향후 Target은 실제 연속 시퀀스의 미래 기록에서 별도로 생성해야 한다.

첫 Frame IMU 가속도 Norm은 **15,145.204m/s²**로 비정상적이다. 값은 임의 교체하지 않으며 품질 경고 대상으로 분류한다. 모든 파일명이 맞아도 Timestamp/센서 Frame ID가 없으므로 물리적 동기화가 검증되었다고 주장하지 않는다.

## 확인 범위

- 실제 업로드 데이터 전체 200 Frame, 2,600개 모달리티 파일의 존재·형태·Semantic/Depth/Box 통계 검사.
- 실제 기존 Loader와 기존 모니터의 동일 샘플 GT 비교.
- 수정된 실제 Loader 200 Frame ID/색상 정합성 및 Depth 첫 샘플 단위·Mask 비교.
- CPU Label 계약 테스트; 동적 Head/Loss/Metric 검사와 UI/통합 검증은 상위 검증 보고서 참조.
- CARLA 서버·GPU로 재수집/전체 학습은 미검증. TCP 학습 코드와의 완전 호환성도 주장하지 않는다.

코드 근거: `sdv/collection/encoding.py`, `sdv/collection/workflow.py`, `sdv/collection/carla_engine/main.py`, `sdv/collection/carla_engine/agents/navigation/basic_agent.py`, `sdv/training/modeling/data.py`, `sdv/training/predictions.py`. CARLA 공식 센서의 Depth BGRA 및 단위 설명: https://carla.readthedocs.io/en/0.9.15/ref_sensors/ . 최신 ID 표 변경 설명: https://carla.readthedocs.io/en/0.9.16/ref_sensors/#semantic-segmentation-camera . 클래스 목록은 외부 표를 추측하여 주입한 것이 아니라 본 프로젝트 수집 팔레트와 실제 파일의 색상에서 확인했다.
