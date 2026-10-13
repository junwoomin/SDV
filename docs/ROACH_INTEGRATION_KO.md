# SDV CARLA–LRS Roach 통합 결과

## 완료 범위와 현재 판정

기존 수집기의 Ego 제어를 Roach PPO → CARLA `VehicleControl` 경로로 교체했고, 기존 SDV 제어기·TrafficManager Autopilot·웹 수동 제어를 비교 모드로 추가했다. Roach 입력과 RGB/Segmentation/Depth/OD/BEV 및 정책 원본·최종 제어를 같은 CARLA Frame의 메타데이터에 연결했다.

**실제 CARLA 주행 검증은 완료하지 못했다.** 이 실행 환경에는 CARLA 서버와 GPU가 없으며, 실제 CARLA 0.9.15 Python 클라이언트의 `127.0.0.1:2000` 접속도 2초 타임아웃으로 실패했다. 따라서 Roach가 기존 제어기보다 안정적이라는 결론, 실제 Route Completion/충돌/신호위반 수치, 실제 센서 수집 성공 결과는 없다. 아래 CPU·경계 테스트 결과와 실주행을 구분해야 한다.

## 기존 비정상 주행의 코드상 원인

| 발견 항목 | 기존 동작과 영향 | 변경 |
|---|---|---|
| 무작위 조향 | `workflow.py`가 `BehaviorAgent.run_step()` 이후 조향에 매번 `uniform(-0.2,0.2)` 추가. PID가 만든 경로 추종 명령이 변조됨 | 잡음 제거. 원본 Action과 최종 적용 제어 별도 기록 |
| 시간 간격 불일치 | 수집 기본 10 FPS, LocalPlanner PID 기본 `dt=1/20` | 실제 시뮬레이션·제어 주기로 PID 설정. LRS 입력 History는 30 Hz로 고정 |
| 라벨 불일치 | 기존 `next_control`은 무작위 조향 전 값이어서 실제 적용 조향과 달라질 수 있음 | `raw_action`, `raw_vehicle_control`, `applied_control`, `previous_vehicle_control` 분리 |
| 경로 교차점/고가도로 | 기존 가장 가까운 Global Route 인덱스가 전체 경로 XY만 검색 | 진행 구간을 제한한 단조 증가 XYZ 경로 투영 |
| 프레임 경쟁 | 다른 클라이언트가 tick한 경우 `continue`하여 이전 명령 유지 가능 | WorldSnapshot·센서 Frame 불일치 시 제동 후 오류 종료 |
| 추론 연결 부재 | Ego는 SDV `BehaviorAgent`, TrafficManager는 NPC 제어. Roach가 Ego를 제어하지 않았음 | 기본 Ego controller를 실제 Roach runtime으로 교체 |

이 항목들은 소스에서 확인한 결함이다. 각 결함이 사용자 영상의 특정 회전·이탈을 얼마나 설명하는지는 동일 장면 재현 시험 전에는 확정할 수 없다.

## 확인한 LRS 원본과 입력 계약

기준 저장소: https://github.com/junwoomin/LRS

고정 Commit: `6eaf6c38450efc540f5fb1c86ed2980dd96a59bc`.

`loop.py`, `simulate.py`, `sim/Global_Dynamic_MapManager.py`, `sim/routeplanner.py`, `roach/models/ppo_policy.py`, `torch_layers.py`, `distributions.py`, `roach/rl_birdview_agent.py`, `roach/utils/rl_birdview_wrapper.py`, `checkpoint.py`, `roach/config/config_agent.yaml`을 비교했다. 파일 출처와 변경 범위는 `sdv/driving/vendor/roach/SOURCE.json`에 기록했다.

**중요: LRS 실제 실행 경로와 원래 Roach 학습 Wrapper는 동일하지 않다.** 체크포인트 안에는 History 순서·픽셀 Anchor·기어 정규화의 학습 provenance가 없다. 이번 구현은 사용자가 지정한 LRS 실제 `loop.py` 경로를 보존하는 `lrs_loop` 계약을 사용한다. 원래 Roach 학습 환경과 완전히 일치한다고 주장하지 않는다.

| 항목 | 이번 `lrs_loop` | 원래 YAML/Wrapper |
|---|---|---|
| Birdview | 15 × 192 × 192, 값 0–255 | 같은 Shape |
| 채널 | road, route, lane, vehicle×4, pedestrian×4, traffic-light×4 | 같은 그룹이나 History 순서는 다름 |
| History | `[-1,-5,-10,-15]`, 최신→과거 | `[-16,-11,-6,-1]`, 과거→최신 |
| History 시간 | 30 Hz에서 0, 4/30, 9/30, 14/30초 | YAML만으로 실제 시간 확정 불가 |
| Ego Pixel | `(96,136)` | `(96,152)` |
| State | throttle, steer, brake, LRS gear, ego vx, ego vy | Wrapper는 gear/5 |
| 기어 | P=0, R=1, N=2, D=3; CARLA 방향 상태에서 변환 | CARLA 변속 단수와 혼동 금지 |
| dtype/정규화 | 정확한 픽셀 값을 uint8로 보관하고 추론 경계에서 float32, 정책 내부 `/255` | LRS loop가 먼저 float32로 변환 |

Roach Policy 입력은 항상 5 PPM이다. 데이터셋 지도 5/10 PPM 선택을 유지하며, 10 PPM 지도도 동일한 물리적 범위를 5 PPM 좌표로 Nearest Neighbor 샘플링한다. 일반 SDV 시각화용 BEV를 신경망에 넣지 않는다. Height Layer는 기존 HBLR 도로 선택에 사용하며 채널을 추가하지 않는다.

차량·보행자·신호등은 CARLA 실제 snapshot/actor에서 읽는다. History의 과거 World 좌표를 **현재 Ego 좌표**로 다시 투영한다. 픽셀 의미를 유지하려고 차량 4.69×2 m 및 보행자 9×9 px의 LRS Template을 정책 입력에 사용한다. 실제 Actor Box는 별도 수집 정보에서 보존한다. 네트워크 입력 Template을 실제 차량 크기와 동일하다고 해석하면 안 된다.

신호등은 영향을 받는 차선의 stop waypoint/정지선 정보를 사용하며 Green/Yellow/Red를 80/170/255로 rasterize한다. 도로 높이는 CARLA waypoint와 실제 box 하단을 사용한다. 도로 밖 보행자 등 waypoint ground가 불명확하면 같은 높이의 실제 box를 보수적으로 유지하고 불확실성을 기록한다. 제공된 PNG 지도에는 정밀한 미터 단위 높이 채널이 없으므로 4 m 높이 허용폭과 low/high 선택은 경사로의 전체 연결 관계를 재구성하지 못한다. 고가도로 완전 분리는 실주행 검증 대상이다.

## Checkpoint와 정책 실행

원본 파일: `roach/log/ckpt_11833344.pth` (8,890,031 bytes).

SHA256: `1fecec0c8a206a9ac07a6a9b0be77a5b7e95a073b20d9dfc2a7397354355094a`.

확인된 로컬 파일의 전체 Hash가 LRS 로더에 지정된 Hash와 일치한다. 이 전달물에 포함했지만 공개 배포 허가나 원래 학습 데이터 provenance까지 확인된 것은 아니다.

* `weights_only=True` 제한 로딩. 정확한 원본 Hash에만 Gym/NumPy 메타데이터 allowlist 적용.
* NumPy 2의 `_reconstruct` 경로 변경 처리. 임의 pickle fallback 없음.
* CNN 과거 Sequential 이름 변환과 현재 ModuleList 이름을 구분하며 `strict=True`로 로딩.
* `PpoPolicy.forward()`의 7개 반환값을 정확히 처리. 6개로 unpack하는 원본 `RlBirdviewAgent.run_step()` 경로를 호출하지 않음.
* 파일 없음·손상·Shape 오류·비유한 값·추론 실패는 오류. 임의 초기화 정책으로 주행하지 않음.
* Acceleration ≥0이면 Throttle, 음수이면 Brake, Steering 부호 보존.
* CARLA Physics가 차량을 이동한다. BicycleModel로 Actor pose를 갱신하거나 매 tick teleport하지 않음.

## 실행 방법

CARLA 서버는 GPU가 있는 사용자 장비에서 0.9.15로 실행해야 한다. 수집용 권장 Python은 **3.10**이다. 기존 Python 3.7 연구 환경과 별도 환경을 사용한다.

```bash
cd sdv_studio
python3.10 -m venv .venv-roach
source .venv-roach/bin/activate
python -m pip install --upgrade pip
# CPU 예시. CUDA를 사용할 경우 장비에 맞는 PyTorch wheel을 먼저 설치한다.
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-roach.txt
python -m sdv.driving.policy inspect --checkpoint roach/log/ckpt_11833344.pth --device cpu
SDV_RESEARCH_PYTHON="$PWD/.venv-roach/bin/python" ./start.sh
```

웹 설정의 연구 Python이 `.venv-roach/bin/python`인지 확인한다. `start.sh`는 기존 설계에 따라 시작 시 CARLA 종료 helper를 실행하므로, 웹 시작 후 CARLA를 연결/기동한다. 수집할 Town의 5 또는 10 PPM 지도가 있어야 한다. 없는 Town/해상도는 다른 지도로 대체하지 않고 오류를 표시한다.

1. **Simulation / Data Collection → Driving Agent → Roach Policy** 선택.
2. 원본 Checkpoint, CPU/CUDA, Deterministic Inference 확인 후 **모델 로딩 검사**.
3. Simulation 30 FPS, Control Frequency 30 Hz로 시작. 15/10 Hz 등은 실험용으로 고를 수 있으나 History는 매 30 Hz tick 갱신한다.
4. RGB/Segmentation/Depth/OD 센서를 활성화하고 데이터 수집 시작.
5. Scene Information에서 실제 최신 제어 Tick과 저장 센서 Frame을 각각 확인.

Manual Control은 해당 모드 수집 중 화면의 방향 버튼을 누르고 있을 때 작동한다. 150 ms heartbeat, 0.5초 명령 만료, 버튼 해제·blur·연결 중단 시 제동이다. 키보드 Space/Enter도 지원한다.

## 데이터와 Frame 의미

`info/{sample}.json`은 모든 센서 파일 저장 뒤 작성되는 Frame commit marker다. 저장 이름은 로컬 sample ID이며 실제 CARLA ID는 메타데이터의 `carla_frame`/`observation_frame`을 확인한다.

| 기록 | 의미 |
|---|---|
| RGB, semantic ID PNG, float32 metric Depth NPY, OD NPY, BEV NPZ | 요청한 센서/라벨의 동일 Frame F |
| `roach_observation/{sample}.npz` | Birdview, State, native Frame F, timestamp |
| `previous_vehicle_control` | Frame F 상태를 만들 때 사용하던 이전 명령 |
| `raw_action`, `raw_vehicle_control` | Frame F 관측으로 정책이 만든 명령 |
| `applied_control`, `safety_reason` | 안전 제어 후 실제 제출한 최종 명령 |
| `command_effective_from_frame=F+1` | 제출한 명령이 다음 물리 tick에 영향을 줌 |
| `action_source_observation_frame` | 저주파 제어에서 유지 중인 Action을 실제로 추론한 원본 Frame |
| `routes/{hash}.json`, Navigation | Global Route, Local Waypoints, RoadOption, Target, Road/Lane, 경로 편차 |
| 정책 metadata | 체크포인트 경로/Hash, 입력 계약, 추론/제어 지연, History Frame, 상태 |

기본 저장 주기는 2 tick마다 1회이므로 30 Hz 정책에 대해 샘플은 15 Hz다. 충돌 terminal tick은 정책·충돌 event를 저장하고 재스폰하며, 해당 tick의 이미지가 반드시 저장되지는 않는다. 따라서 충돌 event를 정상 주행 샘플의 연속 구간과 혼동하지 않아야 한다.

센서가 Frame F에 도착하지 않거나 외부 ticker가 Frame을 바꾸면 안전 정지하며 오래된 센서로 샘플을 완성하지 않는다. History warmup에서는 정책 원본 출력과 제동 적용값을 구분한다. TrafficManager 모드는 다음 명령을 외부에서 확정할 수 없으므로 관측된 제어라는 별도 semantics를 남긴다. 종료 시 기존 World 설정을 복원한다. 다만 TrafficManager의 이전 동기 상태를 조회하는 공용 API가 없어 외부 TM 상태 복원까지 보장하지 않으므로 수집·평가는 전용 서버에서 수행한다.

## 실제 비교 시험 실행기

`scripts/validate_roach_carla.py`는 동일 Town·출발/목적지·날씨·Seed·NPC 초기 배치로 Existing/Roach를 비교한다. Existing 비교군은 **무작위 조향 제거 및 PID dt 수정 후**의 SDV 제어기다. 서로 다른 정책에 의해 이후 교통 궤적이 달라지므로 완전히 같은 closed-loop traffic이라고 주장하지 않는다.

```bash
# 현재 Town의 실제 spawn index 조회 (서버 필요)
python scripts/validate_roach_carla.py --list-spawns --output reports/spawns
# 시작/도착 index는 조회 결과에서 고른 실제 값을 넣는다.
python scripts/validate_roach_carla.py --run --town Town03 \
  --start-spawn START_INDEX --destination-spawn DEST_INDEX \
  --agents existing roach --duration 120 --seed 2026 --output reports/roach_live
```

전용 시험 서버에서 실행한다. `--run`은 컨트롤러별 비교 전 Town을 다시 불러오므로 해당 World의 기존 Actor가 사라진다. `configs/evaluation/roach_scenarios.json`은 직선, 좌/우회전, 차선변경, 교차로, 적색 정지, 선행차 감속, 보행자, 경로 이탈, 고가도로, 도착, 연속주행의 12개 설정을 제공한다. 실제 Town의 검증된 출발/도착 index를 채워야 하며 미지정 경로를 성공 처리하지 않는다.

측정: Route Completion, 충돌, 실제 진행 방향/높이를 고려한 앞 범퍼의 적색 정지선 통과, 차선침범/off-road, 평균속도, 경로 편차, 추론 및 제어 지연. 단순히 이동했다고 시나리오 통과 처리하지 않는다. 고가도로 객체 분리는 별도 Frame 검토 전 미검증을 유지한다. 수집기의 일상 telemetry에서 정확히 계산하지 않은 신호위반 수는 `null`이며 0으로 만들지 않는다. 차선 센서 이벤트는 합법적인 차선 변경도 포함하므로 `lane_invasion_event_count`를 불법 차선 이탈 횟수로 해석하지 않는다.

## 검증과 미검증

최신 실행 증거는 `docs/validation/roach_validation_summary.json`, `roach_policy_cpu.json`, `roach-carla-connection.json`, `roach_evaluation/preflight.json`을 참조한다. 기존 ZIP에 있던 다른 검증 문서는 이번 Roach 변경의 실주행 결과가 아니다.

최종 전체 자동 검사: **819 passed, 4 skipped, 0 failed**. 브라우저에서 실제 CPU 모델 로딩 및 4개 모드를 확인했고 1440/1280/900 px 화면의 패널 넘침은 0 px였다.

검증 완료: 원본 Hash/strict 로딩, 원본 정책과 합성 입력 3종의 Action·Value 오차 0, deterministic 반복 추론, LRS raster 함수와 입력 경계 비교, 5/10 PPM/좌표/History/Height 검사, 오류 제동·Frame/제어 인과관계, 웹 API/설정/수동 명령 검사.

미검증: 실제 CARLA 10개 요구 상황의 주행, CUDA, 실센서 동시 저장, 고가도로 복합 교통, 실제 정책 우월성, 원래 학습 Observation profile과 LRS loop의 동일성. 이 항목이 남아 있으므로 요청 전체가 실환경에서 완료됐다고 판정할 수 없다.

수정 파일 목록은 `docs/ROACH_CHANGED_FILES.json`에 포함했다. 핵심 파일은 `sdv/driving/policy.py`, `observation.py`, `sdv/collection/driving_runtime.py`, `workflow.py`, `studio/static/driving-ui.js`, `scripts/validate_roach_carla.py`다.
