# Roach 주행 평가 도구

[Roach 정책 연결](ROACH_INTEGRATION_KO.md) · [환경 준비](SETUP_KO.md)

전용 CARLA 서버에서 동일 경로·초기 조건의 컨트롤러를 비교하고 프레임별 제어와 주행 지표를 저장하는 도구입니다. 평가할 Town, 경로, 지도와 호환 정책 가중치를 별도로 준비합니다.

## 실행

프로젝트 실행용 Python 환경에서 먼저 서버 연결과 의존성을 확인한다.

```bash
python scripts/validate_roach_carla.py --preflight --start-spawn 0 --destination-spawn 5
python scripts/validate_roach_carla.py --list-spawns --output reports/roach_spawns
```

위 `0`, `5`는 CLI 사용 예시이며 특정 주행 상황을 보장하는 검증된 경로가 아니다. 현재 Town의 출력 좌표를 확인하여 실제 출발/목적지를 선택한다.

**`--run`은 각 에피소드마다 지정 Town을 다시 로딩하므로 전용 CARLA 서버에서 실행한다.**

```bash
python scripts/validate_roach_carla.py --run \
  --town Town03 --start-spawn 0 --destination-spawn 5 \
  --checkpoint roach/log/ckpt_11833344.pth --device cuda \
  --agents existing roach --seed 2026 --duration 120 \
  --output reports/roach_comparison
```

정책 입력은 고정 5 PPM이다. 수집기와 동일한 Global Map/Height 설정을 사용할 경우 해당 BEV 설정 JSON을 `--bev-config <파일>`로 전달한다. 기본 시뮬레이션은 30 Hz이며 `--fixed-delta`, `--control-hz`로 명시적으로 변경할 수 있다. 학습 history 시간 간격은 observation adapter가 별도로 검사한다.

전체 상황은 `configs/evaluation/roach_scenarios.json`에서 Town과 각 출발/목적지 인덱스를 채운 후 실행한다.

```bash
python scripts/validate_roach_carla.py --run \
  --scenario-file configs/evaluation/roach_scenarios.json \
  --checkpoint roach/log/ckpt_11833344.pth --device cuda \
  --agents existing roach --output reports/roach_scenarios
```

`--case left_turn --case right_turn`처럼 일부 상황만 선택할 수 있다. 경로에 필요한 좌회전/우회전/차선 변경/교차로가 없으면 오류로 종료한다. 직선 시나리오는 회전/차선 변경이 포함된 경로를 거부한다. 선행차 fixture는 실제 CARLA 차량에 throttle/brake를 적용하며, 보행자는 WalkerControl로 이동한다. 차량 위치를 매 tick 강제로 덮어쓰지 않는다.

## 출력과 판정

- `preflight.json`: Python 모듈, 체크포인트 존재, RPC 포트, 미설정 경로와 차단 원인.
- `<상황>/<agent>/global_route.json`: 실제 CARLA waypoint, 좌표, road option, road/lane ID.
- `<상황>/<agent>/frames.jsonl`: WorldSnapshot frame/time, 속도, 경로 진행/이탈, 정책 원시/적용 제어, inference/control latency, navigation.
- `<상황>/<agent>/report.json`: 충돌/차선 센서 결과 기반 지표, 목적지 도착, 상황 fixture 발생과 미검증 항목.
- `comparison.json`: 동일 route SHA-256 및 초기 NPC spawn/blueprint 일치 확인, 컨트롤러별 실측 지표.

`existing`은 무작위 steering을 제거하고 PID dt를 수정한 현재 SDV BehaviorAgent다. 수정 전 릴리스와 동일하다고 표기하지 않는다. 동일 seed/초기 조건을 사용해도 두 정책의 주행 결과에 따라 이후 TrafficManager 동작은 달라질 수 있다. 안정성 향상 주장은 반복 측정 후 판단한다.

신호 위반은 **앞 범퍼가 CARLA native stopline을 진행 방향으로 통과하고 이동 시작 frame의 신호가 적색인 경우**만 센다. 주변에 적색 신호가 있다는 사실은 위반이 아니다. 반대/수직 진행 방향, 다른 높이의 stopline은 제외한다. stopline API가 제공되지 않으면 결과를 불완전하다고 기록한다.

경로 완주율은 3차원 waypoint 누적 거리의 투영값이다. 앞 40 m 탐색과 높이 차 2.5 m 제한으로 교차로/다른 deck으로의 진행률 점프를 제한한다. 경로와 5 m 이상 떨어지면 진행률을 높이지 않는다. 높이 일치 실패 frame은 안전 기준 통과로 처리하지 않는다.

Off-road는 CARLA Driving waypoint에 ego **중심**이 포함되는지 측정한다. 차선 센서 callback에는 합법적인 차선 변경도 포함되므로 개수를 그대로 위반 횟수라고 부르지 않는다. Collision Count는 동일 상대 actor에 대해 1초 이하 간격으로 이어진 접촉 callback을 한 사고로 묶는다. callback 원본 개수도 남긴다.

정책 inference 시간은 실제 추론한 frame만 집계한다. Control latency는 observation 생성부터 안전 개입 및 `apply_control`까지의 wall time이며 다음 physics tick은 포함하지 않는다. 제어는 observation frame F에서 생성되고 F+1의 물리 상태에 영향을 준다.

## 남아 있는 검증 범위

고가도로 상황은 주행만으로 HBLR actor 포함/제외의 정답을 알 수 없다. 상/하층 actor를 독립적으로 라벨링하고 frame별 observation metadata와 대조하기 전까지 자동 통과 처리하지 않는다. 현재 기본 시나리오 파일의 출발/목적지는 의도적으로 미설정되어 있다.

선행차 상황은 fixture가 제동했고 ego가 5 m 이상 주행했으며, 실제로 이동한 뒤 선행차와 1~20 m bumper gap에서 2초 정차했는지 확인한다. 이는 해당 상황의 최소 발생 증거이며 모든 충돌 회피 성능을 보장하지 않는다. 보행자 상황은 실제 crossing trigger와 목적지 도착을 요구한다.

이 평가기는 actor/충돌/차선 데이터를 사용한다. RGB·Segmentation·Depth·BEV 파일의 저장 동기화는 수집기에서 별도로 검증해야 한다. 실제 센서 callback 전달 지연/드롭, CARLA 렌더링, CPU/CUDA end-to-end 성능, 장시간 안정성과 고가도로 actor 정답 검증은 실제 서버 실험 없이는 확정할 수 없다.
