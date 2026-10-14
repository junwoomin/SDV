# Roach 정책 연결

[환경 준비](SETUP_KO.md) · [주행 평가](ROACH_LIVE_EVALUATION.md)

SDV 수집의 Roach 모드는 LRS 계열 정책으로 CARLA 차량을 제어하는 프로토타입 경로입니다. 호환 체크포인트, 실제 지도와 정책 관측 계약이 필요합니다.

## 구현

| 파일 | 역할 |
| --- | --- |
| `sdv/driving/observation.py` | ego 상태, BEV 마스크와 Actor history를 정책 입력으로 구성 |
| `sdv/driving/policy.py` | 체크포인트 무결성·구조 확인, 정책 추론과 Action 변환 |
| `sdv/collection/driving_runtime.py` | 경로, 제어 주기, CARLA Control 적용과 주행 기록 |
| `sdv/driving/vendor/roach/` | 정책 모델 소스와 원본 라이선스·출처 |

수집 설정에서 Roach 모드, 프로젝트 내부 체크포인트 경로와 지도 경로를 지정합니다. 별도 연구 환경의 [Roach 의존성](../requirements/roach.txt)을 준비하고 모델 로딩 및 수집 준비 검사를 통과한 뒤 실행합니다. 기본 체크포인트 위치는 `roach/log/ckpt_11833344.pth`이며 가중치는 저장소에 포함되어 있지 않습니다.

## 관측과 평가 범위

`lrs_loop` 관측 어댑터는 동기식 30 Hz의 history 시간 계약을 요구합니다. 실제 5/10 PPM 지도를 입력으로 확인한 뒤 정책용 관측을 구성합니다. 관측 채널, ego 위치, history 순서·간격과 상태 값은 정책 학습 계약과 맞아야 합니다.

모델을 로딩할 수 있다는 사실과 CARLA 환경에서 주행이 안정적이라는 판단은 별도입니다. 체크포인트에는 학습 관측 프로파일이 완전히 기록되어 있지 않아 환경 간 이전, 충돌 회피, 차선·신호 준수와 다층 도로 입력은 실제 시나리오 평가가 필요합니다. [평가 도구](ROACH_LIVE_EVALUATION.md)는 프레임별 제어와 주행 지표를 저장합니다.
