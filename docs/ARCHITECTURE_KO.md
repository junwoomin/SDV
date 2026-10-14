# SDV Studio 구현 구조

[문서 안내](README_ko.md) · [저장소 구조](REPOSITORY_LAYOUT.md)

SDV Studio는 웹 관리 환경과 연구 실행 환경을 분리한 자율주행 연구용 프로토타입입니다.

## 웹 작업 공간

`studio/api.py`는 설정, 데이터셋, 수집, 학습, 예측 이미지와 실험 결과 API를 제공합니다. `studio/static/`은 한국어·영어 화면과 상호작용을 구현합니다. `studio/config.py`는 설정을, `studio/jobs.py`와 `studio/worker.py`는 작업 상태와 실행을 관리합니다.

웹 환경은 `.venv-web`에 설치되며 CARLA·PyTorch를 직접 실행하는 연구 Python은 시스템 설정에서 별도로 선택합니다. 수집과 학습 작업에는 설정·데이터 경로·보정 정보의 사본을 전달합니다.

## 연구 실행

| 영역 | 구현 | 역할 |
| --- | --- | --- |
| 수집 | `sdv/collection/` | CARLA 센서 출력, 프레임 정렬, 장면 저장과 메타데이터 |
| BEV | `sdv/bev/` | 지도 catalog 확인, ego 기준 변환, 정적·동적 라벨과 유효성 |
| 주행 | `sdv/driving/` | Roach 입력 구성, 정책 로딩과 제어 변환 |
| 학습 | `sdv/training/` | 모델 구성, 태스크별 데이터 처리, 손실·지표·체크포인트 |
| 실행 관리 | `sdv/runtime/` | 프로세스, 소유 Actor, 포트와 CARLA 설치 관리 |
| 데스크톱 | `sdv/ui/` | 별도 데스크톱 인터페이스와 예제 이미지 |

## 데이터와 결과

수집은 장면별 센서 파일, `sensor_config.yaml`, `collection_manifest.json`, 프레임별 정보를 기록합니다. 데이터셋 준비는 완전한 장면을 TRAIN·VAL로 나누고 원본과 보정을 복사합니다. 학습은 선택한 데이터셋과 태스크 계약을 사용하며 결과 디렉터리에 설정, 지표, 체크포인트와 예측 이력을 저장합니다.

준비 검사는 선택 환경의 의존성, 데이터, 가중치와 자산을 확인합니다. 실제 CARLA 동작, 주행 안정성, 모델 정확도와 BEV 기하 정합은 해당 환경의 별도 실험으로 평가해야 합니다.
