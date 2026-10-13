# 환경과 별도 실행 자산 준비

[소개](../README_ko.md) · [English](SETUP.md) · [실제 사진 사용 안내](USER_GUIDE_KO.md)

## 웹 앱

Linux에서 Python 3.7 이상으로 저장소 안에서 실행합니다.

```bash
./start.sh
# 다른 웹 Python을 명시적으로 사용할 때:
SDV_WEB_PYTHON=/path/to/python ./start.sh
```

`http://127.0.0.1:8765`를 엽니다. 스크립트가 별도 `.venv-web` 또는 `.venv-web-pyXY`를 준비하고 `requirements/web.txt`를 설치합니다. 다른 Python으로 만들어진 기존 환경은 보존합니다. 설정에서 연구 Python을 별도로 연결하며 웹 의존성으로 기존 연구 환경을 덮어쓰지 않습니다. 새 웹 환경 설치에는 네트워크가 필요합니다.

## 연구 Python과 CARLA

검증된 기존 Python/CARLA/PyTorch 환경이 있으면 재사용합니다. 새로 준비할 때는 GPU·PyTorch/torchvision·CARLA wheel이 호환되는 별도 환경을 정한 뒤 [연구 의존성](../requirements/research.txt)을 검토합니다. 이 과거 목록은 환경 재구성을 위한 자료이며 새 PC의 한 번 설치를 검증한 절차가 아닙니다. [Roach 의존성](../requirements/roach.txt)은 별도 런타임의 목표를 설명합니다. 녹화 PC에서는 기존 Python 3.9.25 환경도 실제 검증했습니다. 선택 Python의 wheel과 서버 버전이 맞아야 합니다.

설정에 연구 Python 실행 파일과 기존 CARLA 엔진 폴더를 지정합니다. **서버/API 0.9.15**, 서로 다른 RPC/TM 포트, **Epic + Off-screen**을 유지하고 진단합니다. 엔진이 없을 때만 **설치 후 실행** 동작으로 선택 환경의 클라이언트와 엔진을 준비합니다. 엔진 약 7.81 GiB, 최초 여유 공간 45 GiB가 필요합니다. 버전이 다르거나 손상된 기존 클라이언트는 덮어쓰지 않고 보고합니다. 독립적으로 실행한 서버도 별도 연결할 수 있습니다.

## 자산 준비표

| 자산 | 경로 / 선택 | 확인 |
| --- | --- | --- |
| CARLA 엔진 | `carla/CarlaUE4.sh` 또는 설정의 기존 절대 설치 경로 | 버전·실행 권한·RPC Ready·실제 World Snapshot |
| 연구 Python | 시스템 설정의 Python 실행 파일 | 그 환경의 CARLA/PyTorch/torchvision·필수 연구 import |
| 원본 지도 패키지 | `assets/bev_maps/catalog.json`과 일치하는 `data5/`, `data10/` Town 파일 또는 프로젝트 내부 Map Root | catalog·원본 해시·offset/해상도·선택 Town 지원 |
| Roach 체크포인트 | `roach/log/ckpt_11833344.pth` 또는 프로젝트 내부의 자신의 선택 경로 | 정책 관측 계약 호환·모델 로딩 검사 |
| 학습 데이터셋 | `data/train`, `data/val` 또는 `data/` 아래 준비한 데이터셋 쌍 | 장면 분할·정렬 ID·태스크 라벨·원본 보정 |
| 보정 | 장면/데이터셋과 함께 기록한 `sensor_config.yaml` | 실제 카메라 기하와 라벨·학습 설정 일치 |
| Encoder 가중치 | `.runtime/cache/torch/hub/checkpoints/` | 학습 준비 검사에서 선택 공식 파일만 다운로드·해시 검증 |

지도는 **실제 PNG와 일치하는 catalog 메타데이터를 함께** 준비합니다. PNG 폴더만 복사하면 충분하지 않습니다. 배포 저장소에는 지도 catalog와 재배포 가능한 checkpoint를 포함하지 않습니다. 없는 지도·보정·정밀 높이·라벨을 만들지 않습니다. 자산 재배포 권한은 코드 라이선스와 별도입니다. [로컬 자산 범위](LOCAL_RUNTIME_ASSETS_KO.md), [지도 원본 감사](GLOBAL_MAP_SOURCE_AUDIT_KO.md)를 참고하세요.

데이터셋 준비는 원본 장면과 당시 보정을 바꾸지 않고 복사합니다. 생성 데이터·학습 결과·체크포인트·가상환경·`.runtime`은 Git에서 제외합니다. 소유자가 제공하는 선택적 구형 `img/test.pth` / `img/test.pt` 경로는 유지하며 학습 미리보기는 실제 VAL 샘플도 사용할 수 있습니다. 보존한 데스크톱 예제 이미지는 `legacy/fixtures/test_img/`에 있고 학습 데이터셋을 대체하지 않습니다.

## 실제 실행 전

설정을 저장하고 센서 미리보기를 멈춘 뒤 엔진 Ready와 수집 준비 검사를 확인합니다. 학습은 완전한 TRAIN·VAL·태스크·보정·공식 가중치를 선택하고 준비 검사를 통과해야 합니다. 자동 검사 통과를 실제 환경 준비와 혼동하지 않습니다. [사진 안내](USER_GUIDE_KO.md), [엔진 한계](CARLA_STABILITY_KO.md), [개발 검사](development_ko.md)를 확인하세요.

이 후보를 검토하는 동안 홈 자산·사용자 설정·실험 기록을 이동할 필요가 없습니다. 활성 앱에 적용하기 전 별도 코드로 검토합니다. `trainx2.py`가 가리키는 별도 객체 검출 실험은 제공된 스냅샷에 없어 이 배포로 재현할 수 없습니다.
