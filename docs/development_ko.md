# 개발 메모

[English](development.md) · [프로젝트 소개](../README_ko.md)

개발 당시 설정 → 데이터 수집 → 모델 학습 → 결과 확인의 기본 흐름을 구현하고 실행했던 연구 프로토타입의 소스 공개 버전입니다. 퇴사로 개발이 중단돼 확장 기능과 일부 연동이 미완성 상태로 남아 있습니다.

아래는 당시 구현 범위가 아니라 현재 공개 소스의 실행 연결·재현성 문제를 기록한 내용입니다. 공개 과정에서는 Python 문법만 검사했고 CARLA 서버, CUDA 학습, 데스크톱 UI, 장치 배포를 실행하지 않았습니다.

## 현재 공개 소스에서 확인한 실행 연결·재현성 문제

| 영역 | 확인한 내용 | 필요한 작업 |
| --- | --- | --- |
| UI 학습 실행 | `start.py`가 `python train.py`를 호출하지만 제공된 코드에 파일이 없음 | 의도한 학습 스크립트를 선택해 명시적으로 연결 |
| 학습 설정 | UI는 `data/loop/train_config_*.yaml`에 저장하고 두 학습 스크립트는 `data/train_config.yaml`을 읽음 | 공통 설정 규약을 정하고 실행할 실험 선택 |
| 학습 파라미터 | 두 학습 스크립트에 epochs·batch·optimizer 값이 하드코딩됨 | UI에서 설정한 값을 학습 코드로 전달 |
| CARLA 버전 | 기존 의존성·설치 메모는 0.9.13, UI 설치·다운로드 코드는 0.9.15 사용 | 서버와 Python API 버전을 하나로 맞추고 검증 |
| 의존성 | 의존성 파일에 Torch, Torchvision, PyYAML, Pillow, Pandas, Matplotlib, SciPy, PyQuaternion, EfficientNet-PyTorch, Torch Scatter, Webcolors 등 누락 | 환경을 재구성하고 동작 확인 |
| 저장 테스트 입력 | `train2.py`는 `img/test.pth`, 다른 학습 스크립트는 `utils.py`를 통해 `img/test.pt` 사용 | 호환 테스트 텐서를 다시 생성하고 경로 통일. 저장 텐서는 이번 공개에서 제외 |
| 데이터셋 입력 | 로더가 생성된 `data/train/`, `data/val/`, `data/sensor_config.yaml`을 요구 | 호환 데이터 생성 후 주석·보정값·분할 확인 |
| 예측·화면 표시 | 결과 화면이 생성된 로그·이미지·예측 폴더에 의존 | 학습 출력 구조와 UI 입력 구조를 비교해 검증 |

모든 오류를 나열한 목록은 아닙니다. 문법 검사로는 누락 패키지, 텐서 shape 오류, 잘못된 라벨, CARLA 호환성 문제, 학습 실패를 확인할 수 없습니다.

## 환경 가정

Linux 데스크톱, Pygame, CARLA, PyTorch/CUDA 학습을 전제로 작성된 코드입니다. 저장소 아래 `carla/CarlaUE4.sh`를 서버 실행 파일로 사용하고, UI는 `/usr/share/fonts/truetype/nanum/NanumGothic.ttf` 폰트 경로를 사용합니다.

`start.py`에는 Python·시스템 패키지 자동 설치, CARLA 다운로드·압축 해제, CARLA 프로세스 정리 코드가 들어 있습니다. 기존 환경에서 실행하기 전 이 초기화 동작을 확인하세요. 공개 과정에서 이 동작을 실행하지는 않았습니다.

기존 의존성 목록은 참고용으로 유지했습니다. 이 파일만 설치하면 동작하는 환경이 완성된다고 가정하면 안 됩니다.

## 수동 실행 진입점

버전·설정 문제를 해결한 뒤 조사할 실행 스크립트입니다.

```bash
# 데스크톱 UI
python start.py

# 데이터 생성: data/sensor_config.yaml 및 로컬 CARLA 설치 필요
python carla_run.py

# 학습 실험: 호환 설정·데이터셋·저장 테스트 입력 필요
python train2.py
python trainx2.py
```

실행 대상을 설명하는 명령이며 검증된 빠른 시작 절차는 아닙니다. 루트의 `sensor_config.yaml`은 예시이고 데이터 생성 코드가 자동으로 읽는 설정 경로와는 다릅니다.

## 공개 코드 정리

현재 UI, CARLA 수집 경로, 학습 모델·스크립트, 해당 경로에서 import하는 내비게이션 코드, 작은 UI 이미지·센서 샘플은 유지했습니다.

Python 캐시, `carla_data_gan/main (Copy).py`, `main22.py`, 사용되지 않는 `provider.py`, 연결되지 않은 `agents_map/`, 별도 미사용 CARLA 실행 스크립트, 연결되지 않은 `models/` LightFormer 실험과 로컬 경로 설정, 작업용 노트북, 오래된 설치 메모, 대용량 저장 테스트 텐서는 제외했습니다. 두 학습 스크립트는 서로 다른 실험이므로 모두 유지했습니다.

`utils.py`에서 다른 CUDA Python 프로세스를 종료하는 미사용 함수는 제거했습니다. 핵심 모델·데이터 생성 동작은 이번 공개에서 수정하지 않고 유지했습니다.

확인한 공개 대상 파일에서 하드코딩된 API 키·토큰·비밀번호·개인 키를 발견하지 못했습니다. `.gitignore`에는 일반적인 자격증명 파일, 로컬 환경, 시뮬레이터 설치, 생성 데이터, 체크포인트, 실험 결과를 제외하도록 설정했습니다. 앞으로의 커밋에도 별도 확인은 필요합니다.

## 외부 코드 표기

CARLA에서 가져온 내비게이션 모듈의 MIT 저작권·라이선스 헤더와 `carla_data_gan/map_utils.py`의 출처 표기를 유지했습니다. 프로젝트 [MIT 라이선스](../LICENSE)는 변경하지 않았습니다.
