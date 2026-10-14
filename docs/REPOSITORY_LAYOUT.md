# Repository structure / 저장소 구조

[English overview](../README.md) · [한국어 소개](../README_ko.md)

SDV Studio is a CARLA research prototype. `studio` manages the web workspace; `sdv` implements the research workflow.

## Entrypoints / 실행 진입점

| Command | Implementation | Role / 역할 |
| --- | --- | --- |
| `./start.sh` / `python -m studio` | `studio.__main__` → `studio.api` | Web workspace / 웹 작업 공간 |
| `./start-desktop.sh` / `python start.py` | `sdv.ui.app` → `sdv.ui.workflow` | Desktop interface / 데스크톱 화면 |
| `python carla_run.py` / `python -m sdv.collection.runner` | `sdv.collection.workflow` | Sensor collection / 센서 데이터 수집 |
| `python train.py --check` | `sdv.training.runner` | Queue preflight / 학습 대기열 사전 검사 |
| `python train2.py --config <file>` | `sdv.training.engine` → `sdv.training.experiment` | Configured training / 설정 기반 학습 |

Studio launches `carla_run.py` and `train2.py` as research subprocesses. These small entrypoints remain part of its runtime interface. `train_model/` preserves model import paths for external scripts and checkpoint references.

## Packages / 패키지

| Path | Responsibility / 역할 |
| --- | --- |
| `studio/static/` | Web screens, interaction and local fonts / 웹 화면·상호작용·글꼴 |
| `studio/` | APIs, configuration, datasets, jobs and results / API·설정·데이터셋·작업·결과 |
| `sdv/collection/` | CARLA collection, sensor synchronization and metadata / 수집·센서 동기화·메타데이터 |
| `sdv/bev/` | Global-map loading, coordinate transforms and BEV labels / 지도 로딩·좌표 변환·BEV 라벨 |
| `sdv/driving/`, `sdv/evaluation/` | Roach policy and driving metrics / Roach 정책·주행 지표 |
| `sdv/training/` | Models, data loaders, losses, metrics and checkpoints / 모델·로더·손실·지표·체크포인트 |
| `sdv/ui/` | Desktop UI and its sample images / 데스크톱 UI·예제 이미지 |
| `configs/` | Sensor examples and driving scenarios / 센서 예제·주행 시나리오 |
| `requirements/` | Web, research, Roach, desktop and test dependencies / 목적별 의존성 |
| `scripts/`, `tests/` | Validation tools and regression checks / 검증 도구·회귀검사 |
| `docs/` | User guides and technical reference / 사용 안내·기술 설명 |

## Local runtime files / 로컬 실행 파일

`data/`, `results/`, `.runtime/`, environments, CARLA, BEV maps and policy weights hold local assets or generated records and are excluded from Git. [Setup EN](SETUP.md) / [환경 준비 KO](SETUP_KO.md).

Desktop sample images are under `sdv/ui/assets/samples/`; they illustrate the desktop interface. Research training uses the selected dataset and recorded calibration. Source notices and bundled font licenses remain with their code/assets.
