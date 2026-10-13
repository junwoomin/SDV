# Repository map and compatibility / 저장소 구조와 호환성

[English overview](../README.md) · [한국어 소개](../README_ko.md)

Base: `junwoomin/SDV/main`, `9fb7de2197e91575de818f15f7961ce3f4b05f75` (new ZIP snapshot). All 428 tracked files matched the actual home source before this cleanup. Work is performed in a separate clone; home settings, generated data, maps and running-engine profiles are preserved.

## Current execution paths / 현재 실행 경로

| Entry | Implementation | Role / 역할 |
| --- | --- | --- |
| `./start.sh` / `python -m studio` | `studio/__main__.py`, `studio/api.py` | Main web console / 주 웹 앱 |
| `./start-desktop.sh` / `python start.py` | `sdv.ui.app` → `sdv.ui.workflow` | Existing desktop UI / 기존 데스크톱 |
| `python carla_run.py` / `python -m sdv.collection.runner` | `sdv.collection.workflow` → `sdv.collection.carla_engine` | Actual collector / 실제 수집 |
| `python train.py` | `sdv.training.runner` → `train2.py` | Existing UI queue bridge / 기존 UI 대기열 연결 |
| `python train2.py --config …` | `sdv.training.engine` → `sdv.training.experiment` | Current trainer / 현재 학습 |
| `python trainx2.py` | Target `experiments.training.detection` absent in supplied main | Preserved unresolved historical command / 누락된 과거 실험 명령 |

Studio jobs retain the exact root `carla_run.py` and `train2.py` subprocess interfaces. Existing configuration/data/output paths are unchanged. Root wrappers are intentionally small; implementation belongs in `sdv`/`studio`. Old helper/import names remain available, including `train_model.*` for checkpoint references.

## Moves and reasons / 이동과 근거

| Before | Now | Reason / 근거 |
| --- | --- | --- |
| `config/examples/` | `configs/examples/` | One settings tree alongside evaluation scenarios / 설정 경로 통합 |
| `requirements-{web,dev,roach,ui}.txt` contents | `requirements/{web,dev,roach,ui}.txt` | Dependency groups together; root files are pip compatibility includes / 목적별 통합·기존 pip 명령 유지 |
| `carla_data_gan/` implementation | `legacy/carla_data_gan/` | Active imports already use `sdv.collection.carla_engine`; namespace shim retained / 실제 import 근거로 구형 보존 |
| `train_model/*.py` except namespace initializer | `legacy/training/modeling/` | Initializer already redirected imports to current `sdv/training/modeling`; duplicate source preserved / 이미 우회된 중복 소스 보존 |
| `utils.py` implementation | `legacy/utils.py` | Mixed historical helpers separated; root module aliases it / 구형 보조 코드·import 유지 |
| `img/car_top.jpg`, `img/car_side.jpg` | `sdv/ui/assets/` | Desktop UI assets next to their consumer; both paths updated / 실제 사용 코드 옆 배치 |
| `img/test_img/` | `legacy/fixtures/test_img/` | Small original desktop examples retained; UI references updated / 기존 예제와 모든 참조 보존 |
| `img/vehicle_class_json_file.txt` | `configs/examples/vehicle_classes.txt` | Unimported label-format example / 라벨 형식 예제 |
| Root `PROGRESS.md` | `docs/legacy/PROGRESS_root.md` | Historical progress, existing legacy progress kept separately / 기존 진행 문서 모두 보존 |
| Previous root READMEs | `docs/legacy/README_before_cleanup_{EN,KO}.md` | Full previous text preserved / 이전 설명 전체 보존 |
| Original `docs/development*.md`, `docs/README_ko.md` | `docs/legacy/*original*.md` | Historical desktop/publication assumptions separated from current guides / 과거와 현재 안내 구분 |

`docs/`, `scripts/`, `tests/`, `sdv/`, `studio/`, `configs/`, `requirements/` remain top-level because they have distinct active roles. We do not collapse tested packages into arbitrary folders. `sensor_config.yaml` stays as the old root example/compatibility file; runtime uses recorded `data/sensor_config.yaml` or a job snapshot. The identical example also lives under `configs/examples/`.

## Runtime and assets / 실행 상태와 자산

`.runtime/`, `data/`, `results/`, CARLA, map PNG/catalog, Roach weights and environments are local runtime content. `.gitignore` protects these from accidental staging. [Setup EN](SETUP.md) / [설치 KO](SETUP_KO.md). Static UI fonts remain in `studio/static/fonts/` with their original license. No upstream notices, legacy implementation or small fixture images were discarded.

Prior validation reports contain historical absolute paths and local-only evidence references. They describe the original validation location, not the new clone location. [Current documentation index](README_ko.md) distinguishes current guides from preserved historical documents.
