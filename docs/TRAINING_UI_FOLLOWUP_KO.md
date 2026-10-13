# SDV 학습·내비게이션 후속 개선 결과

검증일: 2026-10-12. 대상: `wo-MS-7D42`, `/home/wo/SDV/sdv_studio`.

요청한 7개 기능을 홈 프로젝트에 적용하고 웹/API·실데이터 CUDA 학습·회귀검사를 확인했다. 최종 전체 검사: **845 passed, 2 skipped**, 113.97초. 기존 CARLA 안정화, 드라이버, 연구 Python, 데이터, 로그, 사용자 저장 설정은 보존했다. 웹은 기존 `.venv-web`, 학습은 기존 `anaconda3/envs/aa`를 사용한다.

## 변경과 검증

| 요청 | 적용 결과 | 실제 확인 |
|---|---|---|
| 공통 단계 메뉴 | 공통 `workflow()`를 모든 페이지에서 표시. Sequence/학습 Monitor에서 숨기던 CSS 제거. 수집·학습 하위 탭과 Sequence/실험 상세를 URL에 반영 | 설정·수집·Sequences·저장 재생·학습 설정·모니터·대기열·결과에서 메뉴 확인. 뒤로가기, 새로고침, 직접 진입 확인. Sequence 선택 후 주소와 선택 복원 확인 |
| 중지 후 다시 시작 | 새 ID/출력으로 **Epoch 1부터 새 학습**, 동일 작업의 완료 체크포인트로 **이어하기** 구분 | 홈에서 Paused → Resume → Stop → Checkpoint Resume → Completed. Fresh run 첫 Step의 Epoch=1. 중복 fresh 요청은 같은 대기/실행 child 반환. 중복 resume는 차단 |
| Seg 클래스 경고 | 클래스별 TRAIN/VAL 픽셀 수와 VAL-only 표시. 추천은 TRAIN 클래스 기준. 저장한 선택을 자동 수정하지 않음 | `mm`: TRAIN 1024, VAL 1998. 클래스 12(Pedestrian): 0 / 214,713 픽셀. 클래스 23(Water): 0 / 229,214 픽셀. 명시적 적용 확인창에 제외 ID를 표시. UI 초안에서 21개 TRAIN 클래스를 적용한 뒤, 저장하지 않고 새로고침하면 기존 12·23 선택이 복원됨 |
| 실제 손실 선택 | 태스크에 맞는 Torch Loss를 학습·검증 계산에 연결. 가중치/파라미터 검사, 저장·복원, immutable job config 및 checkpoint signature 사용 | 네 태스크의 대체 Loss로 4 Epoch, CE/L1 조합으로 2 Epoch, 기본 Loss로 홈 3 Epoch 실행. 양성 Anchor의 OD 회귀/분류 gradient와 ignore/invalid mask 검사 |
| 데이터셋 선택 | 기본은 실제 TRAIN/VAL 쌍 드롭다운. 이름과 경로, 표본/Scene 수, 분할 metadata, 사용 불가 사유 표시. 고급 직접 경로/센서 보정 입력 유지 | `mm`와 validation fixture 인식, 상대/절대 현재 경로 선택 복원. 중복 이름·빈 분할·VAL 누락·외부 링크 회귀검사. 실제 태스크 라벨/보정은 별도 준비 상태 검사 |
| 그래프 범례 | 공통 Chart 범례에 실제 TRAIN/VAL 출처와 X축을 항상 표시. 실험 이름/지표 label이 있어도 split 생략하지 않음 | 실제 학습/기존 실험에서 `Total loss · TRAIN · epoch`, `Total loss · VAL · epoch` 확인. 저장한 실제 값만 사용 |
| 이미지 기록·탐색 | Epoch 내부 기록 기준과 기존 누적 기준을 설정화. Epoch step/global step 저장. 실제 저장 위치만 탐색하는 Epoch/전체 Iteration 슬라이더. 지연 로딩과 PNG 용량 한도 | 새 4 Epoch run에서 TRAIN `(epoch, epoch_step, global_step)` = `(1,1,1)…(4,2,8)`. 기록 252개, PNG 약 3.4 MB. UI에서 Global 8→7의 같은 표본 이미지 확인. 기존 사용자 Epoch 5/global 640 기록도 조회; 없는 local step은 추정하지 않음 |

## 선택할 수 있는 손실

| 태스크 | 선택 목록 | 기본값과 주요 파라미터 |
|---|---|---|
| Seg | Dice+BCE, Cross Entropy, Dice+Cross Entropy | 기본 Dice+BCE. task weight=1, Dice/BCE/CE weight=1, smoothing=1, label smoothing=0. 제외 픽셀을 마스킹. CE는 class logits와 단일-class one-hot 라벨 사용 |
| Depth | SILog, L1/MAE, Smooth L1 | 기본 SILog, variance focus=0.85. Smooth L1 beta=1 m. 유효한 양수 GT와 validity mask 사용 |
| BEV | BCE with logits, Dice+BCE | 기본 BCE, positive weight=2.13, task weight=1. 서로 겹칠 수 있는 binary layer와 NaN validity mask 사용 |
| OD | Focal+Smooth L1, BCE+Smooth L1 | 기본 Focal, alpha=0.25, gamma=2, box beta=1/9. class/regression weight=1. Anchor matching/box encoding 유지 |

각 태스크의 weight는 양수이며 지원하지 않는 이름·파라미터·비유한 값은 거부한다. 예를 들어 BEV Cross Entropy는 선택/저장할 수 없다. Dice+BCE 등은 UI 이름만 바꾼 것이 아니라 실제 계산을 바꾼다. 기본 Seg/Depth/BEV 공식은 기존 구현과 수치 비교했다. CE/Smooth L1 구현은 [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/2.8/generated/torch.nn.CrossEntropyLoss.html), [SmoothL1Loss](https://docs.pytorch.org/docs/2.8/generated/torch.nn.SmoothL1Loss.html)의 계약을 따른다.

새 작업의 `config/train_config.yaml`, `output/loss_configuration.json`과 체크포인트의 config signature로 손실/파라미터를 유지한다. 이어하기는 원본 작업의 snapshot을 사용한다. 모델/손실/데이터/보정을 변경해 signature가 달라지면 재개를 거부한다. 손실 항목이 없는 과거 snapshot은 과거 기본 공식을 사용하며, 새 필드를 강제로 넣어 과거 checkpoint hash를 바꾸지 않는다.

## 기록 간격과 저장 한도

- 새 UI 초안 기본: **매 Epoch 내부 100 Step**, VAL 매 1 Epoch, 고정 표본 최대 4개, PNG 한도 1024 MiB. TRAIN 간격 0은 비활성화. 작은 데이터셋은 간격을 낮춰야 TRAIN 기록이 생긴다.
- 기존 snapshot에 기록 기준이 없으면 과거 **전체 누적 Step 기준**을 유지한다. UI에서 기록 기준을 바꿔 저장한 설정은 새 작업에 적용된다.
- 화면의 **Epoch 내부 Step / 전체 누적 Iteration** 선택은 조회 방식이다. 실제 기록 위치를 선택하고 Input/GT/Prediction을 함께 보여준다. 저장하지 않은 Epoch/Step의 이미지를 만들어 표시하지 않는다.
- 중지 후 checkpoint로 되돌아가면 완료 checkpoint 뒤의 부분 실행 이미지를 `superseded: checkpoint_rewind`로 표시해 활성 timeline에서 제외한다. 파일과 과거 기록은 보존한다.
- PNG 한도에 도달하면 현재 불완전한 새 capture만 정리하고 이후 이미지 저장을 중단한다. 기존 capture, dataset, checkpoint, 학습은 유지한다. Metadata JSON은 PNG 한도와 별도이며 API manifest는 최대 32 MB이다.
- 홈 **1 MiB** 한도 시험: 3 Epoch/global 6 완료, 실제 기록 56개, PNG 집계 780,040 bytes. 다음 capture가 남은 용량을 넘겨 새 저장을 중단했다. 화면에 한도 도달과 학습 계속을 표시했고 기존 Input/GT/Prediction 실이미지(128/200/200 px)를 확인했다.

## 재시작·이어하기 제한

- **새 학습**: Stopped/Failed/Completed/Paused 학습에서 사용. 원본 설정·경로로 새 ID와 새 output을 만들고 Epoch 1부터 시작한다. 원본 결과는 유지한다. 설정을 수정하려면 학습 설정 화면에서 별도 새 작업을 만든다.
- **Checkpoint 이어하기**: Paused/Stopped/Failed, 소유 output 안의 nonempty `resume_checkpoint.pth`, 완료 Epoch ≥1, 남은 Epoch가 있어야 한다. 완료 Epoch 다음부터 재개하므로 중단된 Epoch의 부분 gradient는 복원하지 않는다.
- Epoch 1 완료 전에 중지해 checkpoint가 없으면 새 학습만 가능하다. 대기열에서 제거한 작업은 이어하기를 차단한다. 완료한 모든 Epoch 뒤에 추가 Epoch를 붙이는 기능은 이번 범위에 포함하지 않았다.
- SQLite/프로세스 관리의 단일 실행 slot을 유지한다. Fresh 이중 클릭은 같은 pending child를 돌려주며, 이미 queued/running인 resume 재요청은 오류로 차단해 중복 worker를 만들지 않는다.

## 실데이터 실행 증거

실행 설정: 독립 검증 데이터 **TRAIN 4 / VAL 1**, 전방 카메라, 128×128, batch 2. 운영 데이터 `mm`나 사용자 학습 설정은 덮어쓰지 않았다. 작은 데이터로 학습 경로와 저장 동작을 확인한 것이며 모델 성능 검증은 아니다. Fixture의 OD에는 양성 Box가 없어 실학습은 background 분류를 확인했고, 양성 Anchor/box 회귀는 tensor 회귀검사로 별도 확인했다.

| 위치 / Job | 결과 | Epoch / Global step | 누적 실행 시간 |
|---|---|---|---|
| 별도 테스트 `29b3defce92f4ad8` | 대체 Loss, pause/stop/checkpoint 재개 완료 | 4 / 8 | 21.4초 |
| 별도 테스트 `f276aae40a0e4415` | 새 학습 Epoch 1 시작 및 이미지 기록 | 4 / 8 | 13.9초 |
| 별도 테스트 `29592fa4fe6e4746` | CE/L1, 기본 BEV/OD 완료 | 2 / 4 | 10.3초 |
| 홈 `90338ccb5f7e420c` | 기본 Loss, Epoch 1 pause → resume → stop → checkpoint resume 완료 | 3 / 6 | 15.1초 |
| 홈 `481c4859b443491a` | Fresh Epoch 1 첫 Step, 완료 | 3 / 6 | 11.0초 |
| 홈 `5f8b624a522e4c5e` | 1 MiB 제한 이후 학습 완료·기존 이미지 보존 | 3 / 6 | 10.6초 |

홈 실행 중 `nvidia-smi` 0.5초 간격 관측 최고 VRAM은 **2323 / 16303 MiB**이다. 관측 표본의 최고값이며 순간 최대 사용량을 보장하지 않는다. CARLA/다른 GPU 학습과 동시 실행하지 않았다.

최종 전체 명령:

```bash
cd /home/wo/SDV/sdv_studio
PYTHONPATH= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/home/wo/Documents/Codex/2026-10-12/task/test_env/bin/python -m pytest -q tests
```

결과 **845 passed, 2 skipped, 179 warnings in 113.97s**. 제외 2건은 기존 optional Legacy encoder 참조 검사이다. 경고는 기존 Torch/NumPy API deprecation 등이 포함된다. Python compile과 수정 JS module syntax 검사도 통과했다. 홈 적용 때 예측 기록 모듈 한 파일을 빠뜨려 첫 홈 검사 2건이 실패했고, 누락을 보완한 뒤 최신 전체 검사를 통과했다. 실패 로그도 진단 폴더에 보존했다.

## 파일·보존·남은 범위

주요 변경: `studio/config.py`, `api.py`, `jobs.py`, `dataset_choices.py`, `task_catalog.py`, `predictions.py`; `static/app.js`, `style.css`, `pipeline-ui.js`, `training-enhancements.js`, `index.html`; `sdv/config/validation.py`, `sdv/training/{experiment,predictions,loss_config,losses,preview_config}.py`, `modeling/loss.py`; 새 Loss/후속 회귀검사.

원본 백업: `.runtime/backups/training_followup_20261012_074001/`. 변경 manifest·환경/작업 JSON·GPU 표본·검사 로그·UI 검증 기록: `docs/validation/training_followup_20261012/`. 사용자 `settings.json`, `training.json`, `sensors.json`은 적용 전 SHA-256과 동일하다. 기존 데이터/로그/결과를 삭제하지 않았다. 테스트 결과는 별도 `검증` 이름으로 남겼다.

CARLA의 **Epic + Off-screen** 운영 설정과 이전 **613초·2524샘플, Town05 500/500** 검증은 그대로다. 이번 변경은 학습/UI 계층이며 CARLA 엔진을 추가 패치하거나 장기 수집을 재시험한 작업이 아니다. Low 품질의 UE4 Landscape/Texture crash 내부 결함은 [기존 안정화 보고서](CARLA_STABILITY_KO.md)의 제한을 유지한다. 드라이버/커널/ML 환경 변경, git 원격 작업은 수행하지 않았다.
