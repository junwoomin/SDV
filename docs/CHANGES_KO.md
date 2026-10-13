# 변경 파일과 연결 결과

## 이번 수정: 실제 시퀀스·라벨·태스크 파이프라인

- `sdv/data_contract.py`: 수집·학습·웹 공통 CARLA ID/팔레트·Depth 단위·OD·BEV 계약, 고급 설정 검증.
- `sdv/training/modeling/{data,model,loss}.py`, `experiment.py`, `task_metrics.py`, `metric_registry.py`, `predictions.py`: 실제 클래스 선택과 Head/Loss/Metric/Palette 연결, 정수 라벨 Nearest Resize, Depth 미터·Invalid 처리, Frame basename 정렬, 선택 BEV 레이어, 실험별 Metric 이름/상한.
- `checkpointing.py`: 데이터 처리 계약 버전이 달라진 과거 체크포인트의 무조건 재개 방지. 이전 출력 Head/state_dict 구조 보존과 과거 전체 실험 재현 가능 여부를 구분.
- `studio/sequences.py`, `dataset_quality.py`, `planning.py`, `task_catalog.py`: 시퀀스별 실제 Frame API·카메라/Box/BEV 렌더링·Metadata·실제 클래스 검사·전체 품질 감사·오프라인 미래 관측 Target JSON.
- `studio/{api,catalog,config,jobs,predictions,training_options,dataset_tools}.py`, `sdv/config/validation.py`: 고급 설정 API와 불변 실험 사본, per-job 지표 사전, Step 필터, 저장 완료 Frame의 라이브 정보, 재귀 수집 경로와 신규 Raw 포맷 검증.
- `sdv/collection/{metadata,workflow,encoding}.py`, `carla_engine/main.py`: Raw Seg ID PNG, float32 미터 Depth, 정확한 카메라별 OD 투영, 센서 native Frame 동기화, 실제 Ego/Road/Route/Traffic 정보와 Calibration 출처, 저장 전후 world Frame 검사.
- `studio/static/{app,pipeline-ui,i18n}.js`, `style.css`, `index.html`: 가로 Epoch/Step/Frame Timeline, 동기화 시퀀스 이미지와 Scene 패널, 라이브/저장 재생 구분, 실제 Box/레이어 토글, 확대, 기본 접힌 태스크 설정과 세부 지표, 3자리/원값 Tooltip/Current·Best·Maximum 표, JSON 내보내기.
- `studio/http_protocol.py`, `__main__.py`: Python 3.7용 Uvicorn 0.22의 pipelined SSE 활성 응답에 이전 idle 타이머가 적용되는 문제 수정. 실제 오류 재현·정상 종료 검사 포함.
- `requirements-web.txt`, `start.sh`: 실제 이미지/Annotation 분석용 NumPy의 Python 버전별 웹 가상환경 설치·검사. 연구 환경은 변경하지 않음.

신규 API: `POST /api/training/task-catalog`, `GET /api/datasets/{id}/sequence`, `/frames/{index}`, `/frame`, `/quality`, `POST /api/datasets/{id}/planning`, `GET /api/jobs/{id}/scene`. 예측 API는 `global_step` 필터와 실제 `global_steps` 목록을 추가했습니다.

[기능 사용·제한](PIPELINE_KO.md) · [동일 샘플 정량 감사](DATASET_AUDIT_KO.md) · [수집 스키마](COLLECTION_SCHEMA_V2_KO.md). 정확한 추가·수정 파일과 SHA256은 `pipeline-change-manifest.json`에 기록합니다.

## 이전 수정: 학습 예측·태스크 지표·대기열

- `sdv/training/experiment.py`, 새 `checkpointing.py`: Epoch별 원자적 전체 재개 상태, 완료 Epoch 경계 일시정지, 모델·옵티마이저·StepLR·RNG·지표 복원과 데이터/설정 무결성 검사. 기존 손실과 모델 알고리즘 유지. 실제 검사에서 발견한 Depth 유효 마스크의 float 변환을 bool 유지로 수정.
- 새 `sdv/training/predictions.py`, `studio/predictions.py`: 데이터 내용으로 고정한 VAL 최대 4개, 설정 Step 주기의 고정 Train 추론, 실제 입력/GT/예측 PNG와 불변 Epoch·Step 이력. 실제 이미지 목록에 등록된 프로젝트 내부 파일만 제공. 이전 이미지 기록 호환.
- 새 `task_metrics.py`, `metric_registry.py`: 실제 전체 Epoch의 Seg/BEV/Depth/Detection 통계와 지표 방향·단위. 원본 Anchor–GT IoU와 예측 AP를 구분.
- `studio/jobs.py`, `api.py`, `config.py`, `sdv/config/validation.py`: 영속 학습 대기열·순서·현재 상태·준비 조건·일시정지/재개·대기열 삭제, 실행 중 다음 설정 저장, 불변 사본, 실측 ETA와 전체 이력 Best. 장시간 준비 검사와 실제 실행 예약을 분리해 다음 모델 추가를 막지 않음.
- `studio/catalog.py`: 저장한 사용자 실험명·옵티마이저·백본·디코더·LR·Batch·태스크, 그래프 축소 이전 전체 Epoch Best, 예측 조회 API.
- `studio/static/app.js`, `style.css`, `i18n.js`, `index.html`: 설정/대기열/모니터링, 모델별 실제 예측 갤러리와 동일 VAL 이미지 비교, Best 표, 약 1.5배 그래프, 기본 접힌 로그, ETA, `결과 확인` 문구와 모델 구분 정보.
- 새 테스트: 실제 subprocess 대기열·종료 기록·복구, 재개 후 optimizer/RNG 연속성, 실제 CPU 태스크 통계·SDV 모델 예측 PNG, 안전한 이미지 HTTP 조회·인증·페이징·설정 사본·실험 Best. 브라우저와 버전별 검사 결과는 최신 검증 문서에 기록.

API 연결: `POST /api/training/queue`, `POST /api/training/queue/reorder`, `POST /api/jobs/{id}/pause`, `/resume`, `DELETE /api/jobs/{id}`. 이미지 이력은 `/api/jobs/{id}/predictions` 및 `/api/experiments/{id}/predictions`, 실제 이미지 파일은 각 `/predictions/frame`. 기존 직접 실행 `/api/jobs`와 중지·결과 다운로드는 유지합니다.

## 이전 수정: 시작 정리·수집 진행·접기·모델 확장

- `start.sh`, `studio/stop_carla.py`: 의존성 설치 전 현재 사용자 CARLA 전체 자동 종료. 정확한 실행파일/직접 실행 shell script, UID·PID namespace·시작 시각 확인 후 TERM→KILL. 다른 사용자의 프로세스와 무관한 프로그램은 포함하지 않음.
- `sdv/collection/progress.py`, `workflow.py`, `studio/jobs.py`: 실제 파일 저장 성공 뒤 전 맵 누적 수 증가, 타운별/전체 진행률 이벤트. 종료 코드만으로 목표 수를 임의로 채우던 처리를 수집에서 제거.
- `sensor_preview_service.py`, `sensor_preview_worker.py`, `sensor-preview.js`: 실행 중 Apply에도 저장한 FPS 전달. 센서 sensor_tick 재설정, 목표와 현재 적용값·실측값 구분. 오래된 2 FPS snapshot에서 벗어나는 실제 경로 수정.
- `app.js`, `style.css`, `i18n.js`, `index.html`, `studio/api.py`: 준비 패널 접기·펼치기와 수동 상태 보존, 큰 실제 수집 진행 카드, 센서→태스크→모델→파라미터 순서, registry 기반 선택·환경 갱신, 한국어/영어, 버전별 asset URL 및 JS/CSS no-store.
- `sdv/training/model_registry.py`, `encoder.py`, `decoder.py`, `model.py`: 흔히 쓰는 백본과 UNet 선택 추가, 공통 C5/C4/C3 contract, ResNet/FPN unpack와 Bi-FPN fusion 오류 수정. 기존 encoder state keys·손실·BEV geometry 보존. 모델 장치를 따라 입력 이동.
- `studio/config.py`, `sdv/config/validation.py`, `studio/pretrained_service.py`: 동일한 registry로 설정·공식 가중치 준비 연결, ResNet alias 잠금 공유, 객체 탐지 128배수 크기 검사, 기본 사전학습 유지.
- 새 검증 파일: 모델 catalog/registry/실제 CPU contract, CARLA 종료, 실제 collection commit 테스트. 전체 회귀·실제 모델·브라우저 JSON과 스크린샷 포함.

지원 목록·확장 방법과 검증 한계는 [모델 문서](MODELS_KO.md), 최종 검사 기록은 [검증 문서](VALIDATION_KO.md)에 있습니다. 아래는 기존 기능의 구현 내역이며, 이전 검증 당시의 미검증 표기는 최신 문서의 범위와 구분합니다.

## 웹 CARLA 설치·실행 연결

- 시뮬레이션 최상단에 **설치 후 실행 / 설치만 / 중지 / 로그**를 연결했습니다. 설치 후 실행은 연구 Python의 CARLA 클라이언트를 먼저 준비·검증한 뒤 엔진 다운로드·실행·RPC 자동 연결을 진행합니다. 설치만은 Python 패키지 없이 엔진만 내려받습니다.
- `studio/carla_service.py`, `studio/carla_worker.py`: 브라우저와 독립적인 상태·로그·취소·소유 엔진 관리. 수집·센서 미리보기 정리 전 엔진 종료를 막고 외부 서버는 종료하지 않습니다.
- `sdv/runtime/install.py`: 프로젝트 `carla/` 고정 설치, 8,386,636,048 bytes(약 7.81 GiB) 배포 이어받기, 최초 45 GiB 및 부분 다운로드 반영 공간 검사, 주 주소 403 등의 실패 시 확인된 미러 전환, Python 3.7 호환 압축 경로·링크 검사.
- `studio/carla_worker.py`: 선택한 연구 Python에 클라이언트가 없는 경우 `carla==0.9.15`만 binary wheel·의존성 미설치 옵션으로 추가하고 같은 Python에서 import·버전을 검증합니다. 기존 다른 버전·손상된 carla는 덮어쓰지 않습니다. 오래되었거나 없는 pip는 캐시한 pip 24.0 wheel 실행기를 사용하며 연구 pip는 업그레이드하지 않습니다.
- 기존 `aa` 연구 환경의 CUDA/PyTorch/NumPy와 연구 알고리즘은 변경하지 않습니다.
- 전체 8 GB 다운로드와 실제 GPU 엔진 기동은 아직 미검증입니다. 테스트 세부 결과는 검증 문서에 구분합니다.

## 센서 배치·실제 미리보기와 언어 전환 복원

- 센서 설정의 숫자 입력과 상단·측면 배치도 드래그를 연결했습니다. 점은 센서의 대략적인 위치를 표시하며 정확한 값은 숫자로 확인합니다. 초안을 **프리뷰에 적용**하는 동작과 수집용 **설정 저장**을 분리해 현재 확인 중인 값과 저장된 값을 구분합니다.
- `studio/sensor_preview_service.py`, `studio/sensor_preview_worker.py`: 선택한 연구 Python의 별도 프로세스로 실제 CARLA RGB·LiDAR를 제공합니다. 이전 외부 차량 overview 화면과 추가 카메라는 제거했습니다. 최신 JPEG만 유지하며 요청 속도 기본값은 **24 FPS**, 입력 범위는 **정수 1~60 FPS**입니다. 실제 수신·표시 속도는 시뮬레이션·GPU·인코딩·전송 성능에 제한됩니다. RGB는 긴 변 최대 **640 px**이며 수집용 해상도는 변경하지 않습니다.
- 미리보기는 현재 world/map을 사용하고 맵 로드나 world 설정 변경을 하지 않습니다. 앱 소유 동기 서버만 직접 tick하며 외부 동기 서버는 외부 ticker가 필요합니다. 생성한 센서와 차량만 정리하고, 활성 미리보기 중 수집·학습 시작, 엔진 종료, 연결 해제와 서버 주소·포트·연구 Python 변경을 막습니다.
- `sdv/collection/sensor_attributes.py`: 기존 LiDAR 생성자가 저장 설정을 무시하던 문제를 수정했습니다. 같은 blueprint 설정 함수를 데스크톱·웹 미리보기와 수집의 일반/semantic LiDAR에 사용하여 확인한 센서 속성이 수집에도 반영되도록 했습니다. 누락 필드는 기존 기본값을 유지합니다. 센서 좌표 변환이나 연구 알고리즘 변경은 없습니다.
- `studio/static/i18n.js`, `app.js`: 상단 KO/EN 선택을 `localStorage`에 보존하고 전환 중 폼 초안을 복원합니다. 사용자 데이터·경로·원시 로그는 원문으로 표시합니다. 새로고침·화면 이동·언어 전환은 실행 중인 서버 작업을 중지하지 않습니다.
- 수집 화면의 카메라/BEV는 여전히 **저장 프레임**을 읽습니다. 센서 설정의 실제 미리보기와 별도 기능이며 수집 없이 실행할 수 있습니다.
- 실제 CARLA의 센서 부착·영상·LiDAR 수신·정리와 GPU 실행은 아직 미검증입니다. 검사 항목과 실제 수행 결과는 [검증 문서](VALIDATION_KO.md)에 기록합니다.

## 실제 데이터에 따른 학습 선택과 가중치 자동 준비

- `studio/training_options.py`: train/val 양쪽 모든 장면의 센서·선택 카메라별 출력 라벨 파일, 크기와 front 기준 frame ID를 확인합니다. 일부 장면만 충족하는 데이터는 선택 가능으로 표시하지 않습니다. 센서 선택·데이터 경로 초안에 따라 KO/EN 사유를 반환하며 설정은 변경하지 않습니다. 기존 전체 학습 준비 검사는 그대로 유지합니다.
- `studio/static/app.js`, `style.css`: 데이터가 없는 입력 센서·태스크를 회색으로 비활성화하고, 필수 조건 미충족을 빨간색으로 안내합니다. BEV를 해제하면 BEV 디코더 선택은 None으로 바뀝니다. `studio/config.py`와 연구 설정 검사도 이를 일치시켜 기존 모델의 비활성 forward 분기를 사용합니다. BEV의 카메라 3개 이상 조건은 유지합니다.
- `studio/pretrained_service.py`: 사용자가 학습 준비 상태 검사를 요청하면 공통 모델 registry에 등록된 선택 인코더의 공식 사전학습 파일을 백그라운드에서 준비합니다. 공식 파일명 SHA256 접두사를 검증한 뒤 캐시에 원자적으로 공개하고, 검증된 캐시는 재사용합니다. 중복 다운로드를 막고 검증 실패한 기존 파일은 보존합니다. 일반 상태 조회로 다운로드를 시작하지 않습니다.
- 이전의 수동 가중치 준비 안내를 현재 자동 준비 흐름으로 교체했습니다. 연구 CUDA/PyTorch 패키지, 모델 레이어·손실식·데이터 형식은 이 변경으로 수정하지 않았습니다.
- 실제 공식 파일 두 종류의 다운로드·SHA256 검증과 Python 3.7 네트워크 차단 상태의 캐시 재사용은 [가중치 검증 기록](pretrained-validation.json)으로 확인했습니다. 모델 역직렬화와 CUDA 학습은 미검증이며 QA 가중치는 배포물에 포함하지 않습니다.

## Python 3.7–3.10 호환 수정

- `start.sh`: 최소 버전 3.7, 활성 Python 우선 선택, 기존 웹 가상환경 보존, 버전별 환경 선택, 의존성 파일 변경 감지와 자동 설치.
- `requirements-web.txt`, `requirements-dev.txt`: Python 버전에 맞는 FastAPI/Uvicorn·이미지·테스트 의존성 지정. 웹 환경에만 적용하며 연구 CUDA/PyTorch/NumPy는 변경하지 않음.
- `studio/compat.py`, `studio/api.py`: Python 3.7에서 비동기 SSE·서버 종료가 동작하도록 컨텍스트를 보존하는 스레드 실행 헬퍼 적용.
- `studio/catalog.py`: 최신 문자열·경로 API를 3.7 호환 구현으로 교체. 경로 검증과 임시 파일 정리 유지.
- `tests/test_python_compat.py`, `tests/test_research_python_compat.py`, `tests/test_research_bridge.py`: 버전별 동작·전체 연구 소스 구문 검사와 테스트 자체의 호환성 수정.

이번 호환 수정에서 모델·손실 함수·데이터 형식·CARLA 연구 로직은 변경하지 않았습니다.

## 주요 추가 파일

| 파일 | 역할 |
|---|---|
| `studio/api.py` | FastAPI 조회·저장·작업 API, SSE, CARLA probe, 환경 검사 |
| `studio/config.py` | 검증된 시스템/센서/학습 설정, 기존 설정 가져오기 |
| `studio/jobs.py`, `studio/worker.py` | SQLite·프로세스 감독·실제 종료 기록·복구 |
| `studio/carla_service.py`, `studio/carla_worker.py` | CARLA 설치 작업·RPC 준비·소유 엔진 감독 |
| `studio/sensor_preview_service.py`, `studio/sensor_preview_worker.py` | 실제 센서 미리보기 프로세스·명령·상태·프레임·소유 액터 정리 |
| `studio/catalog.py` | 기존/신규 결과 파일·metric·checkpoint·이미지·궤적 읽기 |
| `studio/dataset_tools.py` | 장면 분할, 원본 복사, 라벨·프레임·보정 정합 검사 |
| `studio/training_options.py` | 실제 파일·ID 정합성에 따른 학습 입력·태스크 선택 가능 상태 |
| `studio/pretrained_service.py` | 선택 인코더의 공식 사전학습 파일 다운로드·검증·캐시 재사용 |
| `studio/static/index.html`, `app.js`, `style.css` | 5개 웹 페이지·폼·제어·실제 그래프 |
| `studio/static/sensor-preview.js`, `studio/static/sensor-preview.css` | 센서 배치도·드래그·미리보기 적용과 실제 프레임 표시 |
| `studio/static/i18n.js` | KO/EN UI 문구·언어 선택 보존 |
| `studio/static/fonts/` | 오프라인 한글 폰트와 OFL 라이선스 |
| `sdv/telemetry.py` | 실제 연구 작업 JSONL 상태/metric 기록 |
| `sdv/collection/session.py` | 외부 CARLA 연결과 소유 액터 정리 |
| `sdv/collection/sensor_attributes.py` | 미리보기·수집에 공통인 카메라/LiDAR blueprint 속성과 입력 검증 |
| `start.sh`, `requirements-web.txt`, `requirements-dev.txt` | 설치·실행·CPU 검증 환경 |
| `tests/` | API·프로세스·카탈로그·연구 경계 회귀 검사 |

## 기능별 실제 연결

| 웹 동작 | 백엔드 및 실제 대상 |
|---|---|
| 시스템 설정 저장/가져오기 | `/api/settings` → 검증 → `.runtime/studio/settings.json` |
| 센서/학습 설정 저장 | `/api/config/sensors`, `/api/config/training` → JSON 저장 |
| 입력 센서·태스크 선택 가능 여부 | `POST /api/training/options` → 선택 센서·경로 초안의 전체 train/val 파일 목록 검사; 설정은 저장하지 않음 |
| 학습 준비 상태 검사·공식 가중치 준비 | `POST /api/preflight`의 `kind=training`, `prepare_weights=true` → 선택 인코더 가중치 백그라운드 다운로드·검증; 상태는 bootstrap/SSE·준비 검사 응답에 반영 |
| 센서 미리보기 시작·초안 적용·중지 | `POST /api/sensor-preview/start`, `PUT /api/sensor-preview`, `POST /api/sensor-preview/stop` → 연구 Python의 실제 CARLA 미리보기; 저장 센서 설정은 별도 유지 |
| 센서 미리보기 상태·로그·프레임 | `/api/sensor-preview/status`, `/api/sensor-preview/logs`, `/api/sensor-preview/frame` → 실제 worker 상태·로그·채널별 최신 JPEG |
| CARLA 설치 후 실행 / 설치만 | `/api/carla/local/start` → 실행 선택 시 연구 Python의 carla 설치·검증 → 프로젝트 `carla/` 준비 → RPC 자동 연결; 설치만은 엔진 다운로드만 수행 |
| CARLA 상태·중지·로그 | `/api/carla/local`, `/api/carla/local/stop`, `/api/carla/local/logs` → 영속 상태·해당 앱 소유 엔진·실행 로그 |
| 외부 CARLA 연결/해제 | `/api/carla/connect`, `/api/carla/disconnect` → 선택 연구 Python의 CARLA RPC 검사; 외부 엔진은 유지 |
| 수집 시작 | `/api/jobs` → 불변 설정 사본 → 실제 `carla_run.py` |
| 학습 시작 | `/api/jobs` → 실제 `train2.py --config ...` |
| 작업 중지 | `/api/jobs/{id}/stop` → 해당 process group 정상 종료 요청 → 기한 후 정리 |
| 진행률/metric/log | 실제 `events.jsonl`·프로세스 로그 → SSE/조회 API |
| 수집 영상·BEV 미리보기 | `/api/jobs/{id}/preview`, `/api/datasets/{id}/preview` → 저장 프레임 |
| 데이터셋 준비 | `/api/datasets/prepare` → 선택 장면 검사·복사·설정 연결 |
| 결과 비교 | `/api/experiments/*` → 실제 CSV/JSON/레거시 log 및 설정 |
| 다운로드/내보내기 | 개별 artifact 또는 설정/수치/로그 ZIP |

## 연구 코드 수정 이유

- `sdv/runtime/install.py`: 웹 설치·취소·이어받기·공간 검사·미러 전환·Python 3.7 압축 해제 호환.
- `sdv/config/validation.py`: 경로 주입, 실제 데이터 입력 검사, 선택적 미리보기 텐서, 지원 checkpoint 옵션 검증.
- `sdv/collection/workflow.py`: 외부 CARLA 실행 경로, 실제 진행 이벤트, 장면별 보정 스냅샷, 센서별 저장 폴더/선행조건.
- `sdv/collection/carla_engine/main.py`: 외부 서버의 OpenDrive 사용, 지정 차량 적용, 공유 blueprint 함수로 저장 센서 속성 적용. Pygame rasterization 유지.
- `sdv/collection/preview.py`: 공유 blueprint 함수로 실제 수집과 동일한 센서 속성을 적용. 기존 데스크톱 진입점은 유지.
- `sdv/training/modeling/data.py`: 데이터 경로·보정 파일 지정, 원본 이진 마스크 호환. 기존 팔레트와 crop 규약 유지.
- `sdv/training/modeling/model.py`: 계산된 BEV pooled tensor의 미정의 변수 연결 수정, 사용하지 않는 PointPillar import 지연.
- `sdv/training/modeling/decoder.py`: 미학습 EfficientNet 생성자의 잘못된 호출 수정.
- `sdv/training/experiment.py`: 선택 입력/라벨 배치 조립, 실제 aggregate metric, CSV 길이/OD 집계 오류, 원자적 checkpoint, 선택 저장 옵션.

원본 모델의 레이어 구성·손실 수식·주행 내비게이션 알고리즘을 새 알고리즘으로 대체하지 않았습니다. 원본에서 확인한 실행/입력/보고 오류를 고쳤으며, 하드웨어 수치 동등성은 아직 검증하지 못했습니다.

기존 Pygame 진입점은 `start-desktop.sh`로 보존합니다. 기존 `sdv_web/`는 수정 없이 보관하며 새 앱에서 제공하지 않습니다. 이전 README와 진행 기록은 `docs/legacy/`에 있습니다.

## 원본 대비 수정한 파일 전체

- `PROGRESS.md`
- `README.md`
- `README_ko.md`
- `requirements-ui.txt`
- `requirements.txt`
- `sdv/collection/carla_engine/main.py`
- `sdv/collection/preview.py`
- `sdv/collection/workflow.py`
- `sdv/config/validation.py`
- `sdv/runtime/install.py`
- `sdv/training/experiment.py`
- `sdv/training/modeling/data.py`
- `sdv/training/modeling/decoder.py`
- `sdv/training/modeling/model.py`
- `start.sh`

추가 파일의 전체 목록은 `change-manifest.json`에 있습니다. 가상환경·생성 데이터·실행 DB·Python cache는 배포 ZIP에서 제외합니다.
