# 검증 결과 · 2026-10-08

## 시퀀스·클래스·고급 설정·수집 Metadata — 최신 검증

| 검사 | 실제 결과 |
|---|---|
| Python 3.7 / 3.8 / 3.9 전체 웹·프로세스 회귀 | 각각 **436 passed, 20 skipped**, 오류 0. Skip은 연구 Torch/OpenCV가 없는 웹 환경의 검사 |
| Python 3.10 전체 웹·프로세스 회귀 | **435 passed, 21 skipped**, 오류 0. 추가 Skip 1개는 Uvicorn 0.22 전용 과거 오류 재현이며 최신 Uvicorn에는 해당하지 않음 |
| 최종 API 보완 검사 | 전방/좌측 카메라만 있는 수집 호환 보완 후 Python3.7–3.9 각각40통과,3.10 39통과/과거 오류 재현1skip. 실제 클래스명 차트·비교 브라우저 추가검사 통과 |
| CPU 연구·수집 계약 | 실제 torch2.2.2+cpu/torchvision0.17.2+cpu에서 **158 passed**, Skip/실패 0. 데이터 계약, 동적 Head/Loss/Metric, 모든 모델 계약, Metadata/Encoding 포함 |
| 데이터 처리 버전 재개 보호 | 추가 집중 **13 passed**. 과거·다른 Pipeline Schema 체크포인트를 상태 변경 전에 거부. 새 버전의 pause/resume 검증 재실행 |
| 실제 Seg 감사 | 200 Frame/24,000,000 픽셀, 실제 19클래스. 수정된 Loader 전 Frame 결과가 정수 원본 ID Nearest Resize와 일치. 팔레트 밖 원본 픽셀 0 |
| 실제 API 렌더링 | 200 Frame × RGB/Seg/Depth/OD/BEV = **1,000개 이미지**. RGB 및 빈 OD Overlay는 원본 RGB와 정확히 일치. 속도/위치/Yaw/조향을 같은 Frame info와 비교 |
| 실제 학습 엔진 CPU | 제공 자료의 명시적 소수 Frame으로 19클래스 Seg·Depth 1–80m·빈 OD·DAS/Lane 실제 2 Epoch. CUDA 이동/guard만 시험 내에서 대체. 연속 실행과 Epoch1 pause→resume의 모델·Optimizer·Scheduler·지표·고정4 GT 일치, 실행별120예측 유지 |
| BEV 카메라 수 | 세 디코더 × 1/2카메라 = 6개 실제 forward/backward 통과. 카메라 축/기하 보정의 정확성을 증명하는 검사는 아님 |
| Planning 실제 Target | 한 실제 시퀀스에서199개 관측/3,790개 미래 Target. 이후 원본 위치/제어와 전부 대조. Timestamp 없는 자료의 offset_s=null 유지 |
| Chromium 화면 | KO/EN ×1920/1366/390 ×시퀀스/고급 설정/모니터 =18화면. JS·서버 오류0, 가로 넘침0, 모든 이미지 영역 내 유지 |
| Timeline·예측·지표 | 실제 Frame 클릭/드래그/화살표/재생/속도/빠른 탐색 동기화, 실제 CPU Epoch1/2/4 PNG·고정4·없는Epoch3 빈 상태, Current/Best/Maximum/3자리/원값 Tooltip 검사 |
| 화면 초기 Timeline 위치 | 페이지 스크롤0에서 하단1037.313px<1080(1920),718.313px<768(1366), KO/EN 동일 |
| 라이브 저장 Frame | 실제 Frame199의 RGB/BEV와 저장 info를 원자적으로 표시. 통제된 작업 완료 기록으로 확인했으며 실제 CARLA 수집으로 계산하지 않음 |
| Python3.7 SSE | Uvicorn0.22 pipelined 요청의 idle 타이머 오류 재현 후 수정. 실제 ASGI 응답·SSE 유지·종료 및 idle 종료 검사 |

**미검증/제한:** CARLA 엔진에서 신규 Metadata·박스·센서 동기화 실제 재수집, CUDA 전체 학습/성능 수렴/장시간 부하, LiDAR/torch_scatter, Optical/CARLA 축에 대한 기존 BEV 보정, 실제 TCP 학습 어댑터. 제공된 OD Annotation은200개 모두 빈 배열이므로 Positive OD는 명시적인 시험 자료로만 검사했습니다. 기존 Depth PNG의38.378% 포화 픽셀과 소수 정밀도는 복원할 수 없습니다. 기존 데이터의 센서 Timestamp 부재는 N/A이며 시간 정렬 검증 완료로 표시하지 않습니다.

데이터 처리 계약이 변경되어 수정 이전의 일시정지 Checkpoint는 이 버전에서 재개를 거부합니다. 이전 코드로 재개하거나 새 실험을 시작해야 합니다. 과거 설정/결과 파일은 변경하지 않습니다.

브라우저의 작업 상태·Metric 수치는 동작 확인용 통제 자료이며 연구 성능이 아닙니다. 시퀀스 이미지는 제공된 실제 데이터, Prediction은 실제 CPU SDV 모델 출력입니다. 전체 CPU 엔진 시험의 같은 시퀀스 소수 Frame 분리는 기능 확인 목적이며 독립 주행 실험의 평가 성적으로 해석하지 않습니다.

기록: `pipeline-validation.json`, `pipeline-pytest-py*.xml`, `pipeline-pytest-cpu.xml`, `pipeline-checkpoint-schema-tests.xml`, `dataset-audit.json`, `pipeline-actual-data-validation.json`, `task-pipeline-cpu-validation.json`, `planning-export-validation.json`, `pipeline-browser-validation.json`. [사용 안내](PIPELINE_KO.md) · [정량 원인 분석](DATASET_AUDIT_KO.md).

## 이전 수정: 학습 예측·지표·대기열

| 검사 | 실제 결과 |
|---|---|
| Python 3.7 / 3.8 / 3.9 / 3.10 전체 웹·프로세스 회귀 | 각각 **365 passed, 13 skipped**. PyTorch 없는 웹 환경의 실제 모델/지표/예측/체크포인트 테스트만 명시적으로 skip |
| 실제 CPU 모델·지표·예측·재개 계약 | Python 3.10.21, torch 2.2.2+cpu, torchvision 0.17.2+cpu에서 **142 passed, 0 skipped** |
| 실제 학습 엔진 CPU 검사 | 테스트에서만 CUDA 이동·장치 guard를 CPU로 대체. 실제 Dataset·ResNet18/FPN·Seg/Depth·Loss·Adam·StepLR로 2 Epoch 연속 실행과 Epoch 1 일시정지→Epoch 2 재개 비교. 최종 가중치·모든 지표/이력·LR **정확히 일치** |
| 고정 표본·실제 이미지 | 동일 데이터 내용에서 VAL 고정 4개, 실제 SDV 모델의 모든 이미지/BEV 출력 PNG 검사. 실제 엔진 재개 후 예측 48개 유지, TRAIN 주기 0이면 TRAIN 예측 없음 |
| 태스크 지표 | 실제 CPU 정답/예측 Tensor의 혼동행렬·Depth 오차·박스 matching·AP, 비활성/정의 불가 지표 제외. Loss와 오차 최소, IoU/AP 등 최대 Best·Epoch 검사 |
| 대기열·프로세스 | 실제 자식 프로세스 시작·에폭 경계 pause·종료 코드 75·유효 저장 파일·resume·완료, 순서/불변 사본/복구/삭제 결과 보존/단일 실행/자원 경합 검사 |
| 실행 중 다음 설정 저장 | 느린 준비 검사와 실제 launch lock을 잡은 상태에서도 다음 데이터 경로 저장·모델 추가 허용. 현재 작업의 사본은 유지, 다른 연구 Python·저장 루트 변경은 차단 |
| 예측 조회 API | 실제 파일 기반 페이지/필터/등록된 PNG·프로젝트 내부 경로·인증·Traversal/Symlink 차단·기존 이미지 호환. 파일 없는 페이지도 다음 metadata offset으로 진행 |
| Chromium 실제 화면 | **9개 기능 검사**, KO/EN × 모니터/대기열/결과 × 1920/1366/390 = **18조합**, 오류·가로 넘침·정적 번역 누락 0. 최종 스크린샷 19개 |
| 브라우저 예측 갤러리 | 고정 4개 각각 입력/GT/Prediction 3개 PNG 디코딩, 주기 갱신 후 유지, Train/VAL/Epoch/샘플/태스크 필터와 실제 카드 일치, 동일 샘플의 모델별 Epoch 비교 |
| Best·그래프·로그·ETA | 전체 이력의 Epoch 2 Best가 최근 299–300 그래프에서도 유지. 그래프 352px, 기본 접힌 로그, 완료 Epoch의 실제 소요시간 기반 ETA와 휴지 시간 제외 |
| 구문 검사 | app.js/i18n.js·bash -n start.sh·Python 3.7 소스 compile 통과 |

CPU 엔진 검사는 **명시적으로 생성한 64×64 검사 데이터**와 테스트 전용 CPU 대체를 사용했습니다. 브라우저 지표·작업 상태는 통제된 QA 데이터이며, 이미지 자체는 CPU PredictionRecorder가 실제로 생성한 파일을 HTTP API로 제공했습니다. 화면 검사 값을 연구 성능으로 해석하지 마세요. 검증 중 발견한 원본 Depth 마스크 float 변환 오류를 bool 유지로 수정했으며, 원래 silog Loss의 forward/backward가 실제 CPU에서 통과합니다.

**실제 CARLA 연결·수집·센서와 CUDA 전체 학습·GPU pause/resume·실제 연구 데이터의 정확도/수렴·장시간 성능·LiDAR/torch_scatter는 미검증입니다.** 원본 학습 알고리즘·Loss·BEV geometry를 변경하지 않았으며 CUDA guard는 운영 코드에 유지했습니다.

검사 기록은 `training-monitor-validation.json`, 버전별 `training-monitor-pytest-py*.xml`, `training-monitor-pytest-cpu.xml`, `training-engine-cpu-validation.json`, `training-monitor-browser-validation.json`에 있습니다. 보완된 동일 Python 경로 저장과 launch 경합 검사는 별도의 API follow-up XML에도 기록합니다. 사용 흐름·표준 지표 정의·재개/저장 제한은 [학습 모니터링 문서](TRAINING_MONITOR_KO.md)를 확인하세요.

## 이전 수정: 시작·진행률·모델 확장 검증

| 검사 | 실제 결과 |
|---|---|
| Python 3.7 / 3.8 / 3.9 / 3.10 전체 웹·프로세스 회귀 | 각각 **281 passed, 1 skipped**. PyTorch 없는 웹 환경에서 실제 모델 모듈만 명시적 skip |
| 버전별 Python 소스 컴파일 | 네 버전 모두 통과 |
| 실제 CPU 모델 계약 | Python3.10, torch2.2.2+cpu, torchvision0.17.2+cpu에서 **46 passed** |
| 인코더 × 이미지 디코더 | 11 선택 ID × FPN/Bi-FPN/UNet = **33조합**, 실제 shape·유한값·입력/파라미터 gradient 확인 |
| 통합 출력 | Seg/Depth/OD 3 디코더, 실제 3069 앵커 일치. BEV 3종 × 이미지 디코더 None/FPN, 독립 200×200 맵 4개와 geometry·역전파 확인 |
| 기존 모델 호환 | 보존한 원본 ResNet/EfficientNet encoder의 state key/shape·strict load·lifted 출력 수치 비교 통과 |
| 실제 사전학습 파일 | ResNet34·기존 EfficientNet-B0 캐시 로딩, MobileNetV3 Small 공식 파일 **10,306,551 bytes** 신규 다운로드·SHA256·네트워크 차단 로딩·forward 통과 |
| CARLA 자동 종료 | 실제 격리 subprocess 정상 TERM·무응답 KILL·무관한 프로세스 보존, PID 재사용·권한 오류·시작 스크립트 실패 경계 검사 |
| 실행 중 FPS 수정 | 2→24/60 적용 시 worker의 sensor_tick·발행 주기 갱신. 실측10 FPS 입력은 목표24/60이라도 실제10으로 표시하는 fixture 검사 |
| 실제 Chromium | KO/EN × 수집/학습/센서 × 1920/1366/390 = **18조합**, 가로 넘침·JS 오류·정적 번역 누락 없음 |
| 브라우저 작업 연결 | 실제 무해한 subprocess의 이벤트→JobManager→SQLite→SSE로 37/120·30.8%·Town02 17/60 표시. 중지·새로고침 후 수 보존 |
| 접기·모델 선택 | 준비 완료 자동 접기, 수동 펼침/접기 우선 및 reload 유지, 화면 순서, 환경별 모델 불가 이유, 실제 API 설정 저장·FPS apply 연결 |
| JS·시작 스크립트 구문 | app.js/i18n.js/sensor-preview.js와 bash -n 통과 |

**CARLA 엔진 실제 종료·기동·프레임 수신·주행 수집, CUDA 전체 Epoch 학습·수렴·실제 연구 데이터셋, LiDAR/torch_scatter, BEV train-mode normalization 및 모든 구버전 연구 패키지 조합은 미검증입니다.** CPU 계약 검사와 통제된 CARLA/카메라 UI fixture를 실제 하드웨어 성공으로 계산하지 않았습니다. 다른 신규 백본의 가중치는 임의 초기화 계약만 확인했습니다.

원본 오류 두 개를 실제 CPU 텐서로 재현한 뒤 수정했습니다: ResNet의 네 특징맵을 FPN의 세 특징맵에 넘긴 unpack 오류, Bi-FPN의 stride16/32 fusion 크기 오류. 기존 손실·BEV 좌표·데이터 형식·주행 로직은 유지합니다. 실제 CPU 검사에는 기존 BEV 모듈의 ctx.saved_variables 및 pretrained=False 관련 비권장 경고 66건이 남지만 테스트 오류는 없고, 이번 인코더의 실제 사전학습 로딩에는 해당 경고가 없었습니다.

현재 기록은 `model-regression-validation.json`, `model-contract-validation.json`, `model-contract-pytest.xml`, `model-ux-browser-validation.json`이며 아래 내용은 이전 수정 당시의 검사 기록입니다. 실행 시 현재 사용자 CARLA를 종료하는 새 동작은 `start.sh`의 시작 단계에 적용합니다. 웹 종료·연결 해제·프리뷰 정리는 기존 소유 범위를 유지합니다.

## 이전 수정: 프리뷰·학습 선택·가중치 준비 개선

외부 차량 카메라는 프리뷰 worker와 화면에서 제거했습니다. 배치도에는 점이 대략적인 센서 위치임을 명시했습니다. 신규 기본 갱신값은 24 FPS이고 입력 범위는 정수 1~60입니다. 기존에 저장한 정수 갱신값은 유지하며, 구버전 소수 FPS는 새 기본값으로 읽습니다. 요청 FPS와 실제 센서 수신·출력 FPS는 분리해 표시하며, CARLA·GPU 성능을 초과하는 실제 출력률을 보장하지 않습니다.

| 최신 검사 | 결과 |
|---|---|
| Python 3.7 / 3.8 / 3.9 / 3.10 전체 회귀 | 각각 **208 passed** |
| 데이터 선택 가능 여부 | train/val 전체 장면의 센서/라벨 파일, 빈 파일, frame ID 불일치, 부분 BEV, 안전한 경로 및 선택 카메라 의존성 검사 |
| BEV 디코더 | BEV를 끄면 저장/조회 값이 null, 켠 경우 지원 디코더 필수. 연구 모델의 기존 None 분기를 사용 |
| 가중치 서비스 | 13개 회귀: 비차단 준비·중복 방지·진행·취소·해시 검증·캐시 보존·실패 |
| 공식 가중치 실제 다운로드 | ResNet34 **87,319,819 bytes**, EfficientNet-B0 **21,388,428 bytes**, 두 파일 SHA256 확인 및 Python 3.7에서 네트워크 없는 캐시 재사용 통과 |
| API 경계 | 준비 검사 명시 요청만 다운로드 시작. 상태 조회/자동 검사에서 다운로드하지 않음. 초안 데이터 경로 검사·CSRF·경로 보호 확인 |
| 실제 Chromium | 11개 기능 묶음 통과. KO/EN × Training/센서 설정 × 1920·1366·390px: 12개 조합 가로 넘침·JS 오류 없음 |
| 프레임 갱신 | 동일 URL의 decoded-image 캐시 문제를 수정. 명시적 6 FPS fixture에서 약 167ms 간격의 HTTP 요청, 동시 요청 최대 1. 화면 숨김·지연된 이전 설정 응답 처리 확인 |

전체 회귀 원문은 `ux-regression-validation.json`, 브라우저 기록은 `ux-training-validation.json`, 실제 공식 파일 다운로드 증거는 `pretrained-validation.json`에 있습니다. 브라우저 데이터 선택 검사는 실제 임시 train/val PNG를 사용했고, 다운로드 상태와 센서 영상 스케줄링 검사는 명시적 controlled fixture입니다. 기능 검증 후 스크린샷도 직접 점검했습니다. 가중치는 테스트용 격리 경로에서 검증했으며 배포 ZIP에 포함하지 않습니다. 모델 역직렬화·CUDA 학습 성공, 실제 CARLA 센서 영상 및 24 FPS 달성은 **미검증**입니다.

## 이전 수정: 센서 구성 실시간 미리보기·한국어/영어 전환


원본의 실제 RGB·LiDAR 미리보기와 센서 위치 적용 기능이 웹에서 누락된 것을 확인해 별도 연구 Python 프로세스로 복원했습니다. 숫자 입력과 상단·측면 배치도 조작은 초안에 반영되며, **프리뷰에 적용**으로 실제 센서를 재부착하고 **센서 설정 저장**으로 구성을 별도 보존합니다. 차량 외부 카메라 영상도 추가했습니다. 배치도는 실제 영상과 구분해 표시합니다.

| 최신 코드 검사 | 결과 |
|---|---|
| Python 3.7.16 / 3.8.20 / 3.9.25 / 3.10.21 전체 회귀 검사 | 각각 **158 passed** |
| 센서 Blueprint 16개 | 설정된 RGB/LiDAR 속성 적용, 미지정 값의 기존 기본값 유지, 잘못된 수치 거부 |
| 프리뷰 worker 15개 | 명시적 CARLA API fixture와 실제 RGB/LiDAR 바이트로 JPEG 생성·좌표 적용·revision·저속 FPS·clock 소유권·자기 액터 정리·실패 확인 |
| 프리뷰 프로세스 서비스 9개 | 실제 fixture subprocess의 시작·중복·새로고침 복원·적용·취소·종료, 외부 PID 보호, 이전 revision 이미지 차단 |
| 프리뷰 API 7개 | 초안/저장 분리, 실행 충돌 방지, 연결 설정 보호, 요청 검증·CSRF·프레임 경로·revision/session 검사 |
| JavaScript | app.js / i18n.js / sensor-preview.js 문법 검사 통과 |
| 언어 사전 | 정적 UI 550개 항목의 영어 변환 확인. 템플릿의 동적 사용자 데이터·로그·경로는 번역하지 않음 |
| 실제 Chromium | 16개 조작 검사 통과. EN 11개 화면·하위탭 정적 번역 누락 0, JS 오류 0. KO/EN × 1920·1366·390px 화면 검사 통과 |
| 화면 상태·충돌 안내 | 대시보드 프리뷰 실행 표시, 실행 조건이 준비돼 있어도 프리뷰 중 수집·학습/연결 변경 차단 및 이유 표시 확인 |

위 전체 회귀 검사 후 추가한 상태 전환 로그 및 오래된 프레임 요청 차단은 Python 3.7의 worker 15개·API 7개, Python 3.10의 해당 22개 검사로 재확인했습니다. 상세 기록은 `sensor-preview-validation.json`에 있습니다. 브라우저 조작·화면 검사 기록은 `sensor-browser-validation.json`에서 확인할 수 있습니다. 브라우저의 센서 영상은 **CONTROLLED HTTP FRAME FIXTURE / NO CAMERA OR GPU OUTPUT**으로 표시한 검사 이미지입니다. 새 revision 적용 후 늦게 도착한 이전 이미지 차단, 실패·오래된 프레임 표시, 언어 전환 후 미저장 폼과 실행 상태 보존을 확인했습니다. 390px 상단 상태 문구 줄바꿈 문제는 수정 후 재확인했습니다.

**실제 CARLA/GPU 센서 영상·센서 부착 결과·물리 환경은 미검증입니다.** API fixture, 작은 프로세스, 입력 이미지 바이트를 실제 시뮬레이터 실행 성공으로 계산하지 않았습니다. 미리보기는 현재 맵과 world 설정을 유지하며, 외부 동기 서버에는 기존 tick 공급 클라이언트가 필요합니다. 학습 알고리즘과 데이터 구조는 변경하지 않았습니다. 기존에 무시되던 LiDAR 설정값은 미리보기와 수집 양쪽에 같은 공통 함수로 적용하도록 수정했습니다.

## 이전 수정: CARLA Python 클라이언트 자동 준비

설치 후 실행이 `No module named 'carla'`에서 멈추던 선행 검사를 수정했습니다. 선택한 연구 Python에 모듈이 없으면 `carla==0.9.15` wheel만 설치하고, 동일한 Python의 실제 import와 클라이언트 버전을 검증한 뒤 엔진 준비로 진행합니다. 기존 다른 버전·손상된 설치는 덮어쓰지 않습니다. 엔진의 `carla/` 폴더만 있는 상황도 클라이언트 미설치로 정확히 구분합니다.

| 최신 수정 검증 | 결과 |
|---|---|
| Python 3.7.16 / 3.8.20 / 3.9.25 / 3.10.21 전체 회귀 검사 | 각각 **111 passed** |
| 위 4개 버전의 깨끗한 연구 환경 | 실제 `ensure_client` → 공식 CARLA 0.9.15 wheel 설치 → 별도 프로세스 import·버전 확인 통과 |
| Python 3.7, pip 없는 연구 환경 | 웹 pip 26.2.1로 pip 24.0 wheel을 캐시하고 연구 Python에서 실행 성공. 연구 환경의 pip는 설치·변경하지 않음 |
| 기존 패키지 보존·재사용 | 네 환경 모두 추가 배포 패키지는 carla 하나. 두 번째 실행은 로컬 검사만 수행하고 설치·다운로드 미호출 |
| 설치 제어·보호 | 소형 wheel/엔진 fixture로 설치 후 엔진 준비, 취소·시간 초과·자식 종료, 기존 다른 버전·손상 보존 검증 |
| 설치 후 상태 동기화 | 완료·실패·중지 시 의존성 캐시 갱신. 브라우저에서 환경 재조회 1회 및 미저장 입력 보존 확인 |
| 실제 Chromium 화면 | Simulation·Settings × 1920 / 1366 / 390: 6개 조합 오류·가로 넘침 없음. 확인창·취소·로그·새로고침 통과 |

**CARLA 엔진 전체 다운로드·GPU 기동·실제 센서 수집은 여전히 미검증입니다.** Python 클라이언트 설치 성공과 실제 시뮬레이터 실행 성공은 구분합니다. 연구 알고리즘과 학습 로직은 이번 수정에서 변경하지 않았습니다.

실제 패키지 설치 기록은 `carla-client-validation.json`, 전체 회귀 검사는 `carla-client-regression.json`, 화면 검사는 `carla-browser-validation.json`에 있습니다. 최초 회귀 검사에서 standalone Python 복사로 인한 테스트용 venv 생성 오류 3건을 발견해 표준 CLI와 같은 symlink 방식으로 수정했습니다. 생산 코드는 이 테스트 환경 문제 때문에 변경하지 않았습니다.

## 이전 수정: CARLA 자동 설치·엔진 실행 연결

기존 Pygame의 설치 함수를 웹에서 호출하지 않던 누락을 수정했습니다. 시뮬레이션 화면의 **CARLA 설치 후 실행**은 미설치 시 다운로드·압축 해제를 수행하고, 설정한 포트로 엔진을 시작해 RPC 준비를 기다립니다. **설치만**도 별도로 사용할 수 있습니다.

| 엔진 설치 수정 당시 검증 | 결과 |
|---|---|
| Python 3.7.16 / 3.8.20 / 3.9.25 / 3.10.21 | 버전별 **102 passed** |
| 설치기 23개 | 실제 소형 tar·로컬 HTTP fixture: 다운로드·Range 이어받기·취소·잘못된 응답·미러 재시도·압축 경로·링크·권한·기존 파일 보존 |
| 엔진 관리 12개 | 실제 fixture subprocess 및 대체 RPC 응답: 실행·준비·실패·중지·중복·상태 복원·외부 프로세스 보호 |
| 웹 제어 경계 6개 | 설치/실행/중지/로그 API, CSRF, 연구 작업 충돌 방지, 연결 설정 잠금, 준비 후 자동 연결 |
| 소스 컴파일 / 실제 HTTP / SSE / 정상 종료 | 위 4개 Python 버전 모두 통과 |
| 실제 Chromium 화면 | Simulation·Settings × 1920×1080 / 1366×768 / 390×844: 6개 조합 가로 넘침·JS 오류 없음 |
| 브라우저 제어 | 설치 용량 확인창·취소 시 시작 미요청·새로고침 상태 유지·외부 호스트 로컬 시작 차단·로그 API 연결 통과 |
| 배포 주소 소량 검사 | 기존 주 주소 HTTP 403, 대체 CARLA 배포 주소 HEAD 200 / Range 206, 총 8,386,636,048바이트 확인. 수신 데이터 1바이트 |

**실제 7.81 GiB 전체 배포 다운로드·압축 해제와 CARLA GPU 기동·데이터 수집은 미검증입니다.** 설치·엔진 테스트의 소형 fixture를 실제 시뮬레이터 성공으로 계산하지 않았습니다. 연구 수치·알고리즘은 이번 수정에서 변경하지 않았습니다.

실행 상태는 별도 작업 프로세스와 원자적으로 저장되는 파일로 관리합니다. 브라우저 새로고침으로 설치·엔진이 중지되지 않습니다. 웹 서버 정상 종료 시 연구 작업의 Actor 정리가 끝난 뒤 소유 CARLA 엔진을 정리합니다. 외부에서 시작한 CARLA는 종료하지 않습니다. 엔진 준비 제한은 120초이며, 준비 실패 후 소유 프로세스를 정리합니다.

당시 상세 버전별 기록은 `carla-validation.json`입니다. 아래 항목은 이전 배포판의 검증 기록이며, 최신 전체 테스트 수는 위의 **208개**입니다.

## Python 3.7–3.10 호환성 재검증

버전 조건만 낮추지 않고 실제 인터프리터와 버전별 의존성으로 검사했습니다.

| Python | 전체 테스트 | 소스 컴파일·API 검사 | 실제 HTTP·SSE·정상 종료 |
|---|---|---|---|
| 3.7.16 | 61 passed | 통과 | 통과 |
| 3.8.20 | 61 passed | 통과 | 통과 |
| 3.9.25 | 61 passed | 통과 | 통과 |
| 3.10.21 | 61 passed | 통과 | 통과 |
| 3.12.14 | 61 passed | 기존 환경 회귀 검사 | 이번 버전 수정에서 별도 재실행하지 않음 |

Python 3.7/3.8에서 `start.sh`의 새 웹 가상환경 생성·pip 준비·의존성 설치·CLI 실행을 확인했습니다. 3.8에서는 재실행 시 설치를 반복하지 않는 것도 확인했습니다. 기존 3.12 웹 환경은 보존되었습니다. 상세 기록은 `python-compatibility.json`에 있습니다.

당시 수정 범위는 웹 실행기·의존성·표준 라이브러리 호환 코드·테스트였습니다. 이후 위 CARLA 연결 수정에서 설치기의 `tarfile` 필터 의존성도 Python 3.7 호환 추출기로 교체했습니다. 아래의 CARLA/CUDA 실제 하드웨어 미검증 항목은 그대로입니다.

## 확인한 결과

아래 표는 최초 통합판의 검증 기록입니다. Python 호환 수정 이후의 전체 테스트 수는 위 표의 61개이며, 웹 화면 코드는 이번에 변경하지 않았습니다.

| 구분 | 실행 내용 | 결과 |
|---|---|---|
| Python 테스트 | `python -m pytest -q tests` | **54 passed**, 7.90초 |
| API/프로세스 | API·설정 22개, 프로세스 수명주기 10개 | 통과 |
| 실제 파일 처리 | Catalog·데이터셋 12개 | 통과 |
| 연구 연결 경계 | 연구 어댑터 10개 | 통과 |
| Python 구문 | `python -m compileall -q studio sdv` | 통과 |
| JS 구문 | `node --check studio/static/app.js` | 통과 |
| 실행 스크립트 | `bash -n start.sh`, `./start.sh --help` | 통과. 별도 가상환경 생성·웹 의존성 설치 확인 |
| 실제 HTTP smoke | `python scripts/check_web.py` | 조회·설정 오류 처리 통과 |
| Chromium 화면 | 5개 페이지 × 1920×1080 / 1366×768 / 390×844 | 15개 조합 모두 가로 넘침·JS 오류 없음 |
| 브라우저 입력 | 시스템/수집/학습 설정 저장, 새로고침, SSE 수신 중 편집 유지 | 통과 |
| 브라우저 결과 | 격리 CSV fixture 상세 그래프·실험 비교·설정 표시 | 통과 |
| 연결 실패 | CARLA 모듈 없는 환경의 연결 오류와 준비 조건 표시 | 통과 |
| 인증 | 키 없는 API 거절, 접속 키 입력, 인증된 저장, 탭 새로고침 | 통과 |

스크린샷은 `screenshots/`에 있습니다. 기본 화면에는 실제 빈 데이터 상태와 실행 환경에서 측정한 시스템 자원만 표시했습니다. 그래프의 UI/파서 테스트는 **명시적인 임시 fixture**를 사용했으며, 이는 연구 성능이나 학습 성공을 입증하지 않습니다. 해당 fixture는 배포 데이터에 포함하지 않았습니다. 상세 브라우저 기록은 `browser-validation.json`, 인증 검증은 `auth-validation.json`입니다.

테스트 경고 1건: 설치된 Starlette가 TestClient의 기존 httpx 지원을 비권장으로 안내합니다. 테스트 실패나 서버 런타임 오류는 아닙니다.

실행기 환경은 `os.getpid()`와 psutil이 읽는 `/proc`의 PID namespace가 달랐습니다. 프로세스 테스트에는 이 환경에만 적용하는 metadata adapter가 명시되어 있습니다. subprocess 생성, 시그널, 중지·실패·종료 코드는 실제 OS 동작으로 검사했습니다. 일반 Ubuntu 호스트에서 같은 테스트를 다시 실행하는 것이 필요합니다.

## 발견하여 수정한 문제

- 기존 웹의 타이머 수집과 난수 학습이 실제 작업처럼 보이던 연결을 새 실행 경로에서 제거.
- 출력 없는 작업을 중지할 수 없던 blocking log 읽기를 파일 tail + supervisor로 대체.
- 부모가 먼저 종료되면 자식 프로세스가 남던 문제를 재현하여 소유 자식 정리로 수정.
- 서버 복구 시 같은 JSONL 이벤트가 중복 반영되던 문제를 공유 event cursor로 수정.
- 실제 종료 코드가 남았는데 DB 최종 상태가 없던 경우 exit receipt로 복원.
- 실험의 `config/`, `output/` 중첩 구조 때문에 설정·CSV가 빠지던 문제 수정.
- 연구 코드의 CSV 길이 불일치, 미정의 BEV 변수, 선택 입력의 배치 조립 오류 수정.
- 카메라마다 다른 출력 설정을 적용할 때 폴더와 OD 선행 센서가 누락되던 문제 수정.
- Checkpoint 토글과 실제 저장 동작 연결.
- 프리뷰 컨트롤 줄바꿈, 한글 로그 font fallback 수정.

## 미검증 항목

| 항목 | 현재 상태 | 사용자 환경에서 필요한 확인 |
|---|---|---|
| CARLA 연결 성공 및 수집 | **미검증**: 실제 서버 없음. Python 모듈 설치는 위 4개 환경에서 검증 | 호환 서버에서 3~10샘플 수집, 프레임/라벨 저장과 액터 정리 |
| CUDA 실제 학습 | **미검증**: PyTorch/CUDA·가중치·데이터 없음 | 준비된 장면으로 1 epoch, forward/backward 및 checkpoint 로드 |
| 모든 모델/태스크 조합 | **미검증** | ResNet/EfficientNet, FPN/Bi-FPN, BEV·Seg·Depth·OD 조합별 실행 |
| 수치 정확성·수렴 | **미검증** | 기존 기준 실행과 Loss/metric/예측 비교 |
| 영상 전송 부하 | **미측정** | 프리뷰 켜기/끄기 수집 처리량 비교 |
| 장시간 대량 연구 | **미검증** | 실제 수시간 작업, 대용량 디스크·GPU 사용량 확인 |
| CARLA 연결 중 서버/지도 변경·정리 | 코드 및 경계 검토, 실제 서버 미검증 | 전용 CARLA 인스턴스에서 테스트 |

웹과 프로세스 계층 검증은 완료했으나, 이 자료를 근거로 전체 자율주행 연구 파이프라인의 하드웨어 실행이 검증되었다고 해석하면 안 됩니다.

## 남은 제한과 개선 항목

- 원본에 독립적인 평가 실행기/모델 추론 서버가 없습니다. 현재 평가 페이지는 학습 중 validation과 저장 결과 분석을 제공합니다.
- 외부 CARLA 연결과 웹에서 관리하는 로컬 설치·엔진 실행을 모두 제공합니다. 전체 배포 다운로드와 실제 GPU 엔진 실행은 사용자 환경에서 추가 확인해야 합니다.
- 과거 수집 장면은 당시 사용한 `sensor_config.yaml`이 필요합니다. 현재 센서 설정을 과거 데이터의 보정값으로 자동 대체하지 않습니다.
- 데이터셋 준비는 원본 보존을 위해 독립 복사합니다. 큰 데이터는 시간과 추가 공간이 필요하며, 바이트 단위 복사 진행률·취소 UI는 아직 없습니다.
- 내부 경로만 허용합니다. 외부 데이터 저장소 연동·다중 사용자 계정·역할 관리·내장 TLS는 후속 작업입니다.
- 텍스트 metric/log 파서는 파일당 16 MiB, 메타데이터 ZIP은 256 MiB 한도입니다. 대형 checkpoint는 개별 다운로드합니다. 2,000점 이상 그래프는 실제 점을 결정적으로 간추려 표시합니다.
- 원본 `OD_iou`는 Anchor–GT IoU입니다. 검출 mAP 또는 최종 예측 박스 IoU로 해석하지 않습니다.
- 장애로 process group과 receipt까지 함께 사라져 원인을 확인할 수 없는 작업은 성공으로 추정하지 않고 실패로 표시합니다.
