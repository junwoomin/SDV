# SDV 실행·UI 개선 진행 기록

기준: 2026-10-05, 원격 HEAD `ad67aa1f412c7428a2a126fe9c00a1726b94d693`.
작업 폴더: `/home/wo/SDV_V1`. 원격 push/PR/merge 없음.

## 확인된 성공

- 신규 clone, AGENTS 없음. README·개발 메모·start·수집·학습 입력 구조 검사.
- 기존 aa Python 3.9.25는 읽기만 사용. 프로젝트 `.venv`에서만 Pygame 2.6.1, webcolors 24.11.1 추가.
- 원본 start의 py_trees 누락 → 전역 pip 설치 시도를 호출 차단으로 재현. sudo·전역 pip·전역 CARLA kill 제거.
- 기존 CARLA 0.9.15 30GB를 독립 복사. 원본/복사본 실행 파일 SHA256 동일. VERSION·README·LICENSE·wheel·파일 날짜는 공식 배포 구조와 일치. 원본 다운로드 영수증이나 공식 파일 체크섬으로 출처를 암호학적으로 입증한 것은 아님.
- 공식 기본 CARLA 배포 크기 8,386,636,048 bytes, 추가맵 7,375,946,087 bytes. 둘 다 이번에 다운로드하지 않음.
- 원본 더미 메뉴 렌더, 개선 UI 실제 X11 창 열기·스크린샷·종료 확인.
- 실제 CARLA 서버/클라이언트 0.9.15 일치. maps, world, map, 프레임 응답 확인.
- 실제 X11 센서 카메라 이미지 수신. 반복 시작 두 번에도 서버 한 개. 창 종료 후 소유 작업 잔존 없음.
- 12개 안전 테스트 통과: 입력/경로, 다운로드 실패·취소·재개, 안전 tar, 시간 제한, 반복 중지, 다른 프로세스 보호, RPC 클라이언트 재생성.
- 경량 UI 30개 동작/19개 판정 통과: 설정 저장→모델→대기열→빈 결과, 음수/NaN, 반복 클릭, 재진입, Tab/Enter, 창 크기 조절·종료.
- `train.py --check`가 실제 `train2.py` 진입점, UI의 epochs=20/batch=4/LR=0.001/AdamW를 읽고 준비 자료 부재로 차단함.

## 실패와 확인한 원인

- 선택 Town01을 나중에 load_world한 실행은 GPU 증가 6,593MiB로 승인된 6GiB 감시 상한을 초과해 중단. 공식 소스에서 원본 `-quality_level=Low` 오타를 확인해 `-quality-level=Low`로 수정하고 선택 맵을 초기 URL로 전달. 같은 예산에서 미리보기→3장 수집 성공. 상한은 늘리지 않음.

- 첫 CARLA 실행은 GPU 메모리 증가 5,339MiB로 초기 3GiB 감시 상한을 초과해 소유 scope만 종료. 실패 로그 보존.
- 부모 허용 후 6GiB 감시 상한에서도 기존 client 반복 호출은 timeout. namespace·포트·IO를 확인한 뒤, 같은 서버에서 시작 전 client 실패 / LISTEN 후 새 client 성공을 비교해 초기 TCP 연결 실패 객체 재사용을 원인으로 확인. 매 재시도 새 client·버전 검사로 수정.
- aa Pygame 2.0.0의 합성 MouseButtonDown 좌표/버튼 손상 재현. 프로젝트 venv의 2.6.1에서 동일 이벤트 정상 확인.
- 원본 모델 설정 `od` 미초기화, 센서 화면 `sensor` 선언 순서 오류를 실제 경로에서 재현하고 최소 수정.
- 센서 구독 정리 전에 서버/인터프리터를 종료할 때 native terminate 재현. 센서 stop→actor destroy→소유 서버 stop 순서로 수정 후 재발 없음.

## 실제 수집 검증

- `docs/qa/tiny-collection.json`: 선택 Town01, 실제 맵 Carla/Maps/Town01 일치.
- `data/qa_collection/20261005_144844/`에 RGB 3장과 info 3개 저장. 수집 종료 코드 0, remaining_owned=[] 확인.
- native ps로 SDV CARLA/수집 프로세스 0, 소유 sdv scope 0, GPU 메모리 508MiB로 복귀 확인 후 부모에 GPU 해제 보고.
- Town01 미리보기 캡처는 첫 센서 이미지 대기 시점이다. 실제 카메라 수신 증거는 별도 live-sensor-preview.png와 저장된 Town01 RGB 파일이다.
- 한국어 실행 안내와 검증 증거를 README_ko.md 및 docs/qa/EVIDENCE.md에 정리.
- 최종 12/12 안전 검사, 30/30 UI 동작·19/19 판정 통과. 실행 옵션/선택 초기맵과 변경 학습 설정 전달 회귀 포함.

## 미실행·남은 자료

- 대규모 수집, 모델 학습, 학습/검증 성능 측정 없음.
- 현재 `data/train`, `data/val`, 호환 `img/test.pth`, 로컬 사전학습 가중치가 없음. 학습 실행 전 차단하고 자동 모델 다운로드하지 않음.
- 시험 3장은 학습 분할이나 성능 근거로 사용하지 않음. 원본 train2 모델/데이터의 전체 의미·텐서 호환성을 실행으로 입증하지 않음.
- 다운로드 재개·취소·실패는 작은 mock 응답으로 검증. 공식 7.81GiB 전체 다운로드·압축 해제는 미실행.

## 실행 예산

- UI: 사용자 scope CPU 100%, RAM 2GiB, nice 10, 20FPS. 학습 GPU 시작 없음.
- CARLA: 별도 소유 scope CPU 400%, RAM 8GiB, swap 0, 저품질, offscreen 640×360, 10FPS, 최대 90초.
- VRAM은 하드 제한 기능이 없어 6GiB 감시 상한으로 중단. PID compute 조회와 전체 GPU 메모리 증가량을 보수적으로 같이 감시. 그래픽 전용 할당도 놓치지 않게 하며, 다른 작업의 증가로 중단될 수 있음. 시작 전 GPU 여유 8GiB 확인.
- 종료/중지/오류/시간 초과는 이번 실행의 unit·카메라·액터만 정리. OS·드라이버·전역 conda·다른 저장소 변경 없음.

## 2026-10-06 사용자 실행 오류 수정

기본 문자열 수집 수→YAML→대기열 TypeError를 격리 UI에서 재현·수정. 모든 센서 설정 읽기/저장 경계에서 검증된 정수를 사용하고 편집 중 빈칸·NaN·음수·소수·초대형 값은 오류로 차단. 사용자 Town02 YAML SHA256 보존. 서버부터 종료하던 수집 전환을 sensor cleanup→server stop 순서로 수정. 실제 카메라 수신 후 0.506초 정리, 반복/죽은 서버 정리, 잔존 0 확인. 단위 22개 및 UI 80동작/51판정 통과. 이번 직접 정리 서버 실제 맵은 Town10HD_Opt로 기록하고 Town01 증거로 사용하지 않음. 상세 재현·실패·통과·미실행·승인 검토 기록: docs/qa/bugfix-20261006/REPORT.md.

## 2026-10-06 기능별 구조 정리

오류 수정 상태를 보존한 후 소스 34개를 sdv 기능 패키지, experiments, config/examples, requirements로 이동. UI·수집·학습 호환 CLI와 옛 모델 import 경로 유지. actor cleanup/live preview/센서 encoding/ports를 추출해 수집의 UI import를 제거. 자산·환경·사용자 설정·vendor 배포 보존, 영구삭제/학습/대량수집/push 없음. 이동 원본·해시·복구 manifest는 .runtime/refactor-current.txt가 가리키는 폴더, 복구법은 docs/layout.md. 과거 PASS 기록과 새 구조의 검증 기록은 docs/qa/refactor-20261006에서 분리.

## 2026-10-06 창·글자·배치 개선

1400×1000 캔버스 전체 확대를 제거하고 sdv/ui/responsive.py에서 창/DPI metric, native 한글 글꼴 계층, 내용 스크롤, 공통 행동/상태 영역을 그린다. 기존 backend의 config→collection→training→inspection과 숫자 검증·소유 process/actor cleanup을 유지한다. 동일 제어 rect를 그리기·pointer·Tab에 사용하고 논리 이벤트로 연결한다.

세 창 크기와 실제 X11, 반복 resize/최대화·복원, 글자150%/기본크기 복원, 긴 오류, 빈 결과, 설정 재진입, 반복 클릭, data_num 회귀 및 종료 후 재실행을 격리 복사본에서 확인한다. 화면 밖 fixture 제어를 스크롤 전 hit 가능으로 기대하던 harness를 분리해 스크롤 후 접근을 검증했다. PASS/실패/표시 fixture/미실행의 최종 숫자와 PNG는 docs/qa/ui-responsive-20261006/REPORT.md 및 summary.json에 남긴다. 이번 범위에서 CARLA/GPU/학습/수집/다운로드/push/Library 업로드는 하지 않았다.

기존 refactor snapshot과 hash manifest는 그대로 보존하고, UI release before/after와 selective restore guard를 별도로 둔다. UI rollback 후 원래 layout rollback 순서만 허용하며 사용자 데이터/자산 sentinel과 외부 변경 차단을 임시 복사본에서 검증한다. 원본 프로젝트의 복원은 수행하지 않는다.
