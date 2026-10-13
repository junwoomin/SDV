# CARLA 0.9.15 엔진 안정화 및 SDV 기능 검증

검증일: 2026-10-12. 대상: `wo-MS-7D42`, `/home/wo/SDV/sdv_studio`.

홈 SDV에 수정본을 적용했다. 기존 CARLA 0.9.15와 연구 Python을 재사용했고, 기본 렌더링을 **Epic + Off-screen**으로 변경했다. 실제 Roach·RGB·Segmentation·Depth·OD·BEV 수집은 **613.25초, 2,524샘플**, 원래 실패한 **Town05는 500/500샘플**을 통과했다. CUDA 학습의 시작·일시정지·재개·중지와 웹 재생도 확인했다.

Low 품질에서 발생하는 UE4 내부 결함 자체를 바이너리 패치한 것은 아니다. 아래 스택과 비교 시험으로 실패 조건을 확인하고, 동일 장비에서 검증한 운영 프로파일로 해당 경로를 피하도록 수정했다. 영구적인 Low 모드 엔진 수정에는 UE4/CARLA 소스 수준의 추가 진단·패치가 필요하다.

## A. 원인 분석

### 실제 환경

| 항목 | 확인 결과 |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti, 16,303 MiB |
| 드라이버 | 595.91.07, NVIDIA open kernel module |
| CUDA | 드라이버가 표시하는 최대 지원 13.2; 기존 PyTorch 런타임 12.8 |
| 연구 환경 | `/home/wo/anaconda3/envs/aa/bin/python`, Python 3.9.25, PyTorch 2.8.0+cu128, CARLA API 0.9.15 |
| Vulkan | Loader 1.3.275; NVIDIA 장치 API 1.4.329. llvmpipe도 열거되지만 CARLA PID는 NVIDIA GPU 0에서 C+G로 실행됨 |
| OS / Kernel | Ubuntu 24.04.4 LTS / 7.0.0-38-generic |
| 화면 | X11, DISPLAY `:1`, WAYLAND_DISPLAY 없음 |
| 메모리 | RAM 약 31 GiB, Swap 약 107 GiB, `/dev/shm` 16 GiB |
| 저장 공간 | 조사 시작 시 홈 파티션 약 475 GiB 여유 |
| 실행 환경 | Docker/가상화 감지 없음; 로컬 Linux 프로세스 |
| 드라이버 충돌 / 시스템 오류 | nvidia 모듈 확인, nouveau 모듈 없음. 확인한 커널 기록에 GPU Xid/OOM killer 근거 없음 |

샌드박스 내부의 최초 `nvidia-smi` 실패는 실제 호스트 GPU 상태와 구분했다. 허용된 호스트 조회에서 GPU/드라이버 및 CARLA GPU 프로세스를 확인했다. `vulkaninfo`와 `coredumpctl`은 설치돼 있지 않아 시스템 패키지를 추가하지 않고 Vulkan API 직접 열거와 기존 GDB/배포 debug 심볼을 사용했다.

### 기존 실패와 재현

기존 worker의 실행 명령은 `-quality-level=Low -RenderOffScreen`을 항상 함께 붙였다. 별도의 관리 실행 경로도 동일 조합을 사용했다. 이전 프로젝트의 수집 로그를 대조하면 Town05 수집에서 마지막 저장 수는 208샘플이고, 엔진 로그에는 Signal 11, worker에는 exit -11이 남아 있다.

단일 서버의 2×2 비교에서는 초기 30초 동안 client를 연결하지 않았고, 이후 API 버전/World/Snapshot → Town01 → Town03 → Ego → 카메라 순으로 진행했다. Low 실패는 센서를 생성하기 전 지도 전환 뒤에도 재현됐다.

| 프로파일 | 실제 결과 | 종료 구분 |
|---|---|---|
| A: Epic + Off-screen | 두 Town 로딩, Ego, RGB/Seg/Depth 같은 Frame 수신, 추가 생존 통과 | 정리 요청 전 종료 없음; SIGTERM 후 143 |
| B: Epic 일반 렌더링 | 같은 지도/차량/3센서 시험 통과 | 정리 요청 전 종료 없음; SIGTERM 15초 불응으로 소유 그룹만 SIGKILL(-9). 기본으로 채택하지 않음 |
| C: Low 일반 렌더링 | 지도 전환 후 SIGSEGV 재현 | 분리 하네스에서 client exit 0, engine exit -11, 정리 신호 전 -11 확인 |
| D: Low + Off-screen | 지도 전환 후 SIGSEGV 재현 | engine exit -11; 별도 GDB 재현 성공 |

초회 A 시험은 30초 client 타임아웃으로 하네스가 종료됐으나 엔진은 생존했고 Town03/Snapshot 재조회가 성공했다. 로딩 대기 시간을 늘린 재시험은 통과했다. 이 client 실패를 engine segfault로 집계하지 않았다. 초기 C 하네스의 client abort와 엔진 코드를 구분하기 위해 독립 client 프로세스로 재시험했다.

GDB가 최초 SIGSEGV에서 중단한 스레드는 `TaskGraphNP 1`이다. 핵심 스택:

```text
UTexture2D::GetNumMips
  Runtime/Engine/Classes/Engine/Texture2D.h:154
FLandscapeRenderSystem::FetchHeightmapLODBiases
  Runtime/Landscape/Private/LandscapeRender.cpp:988
FLandscapeRenderSystem::FGetSectionLODBiasesTask::AnyThreadTask / DoTask
TGraphTask<...>::ExecuteTask
FTaskThreadAnyThread::ProcessTasks
```

실패는 Low 품질의 Landscape/Texture 비동기 작업 경로에 수렴한다. 스택의 비정상 실행 주소는 내부 객체/리소스 수명 또는 경쟁 문제 가능성을 시사하지만, 해제 후 참조 같은 정확한 메모리 결함 유형은 확정하지 않았다. NVIDIA 드라이버 함수나 Vulkan OOM 반환값에서 최초 실패한 증거는 없다. `Texture2D sampler` warning, `g.TimeoutForBlockOnRenderFence`, `Malloc Size=65538`를 원인이나 VRAM 부족으로 단정하지 않았다.

배포본은 상세 Saved/Logs/Crash 파일을 생성하지 않았고 `-Abslog`만으로 상세 스택을 얻지 못했다. stdout의 `engine.log`와 GDB 원문을 보존했다. [GDB 최초 스택](validation/carla_stability_20261012/gdb_crash/backtrace.log), [Low 단독 분리 시험](validation/carla_stability_20261012/C_isolated/result.json), [2×2 로그](validation/carla_stability_20261012/engine_comparison/)를 참조한다.

CARLA 공식 저장소의 [#7675](https://github.com/carla-simulator/carla/issues/7675)에도 Low+Off-screen 실패 보고가 있지만 확정된 원인 진단은 없다. 이번 장비에서는 Low 단독도 실패했으므로 그 보고를 그대로 원인으로 적용하지 않았다. [공식 0.9.15 렌더링 문서](https://carla.readthedocs.io/en/0.9.15/adv_rendering_options/)에 따라 Vulkan과 GPU 센서 렌더링을 유지했다.

## B. 적용한 수정

| 파일 | 변경 |
|---|---|
| `studio/config.py`, `carla_worker.py` | 렌더링 프로파일 설정화. 기본 Epic off-screen. 기존 설치 경로 재사용, 실행 권한/버전/포트 검사 |
| `studio/carla_diagnostics.py` | 실제 NVIDIA/Vulkan 장치·ICD·프로세스·메모리·화면·시스템 진단. 크래시 시 로그/커널/종료 코드/마지막 단계 보존 |
| `studio/carla_worker.py` | 정확한 명령·선택 환경 변수·사전 검사 파일 저장. 실제 version/World/Snapshot 확인. 선택 Town을 API로 한 번 초기화한 뒤 Ready 판정 |
| `studio/carla_service.py` | Starting/Loading Map/Ready/Running/Crashed/Stopped 상태 및 종료 코드. 같은 프로파일의 반복 SIGSEGV 두 번 뒤 재시작 거부. 자동 재시작 없음 |
| `studio/api.py` | 엔진 크래시와 수집 안전 중지 연결. Run Diagnostics/Test Engine/Logs/Export API. 수집 종료 시 Ready, 종료 PID는 현재 PID와 구분 |
| `studio/static/diagnostics.js`, `app.js`, `index.html` | Settings의 System Diagnostics, 설치 경로·프로파일 UI, 실제 오류와 검사 결과 표시 |
| `studio/jobs.py`, `sdv/collection/workflow.py` | 실제 실행 중인 프로파일을 수집 Metadata에 기록. 선택 지도 로딩 120초 대기, 마지막 단계 기록 |
| `sdv/collection/driving_runtime.py` | `destination_reached`의 NumPy bool_을 Python bool로 생성. 실제 JSON commit 실패 재현 후 수정 |
| `sdv/training/modeling/loss.py` | CUDA 가용 여부로 CPU loss 텐서를 GPU에 강제 생성하던 결함 수정. 입력 텐서의 device/dtype을 보존 |
| `studio/catalog.py`, `studio/jobs.py` | 수집 작업을 사용하지 않은 학습 모델의 결과로 표시하던 결함 수정. 작업 kind를 명시해 학습 결과와 구분 |
| `start.sh` | 독립 실행 CARLA 전체 종료 helper 제거. 준비된 웹 가상환경 재사용. 소유 엔진만 lifecycle에서 정리 |
| `sdv/runtime/processes.py` | 관리 실행 경로도 프로파일 사용. HOME 덮어쓰기 제거. 기존 90초 강제 종료와 임의 6 GiB VRAM 제한을 수집 운영 기본에서 제거 |
| `pytest.ini`, 관련 tests | 실행 중 데이터 임시 파일을 pytest 탐색에서 제외. Snapshot, 반복 크래시, JSON, 결과 구분, 소유권 회귀 검사 |

센서 해상도와 수집 FPS는 낮추지 않았다. 활성 front 카메라에 RGB/semantic/depth 3개만 생성하며, OD는 실제 Actor box의 투영·semantic/depth 가시성 검사로 생성한다. 별도 OD GPU 카메라는 없다. 기존 exact-frame history는 센서당 32패킷으로 제한돼 있고 소유 Actor destroy를 유지했다. 데이터 수집은 `no_rendering_mode=False`; `-nullrhi`, OpenGL, texture pool 임의 조정은 사용하지 않았다.

렌더링 품질 변경은 이미지 도메인에 영향을 주므로 `collection_manifest.json`의 `rendering.profile`에 남긴다. 프로파일 변경은 다음 엔진 실행에만 적용되며 실행 중 job의 설정 사본을 바꾸지 않는다.

23개 변경 파일의 해시 및 백업은 [최종 목록](validation/carla_stability_20261012/changed_files_final.json)에 있다. 원본 백업: `.runtime/backups/carla_stability_20261012_145612/`. 드라이버·커널·시스템 패키지·기존 ML 환경·기존 CARLA 배포는 변경하지 않았다. Git push/PR 및 컴퓨터 전원 종료도 수행하지 않았다.

## C. 실행 검증 결과

| 검사 | 결과와 범위 |
|---|---|
| 최종 전체 자동 검사 | **828 passed, 2 skipped**, 115.16초. 2개 skip은 보존된 원본 모델 소스 비교용 `SDV_LEGACY_MODELING_ROOT` 미지정 검사 |
| 문법 검사 | Python compileall, `bash -n start.sh`, 두 JS 모듈 구문 검사 통과 |
| 전체 수집 장시간 | **613.25초**, 2,524 commit. Town01→Town03→Town04, Roach CPU 30Hz, front 400×300 RGB/Seg/Depth/OD, BEV 500×500, NPC 4/보행자 2 |
| 수집 VRAM | 감시 구간 **3,996~5,140 MiB**. 서버 시작/지도 로딩 비교 시험은 최대 약 7.4 GiB가 관측됨 |
| 원래 Town05 회귀 | 홈 SDV에서 **172.06초, 500/500**, exit 0, 엔진 생존. 원래 실패 기록의 208샘플 지점을 넘어 완료 |
| 실제 출력 감사 | RGB/Seg/Depth 100개씩, 400×300. Depth float32 미터값 finite. 같은 CARLA frame, 저장 stride 2. OD 27/100 파일에 실제 box. BEV/정책 관측 NPZ 100개씩 |
| 제어 모드 | manual/existing/autopilot 각각 **12/12**, exit 0. manual heartbeat 명령 56회 수락. Roach는 위 장시간·Town05 시험에서 검증 |
| 프리뷰 | 실제 400×300 JPEG. 요청 24FPS, 수신 약 24.04FPS, 출력 약 23.52FPS. view 적용 및 소유 Actor 정리 통과 |
| CUDA 학습 | 서로 다른 실제 Scene의 TRAIN 4/VAL 4프레임. ResNet18/FPN, Seg/Depth/OD/BEV, 128×128, batch 2, 기능 검증용 pretrained=False |
| 일시정지·재개 | Epoch 1 후 resume checkpoint 저장→재개 횟수 1→Epoch 3 완료, exit 0. 감시 VRAM 최고 약 2,244 MiB |
| 학습 중지 | 별도 100-epoch 설정 작업을 Epoch 1 이후 정상 중지, status Stopped/exit 130. 저장 결과 보존 |
| 재시작/종료 | workspace prepare, 홈 prepare, 홈 Test Engine, 마지막 홈 prepare/stop 확인. 선택 Town01/Snapshot Ready 후 SIGTERM 종료 143. 별도 OS 종료 없음 |
| 브라우저 | 실제 홈 start.sh로 실행. Diagnostics, 실제 500-frame Sequence, 타임라인 0→499에서 RGB/Seg/Depth/OD/BEV 5개 이미지 로딩 확인. 완료 CUDA 결과·CSV·checkpoint·예측 이력 확인 |
| 공식 사전학습 가중치 | 기본 ResNet-34 87,319,819 bytes를 공식 PyTorch에서 준비, SHA256 `b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f` 검증, Ready 확인 |
| 설치/중복/취소/삭제 보호 | 기존 배포 및 wheel 재사용 실제 확인. 다운로드·중복 시작·취소·port·PID 재사용·fixture 삭제 보호는 자동 검사 통과. 8GB 배포 재다운로드 및 사용자 원본 영구 삭제는 수행하지 않음 |

장시간 수집의 첫 저장~종료 요청: **2026-10-12 05:44:15.300~05:54:28.552 UTC**. 전체 job 시간은 약 620초이며 정상 중지 후 저장 샘플을 유지했다. 해당 결과를 `data/collections/validation_soak_dacc112eff5e48b0/`에 복사했고 원본 Codex 작업 폴더도 보존했다. 실제 Town05 및 제어 시험 데이터는 `data/collections/collection_<job id>/`에 있다.

진단·실행 명령·환경·센서 감사·GPU/수집/학습 감시·최종 테스트 로그는 `docs/validation/carla_stability_20261012/`에 보존했다. [장시간 결과](validation/carla_stability_20261012/soak-final.json), [Town05 결과](validation/carla_stability_20261012/town05-final.json), [CUDA 학습 결과](validation/carla_stability_20261012/training-final.json), [전체 테스트 로그](validation/carla_stability_20261012/final-all-828.log)를 참조한다.

## 남은 한계와 사용

1. Low 프로파일의 UE4 내부 메모리 결함은 남는다. 기본 Epic off-screen에서 검증한 범위의 안정화를 완료했으며 모든 GPU/지도/무기한 실행의 보장은 아니다.
2. 제공 PNG BEV 자산에는 정밀 미터 높이와 일부 sidewalk/junction 채널이 없다. 기존 validity/경고/N/A를 유지했다. 생성된 BEV를 완전한 고가도로 분리·높이 ground truth로 주장하지 않는다.
3. 학습 시험은 작은 실제 데이터의 기능 검사다. 모델의 수렴·일반화·Roach의 안전성/정책 우월성이나 12개 정밀 주행 시나리오 성능을 입증하지 않는다. 기존 강화학습 기능은 제품에서 지원하지 않는 상태를 유지했다.
4. 시스템 크래시 기록에 접근할 수 없는 환경에서는 진단의 stderr/미설치 상태를 표시한다. 이번 조사에서 없던 Saved/Crash 파일을 생성했다고 주장하지 않는다.

```bash
cd /home/wo/SDV/sdv_studio
./start.sh
```

웹: `http://127.0.0.1:8765`. 최종 기본은 기존 `/home/wo/carla`, 연구 Python `aa`, Epic off-screen이다. Settings → System Diagnostics에서 환경 점검/최소 기동/로그/내보내기를 사용할 수 있다. 실제 수집은 Simulation의 설치 후 실행 → 준비 검사 → 수집 시작으로 진행한다. 최종 웹 서버는 준비 상태로 유지하고 소유 CARLA·학습·프리뷰 프로세스는 정리한다. 검증용 4+4프레임 fixture는 남기되 기본 학습 경로/설정은 원래 값으로 복원한다.
