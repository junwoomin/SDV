# 개발 검사

[English](development.md) · [소개](../README_ko.md) · [실행 경로](REPOSITORY_LAYOUT.md)

현재 구현은 `studio`와 `sdv`입니다. 과거 공개 소스의 가정은 [원래 개발 노트](legacy/development_original_KO.md)에 보존했습니다. 별도 연구 환경·자산은 [설치 안내](SETUP_KO.md)를 참고하세요.

`requirements/dev.txt`로 분리된 검사 환경을 사용합니다. 기존 준비된 환경은 연구 패키지를 바꾸지 않고 재사용할 수 있습니다.

```bash
python -m pytest -q
bash -n start.sh start-desktop.sh
python -m compileall -q sdv studio legacy train_model carla_data_gan
python -m studio --help
python -m sdv.collection.runner --help
```

ROS launch testing처럼 다른 시스템의 pytest 플러그인이 호환되지 않는 Python에 들어오면 이 저장소 검사 명령에만 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`을 지정합니다. 해당 외부 플러그인은 여기서 필요하지 않습니다. 프로세스·API 검사는 localhost 소켓과 subprocess 통신이 가능한 호스트에서 실행합니다.

원본 지도 통합 검사는 별도 `assets/bev_maps/catalog.json`과 정확한 지도 패키지가 필요합니다. 소스만 있는 복제본에서는 사유와 함께 skip합니다. 다른 선택적 검사도 원본 자산·의존성·비교 소스가 없으면 누락을 보고합니다. 실행하려면 실제 원본을 공급하며 가짜 자산으로 대신하지 않습니다. 이 후보의 검사 기록은 소스만 있는 상태와 원본 지도 상태를 구분합니다.

자동·CPU 검사만으로 실제 CARLA/CUDA 경로의 정합성을 주장하지 않습니다. [실행 기록](SCREENSHOT_EVIDENCE.md), [엔진 검증](CARLA_STABILITY_KO.md), [학습 검증](TRAINING_UI_FOLLOWUP_KO.md)은 별도 범위를 유지합니다. 문서 정리에 엔진 전체 다운로드·추가 학습·드라이버 변경·원격 반영은 포함하지 않습니다.
