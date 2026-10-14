# 개발 검사

[English](development.md) · [소개](../README_ko.md) · [저장소 구조](REPOSITORY_LAYOUT.md)

별도 검사 환경을 사용합니다. 수집·학습 의존성은 [환경 준비](SETUP_KO.md)를 참고하세요.

```bash
python -m venv .venv-dev
.venv-dev/bin/python -m pip install -r requirements/dev.txt
.venv-dev/bin/python -m pytest -q
bash -n start.sh start-desktop.sh
.venv-dev/bin/python -m compileall -q sdv studio train_model
.venv-dev/bin/python -m studio --help
.venv-dev/bin/python -m sdv.collection.runner --help
.venv-dev/bin/python scripts/check_web.py
```

다른 시스템의 pytest 플러그인이 검사 Python에 개입하면 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`을 지정합니다. 프로세스·API 검사는 localhost 소켓과 subprocess 통신이 가능한 호스트에서 실행합니다.

PyTorch, OpenCV, 원본 지도 등 선택 의존성·자산이 필요한 검사는 웹 전용 환경에서 skip할 수 있습니다. 해당 연구 환경과 원본 자산으로 이 경로를 검사하세요. 문법·API·CPU 검사는 각각의 범위를 확인하며 CARLA·CUDA 실행, 주행 안정성, BEV 기하와 모델 수렴은 별도 평가가 필요합니다.
