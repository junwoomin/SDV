# Development checks

[한국어](development_ko.md) · [Overview](../README.md) · [Repository structure](REPOSITORY_LAYOUT.md)

Use a separate test environment; collection/training dependencies are described in [setup](SETUP.md).

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

Use `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` if unrelated system pytest plugins interfere with the test interpreter. Process/API tests need localhost sockets and subprocess communication.

Tests needing PyTorch, OpenCV, native maps or other optional assets may skip in the web-only environment. Use the corresponding research environment and original assets to exercise those paths. Syntax, API and CPU tests cover their stated boundaries; evaluate CARLA/CUDA execution, driving stability, BEV geometry and model convergence separately.
