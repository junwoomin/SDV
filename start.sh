#!/usr/bin/env bash
set -euo pipefail
sdv_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$sdv_root"
sdv_python="${SDV_WEB_PYTHON:-python}"
# Reuse the prepared web runtime unless the user explicitly selects another.
if [[ -z "${SDV_WEB_PYTHON:-}" && -x "$sdv_root/.venv-web/bin/python" ]]; then
  sdv_python="$sdv_root/.venv-web/bin/python"
fi
# Prefer the active conda/venv interpreter (for example the user's aa env).
if [[ -z "${SDV_WEB_PYTHON:-}" ]] && ! command -v "$sdv_python" >/dev/null 2>&1; then
  sdv_python=python3
fi
if ! command -v "$sdv_python" >/dev/null 2>&1; then
  echo "Python을 찾을 수 없습니다. Python 3.7 이상을 설치하거나 SDV_WEB_PYTHON을 지정하세요." >&2
  exit 1
fi
"$sdv_python" -c 'import sys; sys.exit(0 if sys.version_info >= (3,7) else "웹 서버에는 Python 3.7 이상이 필요합니다.")'
# Existing CARLA/training processes belong to their original owner.
# The web supervisor manages only engines launched by this application.
sdv_python_version="$("$sdv_python" -c 'import sys; print("{}.{}".format(*sys.version_info[:2]))')"
sdv_web_env="$sdv_root/.venv-web"
# Keep an existing environment intact when another Python version is selected.
if [[ -e "$sdv_web_env" ]]; then
  sdv_existing_version="$("$sdv_web_env/bin/python" -c 'import sys; print("{}.{}".format(*sys.version_info[:2]))' 2>/dev/null || true)"
  if [[ "$sdv_existing_version" != "$sdv_python_version" ]]; then
    sdv_web_env="$sdv_root/.venv-web-py${sdv_python_version//./}"
  fi
fi
if [[ ! -x "$sdv_web_env/bin/python" ]]; then
  echo "웹 관리용 가상환경을 준비합니다 (Python $sdv_python_version, $(basename "$sdv_web_env"))."
  "$sdv_python" -m venv "$sdv_web_env" || { echo '가상환경 생성 실패: 선택한 Python의 venv/ensurepip 설치 여부를 확인하세요.' >&2; exit 1; }
fi
sdv_web_python="$sdv_web_env/bin/python"
sdv_dependencies_key="$("$sdv_web_python" -c 'import hashlib, sys; print("{}.{}:{}".format(*sys.version_info[:2], hashlib.sha256(open("requirements/web.txt", "rb").read()).hexdigest()))')"
sdv_installed_key="$(cat "$sdv_web_env/.sdv-requirements" 2>/dev/null || true)"
if [[ "$sdv_installed_key" != "$sdv_dependencies_key" ]] || ! "$sdv_web_python" -c 'import fastapi, uvicorn, psutil, yaml, numpy; from PIL import Image' >/dev/null 2>&1; then
  echo '웹 관리용 의존성을 설치합니다. 기존 CUDA/CARLA 환경은 설정에서 연결하세요.'
  # pip 24.0 is the last pip for Python 3.7; upgrade only inside the web venv.
  "$sdv_web_python" -m pip install --upgrade 'pip>=23.2,<24.1; python_version < "3.8"' 'pip>=23.2; python_version >= "3.8"' || { echo '웹 가상환경의 pip 준비 실패: 네트워크 또는 pip 설정을 확인하세요.' >&2; exit 1; }
  "$sdv_web_python" -m pip install -r requirements/web.txt || { echo '의존성 설치 실패: 네트워크 또는 pip 설정을 확인하세요.' >&2; exit 1; }
  printf '%s\n' "$sdv_dependencies_key" > "$sdv_web_env/.sdv-requirements"
fi
echo "SDV Studio · Python $sdv_python_version · $(basename "$sdv_web_env")"
if [[ -z "${SDV_RESEARCH_PYTHON:-}" ]]; then
  if [[ -x "$sdv_root/.venv/bin/python" ]]; then
    export SDV_RESEARCH_PYTHON="$sdv_root/.venv/bin/python"
  else
    export SDV_RESEARCH_PYTHON="$(command -v "$sdv_python")"
  fi
fi
export PYTHONUNBUFFERED=1
exec "$sdv_web_python" -m studio "$@"
