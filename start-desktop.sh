#!/usr/bin/env bash
set -euo pipefail
sdv_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$sdv_root"
if [[ ! -x .venv/bin/python ]]; then
  echo '프로젝트 Python 환경이 없습니다. README_ko.md의 로컬 환경 준비를 확인하세요.' >&2
  exit 1
fi
exec systemd-run --user --scope --quiet -p CPUQuota=100% -p MemoryMax=2G -p MemorySwapMax=0 nice -n 10 .venv/bin/python -B start.py "$@"
