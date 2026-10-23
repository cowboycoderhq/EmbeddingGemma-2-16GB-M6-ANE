#!/bin/bash
# Make the virtual environments.
#   scripts/setup.sh           .venv        runtime only (run the model)
#   scripts/setup.sh --build   .venv-build  runtime + converter + tests (build the model yourself)
# With uv, Python 3.13 is downloaded if needed. Without uv, set PYTHON to a Python 3.11-3.13.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "${1:-}" = "--build" ]; then V=.venv-build; R=requirements-build.txt; else V=.venv; R=requirements.txt; fi
if command -v uv >/dev/null && [ -z "${PYTHON:-}" ]; then
  uv venv -q -p 3.13 "$V"
  uv pip install -q -p "$V/bin/python" -r "$R"
  [ "$V" = .venv-build ] && uv pip install -q -p "$V/bin/python" --no-deps coreai-opt==0.2.1
else
  PY=${PYTHON:-python3.13}
  "$PY" -c 'import sys; assert (3, 11) <= sys.version_info[:2] <= (3, 13), sys.version' \
    || { echo "Python 3.11-3.13 is needed: install uv, or set PYTHON=/path/to/python3.13"; exit 1; }
  "$PY" -m venv "$V"
  "$V/bin/python" -m pip install -q --upgrade pip
  "$V/bin/python" -m pip install -q -r "$R"
  [ "$V" = .venv-build ] && "$V/bin/python" -m pip install -q --no-deps coreai-opt==0.2.1
fi
"$V/bin/python" -c 'from coreai.runtime import AIModel; import tokenizers, numpy; print("runtime ok")'
echo "ready: $V"
