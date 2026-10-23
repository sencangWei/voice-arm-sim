#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -x .venv-models/bin/python ]]; then
    uv venv --python 3.11.15 .venv-models
fi
uv pip install --python .venv-models/bin/python --index-strategy unsafe-best-match -r requirements-models.txt
.venv-models/bin/python model_service.py --download-only
