#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
export MUJOCO_GL="${MUJOCO_GL:-egl}"
if [[ ! -x .venv/bin/python ]]; then
    echo "缺少本项目 .venv，请先完成独立环境安装。" >&2
    exit 1
fi
if [[ "${START_MODELS:-1}" == 1 && -x .venv-models/bin/python ]]; then
    if ! .venv/bin/python -c 'from urllib.request import urlopen; urlopen("http://127.0.0.1:8911/health",timeout=2)' >/dev/null 2>&1; then
        nohup .venv-models/bin/python -u model_service.py >>logs/models.log 2>&1 &
        echo "模型服务启动中，日志：logs/models.log"
    fi
    .venv/bin/python - <<'PY'
import json, time
from urllib.request import urlopen
deadline = time.monotonic() + 180
while time.monotonic() < deadline:
    try:
        with urlopen("http://127.0.0.1:8911/health", timeout=2) as response:
            health = json.load(response)
        if health.get("ready"):
            print("本地语音与视觉模型已就绪")
            break
        if health.get("error"):
            print("模型服务加载失败，将启动文字与颜色 RGB-D 基线：", health["error"])
            break
    except OSError:
        pass
    time.sleep(2)
else:
    print("模型服务仍未就绪，将启动文字与颜色 RGB-D 基线。")
PY
fi
echo "打开 http://127.0.0.1:8910 ；应用日志：logs/app.log"
exec .venv/bin/python -u app.py "$@" > >(tee -a logs/app.log) 2>&1
