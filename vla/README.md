# 真正的 π0.5 多任务仿真入口

本目录使用官方预训练 `pi05_libero`。当前这轮是推理和集成，未重新训练。动作由 π0.5 的 `sample_actions` 生成；ASR/Qwen 翻译负责输入语言，不规划机械臂轨迹。原来的苹果/香蕉程序留作基线，VLA 使用与权重匹配的 LIBERO 场景。

## 输入和执行

- 原生 MuJoCo / robosuite 仿真机械臂是 Franka Panda；π0.5 是策略模型。
- 两路 RGB（外部相机、腕部相机）按官方旋转 180°、缩放补边到 224×224，状态为位置3 + 旋转轴角3 + 夹爪2。
- 英文指令直接进入策略，没有按物品名选择手写抓放程序。模型每次输出 10×7 动作；默认执行前5步后重新观察。
- 标准 LIBERO 任务有各自成功判据和步数上限；手改指令不会改判据，因此自定义指令下的原任务 success 不能代替新指令评分。
- 录音经本地 Qwen ASR、Qwen 翻译生成可编辑英文，核对后点击 Start。说错、翻译错误和训练范围外任务均可能失败。

## 独立安装

需要 Ubuntu 图形桌面、Python3.11、uv、EGL/OpenGL；语音需要 Tk/ALSA `arecord`。首次下载官方权重约12.44GB。源码、环境、模型均放在本项目，原来的 `.pi05-handson` 不被修改。

```bash
cd /home/robot/voice-arm-sim
rtk proxy python3 -m vla.prepare_source
rtk proxy env GIT_LFS_SKIP_SMUDGE=1 UV_PROJECT_ENVIRONMENT="$PWD/.venv-vla" UV_HTTP_TIMEOUT=120 uv sync --frozen --project "$PWD/.local/openpi" --python 3.11.15
rtk proxy uv venv --python 3.11.15 .venv-vla-sim
rtk proxy uv pip install --python .venv-vla-sim/bin/python -r vla/requirements-sim.txt
rtk proxy .venv-vla/bin/python -u -m vla.fetch_checkpoint
```

下载脚本固定每个 GCS 对象 generation，分段恢复下载，核验官方 CRC32C 并记录 SHA256；最终清单在 `models/pi05_libero/manifest.json`。首次在线准备失败可保留已下载分段重跑。`requirements-sim.txt` 是本机已验证版本快照；官方旧 Python3.8 requirements 中的 NumPy1.22/Torch1.11 不适用这个 Python3.11 环境。

## 启动

第一个终端启动 VLA 服务：

```bash
cd /home/robot/voice-arm-sim
rtk proxy .venv-vla/bin/python -u -m vla.serve
```

第二个终端打开原生窗口与面板（默认端口8766）：

```bash
cd /home/robot/voice-arm-sim
rtk proxy env __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia .venv-vla-sim/bin/python -u -m vla.run --suite libero_goal --task-id 8 --panel --viewer --output runs/vla-live
```

选择任务会加载对应场景和默认指令。Start 每次从固定初态重新运行；可编辑指令，Pause/Stop 在当前有界请求返回后生效。左键旋转、右键平移、滚轮缩放。每轮结果写到时间戳子目录，不覆盖旧记录。

语音另需独立模型服务，已有8911服务时不要重复启动；修改服务代码后需先结束这个项目自己的旧服务，再启动新版：

```bash
rtk proxy .venv-models/bin/python -u model_service.py
```

语音处理只填文字，不自动运行。真人麦克风信号和开放场景能力需要另行验收。

## 可复现评测

```bash
rtk proxy .venv-vla-sim/bin/python -u -m vla.run --suite libero_goal --task-id 8 --initial-state 0 --output runs/vla-goal8-init0
rtk proxy .venv-vla-sim/bin/python -u -m vla.probe_language --output runs/vla-language-probe
```

每轮保存原始指令、任务判据、初态数组及hash、服务权重身份、动作窗口、逐步执行动作、推理耗时和双视角视频。`probe_language` 固定同一观察和同一扩散噪声，重复相同指令，再换指令；只检验语言影响动作，不能证明完整任务成功。[实测报告](../evidence/vla/README.md)与[三任务 Demo](../evidence/vla/demo.mp4)已保存：5次预定执行均通过所选任务判据，包含中文合成语音经 ASR/翻译后直接交给 π0.5 的放碗任务。任务判据可能在完整松爪、撤离前触发；这是小样本集成验证，不是开放场景成功率。

视频按每个控制步一帧、10fps保存，播放时长与机器执行耗时不同。原生窗口的自由视角不改变输入模型的固定外部/腕部相机。窗口隐藏了彩色碰撞网格；这是显示设置，不改变物理接触和模型动作。双显卡桌面若默认使用 llvmpipe，上述 NVIDIA 环境变量用于选择硬件渲染。

## 范围与来源

这套预训练权重并不保证拿走香蕉、任意家庭物品、多轮对话、自动学习或真机迁移。新场景和新机械臂需要验证摄像头/状态/动作契约，必要时采集示范微调。恢复能力也需要专门错误初态评测。

- [官方 openpi](https://github.com/Physical-Intelligence/openpi/tree/15a9616a00943ada6c20a0f158e3adb39df2ccac)，固定源码 `15a9616a00943ada6c20a0f158e3adb39df2ccac`。
- [官方 LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO/tree/f78abd68ee283de9f9be3c8f7e2a9ad60246e95c)，固定源码 `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`。
- [官方 LIBERO 接入说明](https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/examples/libero/README.md)。
- 权重源 `gs://openpi-assets/checkpoints/pi05_libero`；openpi代码许可与 [Gemma使用条款](https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/LICENSE_GEMMA.txt)应分别检查，不能把所有权重当成仅受代码Apache2许可。
