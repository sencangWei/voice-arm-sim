# 真正的 π0.5 多任务仿真入口

本目录使用官方预训练 `pi05_libero`。当前这轮是推理和集成，未重新训练。动作由 π0.5 的 `sample_actions` 生成；ASR/Qwen 翻译负责输入语言，不规划机械臂轨迹。原来的苹果/香蕉程序留作基线，VLA 使用与权重匹配的 LIBERO 场景。

## 输入和执行

- 原生 MuJoCo / robosuite 仿真机械臂是 Franka Panda；π0.5 是策略模型。
- 两路 RGB（外部相机、腕部相机）按官方旋转 180°、缩放补边到 224×224，状态为位置3 + 旋转轴角3 + 夹爪2。
- 面板默认中文：任务菜单、可编辑指令和语音识别原话均用中文。中文在后台经本地 Qwen 翻译后进入策略，英文命令行输入仍直接进入策略；没有按物品名选择手写抓放程序。模型每次输出 10×7 动作；默认执行前5步后重新观察。
- 标准 LIBERO 任务有各自成功判据和步数上限；手改指令不会改判据，因此自定义指令下的原任务 success 不能代替新指令评分。
- 录音经本地 Qwen ASR，面板保留可编辑的中文原话；核对后点击“开始／继续模型”，翻译在后台完成。翻译失败会停止本次执行，不沿用旧指令。说错、翻译错误和训练范围外任务均可能失败。

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

选择任务后点击“加载／重置场景”，从指定初态加载并暂停。“开始／继续模型”使用上方最新指令继续当前场景；“暂停模型”在当前有界请求返回后丢弃尚未执行的动作。需要从头运行时重新加载场景。左键旋转、右键平移、滚轮缩放。每轮结果写到时间戳子目录，不覆盖旧记录。

## 中文工作台：人工纠正和微调

面板分为“仿真与语音／人工纠正／模型与训练／结果与录像”四页，顶部始终显示两路模型相机。复用了旧 `.pi05-handson` 的鼠标摇杆实现（仅增加方向标识和调整说明文字），旧项目和旧训练数据均未修改。当前桌面的字体模糊源于 uv Python 所带 Tk 9 只暴露位图字体；`tk_runtime.py` 在 ABI 匹配时优先使用系统 Tk 8.6 的 `_tkinter`，让 Noto 中文字体正常抗锯齿显示，不替换系统或虚拟环境文件。

1. 加载场景并运行模型；需要纠正时打开“人工纠正”，选择训练集或独立布局留出集，点击“开始人工接管”。
2. 摇杆控制世界坐标 XY，按钮控制 Z 和姿态，松手停止。普通幅度 0.6 用于搬运，精细幅度 0.1 用于抓放；连续步数是重复执行同一动作的控制步数。接近、接触、松爪是否适合这些幅度仍需逐场景检查。
3. 点击“保存为合格片段”代表人工审核整段可用于学习；否则保存草稿。暂停等待不记录动作。每一帧保存实际执行前的两路图像／状态、实际动作以及执行后状态，并逐帧校验 SHA256。人工合格和任务成功判据分开记录。
4. “模型与训练”中选择父模型。默认是本项目官方 `pi05_libero`，也可填写本项目完成的 `checkpoints/voice_pi05_lora/<实验>/<数字步数>`。它与旧项目 `pi05_base` 黑碗微调不是同一模型，不能直接混用权重或数据。
5. “准备／校验数据”只收录本项目 `corrections/` 下审核合格的片段；草稿排除，留出集不参与训练，同一布局不得跨组。至少 2 条训练片段和 1 条独立布局留出片段是流程门槛，不是足够学好的数据量。
6. 先点“训练前检查”，查看官方输入转换及数据身份。再关闭三维场景、停止本面板的模型以释放显存，点击“开始微调”。其它终端启动的服务需要在其原终端管理，本面板不会终止它们。日志在本页和 `logs/panel-training-*.log`。关闭面板会停止本面板启动的模型和训练，训练中应先等待完成再关闭。

训练入口是真实官方 JAX 训练器：π0.5 LoRA、动作窗口10、默认 batch1、更新2000次、学习率5e-5、种子42，保留父模型 LIBERO 归一化参数。窗口在片段末尾重复最后一个动作，绝不跨片段；只在合格训练帧上均匀抽样，不加速视频、不静默缩小或放大动作、不自动回放旧数据。默认参数用于首轮试验，不保证成功率或恢复能力。父模型接续仅加载模型参数，新建优化器，不声称恢复旧优化器状态。

本轮验证：15 项定向测试通过；原生窗口人工执行10步、等待不增帧、存草稿后交还真实模型，模型80步／16次预测达到所选任务判据。CPU 训练前检查使用隔离的合成契约样本，通过官方转换得到10×32（模型内部补齐）；没有用这些样本训练，没有生成或验证新的 LoRA 权重，也未验收真人麦克风。界面截图及记录见 [工作台验收](../evidence/vla/workbench.md)。

中文打字与语音均需独立模型服务（任务菜单翻译表只供显示，不选择动作）。已有8911服务时不要重复启动；修改服务代码后需先结束这个项目自己的旧服务，再启动新版：

```bash
rtk proxy .venv-models/bin/python -u model_service.py
```

语音处理只填中文文字，不自动运行。现场输入“帮我把碗放到盘子上”经真实本地翻译及 π0.5 执行，81个控制步／17次预测达到任务判据；20项定向测试与 Ruff 通过。记录同时保留 `user_instruction` 中文原话与 `task` 模型内部译文。见[中文指令验收](../evidence/vla/chinese-command.md)。真人麦克风信号和开放场景能力需要另行验收。

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


## HAMSTER 中文分层实验（固定桌面）

在“结果与录像”页底部，点击“酒瓶放到碗里”或“碗放到盘子上”复现已审核路径。也可在顶部输入自己的中文单次抓放指令，点击“按上方中文指令重新规划并执行”；它会拍摄固定初始场景，运行真实 HAMSTER，再打开独立 MuJoCo 窗口执行。录音先经现有麦克风按钮填写中文，核对原话后使用这个新指令按钮。该实验固定布局0，不读取顶部的布局设置；运行前先暂停原场景、保存人工片段。

规划在 CPU 上运行，模型已缓存时本次文字端到端约64秒，窗口可以旋转视角。抓取来自实际官方 π0.5；后半程是基于深度的通用几何控制，未训练模型，也不是 HAMSTER 论文的学习型底层策略。新指令只报告运动是否完成，不使用原任务判据冒充中文任务成功；必须看录像检查物品、目的地、松爪。完整边界、失败和对比见[物理报告](../evidence/hamster/physical-hybrid/REPORT.md)。

命令行入口（现有8911和8766服务继续使用；每次换新的输出目录）：

```bash
rtk proxy .venv-vla-sim/bin/python -u -m vla.hamster_command --prompt '帮我把酒瓶放进碗里面' --output runs/hamster-my-command-01 --viewer
# 实际 WAV → ASR → 翻译 → HAMSTER → 物理仿真
rtk proxy .venv-vla-sim/bin/python -u -m vla.hamster_command --audio /绝对路径/中文录音.wav --output runs/hamster-my-voice-01 --viewer
```

需要按 [HAMSTER 环境说明](../evidence/hamster/README.md)先准备独立 CPU 规划器。语音合成验收工具使用已有 `.venv-tts/bin/python`，不使用 `.venv-models`；真人麦克风端到端尚未验收。
