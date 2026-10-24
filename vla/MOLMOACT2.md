# MolmoAct2 本地 MuJoCo 接入

使用官方 `allenai/MolmoAct2-LIBERO` 预训练模型，与原有 π0.5 共用原生中文工作台、MuJoCo 场景和相机。此入口进行预训练推理，没有训练新的 MolmoAct2 权重。实际结果与录像见 [测试记录](../evidence/molmoact2/README.md)。

## 独立安装

先按 [原有环境准备](README.md)准备固定版本的 openpi、LIBERO 和 `.venv-vla-sim`。MolmoAct2 使用独立环境，避免替换 π0.5 的 JAX 或语音服务依赖：

```bash
cd /home/robot/voice-arm-sim
rtk proxy uv venv --python 3.11 .venv-molmo
rtk proxy uv pip install --python .venv-molmo/bin/python -r requirements-molmo.txt
rtk proxy .venv-molmo/bin/python -m scripts.prepare_molmo
```

模型固定到 `0d24a92bd1faf321ef497c3bbd5681af97c65aa2`。下载约 22GB；准备脚本检查官方文件长度和 LFS SHA256，并保存完整文件清单。权重、虚拟环境与缓存不提交到 Git。若文件已下载，可用 `--verify-only` 重新校验。

## 服务与中文窗口

```bash
rtk proxy .venv-molmo/bin/python -u -m vla.molmo --port 8777 --placement vision-cpu
```

`vision-cpu` 将视觉计算、词嵌入查表和语言输出头放在 CPU，语言骨干的其余部分和动作专家放在 GPU。使用 BF16，不量化权重，关闭 CUDA Graph；每次返回动作前释放不再使用的 CUDA 缓存，保留有效模型张量，减少共享显卡的闲置缓存。CPU 查表避免上游每次在 GPU 合并约 756MiB 词表；输入和注意力掩码仍明确放在 GPU。CPU 视觉会影响速度；这不是上游全 GPU 的性能基准。`--placement cuda` 可用于显存足够的独立实验，不要因此停止别人的 GPU 进程。

另一个终端打开中文面板和可拖动的原生三维窗口：

```bash
rtk proxy env __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
  .venv-vla-sim/bin/python -u -m vla.run \
  --suite libero_goal --task-id 8 --initial-state 0 --seed 42 \
  --port 8777 --replan-steps 5 --max-steps 300 \
  --panel --viewer --output runs/molmo-live
```

点“加载／重置场景”，再点“开始／继续模型”。中文输入和麦克风仍使用本地 8911 ASR／翻译服务。翻译只转换语言，不把物体名称映射为预写动作；模型接收实际两路图像、状态和译文。窗口的成功显示仍是当前所选 LIBERO 任务判据；改了目的地后，应使用下面的独立评测，不能拿放碗的判据给放酒瓶评分。

工作台可以人工接管、录像和采集纠错；采集格式维持原来的 π0.5 224 像素契约。“模型与训练”页目前仅微调 π0.5，不能用该页直接训练 MolmoAct2。

## 同场景比较

保留 π0.5 服务 8766，MolmoAct2 使用 8777。分别运行两条指令，每个模型使用新的输出目录：

```bash
rtk proxy .venv-vla-sim/bin/python -u -m vla.compare \
  --port 8777 --case bowl-on-plate --output runs/molmo-bowl-new
rtk proxy .venv-vla-sim/bin/python -u -m vla.compare \
  --port 8777 --case bottle-in-bowl --output runs/molmo-bottle-new
```

换成 `--port 8766` 即测试 π0.5。场景、物理初态、环境种子 42、每次执行 5 步和最多 300 个控制步相同。两个模型的动作噪声实现不同，不能声称逐请求随机数完全相同。每条视频以控制步为帧、10fps 保存，播放时长不是模型墙钟耗时。

放碗使用原始任务判据。酒瓶放碗使用单独的近似物理关系检查：瓶碗接触、XY 中心距离小于 3cm、Z 中心差小于 8cm、夹爪不接触瓶子，连续满足 5 步；必须配合录像确认。评分可以读取仿真真值，但不向动作模型提供目标坐标。这两项小样本不能证明任意家庭物体、中文语音准确率、长期恢复能力或产品可靠性。

## 模型契约与来源

- [官方权重与推理说明](https://huggingface.co/allenai/MolmoAct2-LIBERO/tree/0d24a92bd1faf321ef497c3bbd5681af97c65aa2)：执行仓库内固定版本的自定义模型代码。
- [官方源码](https://github.com/allenai/molmoact2)；接口对照 LeRobot `molmoact2-hf-inference` 分支提交 `28b4f721f931aab554cb176412223f098023705f`。
- 外部／腕部 RGB 分别旋转 180°、256×256，状态为末端位置 3、轴角 3、夹爪位置 2。官方连续动作接口返回已反归一化的 10×7 窗口，本地不再缩放动作或翻转夹爪符号。
- 连续动作流匹配 10 次迭代，不启用深度推理。默认随机生成器初始种子 0，随请求推进；诊断时可用 `_probe_noise_seed` 固定一次请求的动作噪声。
- CPU 视觉前置钩子同时迁移图像和图像块索引，输出由官方模型迁回语言骨干所在设备。没有改官方权重。
- CPU 词嵌入使用查表前后钩子迁移输入／输出；覆盖固定版本的输入迁移辅助方法，避免 Hugging Face 根据首个 CPU 参数把注意力掩码留在 CPU。该部署适配仅支持上面固定的连续动作接口，不宣称支持通用文本生成或深度推理。
