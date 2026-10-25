# 中文指令执行、完成核对与现场连续使用（2026-10-25）

## 问题与修改

原“开始／继续 VLA”只在当前 prompt 等于官方任务语言且官方谓词成立时自动停；自定义中文目标没有对应完成谓词，上限 0 时可能一直预测动作。原“视觉理解抓放”是另一个有限抓放入口，原先仅记录流程结束，未验证指令是否完成。两个按钮不是先后两步。

现在日常入口统一为“执行当前指令”：当前图像和中文 → Qwen3-VL 选择源物体／目的地 → π0.5 抓取 → RGB-D/OSC 搬运、松爪和撤手 → 两路最新图像核对原始中文 → 停止并保留现场。改写指令、再次点击同一按钮即可；不加载或重置场景。原连续 VLA 入口改名“仅 VLA 调试”，单独说明自定义目标需手动暂停。

完成核对有 completed / not_completed / uncertain 三类；夹爪未打开不报完成，图像哈希不匹配直接失败；核对期间不发送控制动作。每条新指令先清空上条完成结果；正常结束、失败和暂停均清除动作队列、回到空闲。失败不会被包装成完成，也没有自动无限重试。

这是视觉语言模型 + π0.5 + 几何控制的混合方案，不是 π0.5 自己输出停止信号。视觉判断可误判，不能当物理真值。抓取仍有 180 步候选检测上限，各段运动有收敛保护；这些用于失败停机，不能证明完成。单次支持抓放，复杂容器、遮挡、从堆叠物中取出仍有能力限制。

## 对照开源实现

以下为 2026-10-25 阅读到的文件版本，blob SHA 标识文件内容，不是整个仓库提交。

- [Vector OS Nano executor](https://github.com/VectorRobotics/vector-os-nano/blob/master/vector_os_nano/core/executor.py)，blob `af2103903aa0aa15563fa19f09969f07ccf0a04f`：按依赖执行技能，检查前置和后置条件，再返回任务结果。注意它先应用技能 world-model effects，后置条件不等于独立的视觉证明。
- [Vector OS Nano place](https://github.com/VectorRobotics/vector-os-nano/blob/master/vector_os_nano/skills/place.py)，blob `890a114530bd52d526e5658f405822890afe96f3`：执行放置、打开夹爪、抬起、回位，返回 SkillResult；空夹爪不能单独证明放到了正确目的地。
- [同济子豪兄 VLM Arm](https://github.com/TommyZihao/vlm_arm/blob/main/agent_demo_20250328/utils_vlm_move.py)，blob `ceea641815c013a1a15aff8dfc0d301e54a7a51d`：拍顶视图、VLM 定位、标定转换、运行 pump_move，流程返回；未在这段代码看到放完后以新图像独立核对。机械臂回零不等于重置物体场景。
- [Wheelarm Lab LLM 节点](https://github.com/renyimen2026/wheelarm-lab/blob/main/wheel_arm_navigation/wheel_arm_navigation/llm.py)，blob `7261e6ac27f9819ae3dc5b1a32819576289ac4b8`：这版解析限定名称的导航目标，并明确不处理抓取、视觉、机械臂动作；该文件不能作为抓取完成判断的实现依据。

借鉴的是“任务 → 技能 → 结果检查 → 返回等待下一指令”。本次额外加当前图像核对，没有照抄固定动作后直接宣布成功。

## 验证

- 回归：`MUJOCO_GL=egl .venv-vla-sim/bin/python -m pytest -q tests --ignore=tests/test_vla_artifacts.py`：148 passed，18 subtests passed。覆盖完成/未完成/不确定均回到空闲、同一环境接受下一指令、构造失败不沿用上次成功、夹爪门和旧图像拒绝。
- 相关文件 Ruff E4/E7/E9/F 通过。
- 不加路径限制的 pytest 会误收集下载的第三方 LeRobot 测试并缺少 serial；本仓库 artifacts 模块在仿真环境缺 google_crc32c。因此没有声称全仓库所有测试通过，没有为无关测试改环境。
- 实际本机 π0.5 + Qwen3-VL + LIBERO：`bowl-plate/result.json`，246 个控制步；视觉核对 completed；独立官方 bowl-on-plate 谓词也为 true。相机图和实际 demo 随附。一次成功不是成功率。
- 当前桌面 GUI 已替换；`gui-verification.json`：最新保存场景逐项精确恢复、ignore_done=true、automatic_motion_steps=0。`panel.png` 展示新按钮，面板保持暂停。
- 连续真实指令见 `sequence.json`：第一条碗放盘子视觉核对通过；第二条酒瓶放架子从第一条停止状态启动，起始状态 SHA 与第一条终止状态一致，同一环境且未 reset。第二条在 180 步内未形成抓取候选，保护停止，记录 exception 而非成功。两条结束后各观察一秒，仿真状态保持不变。因此验证了连续接令、停机和保留现场，没有验证连续两次抓放都成功。
- 错误目的地测试 `wrong-destination.json`：对碗已经在盘子上的两张图询问“把碗放到架子上”，模型返回 not_completed，标签正确，但理由“碗仍悬空”与实际场景不符。这说明视觉核对仍会编造理由，不能将它作为可靠的通用完成判据。
- 第二条失败的完整结果及动作记录保留在 `sequence-failure/`，可用于后续抓取可靠性排查；本次没有用固定重试动作掩盖失败。

## 复现

模型服务：`.venv-models/bin/python -u model_service.py --device cpu --vision-only --port 8912`，策略端口 8766 需已有服务。实际视觉推理耗时取决于设备；界面提示 CPU 可能需数分钟，期间机械臂停止。

单条完整动作：`MUJOCO_GL=egl .venv-vla-sim/bin/python -u -m vla.semantic_grasp --prompt 把碗放到盘子上 --output runs/your-new-output --initial-state 0 --score bowl-on-plate`。

连续指令：`MUJOCO_GL=egl .venv-vla-sim/bin/python -u evidence/task-completion-20261025/verify_sequence.py`。首次初始化新仿真，与用户桌面环境隔离；两条指令之间不 reset。输出路径必须为空。

`restore_panel.py` 是本次场景保留部署脚本，会恢复本目录快照，且不会自动运动；它不是日常启动时必须运行的脚本。日常直接使用已打开面板。

## 备份范围

能力代码在本仓库的 model_service.py、vla/semantic.py、vla/semantic_grasp.py、vla/workbench.py、vla/panel.py。π0.5 权重与训练代码本次未改，未将权重加入 Git。分支 feature/visual-semantic-grasp 推送至用户拥有的 sencangWei/voice-arm-sim；远端独立恢复后的逐文件内容比对记录见 backup-verification.json。
