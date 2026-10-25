# 暂停后保留场景无法继续：修复记录（2026-10-25）

用户现象：执行一段时间后暂停，不重新布置场景就反复提示“操作失败”。

## 根因与修复

- 两次失败场景均记录了 990 个控制步；加上加载场景的 10 个稳定步，触发 robosuite 默认累计 `horizon=1000`。
- LIBERO 的返回 `done` 使用任务成功条件，而内部 robosuite 已经将 episode 标为结束；后续 `step()` 抛出 `ValueError: executing action in terminated episode`。暂停没有清除这个累计终止状态，重新布置才会清除。
- 交互面板创建环境时现传入 `ignore_done=True`，由面板自己的每轮步数上限、任务判断和暂停控制执行时长。
- 用户主动暂停产生的 `InterruptedError` 现显示“已暂停”，真正的异常仍报告失败。
- 修改位置：`vla/workbench.py`；回归测试：`tests/test_vla_workbench.py`。不修改权重或训练数据，独立有界评测入口保持原有行为。

## 验证

- 修复前新增测试真实失败：缺少 `ignore_done`；暂停误显示失败。修复后 49 项测试、13 个 subtests 通过。
- `real-sim-verification.json`：真实 LIBERO / MuJoCo 中先用旧设置复现第 1000 步终止；同一环境不重置物理状态，修正终止设置后运行到第 1102 步，并完成一次本机 π0.5 推理动作。暂停、继续时状态逐元素相同。多数步骤使用零位移动作，验证的是生命周期，不代表抓放任务成功率。
- `gui-verification.json`：从旧面板提取原始物理状态，修复版恢复后逐元素相同，自动执行动作数为 0。截图见 `fixed-panel.png`。
- 提取状态时使用一次零动作草稿；该草稿已移到本次诊断临时目录，没有加入人工纠错训练数据。

回归命令（仓库根目录）：

```bash
MUJOCO_GL=egl .venv-vla-sim/bin/python -m pytest tests/test_semantic.py tests/test_vla_workbench.py tests/test_vla_voice.py tests/test_hybrid.py -q
.venv-models/bin/ruff check --isolated --select E4,E7,E9,F vla/workbench.py tests/test_vla_workbench.py evidence/pause-resume-20261025/restore_panel.py
```

## 当前面板

已打开“中文视觉抓放 · 修复版（保留当前位置）”，保留用户的酒瓶、盘子和机械臂位置，保持暂停。输入指令后用“视觉理解抓放”继续。

`retained-scene-state.npy` 与 `retained-scene.json` 保存本次现场。需要再次恢复此固定快照时，在配置好本机依赖、LIBERO 数据和模型服务 8766 后运行：

```bash
DISPLAY=:0 MUJOCO_GL=glfw __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia .venv-vla-sim/bin/python -u evidence/pause-resume-20261025/restore_panel.py
```

此命令仅加载快照，不自动运行策略。正常后续使用原面板入口也会应用本次环境修复。
