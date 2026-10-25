# 连续指令与模型恢复闭环（2026-10-25）

**后续更正**：本页保留当时的成功和失败。随后已精确复现 `panel-v2` 的把手检测错误并修复，原失败现场关闭及原生面板完整流程通过，见[后续修复报告与录像](../drawer-handle-fix-20261025/README.md)。下文 `drawer-v5` 放置失败仍未由该修复解决。

已实现并实际运行：失败现场重新观察 → 模型结合前后图像和历史选择动作 → 执行已有物理技能 → 核验 → 继续原目标或停止。**当前属于可运行的功能原型，抽屉任务尚不稳定。** 最新原生面板复测放入成功、关闭失败，整条指令没有被计为成功。

本轮使用 Qwen3-VL、相机 RGB-D 和 OSC 控制。没有新增训练，也没有把这套模块化控制称为 π0.5 端到端 VLA。模型负责理解、定位、判断和选择技能；抓放、开关抽屉仍依赖已有控制器的实际能力。

## 改了什么

- 恢复接口增加上次行动前和当前两张图片、近期恢复历史、原指令和剩余计划。保存模型原始回答及图片哈希。
- 模型可选择重抓、持物续放、重新核验、重试抽屉，或补一个必要前置动作。例如发现抽屉没开，先开抽屉；核验前置动作成功后，系统自动续接原来的放入目标。
- 持物状态下保留夹持和原目标上下文，避免将“续放”变成“松手重抓”。恢复子目标成功不代表原目标成功。
- 原来固定两次的恢复改为进展监督。连续三次没有进展会停止；另有每次恢复流程 12 次循环、600 秒、最多三层前置任务的保护上限。这些是停止保护，达到上限不会报成功。
- 模型报告进展还需满足图像变化检查：超过 1% 像素的任一 RGB 通道变化超过 12。此检查只能排除完全不变的重复，不能证明任务进展；最终成功仍依赖目标核验，抽屉还结合深度位移。
- 抓起后如果单独的目的地定位返回空框，会用**同一张当前图像和不变的完整指令**重试任务定位；核对图片哈希和目标关系，只更新目的地框，不挪用旧图的坐标。

## 实际仿真记录

这些是逐项功能测试，未构成随机场景成功率统计。恢复决策由实际运行的视觉模型返回。控制器不读取模拟器物体真值作运动决策；`physical.json` 仅用于独立诊断。

| 场景 | 实测结果 | 控制步数 | 证据 |
| --- | --- | ---: | --- |
| 连续两条指令，不重置：酒瓶到柜顶，再从柜顶到灶台 | 两条均通过 | 244 / 292 | [第一条录像](consecutive-v1/command-1/demo.mp4)、[第二条录像](consecutive-v1/command-2/demo.mp4)、[记录](consecutive-v1/receipts.json) |
| 最终代码再测连续两条指令，不重置 | 两条均通过，已查看最终图像 | 244 / 292 | [第一条录像](consecutive-v2/command-1/demo.mp4)、[第二条录像](consecutive-v2/command-2/demo.mp4)、[记录](consecutive-v2/receipts.json) |
| 首次空抓，模型选择重新抓取 | 成功 | 314 | [录像](empty-v1/demo.mp4)、[结果](empty-v1/result.json)、[独立诊断](empty-v1/physical.json) |
| 抓起后运输前中断，模型保留持物继续放置 | 成功 | 264 | [录像](holding-v1/demo.mp4)、[结果](holding-v1/result.json)、[独立诊断](holding-v1/physical.json) |
| 隐含前置条件：奶酪盒放第三层抽屉，再关闭 | v4 完整成功，自动补开抽屉，再放入、关闭 | 970 | [完整录像](drawer-v4/command-1/demo.mp4)、[结果](drawer-v4/command-1/result.json) |
| 同一抽屉指令再次运行 | v5 失败：放置定位偏差、接触后下降不收敛，未恢复成功 | 662 | [结果](drawer-v5/command-1/result.json)、[失败画面](drawer-v5/command-1/final.png) |
| 原生中文面板按钮，第一次复测 | 放置阶段目的地空框，停下并保留现场 | 510 | [面板记录](panel-check.json)、[操作截图](panel-running.png) |
| 原生中文面板，加入完整指令定位回退后 | 自动打开、放入成功；关抽屉把手几何检查失败，完成 1/2 个原步骤，整体未确认 | 698 | [面板记录](panel-check-v2.json)、[原始输入图](panel-v2/scene_1792929512891635032/semantic_1792929515060766786/step-02/input.png) |

[最新面板测试的动作录像](panel-v2/partial-demo.mp4)只展示自动开抽屉、放入、撤手。之后关闭动作在运动前被几何检查拦截，视频没有把任务包装成完整成功。

空抓和持物中断是明确标注的受控故障注入：测试脚本让第一次夹爪在空中闭合，或在正常抓起后抛出运输前中断。之后由真实模型选择恢复、真实控制器执行；没有重置场景或瞬移物体。这不能当成任意自然故障的恢复成功率。

视频是按仿真控制帧以 20 FPS 编码，模型等待期间没有重复填帧，**不是实际反应时间录屏**。`videos.json` 记录部分合并录像的帧数和时长。原生面板复测通过实际“执行当前指令”按钮发起，尚未新增真人麦克风语音验收。

## 仍然失败在哪里

1. `drawer-v5`：模型识别出抽屉，但放置框和内部安全区域不稳定。深度估计的放置点向一侧偏，下降未收敛；没有通过放宽接触容差假装成功。
2. 最新 `panel-v2`：放入已核验成功。关抽屉时 `/drawer` 和独立 `/locate` 选择不同把手框，后者在当前深度里没有通过细长横杆检查。两次重定位仍返回相同区域，第三次无进展判断停止；没有盲目施力，也没有宣称抽屉已关。
3. 大模型可以判断和重新规划，但会看错、重复无效建议。只有可执行技能覆盖了失败情形，且定位、接触控制和核验同时可靠，恢复才能成功。当前没有换视角主动感知或通用接触恢复技能。

较早失败也保留：`drawer-v1` 曾在前置完成后重复父任务（已修）；`drawer-v2` 是服务未就绪；`drawer-v3` 是模型只重抓而未补开抽屉（已补前置优先提示）。这些不能与 v4 成功混作一次全程通过。v4 为中间版本的成功证据，最新代码结果以 v5 和 panel-v2 为准。

## 怎么使用与复现

在原生中文面板加载一次布局，输入完整指令，点“执行当前指令”。系统负责理解、物理动作、核验和必要恢复。完成或停止后保留现场，可以再发新指令；独立“VLA 调试”入口不需要跟着点击。

以下命令在项目根目录执行，端口 8911 的模型服务必须已启动且 `/health` 的 `ready` 为 `true`。输出使用新目录，避免覆盖证据。

```bash
rtk proxy curl -fsS http://127.0.0.1:8911/health

# 两条连续指令；同一个 env，不在指令之间 reset
rtk proxy env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl \
  .venv-vla-sim/bin/python -u evidence/continuous-recovery-20261025/trial.py \
  runs/continuous-recovery-consecutive-new

# 带隐含开抽屉前置条件；当前仍可能失败
rtk proxy env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl \
  .venv-vla-sim/bin/python -u evidence/continuous-recovery-20261025/trial.py \
  runs/continuous-recovery-drawer-new --drawer

# 受控空抓恢复；持物中断恢复在结尾另加 --holding
rtk proxy env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl \
  .venv-vla-sim/bin/python -u evidence/recovery-drawer-20261025/recovery_trial.py \
  runs/continuous-recovery-empty-new

# 实际中文面板按钮验收：最终未完成两步时会断言失败，窗口保留现场
rtk proxy env DISPLAY=:0 MUJOCO_GL=glfw __NV_PRIME_RENDER_OFFLOAD=1 \
  __GLX_VENDOR_LIBRARY_NAME=nvidia OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv-vla-sim/bin/python -u evidence/continuous-recovery-20261025/panel_check.py \
  runs/continuous-recovery-panel-new runs/continuous-recovery-panel-new-receipt.json
```

## 代码验证

`OPENBLAS_NUM_THREADS=1 .venv-vla/bin/python -m pytest -q tests`：250 项通过。覆盖恢复格式兼容、准备子目标、持物约束、超过两次有进展的恢复、无进展停止、取消、保护预算、父子目标续接，以及目的地定位回退拒绝过期图片和错误关系。项目检查不包含第三方 vendor 的自带测试收集。

测试通过只证明这些控制逻辑契约，不代表视觉模型或物理技能的通用可靠性。未触碰真机，没有调整旧 π0.5 训练任务。
