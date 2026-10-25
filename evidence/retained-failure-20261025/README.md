# 连续指令仍失败：现场复现与边界（2026-10-25）

当前结论：修复了本次两个“放柜子”请求启动前的 JSON 格式故障；改善了已放下后误报未完成的已知案例。**实际连续抓取仍未通过，不能宣布连续任务可用。** 无新训练、无换权重、无增加控制上限，没有重置用户桌面的现场。

## 现场与原始故障

原场景：libero_goal / task 9 / init 0 / seed 7。第一条“把酒瓶放灶台上”执行了 210 个控制步、13 次推理，松爪间距约 0.0798 m。全景可见瓶子立在加热板后侧、夹爪远离；旧完成模型仍说“正在夹持”，因此界面显示未完成。这个视觉结果没有独立灶台支撑谓词评分，不能仅凭改后的 VLM 当作物理真值。

随后两次“把酒瓶放柜子上”都是 **0 个控制步、0 次策略推理**，失败于 Qwen3-VL 输出的结构：

```text
{"source":{...},{"destination":{...},"relation":"on","english":"Place the wine bottle on the cabinet."}
```

`destination` 前多了一个开花括号。根因是可复现的模型序列化错误，不是步数不够。原始结果在 `original-cabinet-errors/`；两次试用“重新生成 JSON”及更明确 schema 仍返回同一个错误，记录保留在 `diagnostics/`，重试方案已撤回。

## 本次修复

- `model_service.py` 只对这一种已复现的 JSONDecodeError 删除错误位置的单个 `{`，随后重新通过原有严格 schema 检查。源物体、目的地、框、关系和英文指令均来自模型，逐字不改；原始输出、删除位置和规范化文本留档。未知语法、null、越界框和不支持关系仍拒绝执行。
- 完成模型使用当前固定全景图。腕部画面继续保存并绑定请求 SHA256，但不作为本次核对模型输入。两视角、全景重复的对照表明腕部图会在该案例诱发“仍夹持”误判。仅使用全景也会失去局部可见性；遮挡时仍应 uncertain，不能保证模型总遵守。
- 四个真实图像复测：酒瓶在灶台／询问柜子、碗在盘子／询问架子，分别判 completed / not_completed / completed / not_completed。两个负例理由仍错误提到夹爪未撤离，说明**标签在这四例正确，视觉解释仍会幻觉，尚无通用可靠性证据**。
- CPU 8912 服务已加载这次功能变更。GUI 未重启以保留用户现场；已有窗口缓存的旧“两路画面”进度文案可能到下次启动才更新。其请求已由新服务处理。其他 GPU 服务未被终止。

## 在完全相同现场执行下一条指令

独立 MuJoCo 环境从 `scene-source/initial_state.npy` 初始化，执行 10 个官方静置步后回放 210 个实际动作；**两条指令之间不 reset**。512×512 全景 PNG 与用户最后失败请求逐字节相同：

`4cfb09dbd7416ae70278b428f3e7efb0b29ea4356eaa75b3eb66f096cc191cb0`

证明见 `exact-replay.json`。这是独立复现环境，未操作桌面 GUI。

“把酒瓶放柜子上”现在真实解析成功：模型选 wine bottle → cabinet，实际策略收到 `Place the wine bottle on the cabinet.`。但 π0.5 在 180 个控制步、36 次推理内未形成抬起抓取候选。录像显示靠近偏移并碰倒酒瓶，最后闭合抓空；异常停止，未执行运输放置，也没有宣布成功。见 [实际失败录像](cabinet-execution/demo.mp4)、`result.json`、`trace.json`、`model.json` 和原始预测数组。

同图像、同 checkpoint、同种子规则下另外进行了两个**仅诊断抓取**的对照：

| 传给策略的语言 | 上限 | 结果 |
| --- | --- | --- |
| Place the wine bottle on the cabinet. | 180 | 碰倒酒瓶，没有抬起抓取候选 |
| pick up the wine bottle | 180 | 持续张爪，未抓起 |
| put the wine bottle on the rack | 180 | 持续张爪，未抓起 |

后两项用于定位语言与状态影响，**不是完成用户“放柜子”目标的替代方案，也没有部署为隐藏指令**。录像和动作保存在 `grasp-probes/`。这个结果否定“只换一种说法就能解决当前现场”的解释，不足以单独证明数据量、过拟合、动作上限中的哪一项是最终原因。

还有一个未修复限制：深度连通区域会把相接的酒瓶和灶台合并，源区域 9068 像素，柜子区域也包含柜身。这不能作为精确抓取点或柜顶证明。本次抓取动作本身由 π0.5 输出，并非由该点引导；未到运输阶段，不能把本次抓取失败直接归因于深度点，也没有验证柜顶放置。

## 检查与复现

针对这次缺陷新增 10 个服务测试；仿真环境中：

```bash
rtk proxy env MUJOCO_GL=egl .venv-vla-sim/bin/python -m pytest -q tests --ignore=tests/test_vla_artifacts.py
# 158 passed, 18 subtests passed
rtk proxy .venv-models/bin/ruff check --isolated --select E4,E7,E9,F model_service.py vla/semantic.py vla/semantic_grasp.py tests/test_visual_task.py
```

artifacts 模块原有缺失依赖 google_crc32c，本次未安装，也未称所有测试均通过。

只核验现场回放（新建独立仿真，不调用策略）：

```bash
rtk proxy env MUJOCO_GL=egl .venv-vla-sim/bin/python -u evidence/retained-failure-20261025/replay_command.py --output runs/your-new-replay --replay-only
```

去掉 `--replay-only` 则在回放后的同一环境执行“放柜子”，需要端口 8766 的同一 π0.5 LoRA 和端口 8912 的视觉服务。运行 `probe_grasp.py --output runs/your-new-probes` 可复现两个语言对照。输出路径须全新；不允许覆盖现有证据。

## 下一轮有意义的改进

先建立“从上条动作结束状态重新抓取”的冻结测试集，涵盖桌面／灶台／碗内、直立／倾倒酒瓶和机械臂不同起始姿态，记录抓取、稳定持有、指定放置、撤手后的独立评分。再补这些状态下的多样成功与恢复示范，保留旧任务回放后微调，并与当前权重在相同测试现场对比。需要区分“识别对了”和“动作能做到”；不能靠继续加时间或一个成功录像宣布泛化能力达标。

本次只修软件并保存失败证据，没有在这批失败现场上直接继续训练。模型权重与来源见每次结果中的 policy 字段；代码与原始仿真证据备份到用户拥有的 voice-arm-sim / feature/visual-semantic-grasp。
