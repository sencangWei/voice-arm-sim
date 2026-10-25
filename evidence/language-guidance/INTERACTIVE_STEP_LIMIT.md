# 交互运行在中途暂停：步数上限修复

现象：旧交互窗口每次运行 360 个控制步后暂停。实际场景报告在累计第 720 步记录 `reason=step_limit`、`policy_calls=144`、`official_predicate=false`。用户在第 360 步更换过指令，随后又运行了 360 步。这次停止由运行上限触发，不能据此判断策略已经完成或不会恢复。

原始报告：`interactive-step-limit-original.json`。

修改：

- 原生中文交互面板默认不限步数，显示“本轮步数上限（0＝不限）”。正数表示从本次点击开始计算的控制步数上限；修改后下次点击“开始／继续 VLA”生效。
- “开始／继续”保留现有仿真状态，丢弃旧预测队列，重新观察后预测；不重置场景。
- 手动暂停、人工接管、原任务判据满足、推理异常仍会停止运行。自定义指令没有自动完成判据，应由用户观察后暂停。
- 命令行评测仍保留原有有界设置；不改变模型、控制频率、动作幅度或训练数据。

真实验证：同一 V2 checkpoint、布局 9、中文“帮我把酒瓶放进碗里”，面板默认上限 0，运行到第 362 步时仍在执行。测试程序随后发送普通暂停命令，场景保留并保存动作。这是验证越过旧边界的测试，不是抓取成功率实验。记录：`interactive-unlimited-verification.json`。

软件验证：110 个项目测试通过；相关文件 Ruff 通过。回归用例覆盖交互默认值、评测保持有界、越过旧上限、显式有限运行、保留场景继续、达标停止、面板参数传递及非法值拒绝。

启动不限步数的面板（先确保 8766 的模型服务在运行）：

```bash
LIBERO_CONFIG_PATH="$PWD/.local/libero" \
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
.venv-vla-sim/bin/python -m vla.run \
  --panel --viewer --suite libero_goal --task-id 8 --initial-state 9 \
  --prompt "帮我把酒瓶放进碗里" --seed 42 --port 8766 \
  --max-steps 0 --output runs/panel-unlimited
```

旧窗口的 Python 代码不会自动热更新。现场打开了标题含“不限步数版”的新窗口并在验证后暂停；原窗口及原暂停场景保留。

边界：取消运行上限不会增加模型的语言理解或恢复能力。若控制步仍在增长、机械臂却几乎不动，需要分析模型输出、观测和接触状态，不能再归因于这次已移除的默认步数限制。
