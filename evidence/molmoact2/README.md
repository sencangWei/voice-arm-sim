# MolmoAct2 本地部署与同场景测试

测试日期：2026-10-24。接入和基线已完成，MolmoAct2 权重仍在下载；此阶段尚无 MolmoAct2 动作测试结果。

## 对比条件

两个模型使用 LIBERO Goal 任务 8、初态 0、环境种子 42，每次执行 5 个动作、最多 300 个控制步。初态 SHA256 为 `66b1883f351c4c374c742516714921b55bd5eca7fd6f0fa8e2e0a86abff19ccd`。两者动作噪声实现不同，不是逐请求随机数一致的对比。

| 模型 | 指令 | 结果 | 控制步 / 推理次数 |
| --- | --- | --- | --- |
| 官方 π0.5 LIBERO | put the bowl on the plate | 原始任务判据成功 | 84 / 17 |
| 官方 π0.5 LIBERO | put the wine bottle in the bowl | 独立目标判据失败 | 300 / 60 |
| MolmoAct2 LIBERO | 两条相同指令 | 待测试 | — |

π0.5 的[放碗录像](pi05-bowl-init0/paired_cameras.mp4)、[报告](pi05-bowl-init0/report.json)及[酒瓶放碗录像](pi05-bottle-init0/paired_cameras.mp4)、[报告](pi05-bottle-init0/report.json)均来自实际模型闭环控制。左画面为外部相机，右画面为腕部相机；按控制步以 10fps 保存，播放时长不等于实际推理耗时。

酒瓶放碗使用近似物理关系判据：瓶碗接触、XY 中心距离小于 3cm、Z 中心差小于 8cm、夹爪不接触瓶子，连续满足 5 步；还需看录像确认。原来的放碗判据不能用于该指令。仿真位置只用于评分，不会作为目标坐标传入动作模型。

## 已验证的接入环节

- [中文翻译记录](chinese-translation.json)：实际本地 Qwen 将“帮我把酒瓶放到碗里面”译为 `Please put the wine bottle into the bowl.`，未改变目的地。这不证明动作模型已掌握该动作。
- [CPU 视觉兼容性](cpu-vision-compatibility.json)：BF16 图像编码及图像块索引迁移检查通过。此检查用零权重验证算子，不能证明预训练模型效果。
- [独立环境](runtime.json)、[初始主机信息](environment.json)、[显存规划](memory-plan.json)。显存规划是架构估计，实际占用待加载后记录。
- 使用原有中文面板、原生 MuJoCo 窗口和人工接管入口；训练页仍只支持 π0.5。

复现命令及官方模型来源见 [部署说明](../../vla/MOLMOACT2.md)。本次没有训练 MolmoAct2；小样本测试也不能证明未见家庭物体识别、长期恢复或产品可靠性。
