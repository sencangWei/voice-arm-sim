# 同场景改指令的 π0.5 微调实验

此实验针对“图中物品不变，只改变源物体和目的地”。它是有限任务的模型微调验证，不能代表任意家庭物品或开放指令都能执行。

## 实际执行链
中文文本/语音 → 本地语言模型翻译 → 两路RGB、8维状态、英文指令 → π0.5 LoRA → 7维增量动作 → MuJoCo物理仿真。
评测脚本仅按任务选择判分方式，不选择运动程序；每次调用直接使用模型返回的动作。没有预设搬运、松爪或几何目标引导。

## 数据与训练
数据是明确标注来源的仿真教师示范，不是人工示范，也不是评测结果：酒瓶入碗示范由原模型抓取、仿真几何教师运输和释放；另两类来自模型。完整原始失败记录保留。人工视觉审核排除倾倒及失败示范。
训练21段2881个控制帧，初始布局0–7；留出10段，布局8–11；额外布局12–15没有教师示范。三个任务等概率采样，窗口10步，末端物理静置只留3帧，释放和回退保留。
固定openpi 15a9616、LIBERO f78abd6、seed42；沿用官方LIBERO归一化。仅训练LoRA，batch1、学习率5e-5、AdamW，不更新基础权重。V1实际更新2000次，checkpoint1999。launch、训练曲线和冻结数据清单在本目录。

## 评分边界
酒瓶入碗的程序判据是近距离、接触和短时间无夹爪接触，不能单独排除碗倾斜或被带起。因此所有成功需要结合实际双相机录像复核；原模型原始5/8中两例是视觉假阳性，可信观察为3/8。未声称长时间稳定性。
V1训练布局0实测：酒瓶入碗失败360步、酒瓶入架成功205步、碗上盘成功99步。失败录像显示持瓶停在碗上方。完整留出评测正在执行，不能提前宣称通过。

## 重现
```bash
.venv-vla/bin/python -m vla.train_distillation train --manifest evidence/language-guidance/distillation-manifest.json --experiment counterfactual_v1 --steps 2000
.venv-vla/bin/python -m vla.serve --checkpoint checkpoints/voice_pi05_lora/counterfactual_v1/1999
.venv-vla-sim/bin/python -m vla.evaluate_distillation --output runs/new-evaluation --initial-states 8 9 10 11 12 13 14 15
```
模型导出/恢复只备份推理权重，不包含优化器或加载器进度。恢复必须使用同一官方LIBERO基础模型，并检查完整参数树hash。

## 界面修复
自定义指令不再使用原始“碗上盘”的成功信号来阻止启动或自动宣布完成。自定义指令显示需要观察确认；官方原始指令保留原判据。
