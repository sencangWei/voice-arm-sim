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
V1训练布局0实测：酒瓶入碗失败360步、酒瓶入架成功205步、碗上盘成功99步。失败录像显示持瓶停在碗上方。完整布局8–15评测已完成：V1酒瓶入碗2/8、酒瓶入架7/8、碗上盘8/8；所有24次均保留录像和动作，视觉审核见counterfactual_v1-results.json。相比原模型视觉审核3/8，V1未提升总体入碗成功率，不能算达成要求。

V2在相同数据上从V1增加8000次更新（新优化器，不是续恢复优化器），总计10000次。这个继续训练决定在查看V1留出结果前，依据训练布局0失败作出；后续结果属于开发评测。

## 重现
```bash
.venv-vla/bin/python -m vla.train_distillation train --manifest evidence/language-guidance/distillation-manifest.json --experiment counterfactual_v1 --steps 2000
.venv-vla/bin/python -m vla.serve --checkpoint checkpoints/voice_pi05_lora/counterfactual_v1/1999
.venv-vla-sim/bin/python -m vla.evaluate_distillation --output runs/new-evaluation --initial-states 8 9 10 11 12 13 14 15
```
模型导出/恢复只备份推理权重，不包含优化器或加载器进度。恢复必须使用同一官方LIBERO基础模型，并检查完整参数树hash。

## 界面修复
自定义指令不再使用原始“碗上盘”的成功信号来阻止启动或自动宣布完成。自定义指令显示需要观察确认；官方原始指令保留原判据。

## 实际对比录像
[同一场景不同指令（V1，布局9）](https://github.com/sencangWei/voice-arm-sim/releases/download/language-distillation-v1/same-scene-different-instructions-v1.mp4)左右分别是酒瓶入碗、酒瓶入架，都是纯模型动作。
[原模型与V1对比（布局9）](https://github.com/sencangWei/voice-arm-sim/releases/download/language-distillation-v1/baseline-vs-trained-init009.mp4)。原模型该例虽触发近距离判据，但视觉发现碗被带到柜边，不计成功。此处展示单例，不能代替上述整组成功率。两路按视频控制帧同步，结束后保持尾帧，不用于比较推理耗时。

## 恢复公开模型
Release中的adapter和原始教师数据作为独立附件上传，完成状态以恢复验证记录为准；基础模型使用官方pi05_libero。适配器仅20个LoRA张量，导出时逐张量验证所有冻结权重保持官方基座一致。恢复示例：
```bash
JAX_PLATFORMS=cpu .venv-vla/bin/python -m vla.restore_adapter /path/to/extracted/counterfactual_v1 --base-params models/pi05_libero/params --output .local/restored-counterfactual-v1/1999
```
恢复会从磁盘重新读回完整参数树逐张量检查SHA256；不能只凭上传命令成功就认定模型可恢复。
