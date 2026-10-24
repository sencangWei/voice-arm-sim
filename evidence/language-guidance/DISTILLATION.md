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

V2已完成，最终checkpoint为`checkpoints/voice_pi05_lora/counterfactual_v2/7999`。同样布局8–15、seed42、每次预测10步执行5步、上限360控制步的结果如下。数字以双相机中间帧和结束画面复核为准；样本小且已用于开发，不能解读为可靠性承诺。

| 模型 | 酒瓶入碗 | 酒瓶入架 | 碗上盘 |
| --- | --- | --- | --- |
| 官方π0.5 LIBERO | 3/8 | 8/8 | 8/8 |
| V1，2000次更新 | 2/8 | 7/8 | 8/8 |
| V2，累计10000次更新 | 4/8 | 8/8 | 8/8 |

V2程序原始入碗判据是5/8，其中布局14把碗带到柜边，视觉判失败，因此报告4/8。成功布局为9、11、13、15，失败布局为8、10、12、14。全部24例记录见[counterfactual_v2-results.json](counterfactual_v2-results.json)、[酒瓶逐例画面](candidate-v2-bottles.jpg)与[另外两项结果](candidate-v2-controls.jpg)。4/8对3/8不足以证明统计上的可靠提升，更不能称为通用语义理解。

额外6次**真实中文翻译后执行**：布局0的酒瓶入碗失败360步，另两项成功；布局9三项成功。实际输入分别为“帮我把酒瓶放进碗里”“把酒瓶放到架子上”“把碗放在盘子上”，翻译后的实际字符串完整保存在`runs/distillation-chinese-v2/instructions.json`，没有换成评测预设英文。画面见[中文测试](chinese-v2-review.jpg)。真人麦克风对新模型尚未复验。

额外诊断把执行前缀从5改为1：布局0入碗仍失败360步，布局9程序判成功196步。该结果不支持“只是步数太大导致失败”。默认保留5，没有增加固定搬运、强制松爪或失败时脚本接管。

当前实现证明的是有限任务下指令影响学习策略的动作；尚未证明任意新物体、任意放置关系、连续对话和恢复能力。失败可能涉及有限教师覆盖及闭环偏移，尚无证据将原因归到某一个因素。下一轮应先收集失败布局中完整、安全释放的多样示范，冻结独立新测试布局，再决定是否训练；本轮没有继续无界加步数。

## 重现
```bash
.venv-vla/bin/python -m vla.train_distillation train --manifest evidence/language-guidance/distillation-manifest.json --experiment counterfactual_v1 --steps 2000
.venv-vla/bin/python -m vla.serve --checkpoint checkpoints/voice_pi05_lora/counterfactual_v1/1999
.venv-vla-sim/bin/python -m vla.evaluate_distillation --output runs/new-evaluation --initial-states 8 9 10 11 12 13 14 15
```
V2训练与评测（先退出自己启动的旧推理服务，勿终止其他GPU任务）：
```bash
.venv-vla/bin/python -m vla.train_distillation train --manifest evidence/language-guidance/distillation-manifest.json --experiment counterfactual_v2 --steps 8000 --parent checkpoints/voice_pi05_lora/counterfactual_v1/1999
HF_HUB_OFFLINE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 .venv-vla/bin/python -m vla.serve --checkpoint checkpoints/voice_pi05_lora/counterfactual_v2/7999
.venv-vla-sim/bin/python -m vla.evaluate_distillation --output runs/new-v2-english --initial-states 8 9 10 11 12 13 14 15
.venv-vla-sim/bin/python -m vla.evaluate_distillation --output runs/new-v2-chinese --initial-states 0 9 --chinese
```
启动原生工作台的方法见`vla/README.md`；选择8766端口并加载场景后，页首显示服务返回的实际checkpoint。点“开始／继续VLA”直接执行模型动作。“旧版分层实验（非纯VLA）”单独标出，不能用于验证本轮能力。自定义指令不会冒用原始“碗上盘”完成信号宣布成功。
模型导出/恢复只备份推理权重，不包含优化器或加载器进度。恢复必须使用同一官方LIBERO基础模型，并检查完整参数树hash。

## 界面修复
自定义指令不再使用原始“碗上盘”的成功信号来阻止启动或自动宣布完成。自定义指令显示需要观察确认；官方原始指令保留原判据。

## 实际对比录像
[V2同布局不同中文指令](https://github.com/sencangWei/voice-arm-sim/releases/download/language-distillation-v1/same-scene-chinese-instructions-v2.mp4)：左右分别是实际中文输入经本地翻译后的酒瓶入碗、入架，同一checkpoint、同一初始状态hash。布局9是挑选的成功示例，整体入碗结果仍是4/8。
[原模型与V2同布局对比](https://github.com/sencangWei/voice-arm-sim/releases/download/language-distillation-v1/baseline-vs-trained-v2-init009.mp4)：左侧原模型把碗带到柜边，右侧V2将酒瓶放入桌上碗中。没有视频加速；按控制帧对齐，短的一侧结束后保持尾帧，不比较墙钟推理速度。详见[Demo清单](demo-v2-manifest.json)。

[同一场景不同指令（V1，布局9）](https://github.com/sencangWei/voice-arm-sim/releases/download/language-distillation-v1/same-scene-different-instructions-v1.mp4)左右分别是酒瓶入碗、酒瓶入架，都是纯模型动作。
[原模型与V1对比（布局9）](https://github.com/sencangWei/voice-arm-sim/releases/download/language-distillation-v1/baseline-vs-trained-init009.mp4)。原模型该例虽触发近距离判据，但视觉发现碗被带到柜边，不计成功。此处展示单例，不能代替上述整组成功率。两路按视频控制帧同步，结束后保持尾帧，不用于比较推理耗时。

## 恢复公开模型
Release中的adapter和原始教师数据作为独立附件上传，完成状态以恢复验证记录为准；基础模型使用官方pi05_libero。适配器仅20个LoRA张量，导出时逐张量验证所有冻结权重保持官方基座一致。恢复示例：
```bash
JAX_PLATFORMS=cpu .venv-vla/bin/python -m vla.restore_adapter /path/to/extracted/counterfactual_v1 --base-params models/pi05_libero/params --output .local/restored-counterfactual-v1/1999
```
恢复会从磁盘重新读回完整参数树逐张量检查SHA256；不能只凭上传命令成功就认定模型可恢复。

## 已完成的远端恢复核验与当前窗口
公开Release的V1和V2适配器均已下载到新目录，重建完整模型并从磁盘重新读回71个参数张量，SHA256全部与原checkpoint一致。分别见[恢复V1](remote-model-v1-verification.json)、[恢复V2](remote-model-v2-verification.json)。只备份推理模型，未备份优化器状态。V2需要同一官方`pi05_libero`基座，不能替换成通用`pi05_base`。
原始36段教师数据分22个附件，恢复144个文件并逐文件核对；见[教师数据核验](remote-teacher-verification.json)。V2两份对比录像、完整35次运行的评测归档均从Release重新下载并核对SHA256；见[录像及评测核验](remote-v2-demo-evaluation-verification.json)。原始数据包括被拒绝的失败教师片段，不等同于31段批准训练/留出数据。
下载V2适配器后按编号拼接，先根据parts清单核验每个分片及整包hash，再解压：
```bash
gh release download language-distillation-v1 --repo sencangWei/voice-arm-sim --pattern 'pi05-counterfactual-v2-adapter.tar.part*' --pattern 'pi05-counterfactual-v2-adapter-parts.json' --dir /path/to/new-download
cat /path/to/new-download/pi05-counterfactual-v2-adapter.tar.part* > /path/to/new-download/adapter.tar
tar -xf /path/to/new-download/adapter.tar -C /path/to/new-download
JAX_PLATFORMS=cpu .venv-vla/bin/python -m vla.restore_adapter /path/to/new-download/counterfactual_v2 --base-params models/pi05_libero/params --output .local/new-restored-v2/7999
```
当前中文原生窗口已加载布局9并暂停，端口8766，权重`counterfactual_v2/7999`，输入“帮我把酒瓶放进碗里”。[面板截图](native-panel-v2.png)、[MuJoCo截图](native-scene-v2.png)、[现场核验](native-panel-v2-verification.json)。点击“开始／继续VLA”执行纯模型动作；改变指令前重置相同布局可比较行为。布局9是已见过的开发成功案例，不能当独立泛化测试。
