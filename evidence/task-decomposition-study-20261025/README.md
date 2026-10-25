# 三个开源项目：中文任务拆解与闭环执行复核

日期：2026-10-25。范围：固定提交下的 12 份上游源码、当前项目解析器的离线复现、4 份已有运行记录。源码清单和内容哈希见 [source-manifest.json](source-manifest.json)，复现输入及原始运行结果见 [diagnostics.json](diagnostics.json)。

本次没有运行上游机械臂、重新训练、修改控制代码或移动当前仿真场景。抖音页面访问受限，不能声称完整观看了三条视频；以下结论来自源码，不把视频效果当可靠性测试。此前已经有调研记录，问题在于没有把关键机制落实到当前执行链。

## 1. 实际学到了什么

| 项目 | 源码确认的机制 | 对我们的用途 | 证据边界 |
| --- | --- | --- | --- |
| Vector OS Nano | 当前 native loop 向模型提供已注册技能；执行结果返回对话；调用 verify 核验，失败信息供后续模型决策 | 作为技能调度、反馈、核验机制的主要参考 | 默认 MuJoCo 感知读取场景真值；抓取使用 weld 约束，不能证明真实视觉和摩擦抓取可靠 |
| 子豪兄 vlm_arm | 中文规划输出有序函数列表；语音/文字输入、视觉定位、动作函数分层 | 学习中文复合指令如何调用多个已有能力 | 检查到的视觉搬运函数使用吸盘和像素到机械臂标定；缺乏独立的动作后视觉核验 |
| wheelarm-lab | 指令必须完整落在已支持能力中；结构化输出和参数校验；不支持就整体拒绝 | 防止漏掉前后步骤后仍宣称完成 | 当前公开范围主要是命名航点导航和视觉问答，README 明确不支持语言抓取 |

### Vector：值得学的是执行反馈，不是复制演示捷径

固定提交：[cd7029a](https://github.com/VectorRobotics/vector-os-nano/tree/cd7029aad15d6510c956f504e57181cbb6651996)。

- [native_loop.py](https://github.com/VectorRobotics/vector-os-nano/blob/cd7029aad15d6510c956f504e57181cbb6651996/vector_os_nano/vcli/native_loop.py)：`run_turn_native` 提供真实技能工具及 verify/finish，把工具返回值写回对话。模型据此选择下一步。`dispatch_skill` 返回实际执行错误；`handle_verify` 负责核验。finish 本身不能证明任务成功，仍需看核验和外层判定。
- [skill.py](https://github.com/VectorRobotics/vector-os-nano/blob/cd7029aad15d6510c956f504e57181cbb6651996/vector_os_nano/core/skill.py)：技能注册和参数描述，把机器人实际具备的能力交给模型选择。
- [executor.py](https://github.com/VectorRobotics/vector-os-nano/blob/cd7029aad15d6510c956f504e57181cbb6651996/vector_os_nano/core/executor.py)：另有依赖图执行、前后条件检查；执行器遇错返回，不会独立完成所有重规划。声明的状态更新也不等同于传感器证据。
- [架构说明](https://github.com/VectorRobotics/vector-os-nano/blob/cd7029aad15d6510c956f504e57181cbb6651996/docs/ARCHITECTURE.md)标注了向 native tool-use 转换，不能只拿旧规划器描述当前默认设计。
- [MuJoCo 感知](https://github.com/VectorRobotics/vector-os-nano/blob/cd7029aad15d6510c956f504e57181cbb6651996/vector_os_nano/hardware/sim/mujoco_perception.py)直接读物体位置；native loop 也可从仿真物体位置取得名称词表。[MuJoCo 夹爪](https://github.com/VectorRobotics/vector-os-nano/blob/cd7029aad15d6510c956f504e57181cbb6651996/vector_os_nano/hardware/sim/mujoco_gripper.py)在附近物体上施加 weld，抓取半径为 0.05 m。学习调度不意味着接受这些数据作为视觉抓取成绩。

native loop 的 24 轮上限是模型与工具交互预算，不是机械臂运动 24 个控制步。恢复有反馈渠道，不代表任意失败都能恢复。已查看技能列表也不足以证明其具备抽屉开关能力。

### 子豪兄：一句话可编排多个函数，但物理闭环还要补

固定提交：[3534c4b](https://github.com/TommyZihao/vlm_arm/tree/3534c4b93230a2d5d4398d0f6bd7c43a58c8c8e7)。

[utils_agent.py](https://github.com/TommyZihao/vlm_arm/blob/3534c4b93230a2d5d4398d0f6bd7c43a58c8c8e7/agent_demo_20250328/utils_agent.py)列出可调用函数，并通过中文示例让模型输出有顺序的 `function` 列表。[agent_go.py](https://github.com/TommyZihao/vlm_arm/blob/3534c4b93230a2d5d4398d0f6bd7c43a58c8c8e7/agent_demo_20250328/agent_go.py)依次执行函数。这个结构允许组合已有技能，不需要为每个完整句子单写一个分支。

但该示例使用 `eval` 执行模型文本，我们应改用限定技能 ID 和参数校验；示例任务开始会回零，也不适合直接用于用户要求的“接着当前状态执行”。[utils_vlm_move.py](https://github.com/TommyZihao/vlm_arm/blob/3534c4b93230a2d5d4398d0f6bd7c43a58c8c8e7/agent_demo_20250328/utils_vlm_move.py)中的重试主要针对视觉接口输出，不能当作抓空后重新观察、重新抓取的证明。

### wheelarm：不能悄悄只执行自己会的半句话

固定提交：[25fdb62](https://github.com/renyimen2026/wheelarm-lab/tree/25fdb62fae8ffd340e9e023987a62ff4d0254c45)。

[llm.py](https://github.com/renyimen2026/wheelarm-lab/blob/25fdb62fae8ffd340e9e023987a62ff4d0254c45/wheel_arm_navigation/wheel_arm_navigation/llm.py)要求先审查整句指令，再返回 ready/clarify/unsupported，且校验航点、参数和键名。不能从包含不支持要求的句子中只抽取能执行的片段。该原则直接适用于我们的复合抓放指令。其 [README](https://github.com/renyimen2026/wheelarm-lab/blob/25fdb62fae8ffd340e9e023987a62ff4d0254c45/README.md)明确限定能力，不能把该项目当已有桌面抓取方案。

## 2. 我们的具体错误：语义保留，执行表示丢失

检查时本地代码版本：`f98bb9b6187025dd5e876487c72078f68d5856a1`。

用户指令：**打开柜子第三层，把奶油奶酪放进去，之后合上第三层。**

`vla/semantic.py` 的提示词只允许一次抓取放置。解析器只接收 source、destination、relation、english，没有步骤序列，也没有开关抽屉动作。

实际返回的 english 包含打开、放入、关上三个要求，但可执行字段只有奶酪、第三层柜子、in。离线复现确认：解析器接受了该对象，没有检查三步是否都有对应技能。随后 `vla/semantic_grasp.py` 将 `in` 送进圆形容器边沿拟合，抽屉无法通过便报错。原位置和抓后重定位各有一处分支。

| 已保存运行 | 控制步数 | 记录结果 |
| --- | ---: | --- |
| 碗放盘子 | 337 | visual_completed，属于已有核验输出，未在本次重新人工验收 |
| 酒瓶放灶台 | 275 | visual_completed，同上 |
| 打开第三层→放奶酪→关上，第 1 次 | 0 | 容器圆沿拟合不可靠，停止放置 |
| 同一复合指令，第 2 次 | 0 | 同样错误 |

这两次抽屉失败发生在运动开始前，不是控制步数到上限。它们也不能证明视觉语言模型完全不理解句子，因为原始翻译保存了三个动作。当前可确定的根因是任务接口表达不足、目的地几何类型混用及缺少抽屉技能。

当前语义路线是 Qwen3-VL + RGB-D 几何抓放 + OSC 控制，记录也明确标注非端到端 VLA、非本轮训练策略。不能把该路线的结果算作 π0.5 泛化成绩。

## 3. 后续实现和验收顺序（本次尚未实现）

1. **先保全整句语义。** 用实际技能清单约束结构化计划，包含顺序、对象引用、前置条件和完成条件。缺少开抽屉能力时应在运动前说明整项任务不能完成，不能只执行“放入”。换中文表达后仍应得到等价计划。
2. **再补物理技能。** 抓放、开抽屉、关抽屉分别有明确接口和核验条件。抽屉需要把手、关节和可行轨迹，增加模型规模不能替代这些能力。目的地应区分开放容器、支撑平面和抽屉内部，不能把所有 in 都当碗。
3. **逐步核验，再反馈给模型。** 工具执行结束不等于任务完成；使用新图像、深度及机械臂状态，区分成功、失败和看不清。抓空、掉落等返回实际证据，由模型在可用技能内决定下一步，限制重复失败次数。不能用单一固定恢复轨迹冒充通用恢复。
4. **保留场景，连续接令。** 默认不重置、不回到旧物体位置。上一条改变场景后，下一条重新观察；已有中文面板显示计划、当前步骤、核验与缺失能力，继续使用原生 MuJoCo 视窗。
5. **按真实场景验收。** 同一场景连续两条指令、不重置；物体移动后的再次定位；同义中文；复合指令覆盖；人为制造抓空后的反馈；达到目标后停止。规划轮数、物理控制步数、总时间预算分别记录。不能仅因预算耗尽而宣称完成，也不能无限运行。

选型结论：优先借鉴 Vector 的模型选技能与核验反馈机制，参考子豪兄的中文有序编排，采用 wheelarm 的完整指令能力审查。保留我们的图像/深度输入和物理抓取，不照搬真值定位、weld 抓取或任意文本 eval。
