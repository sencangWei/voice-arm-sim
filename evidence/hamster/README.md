# HAMSTER 指令与目标理解试验（真实模型推理中）

2026-10-25 更新：上一轮真实加载因缺少 protobuf 中断，未生成轨迹。已将 protobuf==4.25.8 加入独立环境依赖，真实官方 LlamaTokenizer（legacy=False）读入与编码通过，34个包兼容检查通过。保留失败目录 `trial-init0/`，在新目录 `trial-init0-protobuf/` 重启；13B六个权重分片已加载，正在生成第一条指令。详见 `tokenizer-smoke.json` 与后台日志，仍未宣称路径语义或物理执行成功。

2026-10-24 23:45 历史记录：约27GB权重已全部下载，18文件校验通过。首次模型启动被接口检查生成的上游 `__pycache__` 拦住（源码无修改）；已归档缓存，修正动态导入避免产生缓存，回归检查13项通过。已重新启动真实模型测试；以下准备阶段记录保留，尚未宣布路径语义或物理执行成功。

本次验证的问题：相同 MuJoCo 画面中，将指令从“酒瓶放到架子上”改成“酒瓶放进碗里”，官方模型是否会改变目标。另加“碗放到盘子上”作为对照。三条英文输入及中文含义见 `commands.json`，翻译预先固定；本轮不测试语音识别，也不训练新模型。

## 当前证据与边界

- 已固定官方模型、源码及环境，并在独立 `.venv-hamster` 安装 CPU 运行时。
- `scene-init0.png` 来自实际 LIBERO/MuJoCo 场景，任务 libero_goal/8、初始状态0、seed42，静置10步，视图旋转180度。独立重新渲染后 SHA256 一致：`7ff9c95921717f7c1c5db93834eaa471dfd06b2fb273e15b4890ae01b5aaab18`。
- `runtime-smoke.json` 仅验证图像 token 拼接、官方降采样和小型随机 BF16 Llama 的运行接口，**不证明真实 HAMSTER 能力**。路径解析、上游干净导入与下载故障分支共13项测试通过。
- 准备阶段曾因下载未完成而无法推理；当前权重已校验，实际进展见文首更新。路径语义判断、机械臂执行及 Demo 均未完成。以本机 `logs/hamster-trial.status` 和输出目录为准；解析合法不能算目标理解成功。
- 后台流程下载完成后先逐文件核对大小及官方 LFS SHA256，再使用真实权重运行三条指令，保存原始文本、生成 token、耗时及路径叠加图。首轮输出目录为 `evidence/hamster/trial-init0/`（依赖失败，已保留）；修复后的新轮次是 `evidence/hamster/trial-init0-protobuf/`。未成功下载则不进入推理。
- 官方公开部分输出二维末端路径与夹爪事件；当前仓库没有完整发布可直接部署的低层控制器。本试验不会将路径直接当成七维机器人动作，也不会用固定动作伪造成功。后续是否接物理抓取，取决于路径的源物体、目标和夹爪事件是否合理。

## 来源与配置

| 组件 | 固定来源 |
|---|---|
| HAMSTER | https://github.com/liyi14/HAMSTER_beta ，`526a37f59f97c445005fcdf28f2cfb81ea742e4b` |
| VILA | https://github.com/NVlabs/VILA ，`a5a380d6d09762d6f3fd0443aac6b475fba84f7e` |
| checkpoint | https://huggingface.co/yili18/Hamster_dev ，`794f1f925c87e861d2f562943e978cc11f8c344d` |
| 架构 | Llama13B + SigLIP384 + 官方 mlp_downsample projector，196个图像 token |
| 独立运行时 | Python3.11.15、Torch2.7.1+cpu、Transformers4.37.2、Accelerate0.27.2；完整版本见 requirements-freeze.txt |

`official_manifest.json` 固定18个文件，共26,988,472,840字节。模型卡许可证字段仍是占位文字，不能据此认定适合商业玩具；工程验证与商业授权需分别处理。

官方服务硬编码 CUDA/FP16，本机现有 Qwen、π0.5、Molmo 与用户面板占用大部分显存。为保留这些服务，本轮适配为 CPU/BF16、8线程、贪心生成、每条最多300 token、seed42、开启 KV cache。使用官方视觉层选择、投影器、vicuna_v1 和原始提示模板，无额外 assistant 前缀。CPU/BF16 与官方 GPU/FP16 存在数值差异，不宣称逐位等价；CPU 推理可能很慢。此次只改独立新项目，未修改旧 pi05-handson 项目，未操作真机。

## 复现

在 `/home/robot/voice-arm-sim` 执行（环境、模型权重、上游克隆不提交到 Git）：

```bash
uv venv --python 3.11.15 .venv-hamster
uv pip install --python .venv-hamster/bin/python -r requirements-hamster.txt
mkdir -p .local
# 下列 clone 仅在对应目录不存在时执行
 git clone https://github.com/liyi14/HAMSTER_beta .local/hamster-upstream
 git -C .local/hamster-upstream checkout 526a37f59f97c445005fcdf28f2cfb81ea742e4b
 git clone https://github.com/NVlabs/VILA .local/hamster-vila
 git -C .local/hamster-vila checkout a5a380d6d09762d6f3fd0443aac6b475fba84f7e
# 若上一轮还在运行，不要重复启动；新试验使用新的输出目录
bash scripts/run_hamster_trial.sh evidence/hamster/trial-new
```

流程下载阶段限3小时，推理阶段限2小时，超时返回非零并保留已验证分块，不无限等待。脚本默认前台；本机本次已通过独立后台会话启动，PID记在 `logs/hamster-trial.pid`。

```bash
cat logs/hamster-trial.status
tail -n 10 logs/hamster-trial.log
# 独立核对下载完成后的所有文件
.venv-hamster/bin/python scripts/prepare_hamster.py --verify-only
# 纯解析器检查，不代表模型成功
.venv-vla/bin/python -m pytest -q tests/test_hamster_path.py tests/test_hamster_download.py
# 在现有隔离仿真环境重新生成同一画面，不移动用户正在查看的环境
.venv-vla-sim/bin/python -m vla.capture_hamster_scene --output .local/hamster-scene-new
```

初次 Hugging Face/Xet 下载遇到 TLS 中断及连接超时。保留原缓存，改用 curl 并发 HTTP Range（24连接、32MiB分块），每段校验 Content-Range/长度，拼接后核对完整官方 SHA256；最终由 prepare_hamster.py 再核验。Python依赖下载在当前网络使用阿里云 PyPI 镜像完成，没有修改全局代理或其它虚拟环境。大模型下载和 CPU 推理时间均不代表机械臂动作速度。

## 输出审查

先查看 `command-00/01/02.raw.txt` 与 `.overlay.png`，确认抓的是哪一个物体、最终指向碗还是架子、闭合和松开的位置。`.json` 中 `syntax_valid` 只表示格式有效，`semantic_success` 初始为 null，`physical_execution` 为 false。若生成被300 token截断，原样记录失败，不补造路径。只有真实路径目标通过审查，才另行评估低层控制方案。
