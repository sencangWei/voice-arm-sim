"""Readable Chinese desktop workbench, reusing the earlier correction workflow."""
# ruff: noqa: E402 -- select the compatible antialiased Tk runtime before Tk imports
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import tempfile
import threading
import time
import wave

from vla.tk_runtime import prefer_system_tk

prefer_system_tk()

import tkinter as tk  # noqa: E402 -- choose Tk with antialiased CJK fonts first
from tkinter import font as tkfont, messagebox, ttk  # noqa: E402

import numpy as np

from vla.manual_joystick import TranslationJoystick
from vla.run import ROOT, configure_libero, voice_prompt
from vla.workbench import Workbench


class ChinesePanel:
    def __init__(self, args):
        self.args = args
        self.root = tk.Tk()
        self.root.title("VLA 语音机械臂 · 中文仿真与学习工作台")
        self.root.geometry("900x1260+30+40")
        self.root.minsize(880, 1200)
        # Same explicit pixel font as the earlier Chinese LabPanel. Tk's default
        # small mixed CJK/Latin fallback was the source of the illegible layout.
        families = tkfont.families(self.root)
        family = "Noto Sans CJK SC" if "Noto Sans CJK SC" in families else "song ti"
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
            tkfont.nametofont(name).configure(family=family, size=-18)
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", font="TkDefaultFont")
        style.configure("TButton", padding=(10, 7))
        style.configure("TNotebook.Tab", padding=(14, 8))
        style.configure("Treeview", rowheight=30)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)
        self.lab = Workbench(args)
        self.events = queue.Queue()
        self.recording = self.voice_worker = self.training = self.model = None
        self.record_temp = self.training_log = self.model_log = None
        self.hybrid = self.hybrid_output = None
        self.closing = False
        self.grip = -1.
        self.preview_identity = None
        self.was_manual = False
        self.photo_refs = []
        configure_libero()
        from libero.libero import benchmark
        self.tasks = benchmark.get_benchmark_dict()[args.suite]()
        names = json.loads((ROOT / "vla/task_names_zh.json").read_text())
        self.task_names = [names[self.tasks.get_task(i).language] for i in range(self.tasks.n_tasks)]
        header = ttk.Frame(self.root, padding=12)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)
        self.model_title = tk.StringVar(value="视觉指令控制 · MuJoCo 本地仿真")
        ttk.Label(header, textvariable=self.model_title, font=(family, -24, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
        ttk.Label(header, text="仿真任务").grid(row=1, column=0, sticky="w")
        self.task = ttk.Combobox(header, state="readonly", values=[f"{i} · {self.task_names[i]}"
            for i in range(self.tasks.n_tasks)], width=45)
        self.task.current(args.task_id)
        self.task.grid(row=1, column=1, columnspan=2, sticky="ew", padx=8)
        self.task.bind("<<ComboboxSelected>>", lambda _: self.prompt.set(self.task_names[self.task.current()]))
        self.initial = tk.StringVar(value=str(args.initial_state))
        self.port = tk.StringVar(value=str(args.port))
        options = ttk.Frame(header)
        options.grid(row=2, column=0, columnspan=3, sticky="w", pady=7)
        ttk.Label(options, text="初始布局编号").pack(side="left")
        ttk.Spinbox(options, textvariable=self.initial, from_=0, to=49, width=5).pack(side="left", padx=8)
        ttk.Label(options, text="模型端口").pack(side="left")
        ttk.Entry(options, textvariable=self.port, width=7).pack(side="left", padx=8)
        ttk.Button(options, text="加载／重置场景", command=self.load).pack(side="left", padx=8)
        self.prompt = tk.StringVar(value=args.prompt or self.task_names[args.task_id])
        ttk.Label(header, text="告诉机械臂要做什么（可以打中文，也可以说中文）").grid(row=3, column=0, columnspan=3, sticky="w")
        ttk.Entry(header, textvariable=self.prompt).grid(row=4, column=0, columnspan=3, sticky="ew", pady=6)
        ttk.Label(header, text="首次加载场景；后续直接换指令并执行，无需重置物体和机械臂。", foreground="#555555").grid(
            row=5, column=0, columnspan=3, sticky="w")
        self.checkpoint_text = tk.StringVar(value="模型权重：加载场景后显示")
        ttk.Label(header, textvariable=self.checkpoint_text, wraplength=850, foreground="#555555").grid(
            row=6, column=0, columnspan=3, sticky="w")
        cameras = ttk.Frame(self.root, padding=(12, 0))
        cameras.grid(row=1, column=0, sticky="ew")
        self.cameras = []
        for index, title in enumerate(("场景相机（模型所见）", "腕部相机（模型所见）")):
            cameras.columnconfigure(index, weight=1)
            frame = ttk.LabelFrame(cameras, text=title, padding=4)
            frame.grid(row=0, column=index, padx=5, sticky="ew")
            label = ttk.Label(frame, text="加载场景后显示", anchor="center")
            label.pack(fill="both", expand=True)
            self.cameras.append(label)
        self.book = ttk.Notebook(self.root)
        self.book.grid(row=2, column=0, sticky="nsew", padx=12, pady=8)
        self.hybrid_status = tk.StringVar(value="未启动；使用已有官方 π0.5 服务 8766。")
        self._operation_page()
        self._correction_page()
        self._training_page()
        self._results_page()
        self.status = tk.StringVar(value="请选择任务并加载场景。")
        ttk.Label(self.root, textvariable=self.status, wraplength=850, padding=12, foreground="#174f80").grid(
            row=3, column=0, sticky="ew")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.refresh()

    def page(self, name):
        frame = ttk.Frame(self.book, padding=12)
        frame.columnconfigure(0, weight=1)
        self.book.add(frame, text=name)
        return frame

    def row_buttons(self, parent, row, buttons):
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, sticky="w", pady=7)
        result = []
        for text, callback in buttons:
            button = ttk.Button(frame, text=text, command=lambda callback=callback: self.perform(callback))
            button.pack(side="left", padx=(0, 8))
            result.append(button)
        return result

    def perform(self, operation):
        try:
            operation()
        except Exception as error:
            self.lab._update(status="操作失败：" + str(error))

    def send(self, command, *values):
        if self.hybrid and self.hybrid.poll() is None:
            raise ValueError("请先完成或停止分层实验，再操作原场景")
        if self.training and self.training.poll() is None:
            raise ValueError("训练进行中，请等待完成后再操作场景")
        if not self.lab.send(command, *values):
            raise ValueError("上一动作还未完成，请松开摇杆稍候")

    def load(self):
        self.perform(lambda: self.send("load", self.task.current(), int(self.initial.get()),
                                      self.prompt.get(), int(self.port.get())))

    def run_model(self):
        limit = int(self.step_limit.get())
        if not 0 <= limit <= 10000:
            raise ValueError("控制步数上限须为 0～10000；0 表示不限步数")
        self.send("run", self.prompt.get(), limit)

    def _operation_page(self):
        page = self.page("仿真与语音")
        self.step_limit = tk.StringVar(value=str(self.args.max_steps or 0))
        self.row_buttons(page, 0, (("执行当前指令", lambda: self.send("semantic", self.prompt.get())),
                                   ("暂停执行", lambda: self.send("pause")),
                                   ("保存本轮结果", lambda: self.send("save"))))
        self.row_buttons(page, 1, (("重置三维视角", lambda: self.send("camera", False)),
                                   ("俯视对准", lambda: self.send("camera", True)),
                                   ("开始／结束 Demo", lambda: self.send("video"))))
        ttk.Label(page, text="使用顺序：首次加载场景 → 输入／说出指令 → 点击“执行当前指令”。\n"
                  "视觉模型理解中文、选物体 → RGB-D 定位与通用抓放控制 → 松爪撤手 → 视觉核对后停止。\n"
                  "完成或未确认都保留现场；改写指令再次执行，不用重置，不用再点 VLA。\n"
                  "视觉解析与核对使用本机 CPU，各可能需 2～3 分钟；等待时不发送动作。\n"
                  "上方采用视觉＋几何控制，不是 VLA；适用于分离、直立且夹爪可容纳的物体。复杂摆放可能失败。",
                  wraplength=835).grid(row=2, column=0, sticky="w", pady=8)
        limit = ttk.Frame(page)
        limit.grid(row=3, column=0, sticky="w", pady=6)
        ttk.Label(limit, text="VLA 调试上限（0＝不限；不影响上方指令执行）").pack(side="left")
        ttk.Entry(limit, textvariable=self.step_limit, width=7).pack(side="left", padx=8)
        from desktop_panel import Panel
        devices = Panel.capture_devices()
        self.device = tk.StringVar(value=devices[0] if devices else "")
        ttk.Label(page, text="麦克风设备（ALSA）").grid(row=4, column=0, sticky="w", pady=(10, 0))
        ttk.Combobox(page, textvariable=self.device, values=devices).grid(row=5, column=0, sticky="ew", pady=5)
        self.mic = ttk.Button(page, text="开始录音", command=lambda: self.perform(self.toggle_recording))
        self.mic.grid(row=6, column=0, sticky="w", pady=6)
        self.voice_text = tk.StringVar(value="直接说中文，例如：“帮我把碗放到盘子上。” 识别后核对，再点击“执行当前指令”。")
        ttk.Label(page, textvariable=self.voice_text, wraplength=835).grid(row=7, column=0, sticky="w", pady=6)
        self.row_buttons(page, 8, (("仅 VLA 调试", self.run_model),
                                   ("旧版分层实验", lambda: self.start_hybrid("fresh")),
                                   ("停止规划执行", self.stop_hybrid)))
        ttk.Label(page, text="下方为独立调试入口，不是上方执行按钮的下一步。\n"
                  "仅 VLA 调试：直接连续预测动作，自定义指令没有自动完成核对，需手动暂停。\n"
                  "旧版分层实验会打开独立窗口、固定布局 0；日常连续指令请用上方按钮。",
                  wraplength=835).grid(row=9, column=0, sticky="w", pady=6)
        ttk.Label(page, textvariable=self.hybrid_status, wraplength=835, foreground="#174f80").grid(
            row=10, column=0, sticky="w", pady=6)

    def _correction_page(self):
        page = self.page("人工纠正")
        ttk.Label(page, text="暂停后接管，像之前一样用摇杆／按钮纠正。保存合格前请核对动作。\n"
                  "世界坐标：红 X、绿 Y、蓝 Z；摇杆向右为 +X，向上为 +Y。\n"
                  "草稿不训练；评测片段也不训练，同一布局不能跨两组。", wraplength=835).grid(row=0, column=0, sticky="w")
        line = ttk.Frame(page)
        line.grid(row=1, column=0, sticky="w", pady=8)
        self.split = tk.StringVar(value="训练集")
        ttk.Combobox(line, textvariable=self.split, values=("训练集", "留出评测集"), state="readonly", width=12).pack(side="left")
        ttk.Button(line, text="开始人工接管", command=lambda: self.perform(lambda: self.send(
            "begin", "train" if self.split.get() == "训练集" else "holdout"))).pack(side="left", padx=10)
        controls = ttk.Frame(page)
        controls.grid(row=2, column=0, sticky="w")
        self.joystick = TranslationJoystick(controls, self.joystick_move, lambda error: self.status.set(str(error)))
        self.joystick.grid(row=0, column=0, rowspan=4, sticky="n", padx=(0, 12))
        self.amplitude = tk.DoubleVar(value=.6)
        self.repeats = tk.IntVar(value=5)
        options = ttk.Frame(controls)
        options.grid(row=0, column=1, sticky="w")
        for title, value in (("普通移动", .6), ("精细抓放", .1)):
            ttk.Radiobutton(options, text=title, variable=self.amplitude, value=value).pack(side="left", padx=4)
        ttk.Label(options, text="连续步数").pack(side="left", padx=5)
        ttk.Spinbox(options, textvariable=self.repeats, from_=1, to=20, width=3).pack(side="left")
        jog = ttk.Frame(controls)
        jog.grid(row=1, column=1, sticky="w")
        self.manual_buttons = []
        for axis, name in enumerate(("X", "Y", "Z", "Rx", "Ry", "Rz")):
            for row, sign in enumerate((1, -1)):
                button = ttk.Button(jog, text=f"{name}{'+' if sign > 0 else '−'}", width=4,
                    command=lambda axis=axis, sign=sign: self.perform(lambda: self.jog(axis, sign)))
                button.grid(row=row, column=axis, padx=2, pady=4)
                self.manual_buttons.append(button)
        grip = ttk.Frame(controls)
        grip.grid(row=2, column=1, sticky="w")
        for title, value in (("松爪 10 步", -1.), ("闭爪 10 步", 1.), ("保持 10 步", None)):
            button = ttk.Button(grip, text=title, command=lambda value=value: self.perform(lambda: self.gripper(value)))
            button.pack(side="left", padx=3, pady=5)
            self.manual_buttons.append(button)
        self.row_buttons(page, 3, (("保存为合格片段", lambda: self.finish(True)),
                                   ("结束并存草稿", lambda: self.finish(False)),
                                   ("打开纠正目录", lambda: self.open_path(ROOT / "corrections"))))
        self.frame_text = tk.StringVar(value="尚未开始人工接管")
        ttk.Label(page, textvariable=self.frame_text, wraplength=835).grid(row=4, column=0, sticky="w", pady=6)
        ttk.Label(page, text="合格＝你已审核的训练示范，不等于自动证明任务成功。\n"
                  "普通幅度 0.6 用于空中搬运，精细幅度 0.1 用于接近和抓放。\n"
                  "暂停不记帧，不对旧视频加速，不改写真实控制动作。", wraplength=835).grid(row=5, column=0, sticky="w")

    def jog(self, axis, sign):
        action = np.zeros(7)
        action[-1] = self.grip
        action[axis] = sign * (self.amplitude.get() if axis < 3 else .1)
        self.send("manual", action.tolist(), self.repeats.get())

    def joystick_move(self, x, y):
        if not self.lab.state()["manual"]:
            return False
        action = [x * self.amplitude.get(), y * self.amplitude.get(), 0., 0., 0., 0., self.grip]
        return self.lab.send("manual", action, 1)

    def gripper(self, value):
        if value is not None:
            self.grip = value
        self.send("manual", [0.] * 6 + [self.grip], 10)

    def finish(self, approved):
        self.joystick.stop()
        if approved and not messagebox.askyesno("审核人工片段", "确认这一整段动作可用于学习？\n多余或错误动作请先存草稿。", parent=self.root):
            return
        self.send("finish", approved)

    def _training_page(self):
        page = self.page("模型与训练")
        self.parent = tk.StringVar(value=str(ROOT / "models/pi05_libero"))
        ttk.Label(page, text="π0.5 父模型目录（本页微调仅适用于 π0.5）").grid(row=0, column=0, sticky="w")
        ttk.Entry(page, textvariable=self.parent).grid(row=1, column=0, sticky="ew", pady=5)
        self.row_buttons(page, 2, (("启动所选模型", self.start_model),
                                   ("停止本面板模型", self.stop_model),
                                   ("打开模型目录", lambda: self.open_path(ROOT / "checkpoints"))))
        opts = ttk.Frame(page)
        opts.grid(row=3, column=0, sticky="w", pady=6)
        self.train_values = {}
        for index, (name, title, value) in enumerate((("steps", "更新次数", "2000"), ("batch", "每批窗口", "1"),
                                                    ("lr", "学习率", "5e-5"))):
            ttk.Label(opts, text=title).grid(row=0, column=index*2)
            variable = self.train_values[name] = tk.StringVar(value=value)
            ttk.Entry(opts, textvariable=variable, width=8).grid(row=0, column=index*2+1, padx=6)
        self.dataset_hint = tk.StringVar(value="先准备数据：仅收录本项目审核合格的片段，训练与评测按场景隔离。")
        ttk.Label(page, textvariable=self.dataset_hint, wraplength=835).grid(row=4, column=0, sticky="w", pady=6)
        self.row_buttons(page, 5, (("准备／校验数据", self.prepare), ("训练前检查", lambda: self.train(True)),
                                   ("开始微调", lambda: self.train(False)), ("停止本面板训练", self.stop_training)))
        ttk.Label(page, text="至少 2 条训练片段＋1 条独立布局评测片段。新项目不自动混入旧黑碗数据。\n"
                  "仅 LoRA 参数更新；真实训练使用本机 GPU。先检查，点击开始才训练。\n"
                  "这里只管理本面板启动的进程；不会结束其它 GPU 任务。", wraplength=835).grid(row=6, column=0, sticky="w", pady=5)
        self.log = tk.Text(page, height=7, wrap="word", state="disabled", font="TkDefaultFont")
        self.log.grid(row=7, column=0, sticky="nsew", pady=7)
        page.rowconfigure(7, weight=1)
        self.manifest = None

    def start_model(self):
        if self.lab.state()["ready"]:
            raise ValueError("请先关闭三维场景再切换模型；保存片段后关闭 MuJoCo 窗口即可")
        if self.training and self.training.poll() is None:
            raise ValueError("训练运行中")
        if self.model and self.model.poll() is None:
            raise ValueError("本面板模型已运行，请先停止")
        parent = Path(self.parent.get()).expanduser().resolve()
        if not (parent / "params").is_dir():
            raise ValueError("模型目录缺少 params")
        port = 8767
        import socket
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                raise ValueError("8767 端口被占用，请在原面板管理该进程")
        (ROOT / "logs").mkdir(exist_ok=True)
        self.model_log = ROOT / "logs" / f"panel-model-{time.time_ns()}.log"
        with self.model_log.open("w") as stream:
            self.model = subprocess.Popen([str(ROOT / ".venv-vla/bin/python"), "-u", "-m", "vla.serve",
                "--checkpoint", str(parent), "--port", str(port)], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        self.port.set(str(port))
        self.lab._update(status="模型正在加载到 8767；等日志出现 server listening 后加载场景。")

    def stop_model(self):
        if self.lab.state()["ready"]:
            raise ValueError("先保存片段并关闭 MuJoCo 窗口")
        if self.model and self.model.poll() is None:
            self.model.terminate()

    def prepare(self):
        from vla.train_corrections import prepare_dataset
        self.manifest = prepare_dataset(ROOT / "corrections", ROOT / "runs" / f"dataset_{time.time_ns()}")
        self.dataset_hint.set("数据已校验：" + str(self.manifest))

    def train(self, dry_run):
        if self.hybrid and self.hybrid.poll() is None:
            raise ValueError("分层实验进行中，请先停止")
        if self.training and self.training.poll() is None:
            raise ValueError("训练／检查仍在进行")
        if self.lab.state()["manual"]:
            raise ValueError("先保存人工纠正片段")
        if self.manifest is None:
            self.prepare()
        if not dry_run:
            if self.lab.state()["ready"]:
                raise ValueError("请先关闭 MuJoCo 场景，再训练")
            if self.model and self.model.poll() is None:
                raise ValueError("请先停止本面板模型，释放显存")
            memory = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
            if int(memory.splitlines()[0]) < 22000:
                raise ValueError("可用显存不足 22GB；其它服务请在其原终端退出，本面板不结束外部进程")
        command = [str(ROOT / ".venv-vla/bin/python"), "-u", "-m", "vla.train_corrections", "--manifest", str(self.manifest),
            "--parent", self.parent.get(), "--experiment", f"human_{time.time_ns()}",
            "--steps", self.train_values["steps"].get(), "--batch-size", self.train_values["batch"].get(),
            "--learning-rate", self.train_values["lr"].get()]
        if dry_run:
            command.append("--dry-run")
        self.training_log = ROOT / "logs" / f"panel-training-{time.time_ns()}.log"
        self.training_log.parent.mkdir(exist_ok=True)
        environment = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false", OMP_NUM_THREADS="4")
        if dry_run:
            environment["JAX_PLATFORMS"] = "cpu"
        with self.training_log.open("w") as stream:
            self.training = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        self.lab._update(status="训练前检查进行中" if dry_run else "LoRA 微调进行中；日志见本页")

    def stop_training(self):
        if self.training and self.training.poll() is None:
            self.training.send_signal(signal.SIGINT)

    def _results_page(self):
        page = self.page("结果与录像")
        self.row_buttons(page, 0, (("打开本轮结果", self.open_result),
                                   ("打开所有结果", lambda: self.open_path(Path(self.args.output))),
                                   ("打开数据目录", lambda: self.open_path(ROOT / "corrections"))))
        ttk.Label(page, text="每个场景单独保存初始状态、模型来源、预测动作和已执行动作。\n"
                  "报告中的官方判据与人工审核分开记录。\n\n"
                  "三维窗口关闭后可重新加载；关闭面板会存草稿并停止本面板启动的模型／训练。\n"
                  "旧训练工程和旧纠正数据保持原样。", wraplength=835).grid(row=1, column=0, sticky="w", pady=10)
        hybrid = ttk.LabelFrame(page, text="HAMSTER 分层模型实验 · 独立 MuJoCo 场景", padding=10)
        hybrid.grid(row=2, column=0, sticky="ew", pady=10)
        ttk.Label(hybrid, text="HAMSTER 选目标，π0.5 抓取，RGB-D／OSC 搬运和松爪。\n"
                  "前两个按钮复现已审核路径；新指令按钮会重新运行真实 HAMSTER。\n"
                  "固定桌面布局 0，仅单次抓放；不使用上方布局，不训练模型。\n"
                  "新指令无自动语义评分，请检查录像；CPU 规划可能需要几分钟。", wraplength=790).grid(row=0, column=0, sticky="w")
        self.row_buttons(hybrid, 1, (("酒瓶放到碗里", lambda: self.start_hybrid("01")),
                                     ("碗放到盘子上", lambda: self.start_hybrid("02")),
                                     ("停止本次实验", self.stop_hybrid)))
        self.row_buttons(hybrid, 2, (("按上方中文指令重新规划并执行", lambda: self.start_hybrid("fresh")),))
        self.row_buttons(hybrid, 3, (("打开实验结果", self.open_hybrid),))
        ttk.Label(hybrid, textvariable=self.hybrid_status, wraplength=790).grid(row=4, column=0, sticky="w")

    def start_hybrid(self, case):
        if self.hybrid and self.hybrid.poll() is None:
            raise ValueError("分层模型实验仍在运行")
        if self.training and self.training.poll() is None:
            raise ValueError("训练进行中，请等待完成")
        if self.lab.state()["running"] or self.lab.state()["manual"]:
            raise ValueError("请先暂停当前模型并保存人工片段")
        if case == "fresh":
            prompt = self.prompt.get().strip()
            if not prompt or len(prompt) > 500:
                raise ValueError("请在上方输入 1～500 字的单次抓放指令，也可先录音填写")
        else:
            plan = ROOT / "evidence/hamster/trial-init0-protobuf" / f"command-{case}.json"
            if case not in ("01", "02") or not plan.is_file():
                raise ValueError("没有对应的已审核模型路径")
        self.hybrid_output = ROOT / "runs" / f"hybrid-panel-{case}-{time.time_ns()}"
        self.hybrid_output.parent.mkdir(parents=True, exist_ok=True)
        # Keep the log outside: the runner refuses nonempty output directories.
        log_path = self.hybrid_output.with_suffix(".log")
        command = [str(ROOT / ".venv-vla-sim/bin/python"), "-u", "-m"]
        if case == "fresh":
            command += ["vla.hamster_command", "--prompt", prompt]
        else:
            command += ["vla.hybrid", "--plan", str(plan)]
        command += ["--output", str(self.hybrid_output), "--viewer"]
        with log_path.open("w") as stream:
            self.hybrid = subprocess.Popen(command, start_new_session=True,
                cwd=ROOT, env=dict(os.environ, MUJOCO_GL="glfw"), stdout=stream, stderr=subprocess.STDOUT)
        self.hybrid_status.set("运行中：三维窗口可旋转视角，完成后自动保存录像。日志：" + str(log_path))

    def stop_hybrid(self):
        if self.hybrid and self.hybrid.poll() is None:
            # Every experiment owns a new session; include its CPU planner child.
            try:
                os.killpg(self.hybrid.pid, signal.SIGINT)
            except ProcessLookupError:
                pass

    def open_hybrid(self):
        if self.hybrid_output is None:
            raise ValueError("尚未运行分层实验")
        self.open_path(self.hybrid_output)

    def open_result(self):
        output = self.lab.state().get("output")
        if not output:
            raise ValueError("尚未加载场景")
        self.open_path(Path(output))

    @staticmethod
    def open_path(path):
        path.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(["xdg-open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def toggle_recording(self):
        if self.recording:
            self.finish_recording()
            return
        if self.voice_worker and self.voice_worker.is_alive():
            raise ValueError("语音识别中")
        if not self.device.get().strip():
            raise ValueError("请连接麦克风，或填写 ALSA 设备名")
        self.record_temp = tempfile.TemporaryDirectory(prefix="vla-voice-")
        self.recording = subprocess.Popen(["arecord", "-D", self.device.get().strip(), "-f", "S16_LE",
            "-r", "16000", "-c", "1", "-d", "30", "-t", "wav", "-q", str(Path(self.record_temp.name) / "instruction.wav")],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.mic.configure(text="结束录音并填写指令")
        self.voice_text.set("录音中，最长 30 秒；录音不会直接执行动作。")

    def finish_recording(self, submit=True):
        process, temp = self.recording, self.record_temp
        self.recording = self.record_temp = None
        self.mic.configure(text="识别中……", state="disabled")
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
        def finish():
            try:
                process.wait(timeout=3)
                if submit:
                    path = Path(temp.name) / "instruction.wav"
                    with wave.open(str(path)) as audio:
                        if audio.getnframes() < 4000:
                            raise ValueError("录音不足 0.25 秒")
                    self.events.put(("voice", voice_prompt(path.read_bytes())))
            except Exception as error:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                self.events.put(("error", str(error)))
            finally:
                temp.cleanup()
                self.events.put(("voice_done", None))
        self.voice_worker = threading.Thread(target=finish, daemon=True)
        self.voice_worker.start()

    def refresh(self):
        state = self.lab.state()
        backend = state.get("policy_backend")
        if backend:
            model = "MolmoAct2" if backend == "official-molmoact2-libero-vla" else "π0.5"
            self.model_title.set("视觉指令控制 · MuJoCo 本地仿真")
            self.checkpoint_text.set(model + " 独立 VLA 调试权重：" + (state.get("checkpoint_source") or backend))
        if state["manual"] and not self.was_manual:
            self.grip = state.get("grip", -1.)
        self.was_manual = state["manual"]
        self.status.set(state["status"])
        if self.hybrid and self.hybrid.poll() is not None:
            result_path = self.hybrid_output / "result.json"
            if not result_path.is_file():
                result_path = self.hybrid_output / "execution/result.json"
            if result_path.is_file():
                result = json.loads(result_path.read_text())
                self.hybrid_status.set(f"实验结束：{result['result']}；{result['control_steps']} 个控制步。"
                                       "点击打开实验结果查看 rollout.mp4。")
            else:
                self.hybrid_status.set(f"实验退出（{self.hybrid.returncode}），请检查相邻 .log 文件。")
            self.hybrid = None
        elif self.hybrid:
            workflow = self.hybrid_output / "workflow.json"
            if workflow.is_file():
                progress = json.loads(workflow.read_text())
                stages = progress.get("stages", [])
                stage = stages[-1]["stage"] if stages else "translation"
                self.hybrid_status.set("新指令运行中：" + {"capture": "拍摄初始场景", "planner": "CPU 模型规划",
                    "execution": "MuJoCo 抓放", "translation": "中文识别／翻译"}.get(stage, stage)
                    + "；结果：" + str(self.hybrid_output))
        self.frame_text.set(f"已执行 {state['steps']} 个控制步；本段人工动作 {state['frames']} 帧。")
        self.joystick.set_enabled(state["manual"] and not self.closing)
        for button in self.manual_buttons:
            button.configure(state="normal" if state["manual"] else "disabled")
        images = state["images"]
        if images is not None and images is not self.preview_identity:
            from PIL import Image, ImageTk
            self.photo_refs = [ImageTk.PhotoImage(Image.fromarray(array)) for array in images]
            for label, photo in zip(self.cameras, self.photo_refs):
                label.configure(image=photo, text="")
            self.preview_identity = images
        while not self.events.empty():
            kind, value = self.events.get_nowait()
            if kind == "voice":
                self.voice_text.set("识别原文：" + value[0])
                self.prompt.set(value[0])
            elif kind == "voice_done":
                self.mic.configure(text="开始录音", state="normal")
            else:
                self.lab._update(status="语音失败：" + value)
        if self.recording and self.recording.poll() is not None:
            self.finish_recording()
        if self.training_log and self.training_log.exists():
            text = self.training_log.read_text(errors="replace")[-10000:]
            if self.training and self.training.poll() is not None:
                text += f"\n进程退出码：{self.training.returncode}"
            self.log.configure(state="normal")
            self.log.delete("1.0", "end")
            self.log.insert("end", text)
            self.log.see("end")
            self.log.configure(state="disabled")
        elif self.model_log and self.model_log.exists():
            self.log.configure(state="normal")
            self.log.delete("1.0", "end")
            self.log.insert("end", self.model_log.read_text(errors="replace")[-10000:])
            self.log.configure(state="disabled")
        while not self.lab.events.empty():
            print(self.lab.events.get_nowait(), flush=True)
        if self.closing and not self.lab.thread.is_alive() and not (self.voice_worker and self.voice_worker.is_alive()):
            if self.model and self.model.poll() is None:
                self.model.terminate()
            self.root.destroy()
            return
        self.root.after(150, self.refresh)

    def close(self):
        self.closing = True
        self.joystick.stop()
        if self.recording:
            self.finish_recording(False)
        self.stop_training()
        self.stop_hybrid()
        self.lab.send("close")
        # Only processes created by this panel are stopped. External services stay up.


def panel(args):
    ChinesePanel(args).root.mainloop()
