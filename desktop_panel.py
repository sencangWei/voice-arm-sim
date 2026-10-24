"""Native Tk controls; simulator rendering stays in the MuJoCo window."""
import argparse
import json
import os
from pathlib import Path
import queue
import random
import re
import signal
import subprocess
import tempfile
import threading
import tkinter as tk
from tkinter import ttk
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import wave


class Panel:
    def __init__(self, root, url):
        self.root, self.url = root, url
        self.updates = queue.Queue()
        self.closing = threading.Event()
        self.state = {}
        self.pending = False
        self.recording = None
        self.record_temp = None
        self.record_error = None
        root.title("语音机械臂 · MuJoCo 原生控制")
        root.geometry("570x680+30+50")
        root.option_add("*Font", "{Noto Sans CJK SC} 11")
        root.protocol("WM_DELETE_WINDOW", self.close)
        main = ttk.Frame(root, padding=18)
        main.pack(fill="both", expand=True)
        ttk.Label(main, text="MuJoCo 桌面机械臂", font=("Noto Sans CJK SC", 17, "bold")).pack(anchor="w")
        ttk.Label(main, text="在 MuJoCo 窗口拖动旋转视角、右键平移、滚轮缩放。\n本面板只发送指令，不传输仿真图片。", wraplength=520).pack(anchor="w", pady=(6, 14))
        self.text = tk.StringVar(value="请把绿色苹果放到蓝色托盘里面")
        self.entry = ttk.Entry(main, textvariable=self.text)
        self.entry.pack(fill="x")
        self.entry.bind("<Return>", lambda event: self.command())
        self.send = ttk.Button(main, text="执行指令", command=self.command)
        self.send.pack(fill="x", pady=6)
        row = ttk.Frame(main)
        row.pack(fill="x")
        self.task_buttons = []
        for name in ("红色方块", "苹果", "香蕉"):
            button = ttk.Button(row, text=name + " → 托盘", command=lambda n=name: self.command("把" + n + "放进蓝色托盘"))
            button.pack(side="left", expand=True, fill="x", padx=2)
            self.task_buttons.append(button)
        take_out = ttk.Button(main, text="香蕉：从盘子取出 → 桌面空位",
                              command=lambda: self.command("把香蕉从盘子取出来"))
        take_out.pack(fill="x", pady=4)
        self.task_buttons.append(take_out)
        settings = ttk.Frame(main)
        settings.pack(fill="x", pady=10)
        ttk.Label(settings, text="定位方式").pack(side="left")
        self.mode = tk.StringVar(value="自动（优先 Qwen）")
        self.modes = {"自动（优先 Qwen）": "auto", "Qwen + RGB-D": "vlm", "颜色 RGB-D 基线": "color"}
        self.mode_box = ttk.Combobox(settings, textvariable=self.mode, values=list(self.modes), state="readonly", width=23)
        self.mode_box.pack(side="right")
        microw = ttk.Frame(main)
        microw.pack(fill="x")
        ttk.Label(microw, text="麦克风设备").pack(side="left")
        devices = self.capture_devices()
        self.device = tk.StringVar(value=devices[0] if devices else "")
        self.device_box = ttk.Combobox(microw, textvariable=self.device, values=devices, width=25)
        self.device_box.pack(side="right")
        self.mic = ttk.Button(main, text="开始录音", command=self.toggle_recording)
        self.mic.pack(fill="x", pady=6)
        ttk.Label(main, text="说完再次点击结束并执行，最长30秒。需连接真实麦克风。", wraplength=520).pack(anchor="w")
        actions = ttk.Frame(main)
        actions.pack(fill="x", pady=12)
        self.pause = ttk.Button(actions, text="暂停 / 继续", command=lambda: self.action("/api/pause", {}))
        self.pause.pack(side="left", expand=True, fill="x")
        ttk.Button(actions, text="停止", command=self.stop).pack(side="left", expand=True, fill="x", padx=5)
        self.reset = ttk.Button(actions, text="随机布局", command=lambda: self.action("/api/reset", {"seed": random.randrange(2**31), "mode": self.modes[self.mode.get()]}))
        self.reset.pack(side="left", expand=True, fill="x")
        self.status = tk.StringVar(value="正在连接仿真…")
        ttk.Label(main, textvariable=self.status, wraplength=520, justify="left").pack(anchor="w", pady=8)
        self.feedback = tk.StringVar(value="支持放进托盘、从盘子取出。取出默认放到相机确认的桌面空位。SPACE暂停/继续，S停止。")
        ttk.Label(main, textvariable=self.feedback, wraplength=520, foreground="#145b8c").pack(anchor="w", pady=8)
        ttk.Label(main, text="此窗口运行仿真推理；尚未加入策略训练。", foreground="#666666").pack(side="bottom", anchor="w")
        threading.Thread(target=self.poll, daemon=True).start()
        root.after(100, self.consume)

    @staticmethod
    def capture_devices():
        try:
            result = subprocess.run(["arecord", "-l"], env={**os.environ, "LC_ALL": "C"}, capture_output=True, text=True, timeout=2)
            return [f"plughw:{card},{device}" for card, device in re.findall(r"card (\d+):.*?device (\d+):", result.stdout)]
        except (OSError, subprocess.TimeoutExpired):
            return []

    def request(self, path, data=None, audio=False, mode=None):
        headers = {"Content-Type": "audio/wav" if audio else "application/json"}
        if mode:
            headers["X-Vision-Mode"] = mode
        body = data if audio else json.dumps(data).encode() if data is not None else None
        try:
            with urlopen(Request(self.url + path, data=body, headers=headers), timeout=125 if audio else 3) as response:
                return json.load(response)
        except HTTPError as exc:
            raise ValueError(json.load(exc).get("error", str(exc))) from exc

    def poll(self):
        while not self.closing.is_set():
            try:
                self.updates.put(("state", self.request("/api/state")))
            except Exception as exc:
                self.updates.put(("connection", str(exc)))
            self.closing.wait(.3)

    def action(self, path, data, audio=False):
        if self.pending and path not in ("/api/stop", "/api/pause"):
            return
        blocking = path not in ("/api/stop", "/api/pause")
        if blocking:
            self.pending = True
        mode = self.modes[self.mode.get()]
        self.refresh_controls()
        def work():
            try:
                result = self.request(path, data, audio=audio, mode=mode)
                self.updates.put(("reply", result.get("text") or result.get("message", "已发送")))
            except Exception as exc:
                self.updates.put(("reply", "操作失败：" + str(exc)))
            finally:
                if blocking:
                    self.updates.put(("released", None))
        threading.Thread(target=work, daemon=True).start()

    def command(self, text=None):
        if self.state.get("busy", True) or self.pending or self.recording is not None:
            return
        command = (text if text is not None else self.text.get()).strip()
        if command:
            self.text.set(command)
            self.action("/api/command", {"text": command, "mode": self.modes[self.mode.get()]})

    def refresh_controls(self):
        busy = self.state.get("busy", True) or self.pending or self.recording is not None
        for widget in [self.entry, self.send, self.reset, *self.task_buttons]:
            widget.configure(state="disabled" if busy else "normal")
        self.mode_box.configure(state="disabled" if busy else "readonly")
        self.device_box.configure(state="disabled" if busy else "normal")
        self.mic.configure(state="normal" if self.recording is not None or not busy else "disabled",
                           text="结束录音并执行" if self.recording is not None else "开始录音")
        self.pause.configure(state="normal" if self.state.get("busy") and not self.pending else "disabled")

    def consume(self):
        while True:
            try:
                kind, value = self.updates.get_nowait()
            except queue.Empty:
                break
            if kind == "state":
                self.state = value
                models = value.get("models", {})
                result = value.get("last_result") or {}
                self.status.set(f"阶段：{value.get('phase')}  |  控制步：{value.get('steps', 0)}\n{value.get('message', '')}\n定位：{value.get('detection_source', '')}\n{models.get('asr', '')}\n{models.get('vision', '')}" + (f"\n结果：{result.get('message', '')}" if result else ""))
            elif kind == "connection":
                self.state = {"busy": True}
                self.status.set("连接仿真中：" + value)
            elif kind == "released":
                self.pending = False
            else:
                self.feedback.set(value)
        if self.recording is not None and self.recording.poll() is not None:
            self.finish_recording(True)
        self.refresh_controls()
        if not self.closing.is_set():
            self.root.after(100, self.consume)

    def toggle_recording(self):
        if self.recording is not None:
            self.finish_recording(True)
            return
        if self.state.get("busy", True) or self.pending:
            return
        device = self.device.get().strip()
        if not device:
            self.feedback.set("未检测到采集设备。请连接麦克风后重启，或输入ALSA设备名。")
            return
        self.record_temp = tempfile.TemporaryDirectory(prefix="voice-arm-")
        self.record_error = open(Path(self.record_temp.name) / "error.log", "wb")
        try:
            self.recording = subprocess.Popen(["arecord", "-D", device, "-f", "S16_LE", "-r", "16000", "-c", "1", "-d", "30", "-t", "wav", "-q", str(Path(self.record_temp.name) / "command.wav")], stdout=subprocess.DEVNULL, stderr=self.record_error)
            self.feedback.set("正在录音，请说指令；再次点击结束并执行。")
        except OSError as exc:
            self.record_error.close()
            self.record_temp.cleanup()
            self.record_temp = None
            self.feedback.set("录音无法启动：" + str(exc))
        self.refresh_controls()

    def finish_recording(self, submit):
        process, temp, error = self.recording, self.record_temp, self.record_error
        self.recording = self.record_temp = self.record_error = None
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
        self.pending = True
        self.refresh_controls()
        def finish():
            try:
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    raise ValueError("录音进程未正常结束")
                error.close()
                if not submit:
                    return
                path = Path(temp.name) / "command.wav"
                if not path.exists():
                    raise ValueError((Path(temp.name) / "error.log").read_text(errors="replace")[:300] or "未生成录音")
                with wave.open(str(path)) as audio:
                    if audio.getnframes() < 4000:
                        raise ValueError("录音不足0.25秒或设备不可用，请检查麦克风")
                result = self.request("/api/audio", path.read_bytes(), audio=True, mode=mode)
                self.updates.put(("reply", "语音识别：" + result.get("text", result.get("message", ""))))
            except Exception as exc:
                self.updates.put(("reply", "录音失败：" + str(exc)))
            finally:
                temp.cleanup()
                self.updates.put(("released", None))
        mode = self.modes[self.mode.get()]
        threading.Thread(target=finish, daemon=True).start()

    def stop(self):
        if self.recording is not None:
            self.finish_recording(False)
        self.action("/api/stop", {})

    def close(self):
        self.closing.set()
        if self.recording is not None:
            self.finish_recording(False)
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8910")
    args = parser.parse_args()
    root = tk.Tk()
    Panel(root, args.url.rstrip("/"))
    root.mainloop()


if __name__ == "__main__":
    main()
