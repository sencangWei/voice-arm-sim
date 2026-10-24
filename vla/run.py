#!/usr/bin/env python3
"""Bounded genuine VLA rollout; no hardware or scripted command mapping."""
import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import sys
import threading
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".local/openpi"
for path in (SOURCE / "packages/openpi-client/src", SOURCE / "third_party/libero"):
    sys.path.insert(0, str(path))

LIMITS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300,
          "libero_10": 520, "libero_90": 400}


def configure_libero():
    config = ROOT / ".local/vla-libero"
    config.mkdir(parents=True, exist_ok=True)
    base = SOURCE / "third_party/libero/libero/libero"
    content = "\n".join(f"{key}: {value}" for key, value in {
        "benchmark_root": base, "bddl_files": base / "bddl_files",
        "init_states": base / "init_files", "assets": base / "assets",
        "datasets": ROOT / ".local/libero-datasets"}.items()) + "\n"
    (ROOT / ".local/libero-datasets").mkdir(parents=True, exist_ok=True)
    (config / "config.yaml").write_text(content)
    os.environ["LIBERO_CONFIG_PATH"] = str(config)


def load_initial_states(task):
    # Pinned official LIBERO files contain numpy pickle arrays (PyTorch >=2.6).
    # Never deserialize a user-supplied state/checkpoint path here.
    import torch
    base = (SOURCE / "third_party/libero/libero/libero/init_files").resolve()
    path = (base / task.problem_folder / task.init_states_file).resolve()
    if not path.is_relative_to(base):
        raise ValueError("initial state must belong to pinned LIBERO source")
    return torch.load(path, weights_only=False)


def observation_payload(obs, prompt, image_tools):
    """Official rotate180, resize/pad224, position+axis-angle+gripper state8."""
    images = [image_tools.convert_to_uint8(image_tools.resize_with_pad(
        np.ascontiguousarray(obs[key][::-1, ::-1]), 224, 224))
        for key in ("agentview_image", "robot0_eye_in_hand_image")]
    quat = np.asarray(obs["robot0_eef_quat"], dtype=float).copy()
    w = np.clip(quat[3], -1., 1.)
    den = np.sqrt(1. - w * w)
    rotation = np.zeros(3) if math.isclose(den, 0.) else quat[:3] * 2. * math.acos(w) / den
    state = np.concatenate((obs["robot0_eef_pos"], rotation, obs["robot0_gripper_qpos"]))
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError("LIBERO state must be finite shape (8,)")
    return {"observation/image": images[0], "observation/wrist_image": images[1],
            "observation/state": state, "prompt": prompt}


def validate_actions(actions, replan_steps):
    actions = np.asarray(actions)
    if (actions.ndim != 2 or actions.shape[1] != 7 or len(actions) < replan_steps
            or len(actions) > 10000 or not np.isfinite(actions).all()):
        raise ValueError("policy actions must be finite Nx7, replan_steps <= N <= 10000")
    return actions


def connect_policy(port):
    from openpi_client import websocket_client_policy, msgpack_numpy
    import websockets.sync.client

    class BoundedPolicy(websocket_client_policy.WebsocketClientPolicy):
        def _wait_for_server(self):
            conn = websockets.sync.client.connect(self._uri, compression=None,
                max_size=64 * 1024 * 1024, open_timeout=15)
            try:
                return conn, msgpack_numpy.unpackb(conn.recv(timeout=120))
            except BaseException:
                conn.close()
                raise

        def infer(self, obs):
            self._ws.send(self._packer.pack(obs))
            response = self._ws.recv(timeout=120)
            if isinstance(response, str):
                raise RuntimeError(response)
            return msgpack_numpy.unpackb(response)

    return BoundedPolicy("127.0.0.1", port)


def rollout(args, stop=None, paused=None, status=print):
    """Run one episode; only ten official settling steps use dummy actions."""
    configure_libero()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    import imageio.v2 as imageio
    stop = stop or threading.Event()
    paused = paused or threading.Event()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(suite=args.suite, task_id=args.task_id, initial_state=args.initial_state,
        seed=args.seed, replan_steps=args.replan_steps, max_steps=args.max_steps,
        score_definition="selected LIBERO task predicates, independent of prompt override",
        result="exception", steps=0, policy_calls=0)
    env = viewer = client = video = None
    executed_actions = []
    started = time.monotonic()
    try:
        np.random.seed(args.seed)
        suite = benchmark.get_benchmark_dict()[args.suite]()
        if not 0 <= args.task_id < suite.n_tasks:
            raise ValueError("task-id outside suite")
        task = suite.get_task(args.task_id)
        states = load_initial_states(task)
        if not 0 <= args.initial_state < len(states):
            raise ValueError("initial-state outside task states")
        initial = np.asarray(states[args.initial_state])
        report.update(task_language=task.language,
            prompt=task.language if args.prompt is None else args.prompt,
            initial_state_sha256=hashlib.sha256(initial.tobytes()).hexdigest(),
            initial_state_dtype=str(initial.dtype), initial_state_shape=list(initial.shape))
        np.save(output / "initial_state.npy", initial)
        status("Connecting to genuine VLA policy")
        client = connect_policy(args.port)
        report["server_metadata"] = client.get_server_metadata()
        if report["server_metadata"].get("backend") != "official-pi05-libero-vla":
            raise ValueError("Expected the official multitask VLA server on this port")
        env = OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path("bddl_files")) /
            task.problem_folder / task.bddl_file), camera_heights=256, camera_widths=256)
        env.seed(args.seed)
        env.reset()
        obs = env.set_init_state(initial)
        if args.viewer:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(env.sim.model._model, env.sim.data._data)
        video = imageio.get_writer(output / "paired_cameras.mp4", fps=10)
        for _ in range(10):
            if stop.is_set():
                break
            obs, _, _, _ = env.step([0.] * 6 + [-1.])
        plan = collections.deque()
        report["result"] = "failure"
        with (output / "inference.jsonl").open("w") as calls:
            for step in range(args.max_steps):
                while paused.is_set() and not stop.wait(.05):
                    if viewer is not None:
                        if not viewer.is_running():
                            stop.set()
                            break
                        viewer.sync()
                if stop.is_set() or (viewer is not None and not viewer.is_running()):
                    report["result"] = "stopped"
                    break
                payload = observation_payload(obs, report["prompt"], image_tools)
                video.append_data(np.concatenate((payload["observation/image"], payload["observation/wrist_image"]), axis=1))
                if not plan:
                    status(f"Inferring step {step}")
                    tick = time.monotonic()
                    response = client.infer(payload)
                    elapsed = time.monotonic() - tick
                    actions = validate_actions(response["actions"], args.replan_steps)
                    index = report["policy_calls"]
                    np.save(output / f"actions_{index:04d}.npy", actions)
                    calls.write(json.dumps(dict(call=index, step=step, seconds=elapsed,
                        shape=list(actions.shape), queued_prefix=args.replan_steps)) + "\n")
                    calls.flush()
                    report["policy_calls"] += 1
                    plan.extend(actions[:args.replan_steps])
                if stop.is_set():
                    report["result"] = "stopped"
                    break
                action = plan.popleft()
                obs, _, done, _ = env.step(action.tolist())
                executed_actions.append(action.copy())
                report["steps"] = step + 1
                if viewer is not None:
                    viewer.sync()
                if done:
                    report["result"] = "success"
                    break
        payload = observation_payload(obs, report["prompt"], image_tools)
        video.append_data(np.concatenate((payload["observation/image"], payload["observation/wrist_image"]), axis=1))
    except Exception:
        report["result"] = "exception"
        report["exception"] = traceback.format_exc()
    finally:
        for resource in (video, viewer, env, getattr(client, "_ws", None)):
            if resource is not None:
                try:
                    resource.close()
                except Exception as error:
                    report.setdefault("cleanup_errors", []).append(str(error))
        np.save(output / "executed_actions.npy", np.asarray(executed_actions).reshape(-1, 7))
        report["elapsed_seconds"] = time.monotonic() - started
        (output / "report.json").write_text(json.dumps(report, indent=2,
            default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value)) + "\n")
    status(str(report["result"]) + " · " + str(output / "report.json"))
    return report


def voice_prompt(audio, url="http://127.0.0.1:8911"):
    """Prepare editable English text; never invoke a policy or simulator."""
    from urllib.request import Request, urlopen
    def request(path, body, content_type):
        with urlopen(Request(url + path, data=body,
                headers={"Content-Type": content_type}), timeout=125) as response:
            return json.load(response)
    original = request("/transcribe", audio, "audio/wav")["text"]
    result = request("/translate", json.dumps({"text": original}).encode(), "application/json")
    return original, result["text"]


def panel(args):
    """Tk on main thread; one worker owns simulator and bounded network IO."""
    import tkinter as tk
    from tkinter import ttk
    configure_libero()
    from libero.libero import benchmark
    tasks = benchmark.get_benchmark_dict()[args.suite]()
    if not 0 <= args.task_id < tasks.n_tasks:
        raise ValueError("task-id outside suite")
    window = tk.Tk()
    window.title("Genuine VLA · LIBERO simulation")
    selected = tk.StringVar(value=f"{args.task_id}: {tasks.get_task(args.task_id).language}")
    taskbox = ttk.Combobox(window, textvariable=selected, state="readonly", width=95,
        values=[f"{i}: {tasks.get_task(i).language}" for i in range(tasks.n_tasks)])
    taskbox.pack(padx=12, pady=8)
    prompt = tk.StringVar(value=args.prompt if args.prompt is not None else tasks.get_task(args.task_id).language)
    taskbox.bind("<<ComboboxSelected>>", lambda event: prompt.set(tasks.get_task(taskbox.current()).language))
    ttk.Label(window, text="Raw English instruction to VLA (no scripted action mapping)").pack()
    ttk.Entry(window, textvariable=prompt, width=95).pack(padx=12, pady=8)
    ttk.Label(window, text=f"Each Start resets initial state {args.initial_state}; score uses selected task predicates.").pack()
    events = queue.Queue()
    label = tk.StringVar(value="Ready")
    ttk.Label(window, textvariable=label, wraplength=760).pack(pady=8)
    stop, paused = threading.Event(), threading.Event()
    worker = None
    import signal
    import subprocess
    import tempfile
    import wave
    sys.path.insert(0, str(ROOT))
    from desktop_panel import Panel
    devices = Panel.capture_devices()
    device = tk.StringVar(value=devices[0] if devices else "")
    ttk.Label(window, text="Microphone ALSA device").pack()
    ttk.Combobox(window, textvariable=device, values=devices, width=50).pack()
    original_text = tk.StringVar(value="录音只填写指令；核对英文后点击 Start。")
    ttk.Label(window, textvariable=original_text, wraplength=760).pack()
    recording = record_temp = voice_worker = None

    def finish_recording(submit=True):
        nonlocal recording, record_temp, voice_worker
        process, temp = recording, record_temp
        recording = record_temp = None
        mic.configure(text="开始录音", state="disabled")
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
        def finish():
            try:
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    raise ValueError("Recording did not stop")
                if not submit:
                    return
                path = Path(temp.name) / "instruction.wav"
                with wave.open(str(path)) as audio:
                    if audio.getnframes() < 4000:
                        raise ValueError("Recording shorter than 0.25 seconds")
                events.put(("voice", voice_prompt(path.read_bytes())))
            except Exception as error:
                events.put("Voice failed: " + str(error))
            finally:
                temp.cleanup()
                events.put(("voice_done", None))
        voice_worker = threading.Thread(target=finish, daemon=True)
        voice_worker.start()

    def toggle_recording():
        nonlocal recording, record_temp
        if recording is not None:
            finish_recording()
            return
        if ((worker is not None and worker.is_alive()) or
                (voice_worker is not None and voice_worker.is_alive())):
            return
        if not device.get().strip():
            label.set("Connect a microphone or enter an ALSA device")
            return
        record_temp = tempfile.TemporaryDirectory(prefix="vla-voice-")
        try:
            recording = subprocess.Popen(["arecord", "-D", device.get().strip(),
                "-f", "S16_LE", "-r", "16000", "-c", "1", "-d", "30",
                "-t", "wav", "-q", str(Path(record_temp.name) / "instruction.wav")],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            mic.configure(text="结束录音并填写指令")
            label.set("Recording (maximum 30 seconds); no policy execution")
        except OSError as error:
            record_temp.cleanup()
            record_temp = None
            label.set(str(error))

    mic = ttk.Button(window, text="开始录音", command=toggle_recording)
    mic.pack(pady=8)

    def start():
        nonlocal worker
        if (recording is not None or (voice_worker is not None and voice_worker.is_alive())
                or (worker is not None and worker.is_alive())):
            return
        stop.clear()
        paused.clear()
        options = argparse.Namespace(**vars(args))
        options.task_id = int(selected.get().split(":", 1)[0])
        options.prompt = prompt.get()
        options.output = str(Path(args.output) / f"episode_{time.time_ns()}")
        worker = threading.Thread(target=rollout, args=(options, stop, paused, events.put), daemon=True)
        worker.start()

    def pause():
        if paused.is_set():
            paused.clear()
            label.set("Resuming")
        else:
            paused.set()
            label.set("Paused after current inference/step")

    buttons = ttk.Frame(window)
    buttons.pack(pady=8)
    for name, callback in (("Start / reset", start), ("Pause / resume", pause), ("Stop", stop.set)):
        ttk.Button(buttons, text=name, command=callback).pack(side="left", padx=6)
    closing = False

    def close():
        nonlocal closing
        closing = True
        if recording is not None:
            finish_recording(False)
        stop.set()
        label.set("Stopping; waiting for bounded inference to finish…")

    def poll():
        while not events.empty():
            event = events.get_nowait()
            if isinstance(event, tuple):
                if event[0] == "voice":
                    original, english = event[1]
                    original_text.set("识别原文：" + original)
                    prompt.set(english)
                    label.set("Review/edit English instruction, then click Start")
                elif event[0] == "voice_done":
                    mic.configure(state="normal")
            else:
                label.set(event)
        if recording is not None and recording.poll() is not None:
            finish_recording()
        if (closing and (worker is None or not worker.is_alive())
                and (voice_worker is None or not voice_worker.is_alive())):
            window.destroy()
        else:
            window.after(100, poll)

    window.protocol("WM_DELETE_WINDOW", close)
    poll()
    window.mainloop()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=LIMITS, default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--initial-state", type=int, default=0)
    parser.add_argument("--prompt", help="Raw English instruction; defaults to task language")
    parser.add_argument("--output", required=True, help="New directory; panel writes episode subdirectories")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--panel", action="store_true")
    args = parser.parse_args(argv)
    args.max_steps = LIMITS[args.suite] if args.max_steps is None else args.max_steps
    if not (1 <= args.max_steps <= 10000 and 1 <= args.replan_steps <= 1000
            and 0 <= args.task_id and 0 <= args.initial_state and 1 <= args.port <= 65535):
        parser.error("invalid bounded step counts, indices, or port")
    return args


if __name__ == "__main__":
    options = parse_args()
    os.environ.setdefault("MUJOCO_GL", "glfw" if options.viewer else "egl")
    if options.panel:
        panel(options)
    else:
        result = rollout(options)
        sys.exit(1 if result["result"] == "exception" else 0)
