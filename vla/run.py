#!/usr/bin/env python3
"""Bounded genuine VLA rollout; no hardware or scripted command mapping."""
import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
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


def observation_payload(obs, prompt, image_tools, image_size=224):
    """Rotate180, resize/pad to model resolution, position+axis-angle+gripper8."""
    images = [image_tools.convert_to_uint8(image_tools.resize_with_pad(
        np.ascontiguousarray(obs[key][::-1, ::-1]), image_size, image_size))
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
            # Synchronous server inference can delay pong; recv remains bounded below.
            conn = websockets.sync.client.connect(self._uri, compression=None,
                max_size=64 * 1024 * 1024, open_timeout=15, ping_timeout=None)
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


def rollout(args, stop=None, paused=None, status=print, success_check=None, policy_factory=None):
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
    if success_check is not None:
        report["score_definition"] = success_check.definition
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
        client = (policy_factory or connect_policy)(args.port)
        report["server_metadata"] = client.get_server_metadata()
        if report["server_metadata"].get("backend") not in ("official-pi05-libero-vla", "local-pi05-libero-lora", "official-molmoact2-libero-vla"):
            raise ValueError("Expected this project's official or LoRA VLA server on this port")
        image_size = report["server_metadata"].get("image_size", 224)
        env = OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path("bddl_files")) /
            task.problem_folder / task.bddl_file), camera_heights=256, camera_widths=256)
        env.seed(args.seed)
        env.reset()
        obs = env.set_init_state(initial)
        if args.viewer:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(env.sim.model._model, env.sim.data._data)
            with viewer.lock():
                # Match robosuite's RGB cameras: hide colored collision meshes.
                viewer.opt.geomgroup[0] = 0
                viewer.opt.geomgroup[1] = 1
                viewer.cam.lookat[:] = [0., 0., 1.]
                viewer.cam.distance = 2.4
                viewer.cam.azimuth = 225
                viewer.cam.elevation = -30
            viewer.sync()
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
                payload = observation_payload(obs, report["prompt"], image_tools, image_size)
                video.append_data(np.concatenate((payload["observation/image"], payload["observation/wrist_image"]), axis=1))
                if not plan:
                    status(f"Inferring step {step}")
                    tick = time.monotonic()
                    response = client.infer(payload)
                    elapsed = time.monotonic() - tick
                    actions = validate_actions(response["actions"], args.replan_steps)
                    index = report["policy_calls"]
                    np.save(output / f"actions_{index:04d}.npy", actions)
                    for branch in ("conditional_actions", "unconditional_actions"):
                        if branch in response:
                            np.save(output / f"{branch}_{index:04d}.npy", response[branch])
                    calls.write(json.dumps(dict(call=index, step=step, seconds=elapsed,
                        shape=list(actions.shape), queued_prefix=args.replan_steps,
                        guidance_noise_seed=response.get("guidance_noise_seed"))) + "\n")
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
                report["original_task_predicate"] = bool(done)
                satisfied = bool(done) if success_check is None else success_check(env)
                if satisfied:
                    report["result"] = "success"
                    break
        payload = observation_payload(obs, report["prompt"], image_tools, image_size)
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


def translate_prompt(text, url="http://127.0.0.1:8911"):
    """Keep user language outside the action model; never map words to actions."""
    import re
    from urllib.request import Request, urlopen
    if not isinstance(text, str) or not text.strip() or len(text) > 500:
        raise ValueError("请输入 1～500 字的指令")
    text = text.strip()
    if not re.search(r"[\u3400-\u9fff]", text):
        return text
    request = Request(url + "/translate", data=json.dumps({"text": text}).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=125) as response:
        translated = json.load(response).get("text")
    if (not isinstance(translated, str) or not translated.strip()
            or len(translated) > 2000 or re.search(r"[\u3400-\u9fff]", translated)):
        raise ValueError("中文指令处理失败，请重新说一遍或稍后重试")
    return translated.strip()


def voice_prompt(audio, url="http://127.0.0.1:8911"):
    """Return original speech and internal translation; never execute actions."""
    from urllib.request import Request, urlopen
    with urlopen(Request(url + "/transcribe", data=audio,
            headers={"Content-Type": "audio/wav"}), timeout=125) as response:
        original = json.load(response)["text"]
    return original, translate_prompt(original, url)


def panel(args):
    from vla.panel import panel as chinese_panel
    return chinese_panel(args)


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
