"""Single simulation owner for the Chinese native workbench."""
import collections
import hashlib
from pathlib import Path
import queue
import threading
import time
import traceback

import numpy as np

from vla.corrections import CorrectionRecorder, write_json
from vla.run import ROOT, configure_libero, connect_policy, load_initial_states, observation_payload, translate_prompt, validate_actions


class Workbench:
    def __init__(self, args):
        self.args = args
        self.commands = queue.Queue(maxsize=4)
        self.events = queue.Queue()
        self.interrupt = threading.Event()
        self.quit = threading.Event()
        self.lock = threading.Lock()
        self.snapshot = {"ready": False, "running": False, "manual": False, "steps": 0,
                         "frames": 0, "status": "请选择任务，加载场景。", "images": None}
        self.env = self.viewer = self.client = self.recorder = self.video = None
        self.running = False
        self.plan = collections.deque()
        self.grip = -1.
        self.output = None
        self.actions = []
        self.calls, self.done = 0, False
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def state(self):
        with self.lock:
            return dict(self.snapshot)

    def _update(self, **values):
        with self.lock:
            self.snapshot.update(values)

    def send(self, command, *values):
        if command in ("load", "pause", "begin", "run", "close"):
            self.interrupt.set()  # Discard an in-flight prediction before executing it.
        if command == "close":
            self.quit.set()
            return True
        try:
            self.commands.put_nowait((command, values))
            return True
        except queue.Full:
            return False  # Never accumulate a long joystick movement backlog.

    def _payload(self, image_size=None):
        from openpi_client import image_tools
        size = image_size or self.scene.get("source_model", {}).get("image_size", 224)
        return observation_payload(self.obs, self.prompt, image_tools, size)

    def _publish(self):
        payload = self._payload()
        self._update(ready=True, running=self.running, manual=self.recorder is not None,
                     policy_backend=self.scene.get("source_model", {}).get("backend"),
                     steps=len(self.actions), frames=self.recorder.meta["frame_count"] if self.recorder else 0,
                     grip=self.grip,
                     images=(payload["observation/image"], payload["observation/wrist_image"]),
                     output=str(self.output), state=payload["observation/state"].tolist())
        if self.viewer is not None:
            self.viewer.sync()

    def _save_scene(self, reason):
        if self.output is not None:
            np.save(self.output / "executed_actions.npy", np.asarray(self.actions).reshape(-1, 7))
            write_json(self.output / "report.json", dict(self.scene, reason=reason,
                steps=len(self.actions), policy_calls=self.calls, official_predicate=bool(self.done)))

    def _close_scene(self):
        self.running = False
        if self.recorder:
            try:
                self.recorder.finish(False)
            except Exception:
                self.events.put(traceback.format_exc())
            self.recorder = None
        try:
            self._save_scene("scene_closed")
        except Exception:
            self.events.put(traceback.format_exc())
        for name in ("video", "viewer", "env", "client"):
            resource = getattr(self, name)
            if resource is not None:
                try:
                    (resource._ws if name == "client" else resource).close()
                except Exception:
                    self.events.put(traceback.format_exc())
                setattr(self, name, None)
        self.output = None
        self.plan.clear()
        self._update(ready=False, running=False, manual=False, images=None)

    def _load(self, task_id, initial_state, prompt, port):
        if self.recorder:
            raise ValueError("请先保存纠正片段，再切换场景")
        self._update(status="正在理解你的指令……")
        user_instruction = prompt
        prompt = translate_prompt(prompt)
        self._close_scene()
        configure_libero()
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        task = benchmark.get_benchmark_dict()[self.args.suite]().get_task(task_id)
        states = load_initial_states(task)
        if not 0 <= initial_state < len(states):
            raise ValueError(f"初始布局范围：0～{len(states)-1}")
        if not prompt.strip():
            raise ValueError("指令不能为空")
        self.prompt = prompt
        self._update(status="正在连接模型、加载 MuJoCo 场景……")
        self.client = connect_policy(port)
        model = self.client.get_server_metadata()
        if model.get("backend") not in ("official-pi05-libero-vla", "local-pi05-libero-lora", "official-molmoact2-libero-vla"):
            raise ValueError("当前端口不是此项目支持的 VLA 服务")
        initial = np.asarray(states[initial_state])
        identity = dict(suite=self.args.suite, task_id=task_id, initial_state=initial_state,
                        initial_state_sha256=hashlib.sha256(initial.tobytes()).hexdigest())
        import json
        self.scene = dict(identity, task=prompt, user_instruction=user_instruction, task_language=task.language, source_model=model,
                          seed=self.args.seed, replan_steps=self.args.replan_steps, policy_port=port)
        self.scene["scenario_key"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.actions, self.calls, self.done = [], 0, False
        self.output = Path(self.args.output).resolve() / f"scene_{time.time_ns()}"
        self.output.mkdir(parents=True, exist_ok=False)
        np.save(self.output / "initial_state.npy", initial)
        self.env = OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path("bddl_files")) /
            task.problem_folder / task.bddl_file), camera_heights=256, camera_widths=256)
        self.env.seed(self.args.seed)
        self.env.reset()
        self.obs = self.env.set_init_state(initial)
        self.actions, self.calls, self.done = [], 0, False
        for _ in range(10):
            self.obs, _, _, _ = self.env.step([0.] * 6 + [-1.])
        self.grip = -1.
        if self.args.viewer:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.env.sim.model._model, self.env.sim.data._data)
            self._camera(False)
        self.interrupt.clear()
        self._publish()
        self._update(status="场景已加载并暂停。可开始模型，也可人工接管。")

    def _ensure_policy(self):
        from websockets.protocol import State
        if self.client is not None and self.client._ws.state is State.OPEN:
            return
        if self.client is not None:
            self.client._ws.close()
            self.client = None
        self._update(status="连接已断开，正在重新连接原模型……")
        client = connect_policy(self.scene["policy_port"])
        try:
            if client.get_server_metadata() != self.scene["source_model"]:
                raise ValueError("服务端模型已变化，请加载新场景后再运行")
        except BaseException:
            client._ws.close()
            raise
        self.client = client

    def _camera(self, top):
        if self.viewer:
            with self.viewer.lock():
                self.viewer.opt.geomgroup[0] = 0
                self.viewer.opt.geomgroup[1] = 1
                self.viewer.cam.lookat[:] = [0., 0., 1.]
                self.viewer.cam.distance = 1.9 if top else 2.4
                self.viewer.cam.azimuth = 90 if top else 225
                self.viewer.cam.elevation = -89 if top else -30
            self.viewer.sync()

    def _step(self, action, human=False):
        # Corrections retain the existing training format, independent of the
        # inference model's camera resolution.
        payload = self._payload(224) if human else None
        self.obs, _, self.done, _ = self.env.step(np.asarray(action).tolist())
        self.actions.append(np.asarray(action).copy())
        if human:
            self.recorder.append(payload, action, self._payload()["observation/state"], self.done)
        if self.video:
            self.video.append_data(np.concatenate(self._payload_images(), axis=1))
        self._publish()

    def _payload_images(self):
        payload = self._payload()
        return payload["observation/image"], payload["observation/wrist_image"]

    def _command(self, name, values):
        if name == "load":
            return self._load(*values)
        if self.env is None:
            raise ValueError("请先加载场景")
        if name == "pause":
            self.running = False
            self.plan.clear()
            self._update(status="已暂停；模型动作队列已清空。", running=False)
        elif name == "run":
            if self.recorder:
                raise ValueError("请先保存人工片段，再交还模型")
            if self.done:
                raise ValueError("当前任务判据已满足；可人工补收尾或加载新布局")
            user_instruction = str(values[0]).strip() if values else self.scene.get("user_instruction", self.prompt)
            if not user_instruction:
                raise ValueError("指令不能为空")
            self._update(status="正在理解你的指令……")
            prompt = translate_prompt(user_instruction)
            self.running = False
            self.plan.clear()
            self._ensure_policy()
            self.scene["user_instruction"] = user_instruction
            if prompt != self.prompt:
                self.scene.setdefault("instruction_changes", []).append(
                    dict(control_step=len(self.actions), previous=self.prompt, prompt=prompt))
                self.prompt = prompt
                self.scene["task"] = prompt
            self.interrupt.clear()
            self.plan.clear()
            self.running = True
            self.auto_steps = 0
            self._update(running=True, status="模型运行中")
        elif name == "begin":
            if self.recorder:
                raise ValueError("已经处于人工接管")
            self.running = False
            self.plan.clear()
            split, = values
            self.recorder = CorrectionRecorder(ROOT / "corrections" / f"human_{time.time_ns()}",
                dict(self.scene, split=split, source_scene=str(self.output)))
            np.save(self.recorder.path / "initial_sim_state.npy", np.asarray(self.env.sim.get_state().flatten()))
            self.grip = -1. if np.abs(self.obs["robot0_gripper_qpos"]).sum() >= .06 else 1.
            self._update(status="人工接管：实际执行的每一步都会保存；松开摇杆即停止。")
            self._publish()
        elif name == "manual":
            if not self.recorder:
                raise ValueError("请先开始人工接管")
            action, repeats = values
            action = np.asarray(action, dtype=float)
            validate_actions(action[None], 1)
            if np.abs(action).max() > 1 or not 1 <= repeats <= 20:
                raise ValueError("人工幅度或重复步数越界")
            self.grip = float(action[-1])
            for _ in range(repeats):
                if self.quit.is_set():
                    break
                self._step(action, human=True)
        elif name == "finish":
            if not self.recorder:
                raise ValueError("没有录制中的片段")
            path = self.recorder.finish(values[0])
            self.recorder = None
            self._publish()
            self._update(status=f"已保存{'合格片段' if values[0] else '草稿'}：{path.name}")
        elif name == "camera":
            self._camera(values[0])
        elif name == "video":
            if self.video:
                self.video.close()
                self.video = None
                self._update(status="双相机 Demo 已保存到当前场景目录。")
            else:
                import imageio.v2 as imageio
                self.video = imageio.get_writer(self.output / f"demo_{time.time_ns()}.mp4", fps=20)
                self.video.append_data(np.concatenate(self._payload_images(), axis=1))
                self._update(status="正在录制双相机 Demo（20 Hz 仿真时间）。")
        elif name == "save":
            self._save_scene("user_saved")
            self._update(status="当前结果及动作记录已保存。")

    def _infer_step(self):
        if not self.plan:
            self._update(status=f"模型预测中：控制步 {len(self.actions)}")
            response = self.client.infer(self._payload())
            if self.interrupt.is_set():
                self.running = False
                return
            actions = validate_actions(response["actions"], self.args.replan_steps)
            np.save(self.output / f"actions_{self.calls:04d}.npy", actions)
            self.calls += 1
            self.plan.extend(actions[:self.args.replan_steps])
        if not self.interrupt.is_set():
            self._step(self.plan.popleft())
            self.auto_steps += 1
        if self.done or self.auto_steps >= self.args.max_steps:
            self.running = False
            self.plan.clear()
            self._save_scene("predicate_satisfied" if self.done else "step_limit")
            self._update(running=False, status=("任务判据已满足，已暂停。可人工补充松爪、撤手。" if self.done
                else "达到本轮控制步数上限，已暂停；可继续运行或人工接管。"))

    def _loop(self):
        try:
            while not self.quit.is_set():
                try:
                    command = self.commands.get(timeout=.02 if self.running else .05)
                except queue.Empty:
                    command = None
                try:
                    if command:
                        self._command(*command)
                    elif self.running:
                        self._infer_step()
                    elif self.viewer:
                        self.viewer.sync()
                    if self.viewer and not self.viewer.is_running():
                        self._close_scene()
                        self._update(status="三维窗口已关闭；可重新加载场景。")
                except Exception as error:
                    self.running = False
                    self.plan.clear()
                    if command and command[0] == "load":
                        self._close_scene()
                    self._update(running=False, status="操作失败：" + str(error))
                    self.events.put(traceback.format_exc())
        finally:
            self._close_scene()
