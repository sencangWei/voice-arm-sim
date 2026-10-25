"""Experimental VLM intent + pi05 grasp + RGB-D/OSC place (not end-to-end VLA).

No simulator object identities/poses enter the controller. Geometry assumes a
separated tabletop scene. Gripper width/lift is only a grasp candidate, not proof
of contact. Completion is unscored unless an independent evaluator is requested.
"""
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import queue
import threading
import time
import urllib.request

import numpy as np

from vla.hybrid import locate_surface, servo_action, world_cloud
from vla.run import configure_libero, connect_policy, load_initial_states, observation_payload, validate_actions
from vla.semantic import parse_task


def post(port, endpoint, payload, timeout=300):
    request = urllib.request.Request(f"http://127.0.0.1:{port}/{endpoint}",
        data=json.dumps(payload, ensure_ascii=False).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(exc.read().decode()) from exc


def image_bytes(rgb):
    from PIL import Image
    stream = io.BytesIO()
    Image.fromarray(rgb).save(stream, format="PNG")
    return stream.getvalue()


def region_point(cloud, box, allow_table=False):
    """Depth-component refinement at the model's normalized bounding-box center."""
    point = {"x": (box[0] + box[2]) / 2000, "y": (box[1] + box[3]) / 2000}
    if allow_table:
        height, width = cloud.shape[:2]
        u, v = round(point["x"] * (width - 1)), round(point["y"] * (height - 1))
        z = cloud[..., 2]
        values, counts = np.unique(np.round(z[(z > .7) & (z < 1.05)], 3), return_counts=True)
        if len(values):
            tabletop = float(values[np.argmax(counts)])
            patch = np.zeros(z.shape, dtype=bool)
            patch[max(0, v-4):v+5, max(0, u-4):u+5] = True
            visible = cloud[patch]
            if np.isfinite(visible).all() and np.max(np.abs(visible[:, 2] - tabletop)) < .006:
                target = np.median(visible, axis=0)
                return target, patch, dict(raw_point=cloud[v, u].tolist(), refined_point=target.tolist(),
                    tabletop=tabletop, pixels=int(patch.sum()), method="visible planar table patch at model-selected pixel")
    target, mask, metadata = locate_surface(cloud, point)
    if not np.isfinite(target).all():
        raise ValueError("Invalid RGB-D target")
    return target, mask, metadata


class VisualSkill:
    def __init__(self, env, obs, client, output, *, advance=None, stop=lambda: False, status=print,
                 heartbeat=lambda: None, on_task=lambda task: None):
        self.env, self.obs, self.client = env, obs, client
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.advance, self.stop, self.status = advance, stop, status
        self.heartbeat, self.on_task = heartbeat, on_task
        self.trace, self.writer = [], None
        self.report = dict(backend="Qwen3-VL intent + pi05 grasp + RGB-D/OSC place",
            end_to_end_vla=False, controller_object_truth=False, semantic_success=None,
            result="not_started", grasp_verification="proprioceptive candidate only; requires video review")
        self.report["inference_calls"] = 0

    def request(self, port, endpoint, payload, timeout=300):
        # Only network IO runs elsewhere; the scene remains owned by this thread.
        replies = queue.Queue(maxsize=1)
        def worker():
            try:
                replies.put((post(port, endpoint, payload, timeout), None))
            except Exception as exc:
                replies.put((None, exc))
        threading.Thread(target=worker, daemon=True).start()
        while True:
            if self.stop():
                raise InterruptedError("用户已暂停；未发送后续动作")
            self.heartbeat()
            try:
                response, error = replies.get(timeout=.05)
            except queue.Empty:
                continue
            if error:
                raise error
            return response

    def capture(self):
        from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, get_camera_intrinsic_matrix, get_real_depth_map
        rgb, depth = self.env.sim.render(width=512, height=512, camera_name="agentview", depth=True)
        rgb = np.ascontiguousarray(rgb[::-1, ::-1])
        depth = get_real_depth_map(self.env.sim, depth[::-1])
        intrinsic = get_camera_intrinsic_matrix(self.env.sim, "agentview", 512, 512)
        extrinsic = get_camera_extrinsic_matrix(self.env.sim, "agentview")
        return rgb, world_cloud(depth, intrinsic, extrinsic)

    def step(self, action, phase):
        if self.stop():
            raise InterruptedError("用户已暂停")
        action = np.asarray(action, dtype=float)
        validate_actions(action[None], 1)
        self.obs = self.advance(action) if self.advance else self.env.step(action.tolist())[0]
        self.trace.append(dict(step=len(self.trace), phase=phase, action=action.tolist(),
            eef=self.obs["robot0_eef_pos"].tolist(), gripper=self.obs["robot0_gripper_qpos"].tolist()))
        self.writer.append_data(np.ascontiguousarray(self.obs["agentview_image"][::-1, ::-1]))
        if len(self.trace) % 20 == 0:
            self.status(f"{phase}：控制步 {len(self.trace)}")

    def move(self, target, grip, phase, limit=80):
        self.status(phase)
        for _ in range(limit):
            eef = np.asarray(self.obs["robot0_eef_pos"])
            self.step(servo_action(eef, target, grip), phase)
            if np.linalg.norm(np.asarray(self.obs["robot0_eef_pos"]) - target) < .008:
                return
        raise TimeoutError(f"{phase} 未收敛；保留现场，请人工检查")

    def run(self, text, task_port=8912):
        import imageio.v2 as imageio
        from openpi_client import image_tools
        started = time.monotonic()
        self.report.update(chinese=text, policy=self.client.get_server_metadata())
        try:
            if self.report["policy"].get("backend") not in (
                    "official-pi05-libero-vla", "local-pi05-libero-lora"):
                raise ValueError("This experimental skill requires a pi05 policy")
            rgb, cloud = self.capture()
            raw = image_bytes(rgb)
            (self.output / "input.png").write_bytes(raw)
            self.status("模型正在看当前画面并解析中文指令（CPU 服务约需 2～3 分钟）")
            response = self.request(task_port, "task", {"text": text, "image": base64.b64encode(raw).decode()})
            (self.output / "model.json").write_text(json.dumps(response, ensure_ascii=False, indent=2))
            if response.get("image_sha256") != hashlib.sha256(raw).hexdigest():
                raise ValueError("Model response image does not match current observation")
            task = parse_task(json.dumps(response["task"]))
            self.report["task"] = task
            self.on_task(task)
            if self.stop():
                raise InterruptedError("用户已暂停")
            source, source_mask, source_meta = region_point(cloud, task["source"]["bbox_2d"])
            destination, goal_mask, goal_meta = region_point(cloud, task["destination"]["bbox_2d"], task["relation"] == "on")
            if (source_mask & goal_mask).sum() > .5 * min(source_mask.sum(), goal_mask.sum()):
                raise ValueError("源物体和目的地落在同一深度区域，停止以免搬错目标")
            np.savez_compressed(self.output / "geometry.npz", cloud=cloud,
                source_mask=source_mask, destination_mask=goal_mask)
            self.report.update(source=source_meta, destination=goal_meta)
            self.status(f"模型选择：{task['source']['label']} → {task['destination']['label']} ({task['relation']})")
            self.writer = imageio.get_writer(self.output / "demo.mp4", fps=20)
            self.writer.append_data(np.ascontiguousarray(self.obs["agentview_image"][::-1, ::-1]))
            queue, grasp_eef, min_z = [], None, None
            self.status("π0.5 正在抓取；尚未接管运输")
            for step in range(180):
                if not queue:
                    payload = observation_payload(self.obs, task["english"], image_tools)
                    payload["_probe_noise_seed"] = 42 + step // 5
                    prediction = validate_actions(self.client.infer(payload)["actions"], 5)
                    self.report["inference_calls"] += 1
                    np.save(self.output / f"prediction-{step:03d}.npy", prediction)
                    queue.extend(prediction[:5])
                action = np.asarray(queue.pop(0))
                eef = np.asarray(self.obs["robot0_eef_pos"])
                gap = float(np.abs(self.obs["robot0_gripper_qpos"]).sum())
                if action[-1] > .5 and (min_z is None or eef[2] < min_z):
                    min_z, grasp_eef = float(eef[2]), eef.copy()
                if min_z is not None and eef[2] > min_z + .07 and .003 < gap < .075 and action[-1] > .5:
                    break
                self.step(action, "VLA 抓取")
            else:
                raise TimeoutError("180 步内未检测到抬起的抓取候选；请检查抓取")
            offset = source[:2] - grasp_eef[:2]
            if np.linalg.norm(offset) > .10:
                raise ValueError("抓取位置偏离视觉源物体，停止运输")
            self.report.update(handover_step=len(self.trace), grasp_eef=grasp_eef.tolist(), offset_xy=offset.tolist())
            eef = np.asarray(self.obs["robot0_eef_pos"])
            # Clear a raised support with the held object's bottom, not just TCP.
            release_z = min_z + (destination[2] - goal_meta["tabletop"]) + .018
            safe_z = max(1.20, release_z + .035, float(eef[2]))
            self.report["clearance_z"] = safe_z
            self.move(np.r_[eef[:2], safe_z], 1., "抬起避让")
            # Query a fresh image after grasp; the old destination coordinate is
            # never blindly reused when its current visual localization fails.
            rgb, cloud = self.capture()
            (self.output / "reobserve.png").write_bytes(image_bytes(rgb))
            self.status("重新观察并定位目的地")
            location = self.request(8911, "locate", {"target": task["destination"]["label"],
                "image": base64.b64encode(image_bytes(rgb)).decode()}, timeout=90)
            (self.output / "relocalize.json").write_text(json.dumps(location, ensure_ascii=False, indent=2))
            if not location.get("bbox_2d"):
                raise ValueError("当前画面找不到目的地，已停止；不会换成预设位置")
            destination, _, metadata = region_point(cloud, location["bbox_2d"], task["relation"] == "on")
            self.report["destination_reobserved"] = metadata
            target_xy = destination[:2] - offset
            release_z = min_z + (destination[2] - goal_meta["tabletop"]) + .018
            if release_z + .035 > safe_z:
                safe_z = release_z + .035
                self.move(np.r_[self.obs["robot0_eef_pos"][:2], safe_z], 1., "更新避让高度")
            self.move(np.r_[target_xy, safe_z], 1., "搬运到新定位的目的地")
            self.move(np.r_[target_xy, release_z], 1., "下降放置")
            target = np.r_[target_xy, release_z]
            for _ in range(15):
                self.step(servo_action(self.obs["robot0_eef_pos"], target, -1.), "松爪")
            self.move(np.r_[target_xy, safe_z], -1., "撤手")
            for _ in range(30):
                self.step(servo_action(self.obs["robot0_eef_pos"], np.r_[target_xy, safe_z], -1.), "稳定观察")
            self.report["result"] = "motion_completed_unscored"
        except Exception as exc:
            self.report.update(result="interrupted" if isinstance(exc, InterruptedError) else "exception",
                exception=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            if self.writer:
                self.writer.close()
            try:
                rgb, _ = self.capture()
                (self.output / "final.png").write_bytes(image_bytes(rgb))
            except Exception as exc:
                self.report["final_capture_error"] = str(exc)
            self.report.update(control_steps=len(self.trace), wall_duration_s=time.monotonic() - started)
            (self.output / "result.json").write_text(json.dumps(self.report, ensure_ascii=False, indent=2))
            (self.output / "trace.json").write_text(json.dumps(self.trace, ensure_ascii=False, indent=2))
        return self.report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-state", type=int, default=0)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--task-port", type=int, default=8912)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--score", choices=("bottle-in-bowl", "bottle-on-rack", "bowl-on-plate"))
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "glfw" if args.viewer else "egl")
    configure_libero()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    task = benchmark.get_benchmark_dict()["libero_goal"]().get_task(8)
    env = OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
        camera_heights=512, camera_widths=512)
    client, viewer = None, None
    try:
        env.seed(42)
        env.reset()
        obs = env.set_init_state(load_initial_states(task)[args.initial_state])
        for _ in range(10):
            obs = env.step([0.] * 6 + [-1.])[0]
        if args.viewer:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(env.sim.model._model, env.sim.data._data)
        client = connect_policy(args.port)
        evaluator, measured = None, []
        if args.score:
            from vla.compare import BottleInBowlCheck
            from vla.language_guidance import BottleOnRackCheck
            evaluator = BottleInBowlCheck() if args.score == "bottle-in-bowl" else BottleOnRackCheck() if args.score == "bottle-on-rack" else lambda e: bool(e.check_success())
        def advance(action):
            result = env.step(action.tolist())[0]
            # Observe each real control step. The controller never reads scores.
            if evaluator:
                measured.append(bool(evaluator(env)))
            if viewer:
                viewer.sync()
                time.sleep(.05)
            return result
        skill = VisualSkill(env, obs, client, args.output, advance=advance,
            stop=lambda: viewer is not None and not viewer.is_running(),
            heartbeat=lambda: viewer.sync() if viewer else None)
        skill.report.update(initial_state=args.initial_state, seed=42,
            initial_state_sha256=hashlib.sha256(np.asarray(load_initial_states(task)[args.initial_state]).tobytes()).hexdigest(),
            controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        try:
            report = skill.run(args.prompt, args.task_port)
        finally:
            # Evaluator truth remains outside the controller, including on error.
            if args.score:
                score = dict(case=args.score, success=bool(skill.report["result"] == "motion_completed_unscored"
                    and measured and measured[-1]), evaluated_control_steps=len(measured),
                    note="Independent relation monitored at actual control steps; video review required")
                (args.output / "evaluation.json").write_text(json.dumps(score, indent=2))
                if hasattr(evaluator, "history"):
                    (args.output / "evaluation-trace.json").write_text(json.dumps(evaluator.history, indent=2))
        print(json.dumps(report, ensure_ascii=False), flush=True)
    finally:
        if client:
            client._ws.close()
        if viewer:
            viewer.close()
        env.close()


if __name__ == "__main__":
    main()
