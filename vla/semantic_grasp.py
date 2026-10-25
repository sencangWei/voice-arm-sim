"""Experimental VLM intent + pi05 grasp + RGB-D/OSC place (not end-to-end VLA).

No simulator object identities/poses enter the controller. Geometry assumes a
separated tabletop scene. Gripper width/lift is only a grasp candidate, not proof
of contact. Completion is a fallible, fresh-image VLM judgment after release/retreat;
independent physical scoring remains separate.
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
from vla.semantic import parse_completion, parse_task


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
    height, width = cloud.shape[:2]
    x0, y0, x1, y1 = np.rint(np.asarray(box) / 1000 * [width-1, height-1, width-1, height-1]).astype(int)
    bounds = np.zeros((height, width), dtype=bool)
    bounds[y0:y1+1, x0:x1+1] = True
    if allow_table:
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
            # Cabinet sides and support tops can form one depth component.
            # Select measured horizontal pixels inside the model's destination
            # box rather than the median height of the entire cabinet body.
            normal = np.cross(cloud[1:-1, 2:] - cloud[1:-1, :-2],
                              cloud[2:, 1:-1] - cloud[:-2, 1:-1])
            length = np.linalg.norm(normal, axis=-1)
            flat = np.zeros(z.shape, dtype=bool)
            flat[1:-1, 1:-1] = (length > 1e-10) & (np.abs(normal[..., 2]) > .98 * length)
            flat &= bounds & np.isfinite(cloud).all(axis=-1) & (z >= tabletop-.004) & (z < 1.30)
            heights, counts = np.unique(np.round(z[flat], 3), return_counts=True)
            if len(counts) and counts.max() >= 60:
                level = float(heights[np.argmax(counts)])
                plane = flat & (np.abs(z - level) < .002)
                target = np.median(cloud[plane], axis=0)
                # Include a shallow raised rim connected to the selected plane.
                # The flat interior alone underestimates a plate's footprint;
                # bounds and depth separation exclude the surrounding table.
                from scipy.ndimage import label
                connected, _ = label(bounds & np.isfinite(cloud).all(axis=-1) &
                    (z >= level-.002) & (z <= level+.015))
                ids, sizes = np.unique(connected[plane], return_counts=True)
                component = ids[np.argmax(sizes)]
                support = connected == component
                return target, support, dict(raw_point=cloud[v, u].tolist(), refined_point=target.tolist(),
                    tabletop=tabletop, pixels=int(plane.sum()),
                    support_pixels=int(support.sum()), support_plane_z=level,
                    method="visible horizontal support plane inside model destination box")
    target, mask, metadata = (locate_surface(cloud, point) if allow_table else
                              locate_surface(cloud, point, bounds=bounds, ceiling=1.50))
    mask &= bounds
    if mask.sum() < 60:
        raise ValueError("Too few depth pixels inside model object box")
    # A touching support must not pull the source point outside the selected
    # object. Median visible pixels avoid the bounds of an attached stove.
    target = np.median(cloud[mask], axis=0)
    metadata.update(refined_point=target.tolist(), pixels=int(mask.sum()),
                    method="depth component clipped to model box; visible point median")
    if not np.isfinite(target).all():
        raise ValueError("Invalid RGB-D target")
    return target, mask, metadata


def nearby_support_point(cloud, mask, reference, margin=.060):
    """Choose a visible planar landing point with a measured edge margin.

    This only handles convex support tops and small tabletop objects. It avoids
    unnecessarily extending the arm to the center of a large support. The 6 cm
    inset reserves space for a small object's footprint and the observed 2–3 cm
    visible-surface / grasp-offset error; it is not a general collision planner.
    """
    from scipy.spatial import ConvexHull, cKDTree
    visible = cloud[mask]
    edges = ConvexHull(visible[:, :2]).equations
    # A support may have a slightly raised center (e.g. a stove burner).
    # Keep the same edge margin but consider its actual visible top pixels,
    # instead of requiring the landing point to lie on the surrounding plane.
    level = float(np.median(visible[:, 2]))
    measured = cloud[np.isfinite(cloud).all(axis=-1) &
                     (cloud[..., 2] >= level-.002) & (cloud[..., 2] <= level+.015)]
    interior = np.all(measured[:, :2] @ edges[:, :2].T + edges[:, 2] <= -margin, axis=1)
    candidates = measured[interior]
    # A convex table outline also encloses stoves and other obstacles. Reserve
    # the held object's footprint around measured geometry above this support;
    # otherwise the nearest "table" point can still belong to a stove base.
    obstacles = cloud[np.isfinite(cloud).all(axis=-1) & (cloud[..., 2] > level+.015)]
    if len(obstacles) and len(candidates):
        clearance, _ = cKDTree(obstacles[:, :2]).query(candidates[:, :2])
        candidates = candidates[clearance >= margin]
    if not len(candidates):
        raise ValueError("可见支撑面没有留足边缘余量的落点，停止放置")
    return candidates[np.argmin(np.linalg.norm(candidates[:, :2] - np.asarray(reference)[:2], axis=1))].copy()


def circular_container_point(cloud, mask):
    """Fit a visible round rim; a bowl's front surface median is off-center.

    This bounded controller supports round open containers, not arbitrary bins.
    Reject poor fits instead of silently placing against a guessed center.
    """
    visible = cloud[mask]
    # A mildly tilted rim spans several heights. Its upper fifth retains a
    # measurable arc on both sides; the upper tenth can contain only one side.
    rim = visible[visible[:, 2] >= np.quantile(visible[:, 2], .80), :2]
    if len(rim) < 60 or np.ptp(visible[:, 2]) < .01:
        raise ValueError("容器可见圆沿不足，停止放置")
    matrix = np.c_[2 * rim, np.ones(len(rim))]
    fit, _, rank, _ = np.linalg.lstsq(matrix, np.sum(rim**2, axis=1), rcond=None)
    center = fit[:2]
    radius_squared = fit[2] + np.sum(center**2)
    residual = float(np.std(np.linalg.norm(rim - center, axis=1)))
    angles = np.sort(np.arctan2(*(rim - center).T[::-1]))
    coverage = 2 * np.pi - np.max(np.diff(np.r_[angles, angles[0] + 2 * np.pi]))
    if rank != 3 or not .02**2 < radius_squared < .25**2 or residual > .004 or coverage < np.pi:
        raise ValueError("容器圆沿拟合不可靠，停止放置")
    target = np.r_[center, np.quantile(visible[:, 2], .05)]
    return target, dict(method="visible round rim circle fit; low visible interior depth",
        refined_point=target.tolist(), rim_radius=float(np.sqrt(radius_squared)),
        rim_residual=residual, rim_coverage=float(coverage))


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
        started = time.monotonic()
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
            self.report.setdefault("request_timings", []).append(dict(
                endpoint=endpoint, port=port, wall_s=time.monotonic()-started,
                inference_s=response.get("inference_s"), device=response.get("device")))
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
        if not self.trace and hasattr(self, "started"):
            self.report["first_control_wall_s"] = time.monotonic() - self.started
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

    def confirm_hold(self):
        """Reject transient closing widths before transporting a grasp candidate.

        Stable width is still not proof of object identity or contact. In
        particular, fingers obstructed by a support can pass this check.
        """
        gaps = []
        start = np.asarray(self.obs["robot0_eef_pos"]).copy()
        for _ in range(20):
            self.step(np.r_[np.zeros(6), 1.], "检查稳定持有")
            gaps.append(float(np.abs(self.obs["robot0_gripper_qpos"]).sum()))
        settled = np.asarray(gaps[-10:])
        drift = float(np.linalg.norm(np.asarray(self.obs["robot0_eef_pos"]) - start))
        accepted = bool(np.isfinite(settled).all() and np.all((settled > .003) & (settled < .075))
                        and np.ptp(settled) < .002 and drift < .012)
        self.report["hold_check"] = dict(gaps=gaps, eef_drift=drift, accepted=accepted,
            method="20 closed hold controls; final 10 widths stable; proprioceptive candidate only")
        if not accepted:
            raise RuntimeError("抓取候选未通过稳定持有检查（可能抓空或滑落），已停止运输；保留现场")

    def verify_completion(self, text, task_port):
        # No simulator object identities, poses, predicates or planned coordinates.
        self.report["semantic_success"] = None
        gap = float(np.abs(self.obs["robot0_gripper_qpos"]).sum())
        self.report["release_gap"] = gap
        if not np.isfinite(gap) or gap < .06:
            judgment = {"verdict": "not_completed", "reason": "夹爪尚未充分打开，不能确认释放"}
        else:
            rgb, cloud = self.capture()
            wrist = np.ascontiguousarray(self.obs["robot0_eye_in_hand_image"][::-1, ::-1])
            if hasattr(self, 'verification_box'):
                # Keep the whole scene (including the retracted arm), and show
                # a magnified current cabinet beside it. No synthetic geometry.
                from PIL import Image
                source = Image.fromarray(rgb)
                width, height = source.size
                box = np.asarray(self.verification_box) * [width, height, width, height] / 1000
                box += [-10, -10, 10, 10]
                box = np.clip(box, [0, 0, 0, 0], [width, height, width, height]).astype(int)
                detail = source.crop(tuple(box)).resize(source.size)
                combined = Image.new('RGB', (width * 2, height))
                combined.paste(source, (0, 0))
                combined.paste(detail, (width, 0))
                rgb = np.asarray(combined)
            else:
                # The service processes one image for completion. Include
                # both fresh views in that image so the wrist observation is
                # actually seen, rather than merely archived in the request.
                from PIL import Image
                source = Image.fromarray(rgb)
                combined = Image.new('RGB', (source.width * 2, source.height))
                combined.paste(source, (0, 0))
                combined.paste(Image.fromarray(wrist).resize(source.size), (source.width, 0))
                rgb = np.asarray(combined)
            raws = [image_bytes(rgb), image_bytes(wrist)]
            names = ("completion-agent.png", "completion-wrist.png")
            for name, raw in zip(names, raws):
                (self.output / name).write_bytes(raw)
            verification_text = text
            if not hasattr(self, 'verification_box'):
                verification_text += '。图左是当前全景，图右是同一时刻腕部相机画面；两幅画面拍摄同一个场景。'
            if hasattr(self, 'verification_box'):
                verification_text += '。图左是全景，图右是同一时刻局部放大。'
                if self.report.get('verification_action') == 'open':
                    verification_text += ('本步目标是打开。能看到向前伸出的抽屉底板和内部空间，'
                        '是已打开的证据；前板仍与其他抽屉齐平才是未打开。')
                else:
                    verification_text += ('请特别核对：能看到向前伸出的抽屉底板及其内部物体，'
                        '说明抽屉仍打开；底板退入柜内、前板与其他抽屉齐平才算关闭。')
            self.status("已松爪撤手，正在用当前画面核对是否完成；机械臂保持停止")
            response = self.request(task_port, "verify", {"text": verification_text,
                "images": [base64.b64encode(raw).decode() for raw in raws]})
            (self.output / "completion-model.json").write_text(json.dumps(response, ensure_ascii=False, indent=2))
            if response.get("image_sha256") != [hashlib.sha256(raw).hexdigest() for raw in raws]:
                raise ValueError("Completion image does not match current observation")
            judgment = parse_completion(json.dumps(response["judgment"]))
            if self.report.get('drawer_verification'):
                from vla.drawer_control import drawer_depth_check
                depth = drawer_depth_check(cloud, self.report['drawer_verification'],
                                           self.report['verification_action'])
                self.report['drawer_depth_check'] = depth
                self.report['model_completion'] = judgment.copy()
                if depth['accepted'] is not True and judgment['verdict'] == 'completed':
                    judgment = dict(verdict='uncertain' if depth['accepted'] is None else 'not_completed',
                                    reason=depth['reason'])
        self.report["completion"] = judgment
        self.report["semantic_success"] = {"completed": True, "not_completed": False,
                                           "uncertain": None}[judgment["verdict"]]
        self.report["result"] = "visual_" + judgment["verdict"]

    def grasp(self, task, source, source_mask, cloud):
        from openpi_client import image_tools
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
        return min_z, grasp_eef, source[:2] - grasp_eef[:2]

    def release_height(self, min_z, destination_z, tabletop):
        return min_z + (destination_z - tabletop) + .018

    def policy_metadata(self):
        return self.client.get_server_metadata()

    def landing_point(self, cloud, box, relation, destination_kind):
        if destination_kind == 'drawer_interior':
            from vla.drawer_control import box_points
            points = box_points(cloud, box)
            # The model selects the currently exposed interior; its horizontal
            # depth mode supplies a landing floor, not a simulator drawer site.
            bins, counts = np.unique(np.round(points[:, 2] / .002), return_counts=True)
            floor = float(bins[np.argmax(counts)] * .002)
            surface = points[np.abs(points[:, 2] - floor) < .003]
            if len(surface) < 40 or np.min(np.ptp(surface[:, :2], axis=0)) < .025:
                raise ValueError('当前图中抽屉底面不足以安全放置')
            height, width = cloud.shape[:2]
            x1, y1, x2, y2 = np.asarray(box)*[width, height, width, height]/1000
            mask = np.zeros((height, width), dtype=bool)
            mask[max(0,int(y1)):min(height,int(np.ceil(y2))),
                 max(0,int(x1)):min(width,int(np.ceil(x2)))] = True
            mask &= np.abs(cloud[..., 2]-floor) < .003
            # Prefer the exposed near edge: reaching under the upper drawer
            # with the large Panda palm collides before the fingers descend.
            margin = max(.035, self.report.get('geometric_grasp', {}).get('support_margin', .035))
            point = nearby_support_point(cloud, mask, self.obs['robot0_eef_pos'][:2], margin=margin)
            _, axes = np.linalg.eigh(np.cov(surface[:, :2].T))
            jaw_axis = np.r_[axes[:, -1], 0.]
            return point, mask, dict(method='visible drawer floor depth mode',
                tabletop=floor, point=point.tolist(), selection='nearest supported floor interior',
                floor_width_axis=jaw_axis.tolist(), edge_margin=margin)
        destination, mask, meta = region_point(cloud, box, relation == 'on')
        if relation == 'in':
            destination, container_meta = circular_container_point(cloud, mask)
            meta.update(container_meta)
        return destination, mask, meta

    def align_for_place(self, metadata, offset):
        return offset

    def relocalize_destination(self, task, rgb, task_port):
        raw = image_bytes(rgb)
        encoded = base64.b64encode(raw).decode()
        location = self.request(task_port, 'locate', dict(target=task['destination']['label'],
            image=encoded), timeout=90)
        (self.output / 'relocalize.json').write_text(json.dumps(location, ensure_ascii=False, indent=2))
        if not location.get('bbox_2d'):
            # Bare region labels can lose the task context (e.g. drawer interior).
            # Ask again with the unchanged goal and this same current image;
            # never reuse the pre-grasp box or replace the held source/goal.
            response = self.request(task_port, 'task', dict(
                text=self.report.get('chinese') or task['english'], image=encoded))
            (self.output / 'relocalize-task.json').write_text(
                json.dumps(response, ensure_ascii=False, indent=2))
            if response.get('image_sha256') != hashlib.sha256(raw).hexdigest():
                raise ValueError('Destination response image does not match current observation')
            fresh = parse_task(json.dumps(response['task']))
            if fresh['relation'] != task['relation']:
                raise ValueError('目的地重新定位改变了原指令关系；保留夹持等待恢复')
            location = fresh['destination']
        return location

    def place(self, task, min_z, offset, destination, goal_meta, task_port, destination_kind=None):
        eef = np.asarray(self.obs['robot0_eef_pos'])
        release_z = self.release_height(min_z, destination[2], goal_meta['tabletop'])
        safe_z = max(1.20, release_z + .035, float(eef[2]))
        self.report['clearance_z'] = safe_z
        self.move(np.r_[eef[:2], safe_z], 1., '抬起避让')
        rgb, cloud = self.capture()
        (self.output / 'reobserve.png').write_bytes(image_bytes(rgb))
        self.status('重新观察并定位目的地')
        location = self.relocalize_destination(task, rgb, task_port)
        destination, mask, metadata = self.landing_point(cloud, location['bbox_2d'],
                                                       task['relation'], destination_kind)
        if metadata['method'] == 'visible horizontal support plane inside model destination box':
            margin = self.report.get('geometric_grasp', {}).get('support_margin', .060)
            destination = nearby_support_point(cloud, mask,
                np.asarray(self.obs['robot0_eef_pos'])[:2] + offset, margin=margin)
            metadata.update(refined_point=destination.tolist(), edge_margin=margin,
                landing_selection='nearest visible convex support interior to held source; no object truth')
        self.report['destination_reobserved'] = metadata
        offset = self.align_for_place(metadata, offset)
        target_xy = destination[:2] - offset
        release_z = self.release_height(min_z, destination[2], metadata['tabletop'])
        if destination_kind == 'drawer_interior':
            # Release above the shallow rim rather than pushing the wide palm
            # between the side walls; gravity settles the held item on the floor.
            release_z += .035
        if release_z + .035 > safe_z:
            safe_z = release_z + .035
            self.move(np.r_[self.obs['robot0_eef_pos'][:2], safe_z], 1., '更新避让高度')
        self.move(np.r_[target_xy, safe_z], 1., '搬运到新定位的目的地')
        self.move(np.r_[target_xy, release_z], 1., '下降放置')
        for _ in range(15):
            self.step(servo_action(self.obs['robot0_eef_pos'], np.r_[target_xy, release_z], -1.), '松爪')
        self.move(np.r_[target_xy, safe_z], -1., '撤手')
        for _ in range(30):
            self.step(servo_action(self.obs['robot0_eef_pos'], np.r_[target_xy, safe_z], -1.), '稳定观察')
        self.report['motion_completed'] = True

    def run(self, text, task_port=8912, destination_kind=None):
        import imageio.v2 as imageio
        started = time.monotonic()
        self.started = started
        self.report.update(chinese=text, policy=self.policy_metadata())
        try:
            if self.report["policy"].get("backend") not in (
                    "official-pi05-libero-vla", "local-pi05-libero-lora", "visual-geometric-control"):
                raise ValueError("This experimental skill requires a pi05 policy")
            rgb, cloud = self.capture()
            raw = image_bytes(rgb)
            (self.output / "input.png").write_bytes(raw)
            self.status("模型正在看当前画面并解析中文指令；解析完成后开始动作")
            response = self.request(task_port, "task", {"text": text, "image": base64.b64encode(raw).decode()})
            (self.output / "model.json").write_text(json.dumps(response, ensure_ascii=False, indent=2))
            if response.get("image_sha256") != hashlib.sha256(raw).hexdigest():
                raise ValueError("Model response image does not match current observation")
            task = parse_task(json.dumps(response["task"]))
            if destination_kind is not None:
                expected = {"support_surface": "on", "circular_container": "in", "drawer_interior": "in"}
                if destination_kind not in expected or task["relation"] != expected[destination_kind]:
                    raise ValueError("单步目的地类型与完整计划不一致；未发送动作")
            self.report["task"] = task
            self.on_task(task)
            if self.stop():
                raise InterruptedError("用户已暂停")
            source, source_mask, source_meta = region_point(cloud, task["source"]["bbox_2d"])
            destination, goal_mask, goal_meta = self.landing_point(
                cloud, task["destination"]["bbox_2d"], task["relation"], destination_kind)
            if goal_mask is not None and (source_mask & goal_mask).sum() > .5 * min(source_mask.sum(), goal_mask.sum()):
                raise ValueError("源物体和目的地落在同一深度区域，停止以免搬错目标")
            np.savez_compressed(self.output / "geometry.npz", cloud=cloud,
                source_mask=source_mask, destination_mask=goal_mask)
            self.report.update(source=source_meta, destination=goal_meta)
            self.status(f"模型选择：{task['source']['label']} → {task['destination']['label']} ({task['relation']})")
            self.writer = imageio.get_writer(self.output / "demo.mp4", fps=20)
            self.writer.append_data(np.ascontiguousarray(self.obs["agentview_image"][::-1, ::-1]))
            min_z, grasp_eef, offset = self.grasp(task, source, source_mask, cloud)
            if np.linalg.norm(offset) > .10:
                raise ValueError("抓取位置偏离视觉源物体，停止运输")
            self.confirm_hold()
            self.report.update(handover_step=len(self.trace), grasp_eef=grasp_eef.tolist(), offset_xy=offset.tolist())
            self.report["grasp_context"] = dict(task=task, min_z=min_z, offset_xy=offset.tolist(),
                source_base=getattr(self, "source_base", None), destination=destination.tolist(),
                goal_meta=goal_meta, destination_kind=destination_kind)
            self.place(task, min_z, offset, destination, goal_meta, task_port, destination_kind)
            self.report["motion_finished_wall_s"] = time.monotonic() - started
            self.verify_completion(text, task_port)
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
                score = dict(case=args.score, success=bool(skill.report.get("motion_completed")
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
