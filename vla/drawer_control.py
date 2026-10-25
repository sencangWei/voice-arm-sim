"""Image-grounded prismatic drawer interaction, executed through contact/OSC.

No cabinet object identities, joint states or simulator object poses are read.
"""
import json
import math
import base64
import hashlib
import time

import numpy as np
from scipy.spatial.transform import Rotation

from vla.visual_control import VisualControl
from vla.semantic_grasp import image_bytes
from vla.hybrid import servo_action


DRAWER_PROMPT = """根据指令与当前图片定位要操作的抽屉。三层抽屉从上往下计数，第一层=上层，第二层=中层，第三层=下层。
仅输出JSON：{"action":"open或close","level":"top或middle或bottom","handle_bbox_2d":[x1,y1,x2,y2],"cabinet_bbox_2d":[x1,y1,x2,y2]}
坐标为0..1000归一化xyxy。handle_bbox_2d紧贴该层金属把手，不包含其他层的把手；cabinet_bbox_2d覆盖整个三层柜子。
物体或把手看不清返回null，不能换层。指令：
"""


def parse_drawer(answer):
    value = json.loads(answer)
    if (not isinstance(value, dict) or set(value) != {'action', 'level', 'handle_bbox_2d', 'cabinet_bbox_2d'}
            or value['action'] not in ('open', 'close') or value['level'] not in ('top', 'middle', 'bottom')):
        raise ValueError('无法确定抽屉动作和层数')
    for key in ('handle_bbox_2d', 'cabinet_bbox_2d'):
        box = value[key]
        if (not isinstance(box, list) or len(box) != 4 or any(type(v) not in (int, float)
                or not math.isfinite(v) or not 0 <= v <= 1000 for v in box)
                or box[0] >= box[2] or box[1] >= box[3]):
            raise ValueError('Invalid drawer image region')
    return value


def box_points(cloud, box):
    height, width = cloud.shape[:2]
    x1, y1, x2, y2 = np.asarray(box) * [width, height, width, height] / 1000
    points = cloud[max(0, int(y1)):min(height, int(np.ceil(y2))),
                   max(0, int(x1)):min(width, int(np.ceil(x2)))].reshape(-1, 3)
    return points[np.isfinite(points).all(axis=1)]


def handle_geometry(cloud, task):
    points = box_points(cloud, task['handle_bbox_2d'])
    if len(points) < 12:
        raise ValueError('把手有效深度不足')
    # Select the thin horizontal bar before fitting the cabinet. On an open
    # drawer the most common height in the model box can be plate/table pixels.
    closing = task.get('action') == 'close'
    bar = points
    if closing:
        bins, counts = np.unique(np.round(points[:, 2] / .003), return_counts=True)
        for index in np.argsort(-counts, kind='stable'):
            candidate = points[np.abs(points[:, 2] - bins[index] * .003) < .007]
            if len(candidate) < 12:
                continue
            eigenvalues = np.linalg.eigvalsh(np.cov(candidate[:, :2].T))
            if eigenvalues[-1] >= .0001 and eigenvalues[-1] >= 4 * eigenvalues[0]:
                bar = candidate
                break
        else:
            raise ValueError('把手区域没有可确认的细长横杆')
    handle = np.median(bar, axis=0)
    eigenvalues, axes = np.linalg.eigh(np.cov(bar[:, :2].T))
    bar_axis = axes[:, -1]
    cabinet = box_points(cloud, task['cabinet_bbox_2d'])
    cabinet = cabinet[(cabinet[:, 2] > handle[2] - .22) & (cabinet[:, 2] < handle[2] + .22)]
    if len(cabinet) < 100:
        raise ValueError('柜体平面深度不足')
    rng = np.random.default_rng(42)
    sample = cabinet[rng.choice(len(cabinet), min(3000, len(cabinet)), replace=False)]
    best = None
    for _ in range(180):
        pair = sample[rng.choice(len(sample), 2, replace=False), :2]
        edge = pair[1] - pair[0]
        if np.linalg.norm(edge) < .07:
            continue
        normal = np.r_[-edge[1], edge[0], 0.]
        normal /= np.linalg.norm(normal)
        if closing and abs(np.dot(normal[:2], bar_axis)) > .2:
            continue
        distance = float(np.dot(handle - np.r_[pair[0], handle[2]], normal))
        if distance < 0:
            normal, distance = -normal, -distance
        # Drawer face lies immediately behind its protruding handle.
        if not .008 < distance < .055:
            continue
        residual = np.abs((sample[:, :2] - pair[0]) @ normal[:2])
        inliers = residual < .003
        if best is None or inliers.sum() > best[0]:
            best = (int(inliers.sum()), normal, distance)
    if best is None or best[0] < 100:
        if task.get('action') != 'close':
            raise ValueError('无法从当前深度确定把手后方竖直柜面')
        # The open drawer front may lie outside the cabinet's image region.
        # Its horizontal bar still determines the slide axis; choose its sign
        # away from the observed cabinet, never from a fixed world direction.
        normal = np.r_[-bar_axis[1], bar_axis[0], 0.]
        if np.dot(handle - np.median(cabinet, axis=0), normal) < 0:
            normal = -normal
        best = (0, normal, None)
    _, normal, separation = best
    # Protruding points separate the bar from the planar drawer face.
    outward = bar @ normal
    selected = bar[outward >= np.quantile(outward, .65)]
    bins, counts = np.unique(np.round(selected[:, 2] / .003), return_counts=True)
    peak = bins[np.argmax(counts)] * .003
    selected = selected[np.abs(selected[:, 2] - peak) < .007]
    handle = np.median(selected, axis=0)
    return handle, normal, dict(handle=handle.tolist(), outward_normal=normal.tolist(),
        plane_points=best[0], handle_front_separation=separation,
        method=('model handle region + RGB-D bar axis and observed cabinet side' if best[0] == 0
                else 'model handle region + RGB-D vertical front-plane fit; no object truth'))


def ordered_handle_geometry(cloud, task):
    """Check three closed horizontal bars by measured height, not model boxes.

    This bounded three-drawer skill rejects occluded/missing bars. The model
    supplies the cabinet, action and requested level; depth supplies the order.
    """
    anchor, normal, geometry = handle_geometry(cloud, task)
    delta = cloud - anchor
    depth = delta @ normal
    lateral = delta @ np.cross(normal, [0., 0., 1.])
    normals = np.cross(cloud[1:-1, 2:] - cloud[1:-1, :-2],
                       cloud[2:, 1:-1] - cloud[:-2, 1:-1])
    lengths = np.linalg.norm(normals, axis=-1)
    vertical = np.zeros(cloud.shape[:2], dtype=bool)
    vertical[1:-1, 1:-1] = (lengths > 1e-10) & (np.abs(normals[..., 2]) < .3 * lengths)
    cabinet = box_points(cloud, task['cabinet_bbox_2d'])
    near = (np.isfinite(cloud).all(axis=-1) & (np.abs(lateral) < .08)
            & (depth > -.008) & (depth < .025)
            & (cloud[..., 2] >= cabinet[:, 2].min())
            & (cloud[..., 2] <= cabinet[:, 2].max()))
    points = cloud[near & vertical]
    bins, counts = np.unique(np.round(points[:, 2] / .003), return_counts=True)
    heights = []
    for index in np.argsort(-counts, kind='stable'):
        height = bins[index] * .003
        if any(abs(height - old) < .02 for old in heights):
            continue
        bar = points[np.abs(points[:, 2] - height) < .007]
        if len(bar) < 40 or np.ptp(bar[:, 2]) < .002:
            continue
        eigenvalues = np.linalg.eigvalsh(np.cov(bar[:, :2].T))
        if eigenvalues[-1] > .0003 and eigenvalues[-1] > 4 * eigenvalues[0]:
            heights.append(float(height))
    if len(heights) != 3:
        raise ValueError('当前深度无法独立确认三层把手，未发送动作')
    heights.sort(reverse=True)
    level = ('top', 'middle', 'bottom').index(task['level'])
    handle = anchor
    if abs(anchor[2] - heights[level]) >= .02:
        bar = cloud[near & (np.abs(cloud[..., 2] - heights[level]) < .010)]
        bar = bar[bar @ normal >= np.quantile(bar @ normal, .65)]
        handle = np.median(bar, axis=0)
    geometry.update(handle=handle.tolist(), observed_handle_heights=heights,
                    selected_level=task['level'], model_handle=anchor.tolist(),
                    method='model cabinet + three measured horizontal bars ranked by height; no object truth')
    return handle, normal, geometry


def reacquire_open_handle(cloud, context, level):
    """Reacquire a verified opening in fresh depth; prior coordinates are a gate."""
    if (context.get('semantic_success') is not True
            or context.get('task', {}).get('action') != 'open'
            or context['task'].get('level') != level
            or context.get('drawer_depth_check', {}).get('accepted') is not True):
        raise ValueError('缺少同层已核验的打开观测')
    geometry = context['geometry']
    normal = np.asarray(geometry['outward_normal'])
    predicted = (np.asarray(geometry['handle'])
                 + normal * context['drawer_depth_check']['displacement_m'])
    delta = cloud - predicted
    near = (np.isfinite(cloud).all(axis=-1) & (np.abs(delta[..., 2]) < .012)
            & (np.abs(delta @ np.cross(normal, [0., 0., 1.])) < .07)
            & (np.abs(delta @ normal) < .035))
    bar = cloud[near]
    if len(bar) < 40:
        raise ValueError('当前深度没有重找到已打开把手，未发送动作')
    eigenvalues = np.linalg.eigvalsh(np.cov(bar[:, :2].T))
    if eigenvalues[-1] < .0003 or eigenvalues[-1] < 4 * eigenvalues[0]:
        raise ValueError('当前观测不支持细长把手，未发送动作')
    selected = bar[bar @ normal >= np.quantile(bar @ normal, .65)]
    handle = np.median(selected, axis=0)
    return handle, normal, dict(geometry, handle=handle.tolist(),
        predicted_handle=predicted.tolist(), current_bar_points=len(bar),
        method='verified prior opening + reacquired current RGB-D horizontal bar; no object truth')


def drawer_depth_observation(cloud, geometry):
    """Measure the visible handle-height band and a stationary cabinet front."""
    if cloud is None:
        raise ValueError('缺少当前抽屉深度')
    points = np.asarray(cloud).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    handle = np.asarray(geometry['handle'])
    normal = np.asarray(geometry['outward_normal'])
    delta = points - handle
    lateral = delta @ np.cross(normal, [0., 0., 1.])
    depth = delta @ normal
    # Thin band through the handle avoids the table and horizontal drawer floor.
    band = depth[(np.abs(lateral) < .05) & (np.abs(delta[:, 2]) < .008)
                 & (depth > -.25) & (depth < .25)]
    if len(band) < 40 or np.ptp(np.quantile(band, [.25, .75])) > .04:
        raise ValueError('把手高度深度不足或被遮挡，不能确认抽屉移动')
    # A top drawer has no stationary front above it. Use observed vertical
    # surfaces on either side of the height band; horizontal table pixels must
    # never become a closed-front reference.
    vertical = np.zeros(cloud.shape[:2], dtype=bool)
    normals = np.cross(cloud[1:-1, 2:] - cloud[1:-1, :-2],
                       cloud[2:, 1:-1] - cloud[:-2, 1:-1])
    lengths = np.linalg.norm(normals, axis=-1)
    vertical[1:-1, 1:-1] = (lengths > 1e-10) & (np.abs(normals[..., 2]) < .2 * lengths)
    vertical = vertical.reshape(-1)[np.isfinite(np.asarray(cloud).reshape(-1, 3)).all(axis=1)]
    front = depth[vertical & (np.abs(lateral) < .065)
                  & (np.abs(delta[:, 2]) > .06) & (np.abs(delta[:, 2]) < .18)]
    counts, edges = np.histogram(front, bins=np.arange(-.30, .251, .005))
    peak = int(np.argmax(counts))
    reference = float(edges[peak] + .0025) if counts[peak] >= 100 else None
    return dict(handle_band_m=float(np.median(band)), cabinet_front_m=reference,
                band_points=len(band), cabinet_plane_points=int(counts[peak]))


def drawer_depth_check(cloud, baseline, action):
    try:
        observed = drawer_depth_observation(cloud, baseline['geometry'])
        displacement = observed['handle_band_m'] - baseline['initial']['handle_band_m']
        reference = baseline['initial']['cabinet_front_m']
        if reference is None:
            raise ValueError('没有可确认的静止柜面参考')
        offset = observed['handle_band_m'] - reference
        accepted = ((.06 <= displacement <= .23 and offset >= .07) if action == 'open'
                    else (-.23 <= displacement <= -.06 and .005 <= offset <= .05))
        return dict(accepted=bool(accepted), displacement_m=displacement,
                    handle_to_closed_front_m=offset, observation=observed,
                    reason='相机深度确认移动与终态' if accepted else '相机深度不支持预期移动或终态')
    except ValueError as exc:
        return dict(accepted=None, reason=str(exc))


class DrawerControl(VisualControl):
    def pose_move(self, target, desired, grip, phase, limit=120, stop_on_contact=False):
        self.status(phase)
        start = np.asarray(self.obs['robot0_eef_pos']).copy()
        positions = []
        for _ in range(limit):
            current = Rotation.from_quat(self.obs['robot0_eef_quat']).as_matrix()
            error = Rotation.from_matrix(desired @ current.T).as_rotvec()
            action = servo_action(self.obs['robot0_eef_pos'], target, grip)
            action[3:6] = np.clip(error / .5, -.4, .4)
            self.step(action, phase)
            if (np.linalg.norm(np.asarray(self.obs['robot0_eef_pos']) - target) < .006
                    and np.linalg.norm(error) < .05):
                return
            if stop_on_contact:
                position = np.asarray(self.obs['robot0_eef_pos']).copy()
                positions.append(position)
                travel = float(np.linalg.norm(position - start))
                if (len(positions) >= 12 and travel > .04
                        and np.max(np.linalg.norm(np.asarray(positions[-12:]) - position, axis=1)) < .001):
                    # An end stop and an obstruction both stop the TCP. Release
                    # and observe; only semantic verification may accept closure.
                    self.report['contact_stop'] = dict(travel_m=travel,
                        reason='推动停滞，停止施压并撤手进行视觉核对')
                    return
        raise TimeoutError(f'{phase} 未收敛')

    def run(self, text, task_port=8911, skill='open_drawer', drawer_contexts=None):
        import imageio.v2 as imageio
        started = time.monotonic()
        self.started = started
        self.report.update(chinese=text, skill=skill, controller_object_truth=False)
        try:
            rgb, cloud = self.capture()
            raw = image_bytes(rgb)
            (self.output / 'input.png').write_bytes(raw)
            response = self.request(task_port, 'drawer', dict(text=text, image=base64.b64encode(raw).decode()))
            (self.output / 'model.json').write_text(json.dumps(response, ensure_ascii=False, indent=2))
            if response.get('image_sha256') != hashlib.sha256(raw).hexdigest():
                raise ValueError('Drawer response does not match current observation')
            task = parse_drawer(json.dumps(response['drawer']))
            self.verification_box = task['cabinet_bbox_2d']
            self.report['verification_box'] = self.verification_box
            self.report['verification_action'] = task['action']
            if task['action'] != {'open_drawer': 'open', 'close_drawer': 'close'}[skill]:
                raise ValueError('抽屉动作与原计划不一致')
            if task['action'] == 'close':
                # A jointly generated action/box can select a closed neighbour.
                # Ground the open front separately in the same current image.
                level = {'top': 'top', 'middle': 'middle', 'bottom': 'bottom'}[task['level']]
                location = self.request(task_port, 'locate', dict(image=base64.b64encode(raw).decode(),
                    target=f'the metal handle on the front of the open {level} drawer (not the two handles on the closed drawers)'))
                (self.output / 'handle-location.json').write_text(json.dumps(location, ensure_ascii=False, indent=2))
                if not location.get('bbox_2d'):
                    raise ValueError('当前画面无法定位已打开抽屉的把手')
                task['handle_bbox_2d'] = location['bbox_2d']
                task = parse_drawer(json.dumps(task))
            handle, normal, geometry = (ordered_handle_geometry(cloud, task)
                if task['action'] == 'open' else handle_geometry(cloud, task))
            try:
                initial = drawer_depth_observation(cloud, geometry)
            except ValueError:
                context = (drawer_contexts or {}).get(task['level'])
                if task['action'] != 'close' or context is None:
                    raise
                handle, normal, geometry = reacquire_open_handle(cloud, context, task['level'])
                initial = drawer_depth_observation(cloud, geometry)
            self.report.update(task=task, geometry=geometry)
            self.report['drawer_verification'] = dict(geometry=geometry,
                initial=initial)
            np.savez_compressed(self.output / 'geometry.npz', cloud=cloud)
            up = np.array([0., 0., 1.])
            # A horizontal Panda palm hits the table at low drawers. Approach
            # obliquely so the wrist remains above the handle and table.
            angle = np.deg2rad(25)
            direction = -normal
            forward = direction * np.cos(angle) - up * np.sin(angle)
            jaw_axis = up * np.cos(angle) + direction * np.sin(angle)
            desired = np.column_stack((np.cross(jaw_axis, forward), jaw_axis, forward))
            desired = desired @ Rotation.from_rotvec([0., 0., np.deg2rad(60)]).as_matrix()
            pre = handle - forward * .12
            safe_z = max(1.25, float(self.obs['robot0_eef_pos'][2]))
            self.writer = imageio.get_writer(self.output / 'demo.mp4', fps=20)
            self.pose_move(np.r_[self.obs['robot0_eef_pos'][:2], safe_z], desired, -1., '抬高并面向抽屉')
            self.pose_move(np.r_[pre[:2], max(float(pre[2]) + .04, 1.10)], desired, -1., '移动到把手外侧')
            self.pose_move(pre, desired, -1., '沿柜面外侧下降')
            self.pose_move(handle, desired, -1., '靠近当前视觉把手')
            for _ in range(20):
                self.step(np.r_[np.zeros(6), 1.], '夹住抽屉把手')
            distance = .16 if skill == 'open_drawer' else -.16
            target = handle + normal * distance
            self.pose_move(target, desired, 1., '拉开抽屉' if distance > 0 else '推回抽屉',
                           stop_on_contact=distance < 0)
            reached = np.asarray(self.obs['robot0_eef_pos']).copy()
            for _ in range(15):
                self.step(np.r_[np.zeros(6), -1.], '松开把手')
            self.pose_move(reached + normal * .10, desired, -1., '退出柜面')
            self.pose_move(np.r_[self.obs['robot0_eef_pos'][:2], safe_z], desired, -1., '撤手观察抽屉')
            self.report['motion_completed'] = True
            self.verify_completion(text, task_port)
        except Exception as exc:
            self.report.update(result='interrupted' if isinstance(exc, InterruptedError) else 'exception',
                               exception=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            if self.writer:
                self.writer.close()
            rgb, _ = self.capture()
            (self.output / 'final.png').write_bytes(image_bytes(rgb))
            self.report.update(control_steps=len(self.trace), wall_duration_s=time.monotonic() - started)
            (self.output / 'result.json').write_text(json.dumps(self.report, ensure_ascii=False, indent=2))
            (self.output / 'trace.json').write_text(json.dumps(self.trace, ensure_ascii=False, indent=2))
        return self.report
