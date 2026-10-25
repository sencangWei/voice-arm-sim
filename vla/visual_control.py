"""Image-grounded tabletop grasping with calibrated OSC control, not a VLA.

The language model selects objects; RGB-D supplies each new grasp and landing
location. No simulator object identities, poses or success predicates are read.
Limited to separated upright objects, including visible round rims.
"""
import json
import time

import numpy as np
from scipy.spatial.transform import Rotation

from vla.hybrid import servo_action
from vla.semantic_grasp import VisualSkill, circular_container_point, image_bytes


def source_support_height(cloud, visible):
    """Estimate the exposed horizontal support beside an upright source.

    The camera often cannot see a bottle's bottom. Using the lowest visible
    bottle pixels as its physical base otherwise drives it into the destination.
    Only use a nearby measured plane, at most 5 cm below those visible pixels.
    """
    base = float(np.quantile(visible[:, 2], .05))
    normal = np.cross(cloud[1:-1, 2:] - cloud[1:-1, :-2],
                      cloud[2:, 1:-1] - cloud[:-2, 1:-1])
    length = np.linalg.norm(normal, axis=-1)
    center = np.median(visible[:, :2], axis=0)
    points = cloud[1:-1, 1:-1]
    flat = (length > 1e-10) & (np.abs(normal[..., 2]) > .99 * length)
    flat &= np.linalg.norm(points[..., :2] - center, axis=-1) < .12
    flat &= (points[..., 2] <= base + .003) & (points[..., 2] >= base - .05)
    levels, counts = np.unique(np.round(points[..., 2][flat], 3), return_counts=True)
    if len(counts) and counts.max() >= 60:
        return min(base, float(levels[np.argmax(counts)]))
    return base


class VisualControl(VisualSkill):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.report["backend"] = "Qwen3-VL intent + RGB-D geometric grasp/place + OSC"

    def policy_metadata(self):
        return dict(backend="visual-geometric-control", learned_action_policy=False,
                    controller_object_truth=False, training_this_run=False)

    def run(self, text, task_port=8911, destination_kind=None):
        # Share the already-loaded GPU visual model, rather than the CPU probe
        # service used by the earlier experimental VLA grasp route.
        options = {} if destination_kind is None else {"destination_kind": destination_kind}
        return super().run(text, task_port=task_port, **options)

    def recover(self, text, action, previous, task_port=8911, destination_kind=None):
        """Execute a model-selected recovery without resetting the scene."""
        import imageio.v2 as imageio
        self.started = time.monotonic()
        self.report['chinese'] = text
        if self.stop():
            raise InterruptedError('用户已暂停')
        if action == 'retry_grasp':
            gap = float(np.abs(self.obs['robot0_gripper_qpos']).sum())
            if (previous.get('grasp_context') and not previous.get('motion_completed')
                    and .003 < gap < .075):
                raise ValueError('可能仍持有物体，拒绝空中张爪；应继续放置或停止')
            self.writer = imageio.get_writer(self.output / 'retreat.mp4', fps=20)
            try:
                target = np.asarray(self.obs['robot0_eef_pos']).copy()
                target[2] = max(1.20, target[2] + .08)
                self.oriented_move(target, -1., '恢复：张爪抬起脱离失败接触')
            except Exception as exc:
                self.report.update(result='interrupted' if isinstance(exc, InterruptedError) else 'exception',
                    exception=f'{type(exc).__name__}: {exc}', control_steps=len(self.trace),
                    wall_duration_s=time.monotonic()-self.started)
                (self.output / 'result.json').write_text(json.dumps(self.report, ensure_ascii=False, indent=2))
                (self.output / 'trace.json').write_text(json.dumps(self.trace, ensure_ascii=False, indent=2))
                raise
            finally:
                self.writer.close()
                self.writer = None
            return self.run(text, task_port, destination_kind)
        try:
            if action == 'reverify':
                # Observing does not release the object. Preserve this state
                # so an uncertain recheck cannot enable an unsafe regrasp.
                for key in ('grasp_context', 'geometric_grasp', 'motion_completed'):
                    if key in previous:
                        self.report[key] = previous[key]
            if action == 'reverify' and previous.get('verification_box'):
                self.verification_box = previous['verification_box']
                self.report['verification_box'] = self.verification_box
                self.report['verification_action'] = previous.get('verification_action')
                if previous.get('drawer_verification'):
                    self.report['drawer_verification'] = previous['drawer_verification']
            if action == 'resume_place':
                context = previous.get('grasp_context')
                if not context or context.get('source_base') is None:
                    raise ValueError('缺少有效持有上下文，不能继续放置')
                self.report.update(grasp_context=context,
                    geometric_grasp=previous.get('geometric_grasp', {}))
                self.source_base = context['source_base']
                self.writer = imageio.get_writer(self.output / 'demo.mp4', fps=20)
                self.confirm_hold()
                self.place(context['task'], context['min_z'], np.asarray(context['offset_xy']),
                    np.asarray(context['destination']), context['goal_meta'], task_port,
                    context['destination_kind'])
            elif action != 'reverify':
                raise ValueError('Unsupported recovery action')
            self.verify_completion(text, task_port)
            return self.report
        except Exception as exc:
            self.report.update(result='interrupted' if isinstance(exc, InterruptedError) else 'exception',
                               exception=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            if self.writer:
                self.writer.close()
            rgb, _ = self.capture()
            (self.output / 'final.png').write_bytes(image_bytes(rgb))
            self.report.update(control_steps=len(self.trace), wall_duration_s=time.monotonic()-self.started)
            (self.output / 'result.json').write_text(json.dumps(self.report, ensure_ascii=False, indent=2))
            (self.output / 'trace.json').write_text(json.dumps(self.trace, ensure_ascii=False, indent=2))

    def align_for_place(self, metadata, offset):
        if 'floor_width_axis' not in metadata:
            return offset
        # The palm is 20 cm wide along local Y but only 6 cm along X.
        # Align its wide axis with the measured drawer width, at safe height.
        current = Rotation.from_quat(self.obs['robot0_eef_quat']).as_matrix()
        jaw = np.asarray(metadata['floor_width_axis'])
        if jaw @ current[:, 1] < 0:
            jaw = -jaw
        down = np.array([0., 0., -1.])
        desired = np.column_stack((np.cross(jaw, down), jaw, down))
        grasp_rotation = np.asarray(self.report.get('geometric_grasp', {}).get('rotation', current))
        rotated = (desired @ grasp_rotation.T @ np.r_[offset, 0.])[:2]
        self.oriented_move(np.asarray(self.obs['robot0_eef_pos']), 1.,
                           '保持夹持，沿抽屉宽度调整掌部朝向', desired=desired)
        self.report['placement_orientation'] = dict(rotation=desired.tolist(), offset_xy=rotated.tolist(),
            method='visible floor principal axis; short palm dimension toward cabinet')
        return rotated

    def oriented_move(self, target, grip, phase, limit=100, desired=None):
        # Panda gripper's local Z points down; orientation is a robot calibration,
        # never an object-specific trajectory or a simulator object pose.
        if desired is None:
            desired = np.diag([1., -1., -1.])
        self.status(phase)
        for _ in range(limit):
            current = Rotation.from_quat(self.obs["robot0_eef_quat"]).as_matrix()
            error = Rotation.from_matrix(desired @ current.T).as_rotvec()
            action = servo_action(self.obs["robot0_eef_pos"], target, grip)
            action[3:6] = np.clip(error / .5, -.4, .4)
            self.step(action, phase)
            if (np.linalg.norm(np.asarray(self.obs["robot0_eef_pos"]) - target) < .008
                    and np.linalg.norm(error) < .04):
                return
        raise TimeoutError(f"{phase} 未收敛；保留现场")

    def grasp(self, task, source, source_mask, cloud):
        visible = cloud[source_mask]
        top = float(np.quantile(visible[:, 2], .995))
        # Use upper object pixels, excluding an attached tabletop/support at its
        # base. The Panda palm extends to only 3 cm above TCP, so a deep
        # top-down approach hits the object before its fingers can close.
        # A thin box's entire top lies in a narrow height band. A percentile
        # selects just its high edge under even sub-millimetre tilt, shifting
        # the grasp toward neighbouring objects. Keep its measured top face.
        threshold = (top - .003 if top - self.report['source']['tabletop'] < .04
                     else np.quantile(visible[:, 2], .65))
        upper = visible[visible[:, 2] >= threshold]
        xy = np.median(upper[:, :2], axis=0)
        center, mode = xy.copy(), "top_parallel"
        if np.min(np.ptp(upper[:, :2], axis=0)) > .08:
            # The 8 cm Panda opening cannot straddle a wide bowl. Fit its
            # measured rim and pinch one wall with the calibrated Y-axis jaws.
            # Preserve center-to-TCP offset when placing the held container.
            center3, rim = circular_container_point(cloud, source_mask)
            center = center3[:2]
            side = -1. if self.obs["robot0_eef_pos"][1] < center[1] else 1.
            xy = center + [0., side * rim["rim_radius"]]
            mode = "round_rim"
        support = self.report['source']['tabletop']
        # A fixed 4 cm floor clearance closes above thin boxes. The calibrated
        # fingertip reaches 9 mm below TCP; keep 12 mm clearance for low items.
        clearance = .025 if mode == 'round_rim' else .012 if top-support < .04 else .04
        target = np.r_[xy, max(top-(.022 if mode == "round_rim" else .012), support+clearance)]
        safe_z = min(1.34, max(top+.10, float(self.obs["robot0_eef_pos"][2]), 1.20))
        self.source_base = source_support_height(cloud, visible)
        self.report["geometric_grasp"] = dict(target=target.tolist(), top=top, source_base=self.source_base,
            mode=mode, center_xy=center.tolist(),
            method="visible RGB-D top or fitted round rim; parallel jaws; no object truth")
        footprint = float(np.quantile(np.linalg.norm(upper[:, :2]-center, axis=1), .99))
        # Thin objects expose their top footprint directly. Reserve that
        # measured radius plus 12 mm, instead of rejecting small plates with
        # the conservative 6 cm inset retained for tall, partly hidden items.
        margin = (max(.035, footprint+.012) if top-self.source_base < .04
                  else max(.060, footprint+.008))
        self.report['geometric_grasp'].update(footprint_radius=footprint,
            support_margin=margin)
        if mode == "round_rim":
            bottom = visible[visible[:, 2] <= np.quantile(visible[:, 2], .05) + .003]
            radius = float(np.quantile(np.linalg.norm(bottom[:, :2] - center, axis=1), .99))
            self.report["geometric_grasp"].update(bottom_radius=radius, support_margin=radius+.008)
        self.oriented_move(np.r_[self.obs["robot0_eef_pos"][:2], safe_z], -1., "张爪抬起并调整朝向")
        self.oriented_move(np.r_[xy, safe_z], -1., "移动到当前视觉物体上方")
        self.oriented_move(target, -1., "下降到视觉抓取位置")
        grasp_eef = np.asarray(self.obs["robot0_eef_pos"]).copy()
        if 'robot0_eef_quat' in self.obs:
            self.report['geometric_grasp']['rotation'] = Rotation.from_quat(
                self.obs['robot0_eef_quat']).as_matrix().tolist()
        for _ in range(20):
            self.step(servo_action(self.obs["robot0_eef_pos"], target, 1.), "闭爪")
        self.confirm_hold()
        self.oriented_move(np.r_[xy, safe_z], 1., "闭爪抬起")
        return float(target[2]), grasp_eef, center - grasp_eef[:2]

    def release_height(self, min_z, destination_z, tabletop):
        # A new command may start on an elevated support. Translate the measured
        # source-bottom-to-TCP distance, not its distance to the global table.
        return float(min_z + destination_z - self.source_base + .012)
