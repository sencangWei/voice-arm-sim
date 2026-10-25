"""Image-grounded tabletop grasping with calibrated OSC control, not a VLA.

The language model selects objects; RGB-D supplies each new grasp and landing
location. No simulator object identities, poses or success predicates are read.
Limited to separated, upright objects that fit the parallel gripper.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from vla.hybrid import servo_action
from vla.semantic_grasp import VisualSkill


class VisualControl(VisualSkill):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.report["backend"] = "Qwen3-VL intent + RGB-D geometric grasp/place + OSC"

    def policy_metadata(self):
        return dict(backend="visual-geometric-control", learned_action_policy=False,
                    controller_object_truth=False, training_this_run=False)

    def oriented_move(self, target, grip, phase, limit=100):
        # Panda gripper's local Z points down; orientation is a robot calibration,
        # never an object-specific trajectory or a simulator object pose.
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
        upper = visible[visible[:, 2] >= np.quantile(visible[:, 2], .65)]
        xy = np.median(upper[:, :2], axis=0)
        target = np.r_[xy, max(top-.012, self.report["source"]["tabletop"]+.04)]
        safe_z = min(1.34, max(top+.10, float(self.obs["robot0_eef_pos"][2]), 1.20))
        self.source_base = float(np.quantile(visible[:, 2], .05))
        self.report["geometric_grasp"] = dict(target=target.tolist(), top=top, source_base=self.source_base,
            method="upper visible RGB-D median; top-down parallel jaws; no object truth")
        self.oriented_move(np.r_[self.obs["robot0_eef_pos"][:2], safe_z], -1., "张爪抬起并调整朝向")
        self.oriented_move(np.r_[xy, safe_z], -1., "移动到当前视觉物体上方")
        self.oriented_move(target, -1., "下降到视觉抓取位置")
        grasp_eef = np.asarray(self.obs["robot0_eef_pos"]).copy()
        for _ in range(20):
            self.step(np.r_[np.zeros(6), 1.], "闭爪")
        self.confirm_hold()
        self.oriented_move(np.r_[xy, safe_z], 1., "闭爪抬起")
        return float(target[2]), grasp_eef, xy - grasp_eef[:2]

    def release_height(self, min_z, destination_z, tabletop):
        # A new command may start on an elevated support. Translate the measured
        # source-bottom-to-TCP distance, not its distance to the global table.
        return float(min_z + destination_z - self.source_base + .012)
