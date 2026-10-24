"""Independent Panda tabletop. Object truth is exposed only by evaluation_truth()."""

import os
os.environ.setdefault("MUJOCO_GL", "egl")

from fruit_meshes import add_fruit_meshes
from copy import deepcopy
import xml.etree.ElementTree as ET
import numpy as np
from robosuite.controllers import load_controller_config
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.camera_utils import (
    get_camera_extrinsic_matrix, get_camera_intrinsic_matrix, get_real_depth_map,
)

TABLE_Z = 0.8
NAMES = ("cube", "apple", "banana")


class DesktopScene(SingleArmEnv):
    def __init__(self, seed=42):
        self.seed = seed
        config = load_controller_config(default_controller="OSC_POSE")
        config.update(kp=150, damping_ratio=1, control_delta=True)
        super().__init__(
            robots="Panda", controller_configs=config, initialization_noise=None,
            use_camera_obs=False, has_renderer=False, has_offscreen_renderer=True,
            control_freq=20, horizon=4000, ignore_done=True, hard_reset=False,
        )

    def _load_model(self):
        super()._load_model()
        robot = self.robots[0].robot_model
        robot.set_base_xpos(robot.base_xpos_offset["table"](0.8))
        arena = TableArena(table_full_size=(0.8, 0.8, 0.05),
                           table_offset=(0, 0, TABLE_Z), table_friction=(1, .005, .0001))
        arena.set_origin([0, 0, 0])
        ET.SubElement(arena.worldbody, "camera", name="overhead", pos="0 0 1.8",
                      quat="1 0 0 0", fovy="50")
        # Collision proxies stay fixed; separate meshes provide fruit silhouettes.
        add_fruit_meshes(arena.asset)
        for name in NAMES:
            body = ET.SubElement(arena.worldbody, "body", name=name, pos="0 0 .85")
            ET.SubElement(body, "freejoint", name=f"{name}_joint")
            common = dict(friction="1 .005 .003", condim="6", density="350")
            if name == "cube":
                ET.SubElement(body, "geom", name="cube_geom", type="box", size=".022 .022 .022",
                              rgba=".9 .08 .06 1", **common)
            elif name == "apple":
                ET.SubElement(body, "geom", name="apple_geom", type="sphere", size=".027",
                              rgba=".18 .75 .08 1", **common)
                ET.SubElement(body, "geom", name="apple_stem", type="capsule", size=".002 .006",
                              pos="0 0 .032", rgba=".25 .12 .03 1", mass=".000001",
                              contype="0", conaffinity="0")
            else:
                for i, (x, y, angle) in enumerate([(-.027, .007, -.5), (0, 0, 0), (.027, .007, .5)]):
                    ET.SubElement(body, "geom", name=f"banana_{i}", type="capsule",
                                  size=".021 .014", pos=f"{x} {y} 0", euler=f"0 1.570796 {angle}",
                                  rgba=".98 .78 .04 1", **common)
        tray = ET.SubElement(arena.worldbody, "body", name="tray", pos=".12 .18 .805")
        ET.SubElement(tray, "geom", name="tray_bottom", type="box", size=".085 .075 .005",
                      rgba=".05 .3 .85 1", friction="1 .005 .0001")
        for i, (pos, size) in enumerate([
            ((.087, 0, .016), (.004, .079, .02)), ((-.087, 0, .016), (.004, .079, .02)),
            ((0, .077, .016), (.083, .004, .02)), ((0, -.077, .016), (.083, .004, .02)),
        ]):
            ET.SubElement(tray, "geom", name=f"tray_wall_{i}", type="box",
                          pos=" ".join(map(str, pos)), size=" ".join(map(str, size)), rgba=".05 .3 .85 1")
        # Robosuite computes inertia from group 0 and renders group 1. Keep
        # collision / mass geoms and add matching massless visual geoms.
        for body in list(arena.worldbody.iter("body")):
            for geom in list(body.findall("geom")):
                if geom.get("name", "").startswith(("cube", "apple", "banana", "tray")):
                    name = geom.get("name")
                    if name.startswith("banana_") and name != "banana_0":
                        continue
                    visual = deepcopy(geom)
                    if name in ("apple_geom", "banana_0"):
                        for attribute in ("size", "pos", "euler"):
                            visual.attrib.pop(attribute, None)
                        visual.set("type", "mesh")
                        visual.set("mesh", "apple_skin" if name == "apple_geom" else "banana_skin")
                        visual.set("rgba", ".28 .65 .055 1" if name == "apple_geom" else ".97 .76 .055 1")
                    elif name == "apple_stem":
                        visual.set("pos", "0 0 .026")
                        visual.set("size", ".0018 .008")
                    visual.set("name", geom.get("name") + "_visual")
                    visual.set("group", "1")
                    visual.set("contype", "0")
                    visual.set("conaffinity", "0")
                    visual.set("mass", "0")
                    visual.attrib.pop("density", None)
                    body.append(visual)
        # Massless details make the silhouette readable from the side and above.
        for body in arena.worldbody.findall("body"):
            if body.get("name") == "apple":
                ET.SubElement(body, "geom", name="apple_leaf_visual", type="ellipsoid",
                              size=".009 .0035 .0007", pos=".008 0 .030", euler="0 -.3 .3",
                              rgba=".12 .38 .025 1", group="1", contype="0", conaffinity="0", mass="0")
            elif body.get("name") == "banana":
                for sign in (-1, 1):
                    ET.SubElement(body, "geom", name=f"banana_tip_{sign}_visual", type="ellipsoid",
                                  size=".0035 .0025 .0025", pos=f"{sign * .059} .021 0",
                                  rgba=".24 .12 .035 1", group="1", contype="0", conaffinity="0", mass="0")
        self.model = ManipulationTask(arena, [robot], [])

    def _reset_internal(self):
        super()._reset_internal()
        rng = np.random.default_rng(self.seed)
        starts = [(-.10, -.17, .823), (.17, -.14, .829), (.10, .015, .823)]
        for name, xyz in zip(NAMES, starts):
            p = np.array(xyz) + np.r_[rng.uniform(-.025, .025, 2), 0]
            self.sim.data.set_joint_qpos(f"{name}_joint", np.r_[p, 1, 0, 0, 0])

    def reward(self, action=None):
        return 0.

    def _check_success(self):
        return False  # Success is measured by a separate evaluator, not passed to control.

    def rgbd(self, camera="overhead", size=480):
        rgb, depth = self.sim.render(camera_name=camera, width=size, height=size, depth=True)
        return np.ascontiguousarray(rgb[::-1]), get_real_depth_map(self.sim, depth[::-1])

    def calibration(self, camera="overhead", size=480):
        return (get_camera_intrinsic_matrix(self.sim, camera, size, size),
                get_camera_extrinsic_matrix(self.sim, camera))

    def evaluation_truth(self, name):
        """For result scoring only. Never call from perception or motion planning."""
        return self.sim.data.body_xpos[self.sim.model.body_name2id(name)].copy()
