"""Bounded RGB-D pick/place with physical gripper contact (no object pose oracle)."""
import time

import numpy as np
from scipy.spatial.transform import Rotation

from perception import localize, parse_command


class Controller:
    def __init__(self, env, step_callback=None, locate=None, resolve=None):
        self.env = env
        self.callback = step_callback
        self.locate = locate
        self.resolve = resolve
        self.steps = 0
        self.obs = env._get_observations(force_update=True)
        self.down = Rotation.from_euler("x", np.pi).as_matrix()

    def step(self, action, phase):
        # Callback checks stop before motion as well as updating the display.
        if self.callback:
            self.callback(self.obs, phase, self.steps)
        self.obs, _, _, _ = self.env.step(action)
        self.steps += 1

    def move(self, xyz, grip, phase, limit=100, tolerance=.006):
        xyz = np.asarray(xyz, dtype=float)
        if not (-.32 <= xyz[0] <= .32 and -.32 <= xyz[1] <= .32 and .81 <= xyz[2] <= 1.2):
            raise ValueError("目标超出桌面操作范围")
        for _ in range(limit):
            error = xyz - self.obs["robot0_eef_pos"]
            current = Rotation.from_quat(self.obs["robot0_eef_quat"]).as_matrix()
            rotation = Rotation.from_matrix(self.down @ current.T).as_rotvec()
            if np.linalg.norm(error) < tolerance and np.linalg.norm(rotation) < .06:
                return
            action = np.r_[np.clip(error / .05, -.5, .5), np.clip(rotation / .5, -.5, .5), grip]
            self.step(action, phase)
        raise RuntimeError(f"{phase} 超时，已停止下压/移动；请检查接触或工作空间")

    def hold(self, grip, phase, count=15):
        for _ in range(count):
            self.step(np.r_[np.zeros(6), grip], phase)

    def detect(self, label):
        if self.locate:
            return self.locate(label)
        rgb, depth = self.env.rgbd()
        return localize(rgb, depth, *self.env.calibration(), label)

    def park(self):
        p = self.obs["robot0_eef_pos"].copy()
        p[2] = 1.1
        self.move(p, -1, "抬起并张开夹爪")
        self.move([-.25, 0, 1.1], -1, "移开相机遮挡")
        self.hold(-1, "观察桌面", 8)

    def run(self, text):
        label = self.resolve(text) if self.resolve else parse_command(text)
        start = time.monotonic()
        self.park()
        target, tray = self.detect(label), self.detect("tray")
        observations = {"target": target.xyz.tolist(), "tray": tray.xyz.tolist()}
        for attempt in range(2):
            if attempt:
                self.park()
                target = self.detect(label)  # Fresh pixels after failed contact.
            x, y = target.xyz[:2]
            # EEF site is at the fingertip grasp centre. Surface depth defines
            # grasp height; approach and transfer occur above the tray walls.
            z = max(.824, .8 + target.height * .50)
            self.move([x, y, 1.07], -1, "移动到物体上方")
            self.move([x, y, z], -1, "缓慢靠近抓取", limit=100, tolerance=.008)
            self.hold(1, "闭合夹爪", 20)
            self.move([x, y, 1.07], 1, "抬起并检查抓取")
            width = float(abs(np.diff(self.obs["robot0_gripper_qpos"])[0]))
            # Empty fully closed Panda fingers have near-zero separation.
            if width > .012:
                break
            self.hold(-1, "空抓：张开重新观察", 15)
        else:
            raise RuntimeError("两次抓取均为空抓，已停止；需要重新定位或人工调整")
        x, y = tray.xyz[:2]
        self.move([x, y, 1.09], 1, "运输到托盘上方")
        self.move([x, y, .84 + target.height * .5], 1, "下降放置", tolerance=.005)
        self.hold(-1, "张开释放", 22)
        self.move([x, y, 1.09], -1, "抬起退出")
        self.move([-.25, 0, 1.1], -1, "移开观察结果")
        self.hold(-1, "等待物体稳定", 15)
        placed = self.detect(label)
        error = np.abs(placed.xyz[:2] - tray.xyz[:2])
        success = bool(error[0] < .065 and error[1] < .055 and placed.xyz[2] < .91)
        return dict(success=success, message="已放入托盘" if success else "放置检查未通过，已停止",
                    steps=self.steps, elapsed_s=time.monotonic()-start, attempts=attempt+1,
                    camera_observations=observations, final_camera_xyz=placed.xyz.tolist(),
                    gripper_width_after_lift=width, command=text,
                    controller="RGB-D geometry / OSC_POSE / physical contact")
