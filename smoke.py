import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from scene import DesktopScene
from perception import localize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="runs/scene-smoke")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    env = DesktopScene(args.seed)
    try:
        obs = env.reset()
        print("eef", obs["robot0_eef_pos"], obs["robot0_eef_quat"])
        for _ in range(20):
            obs, _, _, _ = env.step(np.r_[np.zeros(6), -1])
        rgb, depth = env.rgbd()
        Image.fromarray(rgb).save(out / "overhead.png")
        scene, _ = env.rgbd("agentview")
        Image.fromarray(scene).save(out / "scene.png")
        intrinsic, extrinsic = env.calibration()
        detections = {}
        for name in ("cube", "apple", "banana", "tray"):
            d = localize(rgb, depth, intrinsic, extrinsic, name)
            detections[name] = dict(xyz=d.xyz.tolist(), bbox=d.bbox, pixels=d.pixels, height=d.height,
                                    truth_debug_only=env.evaluation_truth(name).tolist())
        (out / "detections.json").write_text(json.dumps(detections, indent=2))
        print(json.dumps(detections, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
