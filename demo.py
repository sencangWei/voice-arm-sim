"""Headless physical-contact evaluation and actual simulation video."""
import argparse
import json
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from controller import Controller
from scene import DesktopScene


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--command", default="把红色方块放进蓝色托盘")
    p.add_argument("--output", default=None)
    args = p.parse_args()
    out = Path(args.output or f"runs/demo-{time.time_ns()}")
    out.mkdir(parents=True, exist_ok=False)
    env = DesktopScene(args.seed)
    result = {}
    trace = []
    writer = imageio.get_writer(out / "demo.mp4", fps=10)
    try:
        env.reset()
        def callback(obs, phase, steps):
            if not trace or phase != trace[-1]["phase"]:
                print(steps, phase, obs["robot0_eef_pos"].round(3).tolist(), flush=True)
            trace.append(dict(step=steps, phase=phase, eef=obs["robot0_eef_pos"].tolist(),
                              gripper=obs["robot0_gripper_qpos"].tolist()))
            if steps % 2 == 0:
                scene, _ = env.rgbd("agentview", 480)
                top, _ = env.rgbd("overhead", 480)
                writer.append_data(np.concatenate([scene, top], axis=1))
        result = Controller(env, callback).run(args.command)
        # Simulator object poses are used here ONLY for independent scoring.
        label = {"方块":"cube", "苹果":"apple", "香蕉":"banana"}
        name = next(v for k,v in label.items() if k in args.command)
        obj, tray = env.evaluation_truth(name), env.evaluation_truth("tray")
        result["evaluation_truth_only"] = dict(object_xyz=obj.tolist(), tray_xyz=tray.tolist(),
            in_tray=bool(abs(obj[0]-tray[0])<.065 and abs(obj[1]-tray[1])<.055 and .81<obj[2]<.89))
    except Exception as exc:
        result.update(success=False, exception=f"{type(exc).__name__}: {exc}")
        print(result, flush=True)
    finally:
        writer.close()
        env.close()
        result.update(seed=args.seed, command=args.command, video="demo.mp4")
        (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        (out / "trace.json").write_text(json.dumps(trace, ensure_ascii=False))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
