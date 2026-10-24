"""Physical put-in / take-out regression. Truth is used only after execution."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from controller import Controller
from scene import DesktopScene


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 99])
    parser.add_argument("--targets", nargs="+", choices=["banana", "apple", "cube"],
                        default=["banana", "apple", "cube"])
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    env = DesktopScene()
    records = []
    names = {"banana": "香蕉", "apple": "苹果", "cube": "红色方块"}
    try:
        for seed in args.seeds:
            for label in args.targets:
                env.seed = seed
                env.reset()
                record = {"seed": seed, "label": label}
                video = None
                if seed == args.seeds[0] and label == args.targets[0]:
                    video = cv2.VideoWriter(str(output / "demo.mp4"),
                                            cv2.VideoWriter_fourcc(*"mp4v"), 10, (480, 480))
                    if not video.isOpened():
                        raise RuntimeError("Video writer unavailable")

                def capture(obs, phase, steps):
                    if video is not None and steps % 2 == 0:
                        rgb, _ = env.rgbd()
                        video.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

                try:
                    record["put_in"] = Controller(env, capture).run(f"把{names[label]}放进蓝色托盘")
                    obj, tray = env.evaluation_truth(label), env.evaluation_truth("tray")
                    record["truth_before"] = obj.tolist()
                    record["truth_in_tray"] = bool(np.all(np.abs(obj[:2] - tray[:2]) < [.065, .055]))
                    if video is not None:
                        rgb, _ = env.rgbd()
                        cv2.imwrite(str(output / "before.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    record["take_out"] = Controller(env, capture).run(f"把{names[label]}从盘子里面取出来")
                    obj = env.evaluation_truth(label)
                    destination = record["take_out"]["camera_observations"]["destination"]
                    record["truth_after"] = obj.tolist()
                    record["truth_on_table"] = bool(
                        np.any(np.abs(obj[:2] - tray[:2]) > [.15, .14])
                        and np.linalg.norm(obj[:2] - np.array(destination[:2])) < .065
                        and .81 < obj[2] < .85)
                    if video is not None:
                        rgb, _ = env.rgbd()
                        cv2.imwrite(str(output / "after.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                except Exception as exc:
                    record["exception"] = str(exc)
                finally:
                    if video is not None:
                        video.release()
                record["passed"] = bool(record.get("put_in", {}).get("success")
                                        and record.get("take_out", {}).get("success")
                                        and record.get("truth_in_tray") and record.get("truth_on_table"))
                records.append(record)
                (output / "results.json").write_text(json.dumps(records, ensure_ascii=False, indent=2))
                print(seed, label, record["passed"], record.get("exception", ""), flush=True)
    finally:
        env.close()
    assert all(record["passed"] for record in records), "Physical regression failed; see results.json"


if __name__ == "__main__":
    main()
