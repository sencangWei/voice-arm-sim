"""Real model/MuJoCo check: refuse absent skills, then reuse scene for grasping."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evidence/recovery-drawer-20261025"))
from probe import scene
from vla.task_runner import TaskRunner
from vla.semantic_grasp import image_bytes


def main():
    output = Path(sys.argv[1]).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    env, obs = scene()
    receipts = []
    try:
        for index, text in enumerate(("打开炉灶", "把酒瓶放到盘子上", "把酒瓶从盘子上拿到柜子顶部"), 1):
            runner = TaskRunner(env, obs, None, output / f"command-{index}",
                                status=lambda s: print(s, flush=True))
            (runner.output / "start.png").write_bytes(image_bytes(runner.capture()[0]))
            before = np.array(env.sim.data.qpos, copy=True)
            result = runner.run(text)
            after = np.array(env.sim.data.qpos, copy=True)
            (runner.output / "final.png").write_bytes(image_bytes(runner.capture()[0]))
            obs = runner.obs
            receipt = {key: result.get(key) for key in (
                "result", "semantic_success", "completed_steps", "control_steps", "exception")}
            receipt.update(command=text, scene_qpos_unchanged=bool(np.array_equal(before, after)))
            receipts.append(receipt)
            (output / "receipts.json").write_text(json.dumps(receipts, ensure_ascii=False, indent=2))
            print("RECEIPT", json.dumps(receipt, ensure_ascii=False), flush=True)
            if index == 1:
                assert result["result"] == "unsupported" and np.array_equal(before, after)
            elif result.get("semantic_success") is not True:
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
