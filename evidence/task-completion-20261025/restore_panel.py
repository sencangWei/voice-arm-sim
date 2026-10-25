"""One-time GUI recovery from the paused scene captured for this bug report.

Run from repository root using .venv-vla-sim/bin/python with DISPLAY configured.
Does not start any robot motion or training.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from vla.panel import ChinesePanel  # noqa: E402 — repository bootstrap above
from vla.run import parse_args  # noqa: E402 — repository bootstrap above

HERE = Path(__file__).resolve().parent
meta = json.loads((HERE / "retained-scene.json").read_text())
saved = np.load(HERE / "retained-scene-state.npy", allow_pickle=False)
args = parse_args(["--panel", "--viewer", "--suite", meta["suite"],
    "--task-id", str(meta["task_id"]), "--initial-state", str(meta["initial_state"]),
    "--seed", str(meta["seed"]), "--replan-steps", str(meta["replan_steps"]),
    "--prompt", meta["task"], "--port", str(meta["policy_port"]),
    "--output", str(ROOT / "runs/task-completion-20261025/panel")])
panel = ChinesePanel(args)
panel.root.title("中文机械臂 · 指令执行与完成核对")
panel.root.geometry("930x1370+10+20")
load = panel.lab._load
pending = True


def restore_once(*values):
    global pending
    load(*values)
    if not pending:
        return
    lab = panel.lab
    lab.obs = lab.env.set_init_state(saved)
    actual = lab.env.sim.get_state().flatten()
    assert np.array_equal(saved, actual), "Captured scene state did not restore exactly"
    assert lab.env.env.ignore_done is True
    lab.done = lab.env.check_success()
    lab.grip = -1. if np.abs(lab.obs["robot0_gripper_qpos"]).sum() >= .06 else 1.
    lab.scene["restored_from"] = str(HERE / "retained-scene-state.npy")
    lab.scene["user_instruction"] = meta["user_instruction"]
    lab._publish()
    lab._save_scene("retained_scene_restored")
    lab._update(status="已保留物体和机械臂位置，当前暂停。输入指令后点击“执行当前指令”。")
    receipt = dict(result="PASS", state_exact_match=True,
        state_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
        ignore_done=lab.env.env.ignore_done, task_id=meta["task_id"],
        output=str(lab.output), automatic_motion_steps=len(lab.actions))
    receipt_path = ROOT / "evidence/task-completion-20261025/gui-verification.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
    pending = False


panel.lab._load = restore_once
panel.load()
panel.prompt.set(meta["user_instruction"])
panel.root.mainloop()
