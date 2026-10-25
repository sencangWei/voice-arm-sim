"""Replay recorded GUI actions offline; score independently of control/model output."""
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main():
    os.environ.setdefault("MUJOCO_GL", "egl")
    from vla.run import configure_libero, load_initial_states
    configure_libero()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from vla.compare import BottleInBowlCheck
    from vla.semantic_grasp import image_bytes

    scene = HERE / "panel-scene"
    report = json.loads((scene / "report.json").read_text())
    task = benchmark.get_benchmark_dict()[report["suite"]]().get_task(report["task_id"])
    env = OffScreenRenderEnv(
        bddl_file_name=str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
        camera_heights=256, camera_widths=256, ignore_done=True)
    try:
        env.seed(report["seed"])
        env.reset()
        # Match Workbench._load's controller warm-up before its one-time restore.
        env.set_init_state(load_initial_states(task)[report["initial_state"]])
        for _ in range(10):
            env.step([0.] * 6 + [-1.])
        env.set_init_state(np.load(scene / "initial_state.npy", allow_pickle=False))
        actions = np.load(scene / "executed_actions.npy", allow_pickle=False)
        checks = []
        scorer = BottleInBowlCheck()
        start = 0
        for index in range(1, 4):
            saved = json.loads((HERE / f"panel-command-{index}/result.json").read_text())
            end = start + saved["control_steps"]
            for action in actions[start:end]:
                env.step(action)
                scorer(env)
            rgb = env.sim.render(width=512, height=512, camera_name="agentview")
            actual = image_bytes(np.ascontiguousarray(rgb[::-1, ::-1]))
            expected = (HERE / f"panel-command-{index}/completion-agent.png").read_bytes()
            held = scorer.history[-1]["held"]
            if index < 3:
                support = ("wooden_cabinet_1_top_side", "flat_stove_1_cook_region")[index-1]
                placed = bool(env.env._eval_predicate(["on", "wine_bottle_1", support]))
            else:
                placed = scorer.streak >= 5
            checks.append(dict(instruction=saved["chinese"], end_step=end,
                image_exact_match=actual == expected,
                image_sha256=hashlib.sha256(actual).hexdigest(),
                physical_placement=placed, gripper_contact=held,
                passed=actual == expected and placed and not held))
            start = end
        assert start == len(actions)
        receipt = dict(result="PASS" if all(c["passed"] for c in checks) else "FAIL",
            method="one initial restore; replay recorded GUI actions; no inference or reset between commands",
            total_controls=len(actions), commands=checks, final_bowl_streak=scorer.streak,
            scoring_definition=BottleInBowlCheck.definition)
        (HERE / "panel-replay-verification.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(receipt, ensure_ascii=False), flush=True)
        return 0 if receipt["result"] == "PASS" else 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
