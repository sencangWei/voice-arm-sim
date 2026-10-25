"""Real local-model / isolated MuJoCo acceptance; never touches the live scene.

Run from repo root:
  MUJOCO_GL=egl .venv-vla-sim/bin/python evidence/chinese-task-plan-20261025/acceptance.py --output runs/chinese-plan-acceptance
Physical predicates are independent observers; no value feeds a controller.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from vla.run import configure_libero
    configure_libero()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from vla.task_runner import TaskRunner
    import vla.task_runner as runner_module
    from vla.visual_control import VisualControl

    source = ROOT / "evidence/retained-failure-20261025/scene-source"
    meta = json.loads((source / "report.json").read_text())
    task = benchmark.get_benchmark_dict()[meta["suite"]]().get_task(meta["task_id"])
    env = OffScreenRenderEnv(
        bddl_file_name=str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
        camera_heights=256, camera_widths=256, ignore_done=True)
    summary = dict(controller_object_truth=False, end_to_end_vla=False, reset_between_commands=False)
    try:
        env.seed(meta["seed"])
        env.reset()
        obs = env.set_init_state(np.load(source / "initial_state.npy", allow_pickle=False))
        for _ in range(10):
            obs = env.step([0.] * 6 + [-1.])[0]
        for action in np.load(source / "executed_actions.npy", allow_pickle=False):
            obs = env.step(action.tolist())[0]
        before = env.sim.get_state().flatten().copy()
        summary["initial_state_sha256"] = hashlib.sha256(before.tobytes()).hexdigest()
        blocked = TaskRunner(env, obs, None, args.output / "unsupported")
        rejection = blocked.run("打开柜子第三层，把奶油奶酪放进去，之后合上第三层")
        summary["unsupported"] = dict(result=rejection["result"], plan=rejection.get("plan"),
            control_steps=rejection["control_steps"], state_unchanged=bool(np.array_equal(before, env.sim.get_state().flatten())))
        assert rejection["result"] == "unsupported" and not rejection["control_steps"]
        assert summary["unsupported"]["state_unchanged"]
        physical = []
        class ObservedControl(VisualControl):
            def run(self, text, **kwargs):
                result = super().run(text, **kwargs)
                support = ("wooden_cabinet_1_top_side", "flat_stove_1_cook_region")[len(physical)]
                placed = bool(env.env._eval_predicate(["on", "wine_bottle_1", support]))
                held = bool(env.env.check_contact(env.robots[0].gripper.contact_geoms,
                                                  env.env.get_object("wine_bottle_1")))
                physical.append(dict(instruction=text, support=support, placed=placed, held=held))
                return result
        runner_module.VisualControl = ObservedControl
        sequence = TaskRunner(env, obs, None, args.output / "sequence")
        error = None
        try:
            result = sequence.run("先把酒瓶放在柜子顶上，再把同一个酒瓶放到灶台上")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            result = sequence.report
        summary["sequence"] = dict(result=result["result"], semantic_success=result["semantic_success"],
            completed_steps=result["completed_steps"], control_steps=result["control_steps"],
            wall_duration_s=result["wall_duration_s"], physical_observer=physical, exception=error)
        summary["passed"] = (result["semantic_success"] is True and result["completed_steps"] == 2
                             and len(physical) == 2 and all(p["placed"] and not p["held"] for p in physical))
        np.save(args.output / "final-state.npy", env.sim.get_state().flatten())
    finally:
        env.close()
        (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if summary.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
