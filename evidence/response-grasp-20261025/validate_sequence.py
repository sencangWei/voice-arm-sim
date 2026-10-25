"""Real closed-loop rollouts; independent object truth scores, never controls."""
import argparse
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--from-failure', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('MUJOCO_GL', 'egl')
    from vla.run import configure_libero
    configure_libero()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from vla.visual_control import VisualControl
    from vla.compare import BottleInBowlCheck

    scene = HERE / 'before'
    meta = json.loads((scene / 'report.json').read_text())
    task = benchmark.get_benchmark_dict()[meta['suite']]().get_task(meta['task_id'])
    env = OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files')) /
        task.problem_folder / task.bddl_file), camera_heights=256, camera_widths=256, ignore_done=True)
    records = []
    try:
        env.seed(meta['seed'])
        env.reset()
        obs = env.set_init_state(np.load(scene / 'initial_state.npy', allow_pickle=False))
        for _ in range(10):
            obs = env.step([0.] * 6 + [-1.])[0]
        if args.from_failure:
            for action in np.load(scene / 'executed_actions.npy', allow_pickle=False):
                obs = env.step(action.tolist())[0]
        np.save(args.output / 'start-state.npy', env.sim.get_state().flatten())
        commands = [('把碗放到盘子上', 'akita_black_bowl_1', 'plate_1'),
                    ('把酒瓶放到灶台上', 'wine_bottle_1', 'flat_stove_1_cook_region'),
                    ('把酒瓶放到碗里面', 'wine_bottle_1', None)]
        for index, (text, source, support) in enumerate(commands, 1):
            skill = VisualControl(env, obs, None, args.output / f'command-{index}')
            physical = []
            scorer = BottleInBowlCheck()
            original_step = skill.step

            def scored_step(action, phase):
                original_step(action, phase)
                if support is None:
                    placed = bool(scorer(env))
                else:
                    placed = bool(env.env._eval_predicate(['on', source, support]))
                held = bool(env.env.check_contact(env.robots[0].gripper.contact_geoms,
                                                  env.env.get_object(source)))
                physical.append(dict(placed=placed, held=held))

            skill.step = scored_step
            error = None
            try:
                skill.run(text)
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
            obs = skill.obs
            np.save(skill.output / 'final-state.npy', env.sim.get_state().flatten())
            (skill.output / 'physical-observer.json').write_text(json.dumps(physical, indent=2))
            passed = (error is None and skill.report.get('semantic_success') is True and
                      len(physical) >= 10 and all(p['placed'] and not p['held'] for p in physical[-10:]))
            record = dict(instruction=text, passed=passed, exception=error,
                controller_result=skill.report['result'], control_steps=len(skill.trace),
                first_control_wall_s=skill.report.get('first_control_wall_s'),
                wall_duration_s=skill.report['wall_duration_s'], final_physical=physical[-1] if physical else None)
            records.append(record)
            receipt = dict(result='PASS' if len(records) == 3 and all(r['passed'] for r in records) else 'FAIL',
                from_retained_failure=args.from_failure, no_reset_between_commands=True,
                controller_object_truth=False, initial_state_sha256=hashlib.sha256(
                    (args.output / 'start-state.npy').read_bytes()).hexdigest(), commands=records)
            (args.output / 'validation.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
            print(json.dumps(record, ensure_ascii=False), flush=True)
            if not passed:
                return 1
        return 0
    finally:
        env.close()


if __name__ == '__main__':
    raise SystemExit(main())
