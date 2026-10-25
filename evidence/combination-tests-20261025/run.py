"""Live local VLM + MuJoCo combinations; independent physics is never fed to skills."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MUJOCO_GL', 'egl')

from vla.run import configure_libero, load_initial_states
from vla.semantic_grasp import image_bytes
from vla.task_runner import TaskRunner

CASES = {
    'fresh-bottom-drawer': [
        '打开第三层抽屉，然后关上它，再打开第三层抽屉',
    ],
    'bottle-plate-chain': [
        '把酒瓶放到盘子里，然后把它放到柜子顶部',
    ],
    'bottle-chain': [
        '先把酒瓶放到柜子顶部，再把它放到灶台上',
        '把酒瓶从灶台拿到桌面上，然后把蓝色奶酪盒放到柜子顶部',
    ],
    'cheese-chain': [
        '把蓝色奶酪盒放到盘子里，再把它拿到灶台上',
    ],
    'drawer-chain': [
        '打开柜子第二层抽屉，然后关上它，再打开第三层抽屉',
        '把蓝色奶酪盒放进第三层抽屉，然后关闭第三层抽屉',
        '打开第三层抽屉，把里面的奶酪盒拿出来放到桌面上，再关上第三层抽屉',
    ],
}


def physical(env):
    sim = env.sim
    record = dict(joints={n: float(sim.data.get_joint_qpos(n))
                          for n in sim.model.joint_names if 'cabinet' in n},
        objects={}, predicates={}, predicate_errors={})
    for name in ('wine_bottle_1', 'cream_cheese_1'):
        body = name + '_main'
        if body not in sim.model.body_names:
            continue
        record['objects'][name] = dict(position=sim.data.get_body_xpos(body).tolist(),
            held=bool(env.env.check_contact(env.robots[0].gripper.contact_geoms,
                                           env.env.get_object(name))))
        for support in ('wooden_cabinet_1_top_side', 'flat_stove_1_cook_region',
                        'plate_1', 'main_table', 'wooden_cabinet_1_bottom_region'):
            for relation in ('in',) if 'bottom_region' in support else ('on',):
                key = f'{relation} {name} {support}'
                try:
                    record['predicates'][key] = bool(env.env._eval_predicate([relation, name, support]))
                except (KeyError, AttributeError, AssertionError) as exc:
                    record['predicate_errors'][key] = str(exc)
    return record


class ObservedRunner(TaskRunner):
    def execute_leaf(self, *args, **kwargs):
        report = super().execute_leaf(*args, **kwargs)
        leaf = Path(self.report['steps'][-1]['output'])
        (leaf / 'physical-observer.json').write_text(json.dumps(physical(self.env), indent=2))
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--case', choices=CASES)
    parser.add_argument('--initial-state', type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configure_libero()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    task = benchmark.get_benchmark_dict()['libero_goal']().get_task(9)
    summary = dict(controller_object_truth=False, end_to_end_vla=False,
                   reset_between_commands=False, suite='libero_goal', task_id=9,
                   initial_state=args.initial_state, seed=7, commands=[])
    (args.output / 'task.bddl').write_text((Path(get_libero_path('bddl_files')) /
        task.problem_folder / task.bddl_file).read_text())
    cases = {args.case: CASES[args.case]} if args.case else CASES
    for case, commands in cases.items():
        env = OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files')) /
            task.problem_folder / task.bddl_file), camera_heights=256, camera_widths=256, ignore_done=True)
        try:
            env.seed(7)
            env.reset()
            obs = env.set_init_state(load_initial_states(task)[args.initial_state])
            for _ in range(10):
                obs = env.step([0.] * 6 + [-1.])[0]
            for index, text in enumerate(commands, 1):
                out = args.output / case / f'command-{index:02d}'
                started = time.monotonic()
                runner = ObservedRunner(env, obs, None, out,
                    stop=lambda: time.monotonic() - started > 240,
                    status=lambda s: print(s, flush=True))
                before = env.sim.get_state().flatten().copy()
                np.save(out / 'start-state.npy', before)
                (out / 'start.png').write_bytes(image_bytes(runner.capture()[0]))
                print('COMMAND', case, index, text, flush=True)
                error = None
                try:
                    result = runner.run(text)
                except Exception as exc:
                    error = f'{type(exc).__name__}: {exc}'
                    result = runner.report
                obs = runner.obs
                (out / 'final.png').write_bytes(image_bytes(runner.capture()[0]))
                after = env.sim.get_state().flatten().copy()
                np.save(out / 'final-state.npy', after)
                record = dict(case=case, command=index, instruction=text, output=str(out),
                    result=result['result'], semantic_success=result.get('semantic_success'),
                    completed_steps=result['completed_steps'], control_steps=result['control_steps'],
                    wall_s=time.monotonic()-started, exception=error,
                    evaluation_timeout=time.monotonic()-started > 240,
                    start_sha256=hashlib.sha256(before.tobytes()).hexdigest(),
                    final_sha256=hashlib.sha256(after.tobytes()).hexdigest(),
                    physical=physical(env))
                summary['commands'].append(record)
                (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
                print('RECEIPT', json.dumps(record, ensure_ascii=False), flush=True)
        finally:
            env.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
