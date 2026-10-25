"""Isolated physical cabinet probe; ground truth is diagnostic only."""
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MUJOCO_GL', 'egl')


def scene():
    from vla.run import configure_libero, load_initial_states
    configure_libero()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    task = benchmark.get_benchmark_dict()['libero_goal']().get_task(9)
    env = OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files')) /
        task.problem_folder / task.bddl_file), camera_heights=256, camera_widths=256, ignore_done=True)
    env.seed(7)
    env.reset()
    obs = env.set_init_state(load_initial_states(task)[0])
    for _ in range(10):
        obs = env.step([0.] * 6 + [-1.])[0]
    return env, obs


if __name__ == '__main__':
    from vla.visual_control import VisualControl
    from vla.semantic_grasp import image_bytes
    out = Path(sys.argv[1])
    env, obs = scene()
    try:
        skill = VisualControl(env, obs, None, out)
        rgb, cloud = skill.capture()
        (out / 'input.png').write_bytes(image_bytes(rgb))
        np.save(out / 'cloud.npy', cloud)
        diagnostic = {}
        for joint in env.sim.model.joint_names:
            if 'cabinet' in joint:
                diagnostic[joint] = float(env.sim.data.get_joint_qpos(joint))
        for body in env.sim.model.body_names:
            if 'cabinet' in body and ('drawer' in body or body.endswith('main')):
                diagnostic[body] = env.sim.data.get_body_xpos(body).tolist()
        (out / 'diagnostic.json').write_text(json.dumps(diagnostic, indent=2))
        print(diagnostic)
    finally:
        env.close()
