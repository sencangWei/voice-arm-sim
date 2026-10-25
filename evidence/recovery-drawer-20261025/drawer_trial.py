"""Cabinet motion experiment with physics-only independent scoring."""
from pathlib import Path
import sys
import json
import numpy as np

from probe import scene, ROOT
sys.path.insert(0, str(ROOT))


def main():
    from vla.drawer_control import DrawerControl
    output = Path(sys.argv[1])
    env, obs = scene()
    skill = DrawerControl(env, obs, None, output, status=lambda s: print(s, flush=True))
    try:
        try:
            skill.run('打开柜子第三层抽屉')
        except Exception as exc:
            print(type(exc).__name__, str(exc), flush=True)
        joints = {name: float(env.sim.data.get_joint_qpos(name)) for name in env.sim.model.joint_names if 'cabinet' in name}
        contacts = [(env.sim.model.geom_id2name(c.geom1), env.sim.model.geom_id2name(c.geom2))
                    for c in env.sim.data.contact[:env.sim.data.ncon]
                    if 'gripper' in str(env.sim.model.geom_id2name(c.geom1)) + str(env.sim.model.geom_id2name(c.geom2))]
        print('CONTACTS', contacts, flush=True)
        for name in ('wooden_cabinet_1_g40', 'gripper0_finger1_collision', 'gripper0_finger2_collision'):
            i = env.sim.model.geom_name2id(name)
            print('GEOM', name, 'position', env.sim.data.geom_xpos[i].tolist(),
                  'local', env.sim.model.geom_pos[i].tolist(), 'size', env.sim.model.geom_size[i].tolist(), flush=True)
        print('EEF', skill.obs['robot0_eef_pos'].tolist(), 'GAP', skill.obs['robot0_gripper_qpos'].tolist(), flush=True)
        print('INDEPENDENT', joints, flush=True)
        (output / 'physical.json').write_text(json.dumps(joints, indent=2))
        np.save(output / 'final-state.npy', env.sim.get_state().flatten())
    finally:
        env.close()


if __name__ == '__main__':
    main()
