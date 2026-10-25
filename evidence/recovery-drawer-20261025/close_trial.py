"""Diagnostic-only restore of a saved placement, then normal visual close."""
import sys
import json
import numpy as np
from pathlib import Path
from probe import scene, ROOT
sys.path.insert(0, str(ROOT))
from vla.drawer_control import DrawerControl

env, obs = scene()
output = Path(sys.argv[1])
try:
    obs = env.set_init_state(np.load(sys.argv[2]))
    skill = DrawerControl(env, obs, None, output, status=lambda s: print(s, flush=True))
    try:
        skill.run('关闭柜子第三层抽屉', skill='close_drawer')
    except Exception as exc:
        print(type(exc).__name__, str(exc), flush=True)
    contacts = []
    for i in range(env.sim.data.ncon):
        c = env.sim.data.contact[i]
        contacts.append(dict(names=[env.sim.model.geom_id2name(int(v)) for v in (c.geom1, c.geom2)],
                             distance=float(c.dist)))
    (output / 'contacts.json').write_text(json.dumps(contacts, indent=2))
    (output / 'physical.json').write_text(json.dumps({name: float(env.sim.data.get_joint_qpos(name))
        for name in env.sim.model.joint_names if 'cabinet' in name}, indent=2))
    print('CONTACTS', contacts, flush=True)
finally:
    env.close()
