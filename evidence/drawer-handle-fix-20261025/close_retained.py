"""Close the exact recorded failed scene using fresh VLM/RGB-D and physics.

Replay is only fixture construction; new control reads images/proprioception,
never diagnostic object poses or drawer joint states.
"""
from pathlib import Path
import sys, json
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'evidence/recovery-drawer-20261025'))
from probe import scene
from vla.drawer_control import DrawerControl

source=Path(__file__).resolve().parent/'retained'
output=Path(sys.argv[1]);output.mkdir(parents=True,exist_ok=False)
env,obs=scene()
def physical():
    return dict(joints={n:float(env.sim.data.get_joint_qpos(n)) for n in env.sim.model.joint_names if 'cabinet' in n},cheese=env.sim.data.get_body_xpos('cream_cheese_1_main').tolist())
try:
    for action in np.load(source/'executed_actions.npy'):
        obs=env.step(action)[0]
    before=physical()
    skill=DrawerControl(env,obs,None,output/'close',status=lambda s:print(s,flush=True))
    error=None
    try:
        skill.run('关闭柜子第三层抽屉',skill='close_drawer')
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'
    after=physical()
    receipt=dict(before=before,after=after,report=skill.report,exception=error,independent_closed=abs(after['joints']['wooden_cabinet_1_bottom_level'])<.015)
    (output/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
    np.save(output/'final-state.npy',env.sim.get_state().flatten())
    print('RECEIPT',json.dumps(receipt,ensure_ascii=False),flush=True)
finally:
    env.close()
