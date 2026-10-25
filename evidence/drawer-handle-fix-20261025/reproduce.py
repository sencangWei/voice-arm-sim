"""Reproduce retained panel failure via recorded physics actions; diagnostic only."""
from pathlib import Path
import sys, json, hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'evidence/recovery-drawer-20261025'))
from probe import scene
from vla.visual_control import VisualControl
from vla.drawer_control import box_points, handle_geometry
from vla.semantic_grasp import image_bytes

source=Path(__file__).resolve().parent/'retained'
output=Path(sys.argv[1]);output.mkdir(parents=True,exist_ok=False)
env,obs=scene()
try:
    actions=np.load(source/'executed_actions.npy')
    for action in actions:
        obs=env.step(action)[0]
    skill=VisualControl(env,obs,None,output/'capture')
    rgb,cloud=skill.capture()
    raw=image_bytes(rgb);(output/'input.png').write_bytes(raw)
    np.savez_compressed(output/'geometry.npz',cloud=cloud)
    np.save(output/'state.npy',env.sim.get_state().flatten())
    previous=(source/'input.png').read_bytes()
    result=dict(actions=len(actions),image_sha256=hashlib.sha256(raw).hexdigest(),recorded_image_sha256=hashlib.sha256(previous).hexdigest(),image_exact_match=raw==previous)
    task=json.loads((source/'model.json').read_text())['drawer']
    location=json.loads((source/'handle-location.json').read_text())
    task['handle_bbox_2d']=location['bbox_2d']
    result['task']=task
    points=box_points(cloud,task['handle_bbox_2d'])
    bins,counts=np.unique(np.round(points[:,2]/.003),return_counts=True)
    peak=bins[np.argmax(counts)]*.003
    bar=points[np.abs(points[:,2]-peak)<.007]
    eig,axes=np.linalg.eigh(np.cov(bar[:,:2].T))
    result.update(points=len(points),peak_z=peak,bar_points=len(bar),eigenvalues=eig.tolist(),bar_bounds=np.quantile(bar,[0,.5,1],axis=0).tolist(),height_modes=[(float(z),int(n)) for z,n in sorted(zip(bins*.003,counts),key=lambda x:-x[1])[:12]])
    try:
        handle,normal,geometry=handle_geometry(cloud,task);result['geometry']=geometry
    except Exception as exc:
        result['exception']=str(exc)
    result['independent_joints']={n:float(env.sim.data.get_joint_qpos(n)) for n in env.sim.model.joint_names if 'cabinet' in n}
    (output/'diagnosis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))
finally:
    env.close()
