import importlib.util, json, os, sys
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull, cKDTree
ROOT=Path.cwd(); sys.path.insert(0,str(ROOT)); os.environ.setdefault('MUJOCO_GL','egl')
from vla.visual_control import VisualControl
from vla.semantic_grasp import image_bytes
from vla.drawer_control import box_points
spec=importlib.util.spec_from_file_location('probe',ROOT/'evidence/recovery-drawer-20261025/probe.py');probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
src=ROOT/'.local/combination-fix-20261026/drawer-v13/drawer-chain/command-02'
env,obs=probe.scene()
try:
    obs=env.set_init_state(np.load(src/'final-state.npy'))
    skill=VisualControl(env,obs,None,ROOT/'.local/combination-fix-20261026/floor-probe')
    rgb,cloud=skill.capture(); (skill.output/'input.png').write_bytes(image_bytes(rgb))
    box=json.loads((src/'step-01/relocalize.json').read_text())['bbox_2d']
    points=box_points(cloud,box);bins,counts=np.unique(np.round(points[:,2]/.002),return_counts=True)
    floor=float(bins[np.argmax(counts)]*.002)
    h,w=cloud.shape[:2];x1,y1,x2,y2=np.asarray(box)*[w,h,w,h]/1000
    mask=np.zeros((h,w),bool);mask[int(y1):int(np.ceil(y2)),int(x1):int(np.ceil(x2))]=True;mask &= abs(cloud[...,2]-floor)<.003
    visible=cloud[mask]; hull=ConvexHull(visible[:,:2]);edges=hull.equations;level=float(np.median(visible[:,2]))
    measured=cloud[np.isfinite(cloud).all(axis=-1)&(cloud[...,2]>=level-.002)&(cloud[...,2]<=level+.015)]
    obstacles=cloud[np.isfinite(cloud).all(axis=-1)&(cloud[...,2]>level+.015)]
    stats=[]
    for margin in [.035,.04,.045,.05,.0576114]:
        candidates=measured[np.all(measured[:,:2]@edges[:,:2].T+edges[:,2]<=-margin,axis=1)]
        distances=cKDTree(obstacles[:,:2]).query(candidates[:,:2])[0] if len(candidates) else np.array([])
        stats.append(dict(margin=margin,interior=len(candidates),clear=len(distances[distances>=margin]),max_obstacle_distance=float(distances.max()) if len(distances) else None))
    report=dict(box=box,floor=floor,level=level,range=np.ptp(visible,axis=0).tolist(),hull=visible[hull.vertices].tolist(),stats=stats)
    np.savez_compressed(skill.output/'geometry.npz',cloud=cloud,mask=mask)
    (skill.output/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
finally:env.close()
