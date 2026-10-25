"""Read-only RGB-D palm clearance candidates using robot-only calibration."""
import sys
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from probe import scene, ROOT
sys.path.insert(0, str(ROOT))
from vla.drawer_control import handle_geometry
from vla.visual_control import VisualControl
import json

env, obs = scene()
try:
    rgb, cloud = VisualControl.capture(type('Capture', (), {'env': env})())
    task = json.load(open('.local/recovery-drawer-probe-v1/drawer-model.json'))['drawer']
    handle, normal, _ = handle_geometry(cloud, task)
    i = env.sim.model.geom_name2id('gripper0_hand_collision')
    m = env.sim.model.geom_dataid[i]
    a, n = env.sim.model.mesh_vertadr[m], env.sim.model.mesh_vertnum[m]
    world = env.sim.model.mesh_vert[a:a+n] @ env.sim.data.geom_xmat[i].reshape(3, 3).T + env.sim.data.geom_xpos[i]
    rotation = Rotation.from_quat(obs['robot0_eef_quat']).as_matrix()
    vertices = (world - obs['robot0_eef_pos']) @ rotation
    hulls = [ConvexHull(vertices).equations]
    for name in ('gripper0_finger1_collision', 'gripper0_finger2_collision'):
        i = env.sim.model.geom_name2id(name)
        m = env.sim.model.geom_dataid[i]
        a, n = env.sim.model.mesh_vertadr[m], env.sim.model.mesh_vertnum[m]
        world = env.sim.model.mesh_vert[a:a+n] @ env.sim.data.geom_xmat[i].reshape(3,3).T + env.sim.data.geom_xpos[i]
        v = (world-obs['robot0_eef_pos']) @ rotation
        hulls.append(ConvexHull(v).equations)
        print(name,'bounds',v.min(0),v.max(0),flush=True)
    print('robot palm TCP bounds', vertices.min(0), vertices.max(0))
    visible = cloud.reshape(-1,3)[::2]
    up = np.array([0., 0., 1.])
    candidates = []
    for yaw in (0,45):
        direction = Rotation.from_rotvec(up * np.deg2rad(yaw)).apply(-normal)
        for degrees in (25,35,45,55,65):
            angle = np.deg2rad(degrees)
            forward = direction*np.cos(angle)-up*np.sin(angle)
            jaw = up*np.cos(angle)+direction*np.sin(angle)
            desired = np.column_stack((np.cross(jaw,forward),jaw,forward))
            for roll in (-60,-45,-30,0,30,45,60,90):
                r = desired @ Rotation.from_rotvec([0, 0, np.deg2rad(roll)]).as_matrix()
                for out in (0.,.008,.016):
                    for down in (.0,.007,.014):
                        p = (visible-(handle-up*down+normal*out)) @ r
                        counts = [int((np.max(p @ h[:,:3].T+h[:,3],axis=1)<.002).sum()) for h in hulls]
                        candidates.append((sum(counts),yaw,degrees,roll,out,down,counts))
    for candidate in sorted(candidates)[:35]:
        print(candidate,flush=True)
finally:
    env.close()
