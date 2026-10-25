"""Robot-only IK diagnostics. Does not step or mutate the physical scene."""
import json
import sys
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from probe import scene, ROOT
sys.path.insert(0, str(ROOT))
from vla.drawer_control import handle_geometry
from vla.visual_control import VisualControl

env, obs = scene()
try:
    rgb, cloud = VisualControl.capture(type('Capture', (), {'env': env})())
    task = json.load(open('.local/recovery-drawer-probe-v1/drawer-model.json'))['drawer']
    handle, normal, _ = handle_geometry(cloud, task)
    robot = env.robots[0]
    model = env.sim.model._model
    data = mujoco.MjData(model)
    data.qpos[:] = env.sim.data.qpos
    baseline = data.qpos.copy()
    joints = np.asarray(robot._ref_joint_pos_indexes)
    dofs = np.asarray(robot._ref_joint_vel_indexes)
    body = env.sim.model.body_name2id(robot.robot_model.eef_name)
    site = robot.eef_site_id
    limits = model.jnt_range[[model.dof_jntid[i] for i in dofs]]
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    up = np.array([0., 0., 1.])
    for name in ('wooden_cabinet_1_g40', 'gripper0_finger1_collision', 'gripper0_finger2_collision'):
        i = env.sim.model.geom_name2id(name)
        print('GEOM',name,env.sim.model.geom_pos[i],env.sim.model.geom_size[i],flush=True)
    for yaw in (0, -45, 45):
        direction = Rotation.from_rotvec(up*np.deg2rad(yaw)).apply(-normal)
        for pitch in (25,35,45,55):
            a = np.deg2rad(pitch)
            forward = direction*np.cos(a)-up*np.sin(a)
            jaw = up*np.cos(a)+direction*np.sin(a)
            base = np.column_stack((np.cross(jaw,forward),jaw,forward))
            for roll in (-50,-45,40,45,50,60,90):
                desired = base @ Rotation.from_rotvec([0,0,np.deg2rad(roll)]).as_matrix()
                target = handle-up*.007+normal*.008
                data.qpos[:] = baseline
                for _ in range(150):
                    mujoco.mj_forward(model,data)
                    ep = target-data.site_xpos[site]
                    er = Rotation.from_matrix(desired @ data.xmat[body].reshape(3,3).T).as_rotvec()
                    if np.linalg.norm(ep)<.003 and np.linalg.norm(er)<.04:
                        break
                    mujoco.mj_jacSite(model,data,jp,jr,site)
                    j = np.vstack((jp[:,dofs],jr[:,dofs]*.3))
                    e = np.r_[ep,er*.3]
                    dq = j.T @ np.linalg.solve(j@j.T+np.eye(6)*.0001,e)
                    data.qpos[joints] = np.clip(data.qpos[joints]+np.clip(dq,-.15,.15),limits[:,0]+.01,limits[:,1]-.01)
                if np.linalg.norm(ep)<.006 and np.linalg.norm(er)<.06:
                    print(yaw,pitch,roll,'IK reachable',flush=True)
finally:
    env.close()
