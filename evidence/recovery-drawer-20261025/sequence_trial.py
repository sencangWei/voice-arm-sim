"""Run an actual three-step drawer request; score physics independently."""
import sys
import json
import numpy as np
from pathlib import Path
from probe import scene, ROOT
sys.path.insert(0, str(ROOT))
from vla.task_runner import TaskRunner
from vla.semantic_grasp import VisualSkill, image_bytes

output = Path(sys.argv[1])
env, obs = scene()
diagnostics = []
original_step = VisualSkill.step
def observed_step(self, action, phase):
    original_step(self, action, phase)
    sim = env.sim
    body = sim.model.body_name2id('cream_cheese_1_main')
    diagnostics.append(dict(phase=phase, position=sim.data.body_xpos[body].tolist(),
        rotation=sim.data.body_xmat[body].reshape(3,3).tolist(), eef=self.obs['robot0_eef_pos'].tolist()))
    if len(diagnostics) == 1 or diagnostics[-2]['phase'] != phase:
        rgb, _ = self.capture()
        (output / f'phase-{len(diagnostics):04d}.png').write_bytes(image_bytes(rgb))
VisualSkill.step = observed_step
try:
    runner = TaskRunner(env, obs, None, output, status=lambda s: print(s, flush=True))
    execute = runner.execute_leaf
    def capture_leaf(step, number, *args, **kwargs):
        report = execute(step, number, *args, **kwargs)
        if number == 2 and report.get('semantic_success'):
            np.save(output / 'after-place-state.npy', env.sim.get_state().flatten())
        contacts = []
        for i in range(env.sim.data.ncon):
            contact = env.sim.data.contact[i]
            names = [env.sim.model.geom_id2name(int(v)) for v in (contact.geom1, contact.geom2)]
            contacts.append(dict(names=names, distance=float(contact.dist)))
        np.save(output / f'state-{len(runner.report["steps"]):02d}.npy', env.sim.get_state().flatten())
        (output / f'contacts-{len(runner.report["steps"]):02d}.json').write_text(json.dumps(contacts, indent=2))
        return report
    runner.execute_leaf = capture_leaf
    result = runner.run('打开柜子第三层抽屉，把蓝色奶酪盒放进第三层抽屉里面，然后关闭第三层抽屉')
    print('RESULT', json.dumps(result, ensure_ascii=False), flush=True)
    measured = {name: float(env.sim.data.get_joint_qpos(name))
                for name in env.sim.model.joint_names if 'cabinet' in name}
    measured['objects'] = {name: env.sim.data.get_body_xpos(name).tolist()
        for name in env.sim.model.body_names if 'cream_cheese' in name or name == 'wooden_cabinet_1_main'}
    (output / 'physical.json').write_text(json.dumps(measured, indent=2))
    print('INDEPENDENT', measured, flush=True)
finally:
    (output / 'object-motion.json').write_text(json.dumps(diagnostics, indent=2))
    env.close()
