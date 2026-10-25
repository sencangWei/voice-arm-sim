"""Controlled empty-grasp injection; recovery decisions use the live VLM."""
import sys
import json
import numpy as np
from pathlib import Path
from probe import scene, ROOT
sys.path.insert(0, str(ROOT))
import vla.task_runner as runner_module
from vla.visual_control import VisualControl


class EmptyFirstGrasp(VisualControl):
    injected = False

    def grasp(self, task, source, source_mask, cloud):
        if not type(self).injected:
            type(self).injected = True
            self.report['test_fault_injection'] = 'First attempt closes in free space; no scene reset or object teleport'
            for _ in range(30):
                self.step(np.r_[np.zeros(6), 1.], '受控测试：首次空抓')
            self.confirm_hold()
            raise AssertionError('Injected empty grasp should fail the hold check')
        return super().grasp(task, source, source_mask, cloud)


class InterruptFirstPlace(VisualControl):
    injected = False

    def place(self, *args, **kwargs):
        if not type(self).injected:
            type(self).injected = True
            self.report['test_fault_injection'] = 'Interrupt after verified lifted grasp; retain scene and gripper'
            raise RuntimeError('Controlled interruption before first transfer')
        return super().place(*args, **kwargs)


def main():
    output = Path(sys.argv[1])
    env, obs = scene()
    runner_module.VisualControl = InterruptFirstPlace if '--holding' in sys.argv else EmptyFirstGrasp
    controller = runner_module.TaskRunner(env, obs, None, output, status=lambda s: print(s, flush=True))
    try:
        result = controller.run('把酒瓶放到柜子顶部')
        print(json.dumps(result, ensure_ascii=False), flush=True)
        bottle = env.sim.data.get_body_xpos('wine_bottle_1_main')
        cabinet = env.sim.data.get_body_xpos('wooden_cabinet_1_main')
        measured = dict(bottle=bottle.tolist(), cabinet=cabinet.tolist(),
            on_top=bool(abs(bottle[0]-cabinet[0]) < .13 and abs(bottle[1]-cabinet[1]) < .10
                        and bottle[2] > cabinet[2]+.20))
        (output/'physical.json').write_text(json.dumps(measured, indent=2))
        print('INDEPENDENT', measured, flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
