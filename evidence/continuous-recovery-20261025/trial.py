"""Live model + MuJoCo test. No synthetic model decisions or object teleport."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'evidence/recovery-drawer-20261025'))
from probe import scene
from vla.task_runner import TaskRunner
from vla.semantic_grasp import image_bytes


def main():
    output = Path(sys.argv[1]).resolve()
    if output.exists():
        raise FileExistsError(output)
    env, obs = scene()
    commands = (['把蓝色奶酪盒放进柜子第三层抽屉里面，然后关闭第三层抽屉']
                if '--drawer' in sys.argv else
                ['把酒瓶放到柜子顶部', '把酒瓶从柜子顶部拿到灶台上'])
    receipts = []
    try:
        for index, text in enumerate(commands, 1):
            runner = TaskRunner(env, obs, None, output / f'command-{index}',
                                status=lambda s: print(s, flush=True))
            (runner.output / 'start.png').write_bytes(image_bytes(runner.capture()[0]))
            result = runner.run(text)
            (runner.output / 'final.png').write_bytes(image_bytes(runner.capture()[0]))
            obs = runner.obs
            # Object identities/poses are used ONLY for independent diagnostics.
            sim = env.sim
            measured = dict(joints={name: float(sim.data.get_joint_qpos(name))
                                   for name in sim.model.joint_names if 'cabinet' in name},
                bodies={name: sim.data.get_body_xpos(name).tolist() for name in sim.model.body_names
                        if name in ('wine_bottle_1_main', 'cream_cheese_1_main',
                                    'wooden_cabinet_1_main', 'flat_stove_1_main')})
            (runner.output / 'physical.json').write_text(json.dumps(measured, indent=2))
            receipts.append(dict(command=text, result=result['result'],
                semantic_success=result.get('semantic_success'),
                completed_steps=result['completed_steps'], control_steps=result['control_steps'],
                recoveries=result['recoveries'], physical=measured))
            print('RECEIPT', json.dumps(receipts[-1], ensure_ascii=False), flush=True)
            if result.get('semantic_success') is not True:
                break
    finally:
        output.mkdir(parents=True, exist_ok=True)
        (output / 'receipts.json').write_text(json.dumps(receipts, ensure_ascii=False, indent=2))
        env.close()


if __name__ == '__main__':
    main()
