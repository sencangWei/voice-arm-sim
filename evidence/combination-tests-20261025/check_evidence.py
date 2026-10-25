"""Check recorded state continuity and successful commands independently of the VLM."""
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def main():
    summary = read(ROOT / 'summary.json')
    commands = summary['commands']
    assert len(commands) == 8
    assert (summary['full_success'], summary['partial'], summary['uncompleted']) == (2, 1, 5)
    previous = {}
    continuity_checks = 0
    for command in commands:
        output = ROOT / command['archived_output']
        result = read(output / 'result.json')
        assert result['result'] == command['result']
        assert result['completed_steps'] == command['completed_steps']
        assert not command['evaluation_timeout']
        for phase in ('start', 'final'):
            state = np.load(output / f'{phase}-state.npy', allow_pickle=False)
            assert hashlib.sha256(state.tobytes()).hexdigest() == command[f'{phase}_sha256']
        case = command['case']
        if case in previous:
            assert command['start_sha256'] == previous[case]
            continuity_checks += 1
        previous[case] = command['final_sha256']

    bottle = ROOT / 'runs/main/bottle-chain/command-01'
    for step, target in ((1, 'wooden_cabinet_1_top_side'), (2, 'flat_stove_1_cook_region')):
        physics = read(bottle / f'step-{step:02d}/physical-observer.json')
        assert physics['predicates'][f'on wine_bottle_1 {target}']
        assert not physics['objects']['wine_bottle_1']['held']

    drawer = ROOT / 'runs/bottom/fresh-bottom-drawer/command-01'
    openings = []
    for step in range(1, 4):
        joints = read(drawer / f'step-{step:02d}/physical-observer.json')['joints']
        q = joints['wooden_cabinet_1_bottom_level']
        assert q < -0.1 if step in (1, 3) else abs(q) < 0.005
        assert abs(joints['wooden_cabinet_1_top_level']) < 0.005
        assert abs(joints['wooden_cabinet_1_middle_level']) < 0.005
        openings.append(q)

    # Record the actual wrong-layer actuation, rather than trusting the word "middle".
    wrong = read(ROOT / 'runs/main/drawer-chain/command-01/step-01/physical-observer.json')['joints']
    assert wrong['wooden_cabinet_1_top_level'] < -0.1
    assert abs(wrong['wooden_cabinet_1_middle_level']) < 0.005
    receipt = dict(result='PASS', meaning='record integrity and two positive cases verified; six tasks remain incomplete',
        commands=len(commands), state_continuity_checks=continuity_checks,
        full_success=2, partial=1, uncompleted=5, evaluation_timeouts=0,
        bottom_drawer_joint_positions= openings, wrong_layer_failure_confirmed=True)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
