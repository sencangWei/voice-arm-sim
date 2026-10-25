"""Validate recorded physics, input binding and continuity; not a generalization test."""
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
    assert (summary['full_success'], summary['partial'], summary['uncompleted']) == (5, 1, 2)
    previous = {}
    continuity = 0
    completion_inputs = 0
    for command in commands:
        output = ROOT / command['archived_output']
        result = read(output / 'result.json')
        assert result['result'] == command['result']
        assert result['completed_steps'] == command['completed_steps']
        assert not command['evaluation_timeout'] and command['exception'] is None
        for phase in ('start', 'final'):
            state = np.load(output / f'{phase}-state.npy', allow_pickle=False)
            assert hashlib.sha256(state.tobytes()).hexdigest() == command[f'{phase}_sha256']
        case = command['case']
        if case in previous:
            assert previous[case] == command['start_sha256']
            continuity += 1
        previous[case] = command['final_sha256']
        for model_file in output.glob('step-*/completion-model.json'):
            model = read(model_file)
            hashes = [hashlib.sha256((model_file.parent / name).read_bytes()).hexdigest()
                      for name in ('completion-agent.png', 'completion-wrist.png')]
            assert model['image_sha256'] == hashes
            assert model['model_input_image_sha256'] == hashes[:1]
            assert model['model'] == summary['model']
            assert model['revision'] == summary['revision']
            completion_inputs += 1

    runs = ROOT / 'final-v14'
    # Every successful pick/place leaf needs the recorded independent predicate
    # and released gripper, not just a positive model response.
    positive_places = [
        ('bottle-plate-chain/command-01', 1, 'wine_bottle_1', 'plate_1'),
        ('bottle-plate-chain/command-01', 2, 'wine_bottle_1', 'wooden_cabinet_1_top_side'),
        ('bottle-chain/command-01', 1, 'wine_bottle_1', 'wooden_cabinet_1_top_side'),
        ('bottle-chain/command-01', 2, 'wine_bottle_1', 'flat_stove_1_cook_region'),
        ('bottle-chain/command-02', 2, 'cream_cheese_1', 'wooden_cabinet_1_top_side'),
        ('cheese-chain/command-01', 1, 'cream_cheese_1', 'plate_1'),
        ('cheese-chain/command-01', 2, 'cream_cheese_1', 'flat_stove_1_cook_region'),
    ]
    for command, step, source, target in positive_places:
        physics = read(runs / command / f'step-{step:02d}/physical-observer.json')
        assert physics['predicates'][f'on {source} {target}']
        assert not physics['objects'][source]['held']

    # This scene has no main_table predicate. This is a height/non-stove proxy,
    # explicitly weaker than a valid table-contact predicate.
    table = read(runs / 'bottle-chain/command-02/step-01/physical-observer.json')
    assert 'on wine_bottle_1 main_table' in table['predicate_errors']
    bottle = table['objects']['wine_bottle_1']
    assert not bottle['held'] and abs(bottle['position'][2] - .8988609548408878) < .002
    assert not table['predicates']['on wine_bottle_1 flat_stove_1_cook_region']

    for case, layers in [('fresh-bottom-drawer', ('bottom', None, 'bottom')),
                         ('drawer-chain', ('middle', None, 'bottom'))]:
        for step, open_layer in enumerate(layers, 1):
            physics = read(runs / case / 'command-01' / f'step-{step:02d}/physical-observer.json')
            for layer in ('top', 'middle', 'bottom'):
                q = physics['joints'][f'wooden_cabinet_1_{layer}_level']
                assert q < -.1 if layer == open_layer else abs(q) < .005

    failure = read(runs / 'drawer-chain/command-02/step-01/result.json')
    assert failure['semantic_success'] is not True and '边缘余量' in failure['exception']
    assert commands[-2]['physical']['objects']['cream_cheese_1']['held']
    assert commands[-2]['physical']['objects']['cream_cheese_1']['position'][2] > 1.2
    baseline = read(ROOT / 'thin-grasp-baseline/result.json')
    fixed = read(ROOT / 'thin-grasp-topband/result.json')
    assert baseline['physical_lift'] < .01 and not baseline['physical_contact']
    assert fixed['physical_lift'] > .3 and fixed['physical_contact']
    floor = read(ROOT / 'diagnostics/floor-probe/result.json')
    assert floor['stats'][-1]['clear'] == 0
    partial = read(runs / 'cheese-chain/command-01/result.json')
    assert partial['completed_steps'] == 1 and partial['result'] == 'step_unconfirmed'
    assert commands[4]['physical']['predicates']['on cream_cheese_1 flat_stove_1_cook_region']
    assert commands[-2]['completed_steps'] == 0
    # Retrieval was not a valid positive test: the preceding placement failed.
    assert not commands[-2]['physical']['predicates']['in cream_cheese_1 wooden_cabinet_1_bottom_region']
    assert commands[-1]['control_steps'] == 0 and commands[-1]['semantic_success'] is not True
    receipt = dict(result='PASS', meaning='archive integrity and specified physical checks only; one visual false negative and two uncompleted commands remain',
                   commands=8, full_success=5, partial=1, uncompleted=2,
                   state_continuity_checks=continuity, completion_input_bindings=completion_inputs,
                   physical_place_predicates=len(positive_places), drawer_leaf_checks=6,
                   table_check='height/non-stove proxy; table-contact predicate unavailable',
                   retrieval_test='blocked by failed prerequisite; ability not demonstrated')
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
