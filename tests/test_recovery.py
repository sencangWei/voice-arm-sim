import json

import pytest

from vla.recovery import parse_recovery


@pytest.mark.parametrize('action', ['retry_grasp', 'resume_place', 'reverify', 'retry_drawer', 'stop'])
def test_recovery_is_executable_choice(action):
    value = dict(action=action, reason='根据当前画面调整')
    assert parse_recovery(json.dumps(value)) == value


@pytest.mark.parametrize('value', [dict(action='python', reason='x'),
    dict(action='retry_grasp', reason=''), dict(action='stop', reason='x', code='move()')])
def test_recovery_cannot_execute_arbitrary_tools(value):
    with pytest.raises(ValueError):
        parse_recovery(json.dumps(value))


def test_recovery_can_plan_a_concrete_prerequisite():
    value = dict(action='prepare', reason='先打开原目的地抽屉', progress='no_progress', step=dict(
        skill='open_drawer', instruction='打开第三层抽屉', destination_kind='drawer_interior'))
    assert parse_recovery(json.dumps(value)) == value


@pytest.mark.parametrize('mutate', [
    lambda p: p.update(progress='completed'),
    lambda p: p.update(step=None),
    lambda p: p['step'].update(skill='python'),
    lambda p: p['step'].update(skill='unsupported'),
    lambda p: p.update(action='retry_grasp'),
])
def test_invalid_recovery_plan_is_rejected(mutate):
    value = dict(action='prepare', reason='先打开抽屉', progress='uncertain', step=dict(
        skill='open_drawer', instruction='打开第三层', destination_kind='drawer_interior'))
    mutate(value)
    with pytest.raises(ValueError):
        parse_recovery(json.dumps(value))
