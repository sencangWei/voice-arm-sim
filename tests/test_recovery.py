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
