"""Path validation checks; these do not prove pretrained model competence."""
import pytest

from vla.hamster import instruction, parse_path


def test_path_keeps_gripper_actions_at_preceding_point():
    result = parse_path("reasoning <ans>[(0.1, 0.2), <action>Close Gripper</action>, "
                        "(0.8, 0.9), <action>Open Gripper</action>]</ans>")
    assert result == [{"x": .1, "y": .2, "action": "Close Gripper"},
                      {"x": .8, "y": .9, "action": "Open Gripper"}]


@pytest.mark.parametrize("text", [
    "[(0.1, 0.2), (0.8, 0.9)]",  # no answer tags
    "<ans>[(0.1, 0.2)]</ans>",  # one point
    "<ans>[(1.1, 0.2), (0.8, 0.9)]</ans>",
    "<ans>[(True, 0.2), (0.8, 0.9)]</ans>",
    "<ans>[<action>Close Gripper</action>, (0.8, 0.9)]</ans>",
    "<ans>[(0.1, 0.2), 'invented action']</ans>",
    "<ans>[__import__('os').system('false'), (0.8, 0.9)]</ans>",
    "<ans>[(0.1, 0.2), (0.8, 0.9)]</ans><ans>[]</ans>",
])
def test_invalid_paths_are_rejected_without_fallback(text):
    with pytest.raises(ValueError):
        parse_path(text)


def test_changed_destination_is_passed_verbatim():
    assert "<quest>put the wine bottle in the bowl</quest>" in instruction("put the wine bottle in the bowl")
    assert "<quest>put the wine bottle on the rack</quest>" not in instruction("put the wine bottle in the bowl")
