"""Whole-request capability gating and scene-preserving sequential execution."""
import hashlib
import json

import numpy as np
import pytest

from vla.task_plan import parse_plan
from vla.task_runner import TaskRunner


def plan(*instructions, status="ready", missing=None):
    return dict(status=status, reason="按原指令执行" if status == "ready" else "缺少开关抽屉技能",
                missing_skills=missing or [], steps=[dict(skill="pick_place", instruction=text,
                destination_kind="support_surface") for text in instructions])


def test_unsupported_whole_request_keeps_all_steps():
    value = plan("打开第三层抽屉", "把奶酪放进去", "关上第三层抽屉", status="unsupported",
                 missing=["open_drawer", "close_drawer", "place_inside_drawer"])
    for step in value["steps"]:
        step.update(skill="unsupported", destination_kind="unsupported")
    assert parse_plan(json.dumps(value)) == value


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(extra=True),
    lambda p: p.update(steps=[]),
    lambda p: p.update(missing_skills=["open_drawer"]),
    lambda p: p["steps"][0].update(skill="open_drawer"),
    lambda p: p["steps"][0].update(destination_kind="drawer"),
    lambda p: p["steps"][0].update(instruction=""),
    lambda p: p.update(steps=p["steps"] * 9),
])
def test_ready_plan_fails_closed(mutate):
    value = plan("把酒瓶放到柜顶")
    mutate(value)
    with pytest.raises(ValueError):
        parse_plan(json.dumps(value))


def runner(tmp_path, monkeypatch, value, outcomes=(True, True), image_matches=True, recovery='stop',
           decisions=None):
    events = []
    controller = TaskRunner(object(), {"version": 0}, None, tmp_path / "run",
                            stop=lambda: False, status=lambda message: events.append(message))
    controller.capture = lambda: (np.zeros((8, 8, 3), dtype=np.uint8), None)
    choices = iter(decisions) if decisions is not None else None
    def request(port, endpoint, payload):
        import base64
        events.append(dict(payload))
        response = dict(image_sha256=hashlib.sha256(base64.b64decode(payload['image'])).hexdigest()
                        if image_matches else 'stale')
        if endpoint == 'plan':
            response['plan'] = value
        elif endpoint == 'recovery':
            response['recovery'] = next(choices) if choices else dict(action=recovery, reason='现场恢复决策')
        else:
            raise AssertionError(endpoint)
        return response
    controller.request = request
    results = iter(outcomes)
    class Leaf:
        def __init__(self, env, obs, client, output, **callbacks):
            self.output = output
            self.obs = {"version": obs["version"] + 1}
            self.trace = [{}]
            self.report = dict(inference_calls=3, semantic_success=next(results),
                               completion={"reason": "image evidence"}, result="leaf")
            events.append(("leaf", obs["version"], output))
        def run(self, text, **kwargs):
            events.append(("run", text, kwargs))
        def recover(self, text, action, previous, **kwargs):
            events.append(('recover', action, text))
    monkeypatch.setattr("vla.task_runner.VisualControl", Leaf)
    monkeypatch.setattr('vla.task_runner.DrawerControl', Leaf)
    return controller, events


def test_unsupported_is_zero_motion(tmp_path, monkeypatch):
    controller, events = runner(tmp_path, monkeypatch, plan(status="unsupported", missing=["open_drawer"]))
    report = controller.run("打开抽屉再放奶酪进去")
    assert report["result"] == "unsupported"
    assert report["control_steps"] == 0
    assert not any(isinstance(e, tuple) for e in events)
    assert (controller.output / "result.json").is_file()


def test_steps_reuse_updated_observation_and_stop_after_all_verified(tmp_path, monkeypatch):
    controller, events = runner(tmp_path, monkeypatch, plan("酒瓶放柜顶", "酒瓶放灶台"))
    report = controller.run("先把酒瓶放柜顶，再放灶台")
    assert report["semantic_success"] is True
    assert report["control_steps"] == 2 and report["completed_steps"] == 2
    assert [e[1] for e in events if isinstance(e, tuple) and e[0] == "leaf"] == [0, 1]


@pytest.mark.parametrize("outcome", [False, None])
def test_failed_or_uncertain_verification_blocks_next_step_and_feeds_back(tmp_path, monkeypatch, outcome):
    controller, events = runner(tmp_path, monkeypatch, plan("酒瓶放柜顶", "酒瓶放灶台"), (outcome,))
    report = controller.run("先放柜顶再放灶台")
    assert report["semantic_success"] is outcome
    assert report["completed_steps"] == 0
    assert len([e for e in events if isinstance(e, tuple) and e[0] == "leaf"]) == 1
    requests = [e for e in events if isinstance(e, dict)]
    assert len(requests) == 2 and requests[1]["failure"]["step"] == 1
    assert report['recoveries'][0]['executed'] is False


def test_model_retry_executes_then_continues_without_replaying_completed_steps(tmp_path, monkeypatch):
    controller, events = runner(tmp_path, monkeypatch, plan('放到柜顶', '放到桌面'),
                                (True, False, True), recovery='retry_grasp')
    report = controller.run('先放柜顶再放桌面')
    assert report['semantic_success'] is True and report['completed_steps'] == 2
    assert [e[1] for e in events if isinstance(e, tuple) and e[0] == 'leaf'] == [0, 1, 2]
    assert report['recoveries'][0]['executed'] is True
    assert ('recover', 'retry_grasp', '放到桌面') in events


def test_recovery_is_bounded_and_never_advances_on_repeated_failure(tmp_path, monkeypatch):
    controller, events = runner(tmp_path, monkeypatch, plan('放到柜顶', '放到桌面'),
                                (False, False, False), recovery='retry_grasp')
    report = controller.run('先放柜顶再放桌面')
    assert report['completed_steps'] == 0
    assert report['recovery_stop']['kind'] == 'no_progress'
    assert len([r for r in report['recoveries'] if r['executed']]) == 2
    assert all(e['number'] == 1 for e in report['steps'])


def test_productive_recovery_is_not_stopped_after_two_attempts(tmp_path, monkeypatch):
    decisions = [dict(action='retry_grasp', reason='位置已改变，重新抓取', progress='progress', step=None)] * 3
    controller, events = runner(tmp_path, monkeypatch, plan('放到柜顶'),
                                (False, False, False, True), decisions=decisions)
    # Actual visual change, not a model statement alone, is also required.
    frames = iter([0, 0, 50, 100, 150])
    controller.capture = lambda: (np.full((8, 8, 3), next(frames), dtype=np.uint8), None)
    report = controller.run('放到柜顶')
    assert report['semantic_success'] is True
    assert len([r for r in report['recoveries'] if r['executed']]) == 3
    feedback = [e['failure'] for e in events if isinstance(e, dict) and 'failure' in e]
    assert len(feedback[-1]['history']) == 2
    assert feedback[-1]['remaining_steps'] == plan('放到柜顶')['steps']
    assert all('before_image' in e for e in events if isinstance(e, dict) and 'failure' in e)


def test_model_claimed_progress_without_scene_change_does_not_loop(tmp_path, monkeypatch):
    decisions = [dict(action='retry_grasp', reason='有进展', progress='progress', step=None)] * 3
    controller, _ = runner(tmp_path, monkeypatch, plan('放到柜顶'), (False, False, False), decisions=decisions)
    report = controller.run('放到柜顶')
    assert report['semantic_success'] is False
    assert report['recovery_stop']['kind'] == 'no_progress'


def test_model_inserts_verified_prerequisite_without_completing_original_goal(tmp_path, monkeypatch):
    value = plan('奶酪放进第三层抽屉', '关闭第三层抽屉')
    value['steps'][0]['destination_kind'] = 'drawer_interior'
    value['steps'][1].update(skill='close_drawer', destination_kind='drawer_interior')
    helper = dict(skill='open_drawer', instruction='打开柜子第三层抽屉', destination_kind='drawer_interior')
    decisions = [dict(action='prepare', reason='抽屉关闭，先打开再继续放置', progress='no_progress', step=helper)]
    controller, events = runner(tmp_path, monkeypatch, value, (False, True, True, True), decisions=decisions)
    report = controller.run('把奶酪放进第三层抽屉，然后关上')
    assert report['semantic_success'] is True and report['completed_steps'] == 2
    assert report['recoveries'][0]['preparation_success'] is True
    assert [e[1] for e in events if isinstance(e, tuple) and e[0] == 'run'] == [
        '奶酪放进第三层抽屉', '打开柜子第三层抽屉', '奶酪放进第三层抽屉', '关闭第三层抽屉']
    assert report['recoveries'][1]['decision']['action'] == 'continue_original'
    assert len([e for e in events if isinstance(e, dict) and 'failure' in e]) == 1


def test_preparation_that_fails_is_not_counted_as_progress(tmp_path, monkeypatch):
    helper = dict(skill='open_drawer', instruction='打开第三层抽屉', destination_kind='drawer_interior')
    decisions = [dict(action='prepare', reason='打开抽屉', progress='progress', step=helper)]
    decisions += [dict(action='retry_drawer', reason='抽屉未打开', progress='progress', step=None)] * 2
    controller, _ = runner(tmp_path, monkeypatch, plan('奶酪放抽屉'), (False, False, False), decisions=decisions)
    report = controller.run('奶酪放抽屉')
    assert report['completed_steps'] == 0 and report['semantic_success'] is False
    assert report['recovery_stop']['kind'] == 'no_progress'


def test_pause_after_recovery_model_prevents_recovery_motion(tmp_path, monkeypatch):
    controller, events = runner(tmp_path, monkeypatch, plan('放到柜顶'), (False,), recovery='retry_grasp')
    request = controller.request
    def paused(port, endpoint, payload):
        result = request(port, endpoint, payload)
        if endpoint == 'recovery':
            controller.stop = lambda: True
        return result
    controller.request = paused
    with pytest.raises(InterruptedError):
        controller.run('放到柜顶')
    assert len([e for e in events if isinstance(e, tuple) and e[0] == 'leaf']) == 1


def test_held_object_recovery_excludes_opening_and_regrasp(tmp_path, monkeypatch):
    controller, events = runner(tmp_path, monkeypatch, plan('放柜顶'), recovery='resume_place')
    controller.report['recoveries'] = []
    controller.obs['robot0_gripper_qpos'] = np.array([.012, -.012])
    entry = controller.recovery_decision('放柜顶', 8911, 1, plan('放柜顶')['steps'][0],
        dict(grasp_context={'source_base': .90}, motion_completed=False), 1)
    assert entry['decision']['action'] == 'resume_place'
    failure = [e for e in events if isinstance(e, dict)][-1]['failure']
    assert failure['possibly_holding_object'] is True
    assert 'retry_grasp' not in failure['allowed_actions']
    assert 'prepare' not in failure['allowed_actions']


def test_last_allowed_helper_success_does_not_complete_parent(tmp_path, monkeypatch):
    helper = dict(skill='open_drawer', instruction='打开第三层抽屉', destination_kind='drawer_interior')
    decisions = [dict(action='retry_grasp', reason='物体位置已改变', progress='progress', step=None)] * 11
    decisions += [dict(action='prepare', reason='先打开抽屉', progress='progress', step=helper)]
    controller, _ = runner(tmp_path, monkeypatch, plan('奶酪放抽屉'),
                           (False,) * 12 + (True,), decisions=decisions)
    frames = iter(range(0, 210, 15))
    controller.capture = lambda: (np.full((8, 8, 3), next(frames), dtype=np.uint8), None)
    report = controller.run('奶酪放抽屉')
    assert report['completed_steps'] == 0 and report['semantic_success'] is False
    assert report['recoveries'][-1]['preparation_success'] is True
    assert report['recovery_stop']['kind'] == 'safety_budget'


def test_time_budget_keeps_failed_original_goal_while_helper_pending(tmp_path, monkeypatch):
    controller, _ = runner(tmp_path, monkeypatch, plan('奶酪放抽屉'))
    original = dict(semantic_success=False, completion={'reason': '原目标未完成'})
    helper = dict(skill='open_drawer', instruction='打开第三层抽屉', destination_kind='drawer_interior')
    controller.recovery_decision = lambda *args: dict(
        decision=dict(action='prepare', reason='先打开', step=helper), made_progress=False)
    controller.execute_leaf = lambda *args: dict(semantic_success=None)
    times = iter([0, 601])
    monkeypatch.setattr('vla.task_runner.time.monotonic', lambda: next(times))
    result = controller.recover_step('奶酪放抽屉', 8911, 1, plan('奶酪放抽屉')['steps'][0], original, '', 0)
    assert result is original
    assert controller.report['recovery_stop']['kind'] == 'safety_budget'


def test_time_budget_prevents_continuation_after_successful_helper(tmp_path, monkeypatch):
    controller, _ = runner(tmp_path, monkeypatch, plan('奶酪放抽屉'))
    original = dict(semantic_success=False)
    helper = dict(skill='open_drawer', instruction='打开第三层抽屉', destination_kind='drawer_interior')
    controller.recovery_decision = lambda *args: dict(
        decision=dict(action='prepare', reason='先打开', step=helper), made_progress=False)
    executed = []
    def execute(*args):
        executed.append(args)
        return dict(semantic_success=True)
    controller.execute_leaf = execute
    times = iter([0, 601])
    monkeypatch.setattr('vla.task_runner.time.monotonic', lambda: next(times))
    result = controller.recover_step('奶酪放抽屉', 8911, 1, plan('奶酪放抽屉')['steps'][0], original, '', 0)
    assert result is original and len(executed) == 1
    assert controller.report['recovery_stop']['kind'] == 'safety_budget'


def test_original_goal_cannot_be_nested_as_its_own_prerequisite(tmp_path, monkeypatch):
    value = plan('奶酪放抽屉')
    decisions = [dict(action='prepare', reason='重复原步骤', progress='no_progress', step=value['steps'][0])]
    controller, _ = runner(tmp_path, monkeypatch, value, (False,), decisions=decisions)
    report = controller.run('奶酪放抽屉')
    assert report['completed_steps'] == 0
    assert '前置步骤不能重复当前目标' in report['recovery_error']
    assert len(report['steps']) == 1


def test_drawer_plan_dispatches_to_drawer_skill(tmp_path, monkeypatch):
    value = plan('打开第三层抽屉', '奶酪放进第三层', '关闭第三层抽屉')
    for step in value['steps']:
        step['destination_kind'] = 'drawer_interior'
    value['steps'][0]['skill'] = 'open_drawer'
    value['steps'][2]['skill'] = 'close_drawer'
    controller, events = runner(tmp_path, monkeypatch, value, (True, True, True))
    report = controller.run('打开第三层放奶酪再关闭')
    calls = [e[2] for e in events if isinstance(e, tuple) and e[0] == 'run']
    assert calls[0]['skill'] == 'open_drawer' and calls[2]['skill'] == 'close_drawer'
    assert report['completed_steps'] == 3


def test_stale_plan_image_never_moves(tmp_path, monkeypatch):
    controller, events = runner(tmp_path, monkeypatch, plan("酒瓶放柜顶"), image_matches=False)
    with pytest.raises(ValueError, match="observation"):
        controller.run("酒瓶放柜顶")
    assert not any(isinstance(e, tuple) for e in events)
    assert controller.report["result"] == "exception"


def test_pause_during_planning_prevents_first_step(tmp_path, monkeypatch):
    controller, events = runner(tmp_path, monkeypatch, plan("酒瓶放柜顶"))
    request = controller.request
    def paused(*args):
        result = request(*args)
        controller.stop = lambda: True
        return result
    controller.request = paused
    with pytest.raises(InterruptedError):
        controller.run("酒瓶放柜顶")
    assert controller.report["result"] == "interrupted"
    assert not any(isinstance(e, tuple) for e in events)


def test_pause_in_step_callback_never_starts_motion(tmp_path, monkeypatch):
    controller, events = runner(tmp_path, monkeypatch, plan("酒瓶放柜顶", "酒瓶放灶台"))
    def pause_after_first(plan, step):
        if step == 1:
            controller.stop = lambda: True
    controller.on_plan = pause_after_first
    with pytest.raises(InterruptedError):
        controller.run("先放柜顶再放灶台")
    assert not any(isinstance(e, tuple) for e in events)
    assert controller.report["completed_steps"] == 0
