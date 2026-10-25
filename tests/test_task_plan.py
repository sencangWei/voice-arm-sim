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


def runner(tmp_path, monkeypatch, value, outcomes=(True, True), image_matches=True):
    events = []
    controller = TaskRunner(object(), {"version": 0}, None, tmp_path / "run",
                            stop=lambda: False, status=lambda message: events.append(message))
    controller.capture = lambda: (np.zeros((8, 8, 3), dtype=np.uint8), None)
    def request(port, endpoint, payload):
        import base64
        assert endpoint == "plan"
        events.append(dict(payload))
        return dict(plan=value, image_sha256=hashlib.sha256(base64.b64decode(payload["image"])).hexdigest()
                    if image_matches else "stale")
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
    monkeypatch.setattr("vla.task_runner.VisualControl", Leaf)
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
    assert report["failure_feedback"]["executed"] is False


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
