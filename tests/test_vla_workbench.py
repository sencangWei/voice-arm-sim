import collections
import queue
import threading
from types import SimpleNamespace

import pytest
from websockets.protocol import State

from vla.workbench import Workbench
from vla.run import LIMITS, parse_args


def bare_workbench():
    lab = Workbench.__new__(Workbench)
    lab.args = SimpleNamespace(max_steps=0, replan_steps=5)
    lab.env = object()
    lab.viewer = lab.video = lab.client = lab.recorder = None
    lab.output = None
    lab.running = False
    lab.semantic_active = False
    lab.done = False
    lab.plan = collections.deque([[0.] * 7])
    lab.interrupt = threading.Event()
    lab.lock = threading.Lock()
    lab.events = queue.Queue()
    lab.snapshot = {}
    lab.actions = []
    lab.prompt = "old instruction"
    lab.scene = {"task": lab.prompt}
    lab.client = SimpleNamespace(_ws=SimpleNamespace(state=State.OPEN))
    return lab


def semantic_workbench(tmp_path):
    lab = bare_workbench()
    lab.output = tmp_path
    lab.obs = object()
    lab.calls = 0
    lab.quit = threading.Event()
    lab.scene["source_model"] = {"backend": "local-pi05-libero-lora"}
    lab._publish = lambda: None
    lab._save_scene = lambda reason: lab.scene.update(stop_reason=reason)
    return lab


def test_visual_command_uses_same_scene_and_updates_correction_instruction(tmp_path, monkeypatch):
    lab = semantic_workbench(tmp_path)
    env, initial_obs = lab.env, lab.obs
    refreshed = []
    lab.viewer = SimpleNamespace(sync=lambda: refreshed.append(True), is_running=lambda: True)
    lab._step = lambda action: lab.actions.append(action)
    monkeypatch.setattr("vla.workbench.time.sleep", lambda seconds: None)
    model_task = dict(english="Put the bottle on the plate.")
    calls = []
    class Skill:
        def __init__(self, scene, obs, client, output, **callbacks):
            assert scene is env and obs is initial_obs
            self.callbacks = callbacks
            self.report = {"inference_calls": 3}
        def run(self, text):
            assert lab.semantic_active and not lab.running
            assert not self.callbacks["stop"]()
            self.callbacks["on_task"](model_task)
            assert self.callbacks["advance"]([0.] * 7) is initial_obs
            calls.append(text)
    monkeypatch.setattr("vla.semantic_grasp.VisualSkill", Skill)
    lab.interrupt.set()  # send() interrupts the preceding mode.
    lab._command("semantic", ("把酒瓶放到盘子上",))
    assert calls == ["把酒瓶放到盘子上"] and lab.env is env
    assert lab.prompt == model_task["english"] and not lab.plan
    assert lab.scene["user_instruction"] == calls[0]
    assert lab.scene["control_modes"][-1]["semantic_success"] is None
    assert lab.calls == 3 and not lab.semantic_active and not lab.running
    assert refreshed == [True] and len(lab.actions) == 1


def test_visual_command_clears_busy_after_constructor_failure(tmp_path, monkeypatch):
    lab = semantic_workbench(tmp_path)
    def fail(*args, **kwargs):
        raise OSError("output unavailable")
    monkeypatch.setattr("vla.semantic_grasp.VisualSkill", fail)
    with pytest.raises(OSError, match="output unavailable"):
        lab._command("semantic", ("把酒瓶放到盘子上",))
    assert not lab.semantic_active and not lab.running
    assert lab.scene["stop_reason"] == "visual_semantic_finished"


def test_visual_command_does_not_discard_pause_arriving_during_reconnect(tmp_path, monkeypatch):
    lab = semantic_workbench(tmp_path)
    lab._ensure_policy = lab.interrupt.set
    class Skill:
        def __init__(self, *args, **callbacks):
            self.stop = callbacks["stop"]
            self.report = {"inference_calls": 0}
        def run(self, text):
            assert self.stop()
            raise InterruptedError("paused")
    monkeypatch.setattr("vla.semantic_grasp.VisualSkill", Skill)
    with pytest.raises(InterruptedError):
        lab._command("semantic", ("把酒瓶放到盘子上",))
    assert not lab.semantic_active and not lab.actions


def test_visual_command_rejects_unsaved_manual_recording(tmp_path):
    lab = semantic_workbench(tmp_path)
    lab.recorder = object()
    with pytest.raises(ValueError, match="保存人工片段"):
        lab._command("semantic", ("把酒瓶放到盘子上",))
    assert lab.recorder and not lab.semantic_active


def test_start_uses_latest_instruction_and_discards_old_plan():
    lab = bare_workbench()
    lab._command("run", ("new instruction",))
    assert lab.prompt == "new instruction"
    assert lab.running and not lab.plan
    assert lab.scene["instruction_changes"] == [
        {"control_step": 0, "previous": "old instruction", "prompt": "new instruction"}]
    with pytest.raises(ValueError, match="不能为空"):
        lab._command("run", ("  ",))


def test_cleanup_closes_remaining_resources_after_one_failure():
    lab = bare_workbench()
    closed = []

    def broken_close():
        raise RuntimeError("viewer failed to close")

    lab.viewer = SimpleNamespace(close=broken_close)
    lab.env = SimpleNamespace(close=lambda: closed.append("env"))
    lab.client = SimpleNamespace(_ws=SimpleNamespace(close=lambda: closed.append("client")))
    lab._close_scene()
    assert closed == ["env", "client"]
    assert lab.env is lab.client is lab.viewer is None
    assert "viewer failed to close" in lab.events.get_nowait()


def test_chinese_instruction_keeps_original_and_sends_translated_prompt(monkeypatch):
    lab = bare_workbench()
    monkeypatch.setattr("vla.workbench.translate_prompt", lambda text: "put the bowl on the plate")
    lab._command("run", ("帮我把碗放到盘子上",))
    assert lab.prompt == "put the bowl on the plate"
    assert lab.scene["user_instruction"] == "帮我把碗放到盘子上"
    assert lab.running and not lab.plan


def test_translation_failure_does_not_start_old_instruction(monkeypatch):
    lab = bare_workbench()
    def fail(text):
        raise ValueError("中文指令处理失败")
    monkeypatch.setattr("vla.workbench.translate_prompt", fail)
    with pytest.raises(ValueError, match="处理失败"):
        lab._command("run", ("拿起碗",))
    assert not lab.running and lab.prompt == "old instruction"


def test_resume_reconnects_closed_policy_without_resetting_scene(monkeypatch):
    lab = bare_workbench()
    closed = []
    lab.client._ws = SimpleNamespace(state=State.CLOSED, close=lambda: closed.append("old"))
    model = {"backend": "official-molmoact2-libero-vla", "checkpoint": "same-model"}
    lab.scene.update(source_model=model, policy_port=8777)
    replacement = SimpleNamespace(get_server_metadata=lambda: model)
    ports = []
    monkeypatch.setattr("vla.workbench.connect_policy", lambda port: ports.append(port) or replacement)
    env = lab.env
    lab._command("run", ("new instruction",))
    assert ports == [8777] and closed == ["old"]
    assert lab.client is replacement and lab.env is env and lab.running
    assert not lab.plan


def test_resume_rejects_changed_model_and_stays_paused(monkeypatch):
    lab = bare_workbench()
    closed = []
    lab.client = None
    lab.scene.update(source_model={"checkpoint": "original"}, policy_port=8777)
    replacement = SimpleNamespace(get_server_metadata=lambda: {"checkpoint": "different"},
        _ws=SimpleNamespace(close=lambda: closed.append("replacement")))
    monkeypatch.setattr("vla.workbench.connect_policy", lambda port: replacement)
    with pytest.raises(ValueError, match="模型.*变化"):
        lab._command("run", ("new instruction",))
    assert lab.client is None and not lab.running and not lab.plan
    assert closed == ["replacement"]


def test_failed_reconnect_stays_paused_and_discards_queued_actions(monkeypatch):
    lab = bare_workbench()
    lab.client = None
    lab.scene.update(source_model={"checkpoint": "original"}, policy_port=8777)
    def unavailable(port):
        raise ConnectionRefusedError("service unavailable")
    monkeypatch.setattr("vla.workbench.connect_policy", unavailable)
    with pytest.raises(ConnectionRefusedError):
        lab._command("run", ("new instruction",))
    assert lab.client is None and not lab.running and not lab.plan
    assert lab.prompt == "old instruction"


def test_changed_destination_can_run_after_original_task_predicate():
    lab = bare_workbench()
    lab.done = True
    lab.scene["task_language"] = "put the bowl on the plate"
    lab._command("run", ("put the wine bottle in the bowl",))
    assert lab.running
    assert not lab._goal_satisfied()


def test_only_matching_official_instruction_uses_scene_success():
    lab = bare_workbench()
    lab.scene["task_language"] = "put the bowl on the plate"
    lab.done = True
    lab.prompt = "put the bowl on the plate"
    assert lab._goal_satisfied()
    lab.prompt = "put the wine bottle in the bowl"
    assert not lab._goal_satisfied()


def test_panel_defaults_to_unlimited_but_evaluation_remains_bounded():
    assert parse_args(["--panel", "--output", "unused"]).max_steps == 0
    args = parse_args(["--output", "unused"])
    assert args.max_steps == LIMITS[args.suite]
    with pytest.raises(SystemExit):
        parse_args(["--output", "unused", "--max-steps", "0"])


@pytest.mark.parametrize("limit,expected_running", [(0, True), (360, False)])
def test_interactive_limit_is_optional_and_resume_preserves_scene(limit, expected_running):
    lab = bare_workbench()
    lab.args = SimpleNamespace(max_steps=360, replan_steps=5)
    lab._command("run", ("new instruction", limit))
    env = lab.env
    lab.auto_steps = 359
    lab.plan.append([0.] * 7)
    lab._step = lambda action: lab.actions.append(action)
    lab._save_scene = lambda reason: lab.scene.update(stop_reason=reason)
    lab._infer_step()
    assert lab.running is expected_running
    assert lab.scene.get("stop_reason") == (None if expected_running else "step_limit")
    assert len(lab.actions) == 1
    lab._command("pause", ())
    assert not lab.running
    lab._command("run", ("new instruction", 0))
    assert lab.env is env and len(lab.actions) == 1 and lab.running


def test_unlimited_run_still_pauses_when_official_goal_is_satisfied():
    lab = bare_workbench()
    lab.args = SimpleNamespace(max_steps=0, replan_steps=5)
    lab.scene["task_language"] = lab.prompt
    lab._command("run", (lab.prompt, 0))
    lab.plan.append([0.] * 7)
    lab._step = lambda action: setattr(lab, "done", True)
    lab._save_scene = lambda reason: lab.scene.update(stop_reason=reason)
    lab._infer_step()
    assert not lab.running and lab.scene["stop_reason"] == "predicate_satisfied"
