import collections
import queue
import threading
from types import SimpleNamespace

import pytest

from vla.workbench import Workbench


def bare_workbench():
    lab = Workbench.__new__(Workbench)
    lab.env = object()
    lab.viewer = lab.video = lab.client = lab.recorder = None
    lab.output = None
    lab.running = False
    lab.done = False
    lab.plan = collections.deque([[0.] * 7])
    lab.interrupt = threading.Event()
    lab.lock = threading.Lock()
    lab.events = queue.Queue()
    lab.snapshot = {}
    lab.actions = []
    lab.prompt = "old instruction"
    lab.scene = {"task": lab.prompt}
    return lab


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
