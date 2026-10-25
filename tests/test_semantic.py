"""Model boundary and cancellation tests; physical success needs real rollouts."""
import json
import threading

import numpy as np
import pytest

from vla.semantic import parse_task
from vla.semantic_grasp import VisualSkill, region_point


def task():
    return dict(source=dict(label="bottle", bbox_2d=[100, 100, 200, 200]),
                destination=dict(label="bowl", bbox_2d=[500, 500, 700, 700]),
                relation="in", english="Put the bottle into the bowl.")


def test_task_accepts_model_chosen_objects_without_a_vocabulary_list():
    value = task()
    value["source"]["label"] = "unfamiliar blue toy"
    value["destination"]["label"] = "red container"
    assert parse_task("```json\n" + json.dumps(value) + "\n```") == value


@pytest.mark.parametrize("box", [[0, 0, 0, 20], [-1, 0, 10, 20],
    [0, 0, 1001, 20], [True, 0, 20, 30], [0, 0, float("nan"), 30], [1, 2]])
def test_bad_regions_never_reach_motion(box):
    value = task()
    value["source"]["bbox_2d"] = box
    with pytest.raises(ValueError):
        parse_task(json.dumps(value))


@pytest.mark.parametrize("value", [None, [], {"actions": [1, 2]}, dict(task(), relation="under"),
    dict(task(), english=""), dict(task(), destination=task()["source"])])
def test_unsupported_or_ambiguous_outputs_fail_closed(value):
    with pytest.raises(ValueError):
        parse_task(json.dumps(value))


def test_depth_refines_current_model_region_and_supports_clear_table():
    y, x = np.mgrid[:60, :80]
    cloud = np.stack((x*.001, y*.001, np.full(x.shape, .901)), axis=-1)
    cloud[10:20, 10:20, 2] = .94
    cloud[30:40, 50:60, 2] = .96
    a, _, _ = region_point(cloud, [100, 100, 280, 380])
    b, _, _ = region_point(cloud, [600, 450, 800, 720])
    assert b[0] > a[0] and b[2] > a[2]
    table, _, metadata = region_point(cloud, [350, 700, 550, 900], allow_table=True)
    assert table[2] == pytest.approx(.901)
    assert "planar table" in metadata["method"]
    with pytest.raises(ValueError):
        region_point(cloud, [350, 700, 550, 900])


def test_pause_during_slow_model_request_does_not_wait_for_inference(tmp_path, monkeypatch):
    entered, release, stopped = threading.Event(), threading.Event(), threading.Event()
    def slow_request(*args):
        entered.set()
        release.wait(2)
        return {"old": "response"}
    def heartbeat():
        if entered.is_set():
            stopped.set()
    monkeypatch.setattr("vla.semantic_grasp.post", slow_request)
    skill = VisualSkill(None, None, None, tmp_path / "trial", stop=stopped.is_set,
                        heartbeat=heartbeat)
    try:
        with pytest.raises(InterruptedError):
            skill.request(8912, "task", {})
        assert entered.is_set() and not skill.trace
    finally:
        release.set()


def test_paused_skill_never_executes_a_queued_action(tmp_path):
    advanced = []
    skill = VisualSkill(None, None, None, tmp_path / "trial", stop=lambda: True,
                        advance=lambda a: advanced.append(a))
    with pytest.raises(InterruptedError):
        skill.step([0.] * 7, "pending")
    assert not advanced


def test_existing_outputs_are_never_overwritten(tmp_path):
    with pytest.raises(FileExistsError):
        VisualSkill(None, None, None, tmp_path)


def test_panel_observation_size_stays_constant_in_video(tmp_path, monkeypatch):
    import hashlib
    from vla.semantic_grasp import image_bytes
    rgb = np.zeros((512, 512, 3), dtype=np.uint8)
    obs = {"agentview_image": np.zeros((256, 256, 3), dtype=np.uint8),
           "robot0_eye_in_hand_image": np.zeros((256, 256, 3), dtype=np.uint8),
           "robot0_eef_quat": np.array([0., 0., 0., 1.]),
           "robot0_eef_pos": np.array([0., 0., 1.1]),
           "robot0_gripper_qpos": np.array([.04, -.04])}
    class Client:
        def get_server_metadata(self):
            return {"backend": "local-pi05-libero-lora"}
        def infer(self, payload):
            return {"actions": np.zeros((5, 7))}
    frames = []
    class Writer:
        def append_data(self, frame):
            frames.append(frame.shape)
        def close(self):
            pass
    monkeypatch.setattr("imageio.v2.get_writer", lambda *a, **kw: Writer())
    masks = iter([np.array([[True, False]]), np.array([[False, True]])])
    monkeypatch.setattr("vla.semantic_grasp.region_point",
        lambda *a: (np.array([0., 0., 1.]), next(masks), {"tabletop": .9}))
    skill = VisualSkill(None, obs, Client(), tmp_path / "panel", advance=lambda a: obs)
    skill.capture = lambda: (rgb, None)
    skill.request = lambda *a: {"task": task(), "image_sha256": hashlib.sha256(image_bytes(rgb)).hexdigest()}
    skill.stop = lambda: len(skill.trace) >= 1
    with pytest.raises(InterruptedError):
        skill.run("把瓶子放进碗里")
    assert frames == [(256, 256, 3), (256, 256, 3)]
