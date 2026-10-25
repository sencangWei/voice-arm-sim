"""Model boundary and cancellation tests; physical success needs real rollouts."""
import json
import threading

import numpy as np
import pytest

from vla.semantic import parse_task
from vla.semantic_grasp import VisualSkill, nearby_support_point, region_point


def task():
    return dict(source=dict(label="bottle", bbox_2d=[100, 100, 200, 200]),
                destination=dict(label="bowl", bbox_2d=[500, 500, 700, 700]),
                relation="in", english="Put the bottle into the bowl.")


@pytest.mark.parametrize('bad_response', [None, 'stale_image', 'wrong_relation'])
def test_missing_destination_reparses_unchanged_goal_on_current_image(tmp_path, bad_response):
    import hashlib
    from vla.semantic_grasp import image_bytes
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    skill = VisualSkill(None, {}, None, tmp_path / 'relocalize')
    skill.report['chinese'] = '把瓶子放进碗里'
    calls = []
    def request(port, endpoint, payload, **kwargs):
        calls.append(endpoint)
        if endpoint == 'locate':
            return {'bbox_2d': [], 'label': None}
        assert payload['text'] == '把瓶子放进碗里'
        fresh = task()
        fresh['destination']['bbox_2d'] = [450, 450, 650, 650]
        if bad_response == 'wrong_relation':
            fresh['relation'] = 'on'
        return {'task': fresh, 'image_sha256': 'stale' if bad_response == 'stale_image'
                else hashlib.sha256(image_bytes(rgb)).hexdigest()}
    skill.request = request
    if bad_response:
        with pytest.raises(ValueError):
            skill.relocalize_destination(task(), rgb, 8911)
    else:
        location = skill.relocalize_destination(task(), rgb, 8911)
        assert location['bbox_2d'] == [450, 450, 650, 650]
        assert (skill.output / 'relocalize-task.json').exists()
    assert calls == ['locate', 'task'] and not skill.trace


def test_visible_destination_does_not_trigger_task_reparse(tmp_path):
    skill = VisualSkill(None, {}, None, tmp_path / 'visible')
    def request(port, endpoint, payload, **kwargs):
        assert endpoint == 'locate'
        return {'bbox_2d': [400, 400, 600, 600], 'label': 'bowl'}
    skill.request = request
    assert skill.relocalize_destination(task(), np.zeros((8, 8, 3), dtype=np.uint8), 8911)[
        'bbox_2d'] == [400, 400, 600, 600]


def test_leaf_relation_disagreement_with_plan_never_reaches_geometry_or_motion(tmp_path):
    import hashlib
    from vla.semantic_grasp import image_bytes
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    skill = VisualSkill(None, {"agentview_image": rgb}, None, tmp_path / "mismatch")
    skill.policy_metadata = lambda: {"backend": "visual-geometric-control"}
    skill.capture = lambda: (rgb, None)
    skill.request = lambda *a: {"task": task(), "image_sha256": hashlib.sha256(image_bytes(rgb)).hexdigest()}
    with pytest.raises(ValueError, match="目的地类型"):
        skill.run("把瓶子放到柜顶", destination_kind="support_surface")
    assert not skill.trace


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


def test_source_component_cannot_extend_into_touching_support():
    y, x = np.mgrid[:80, :100]
    cloud = np.stack((x*.001, y*.001, np.full(x.shape, .901)), axis=-1)
    cloud[10:30, 10:80, 2] = .95  # object and touching support form one component
    target, mask, _ = region_point(cloud, [100, 125, 300, 375])
    assert .01 <= target[0] <= .03
    assert not mask[:, 32:].any()


def test_on_destination_uses_horizontal_top_instead_of_vertical_cabinet_body():
    y, x = np.mgrid[:100, :100]
    cloud = np.stack((x*.002, y*.002, np.full(x.shape, .901)), axis=-1)
    cloud[10:35, 30:80, 2] = 1.127
    cloud[35:90, 30:80, 0] = .16  # vertical side, varying height
    cloud[35:90, 30:80, 2] = np.linspace(1.127, .92, 55)[:, None]
    target, _, metadata = region_point(cloud, [300, 100, 800, 900], allow_table=True)
    assert target[2] == pytest.approx(1.127)
    assert "horizontal support plane" in metadata["method"]


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


def test_large_support_landing_uses_nearby_interior_and_keeps_edge_margin():
    y, x = np.mgrid[:101, :101]
    cloud = np.stack((x*.002, y*.002, np.full(x.shape, 1.127)), axis=-1)
    target = nearby_support_point(cloud, np.ones(x.shape, dtype=bool), [-.2, .1])
    assert .060 <= target[0] <= .062
    assert target[1] == pytest.approx(.1)
    assert target[2] == pytest.approx(1.127)
    with pytest.raises(ValueError, match="边缘余量"):
        nearby_support_point(cloud, np.ones(x.shape, dtype=bool), [0., 0.], margin=.11)


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


@pytest.mark.parametrize("widths,accepted", [
    (np.linspace(.04, .001, 20), False),  # empty fingers briefly cross plausible widths
    ([.04] * 19 + [.001], False),       # object slips at end of check
    ([.04, .044] * 10, False),          # unstable obstruction
    ([.04] * 20, True),
])
def test_hold_requires_settled_width_not_transient_closure(tmp_path, widths, accepted):
    obs = {"robot0_eef_pos": np.array([0., 0., 1.1]),
           "robot0_gripper_qpos": np.array([.02, -.02])}
    skill = VisualSkill(None, obs, None, tmp_path / "trial")
    sequence = iter(widths)
    def step(action, phase):
        np.testing.assert_array_equal(action, [0.] * 6 + [1.])
        gap = next(sequence)
        obs["robot0_gripper_qpos"] = np.array([gap/2, -gap/2])
    skill.step = step
    if accepted:
        skill.confirm_hold()
    else:
        with pytest.raises(RuntimeError, match="稳定持有"):
            skill.confirm_hold()
    assert skill.report["hold_check"]["accepted"] is accepted
    assert len(skill.report["hold_check"]["gaps"]) == 20


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
    monkeypatch.setattr("vla.semantic_grasp.circular_container_point",
        lambda *a: (np.array([0., 0., 1.]), {}))
    skill = VisualSkill(None, obs, Client(), tmp_path / "panel", advance=lambda a: obs)
    skill.capture = lambda: (rgb, None)
    skill.request = lambda *a: {"task": task(), "image_sha256": hashlib.sha256(image_bytes(rgb)).hexdigest()}
    skill.stop = lambda: len(skill.trace) >= 1
    with pytest.raises(InterruptedError):
        skill.run("把瓶子放进碗里")
    assert frames == [(256, 256, 3), (256, 256, 3)]


@pytest.mark.parametrize("verdict", ["completed", "not_completed", "uncertain"])
def test_completion_requires_explicit_three_way_visual_judgment(verdict):
    from vla.semantic import parse_completion
    value = {"verdict": verdict, "reason": "当前图像的可见证据"}
    assert parse_completion(json.dumps(value)) == value


@pytest.mark.parametrize("value", [{"verdict": True, "reason": "ok"},
    {"verdict": "completed"}, {"verdict": "completed", "reason": ""},
    {"verdict": "success", "reason": "ok"}, None])
def test_malformed_completion_never_counts_as_success(value):
    from vla.semantic import parse_completion
    with pytest.raises(ValueError):
        parse_completion(json.dumps(value))


@pytest.mark.parametrize("verdict,expected", [("completed", True), ("not_completed", False), ("uncertain", None)])
def test_final_verification_uses_current_two_cameras_without_actions(tmp_path, verdict, expected):
    import hashlib
    from vla.semantic_grasp import image_bytes
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    wrist = np.full((16, 16, 3), 123, dtype=np.uint8)
    skill = VisualSkill(None, {"robot0_eye_in_hand_image": wrist,
        "robot0_gripper_qpos": np.array([.04, -.04])}, None, tmp_path / "trial")
    skill.capture = lambda: (rgb, None)
    def request(port, endpoint, payload):
        assert endpoint == "verify"
        assert payload["text"].startswith("把瓶子放进碗里。图左是当前全景")
        assert len(payload["images"]) == 2
        combined = np.concatenate([rgb, np.full_like(rgb, 123)], axis=1)
        return {"judgment": {"verdict": verdict, "reason": "visual evidence"},
            "image_sha256": [hashlib.sha256(image_bytes(image)).hexdigest()
                for image in (combined, np.ascontiguousarray(wrist[::-1, ::-1]))]}
    skill.request = request
    skill.verify_completion("把瓶子放进碗里", 8912)
    assert skill.report["semantic_success"] is expected
    assert not skill.trace


def test_closed_gripper_and_stale_images_cannot_be_completed(tmp_path):
    skill = VisualSkill(None, {"robot0_gripper_qpos": np.array([.02, -.02])}, None, tmp_path / "closed")
    skill.verify_completion("put it down", 8912)
    assert skill.report["semantic_success"] is False
    skill.obs = {"robot0_gripper_qpos": np.array([.04, -.04]),
                 "robot0_eye_in_hand_image": np.zeros((8, 8, 3), dtype=np.uint8)}
    skill.capture = lambda: (np.zeros((8, 8, 3), dtype=np.uint8), None)
    skill.request = lambda *a: {"judgment": {"verdict": "completed", "reason": "ok"},
                                "image_sha256": ["stale", "stale"]}
    with pytest.raises(ValueError, match="image"):
        skill.verify_completion("put it down", 8912)
    assert skill.report["semantic_success"] is not True


def test_source_on_elevated_support_keeps_bottle_neck():
    from vla.semantic_grasp import region_point
    y, x = np.mgrid[:100, :100]
    cloud = np.stack((x * .002, y * .002, np.full_like(x, .90, dtype=float)), axis=-1)
    cloud[20:70, 40:60, 2] = 1.28
    point, mask, _ = region_point(cloud, [400, 200, 600, 700])
    assert point[2] == pytest.approx(1.28)
    assert mask[30, 50]
    assert not mask[0, 0]


def test_support_landing_accepts_visible_raised_center_with_same_edge_margin():
    from vla.semantic_grasp import nearby_support_point
    y, x = np.mgrid[:101, :101]
    cloud = np.stack((x*.002, y*.002, np.full_like(x, .93, dtype=float)), axis=-1)
    center = (x-50)**2 + (y-50)**2 < 25**2
    cloud[center, 2] += .004
    target = nearby_support_point(cloud, ~center, [.1, .1])
    assert np.allclose(target, [.1, .1, .934])
