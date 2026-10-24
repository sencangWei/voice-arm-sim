"""Interface checks only; real model results live in evidence/molmoact2."""
from types import SimpleNamespace

import numpy as np
import pytest

from vla.molmo import validate_observation, vision_inputs_on_device
from vla.run import observation_payload
from vla.workbench import Workbench


def payload():
    return {"observation/image": np.zeros((256, 256, 3), dtype=np.uint8),
            "observation/wrist_image": np.ones((256, 256, 3), dtype=np.uint8),
            "observation/state": np.array([.1, .2, 1., 3.14, 0., 0., .04, -.04]),
            "prompt": "put the wine bottle in the bowl"}


def test_molmo_preserves_raw_state_camera_order_and_prompt():
    obs = payload()
    images, state, text = validate_observation(obs)
    assert images[0] is obs["observation/image"]
    assert images[1] is obs["observation/wrist_image"]
    np.testing.assert_allclose(state, obs["observation/state"])
    assert text == obs["prompt"]


def test_cpu_vision_moves_pixels_and_pooling_indices_together():
    moved = []

    def tensor(name):
        return SimpleNamespace(to=lambda device: moved.append((name, device)) or name)

    result = vision_inputs_on_device(SimpleNamespace(device="cpu"),
                                     (tensor("pixels"), tensor("pooling_indices")))
    assert result == ("pixels", "pooling_indices")
    assert moved == [("pixels", "cpu"), ("pooling_indices", "cpu")]


@pytest.mark.parametrize("key,value", [
    ("observation/image", np.zeros((224, 224, 3), dtype=np.uint8)),
    ("observation/wrist_image", np.zeros((256, 256, 3), dtype=np.float32)),
    ("observation/state", np.zeros(7)),
    ("observation/state", np.full(8, np.nan)),
    ("prompt", " "),
])
def test_molmo_rejects_incompatible_observations(key, value):
    obs = payload()
    obs[key] = value
    with pytest.raises(ValueError):
        validate_observation(obs)


def test_camera_resolution_changes_without_changing_state_or_orientation():
    obs = dict(agentview_image=np.arange(12, dtype=np.uint8).reshape(2, 2, 3),
               robot0_eye_in_hand_image=np.zeros((2, 2, 3), dtype=np.uint8),
               robot0_eef_pos=np.array([1., 2., 3.]),
               robot0_eef_quat=np.array([0., 0., 0., 1.]),
               robot0_gripper_qpos=np.array([.04, -.04]))
    seen = []

    def resize(image, height, width):
        seen.append((height, width))
        return image

    tools = SimpleNamespace(resize_with_pad=resize, convert_to_uint8=lambda x: x)
    result = observation_payload(obs, "instruction", tools, 256)
    assert seen == [(256, 256), (256, 256)]
    np.testing.assert_array_equal(result["observation/image"], obs["agentview_image"][::-1, ::-1])
    np.testing.assert_allclose(result["observation/state"], [1, 2, 3, 0, 0, 0, .04, -.04])


def test_manual_recording_keeps_224_format_for_new_policy():
    lab = Workbench.__new__(Workbench)
    seen = []
    lab._payload = lambda size=None: seen.append(size) or {"observation/state": np.zeros(8)}
    lab.env = SimpleNamespace(step=lambda action: ({}, 0, False, {}))
    recorded = []
    lab.recorder = SimpleNamespace(append=lambda *frame: recorded.append(frame))
    lab.actions, lab.video = [], None
    lab._publish = lambda: None
    lab._step(np.zeros(7), human=True)
    assert seen[0] == 224
    assert len(recorded) == 1


def test_changed_destination_ignores_original_task_success(tmp_path, monkeypatch):
    """An unrelated original predicate must not end a changed-language trial."""
    import sys
    from vla import run

    obs = dict(agentview_image=np.zeros((2, 2, 3), dtype=np.uint8),
               robot0_eye_in_hand_image=np.zeros((2, 2, 3), dtype=np.uint8),
               robot0_eef_pos=np.zeros(3), robot0_eef_quat=np.array([0, 0, 0, 1]),
               robot0_gripper_qpos=np.array([.04, -.04]))
    task = SimpleNamespace(language="bowl on plate", problem_folder="test", bddl_file="test")
    suite = SimpleNamespace(n_tasks=1, get_task=lambda i: task)
    env = SimpleNamespace(seed=lambda seed: None, reset=lambda: None,
                          set_init_state=lambda state: obs,
                          step=lambda action: (obs, 0, True, {}), close=lambda: None)
    modules = {
        "libero.libero": SimpleNamespace(
            benchmark=SimpleNamespace(get_benchmark_dict=lambda: {"libero_goal": lambda: suite}),
            get_libero_path=lambda name: str(tmp_path)),
        "libero.libero.envs": SimpleNamespace(OffScreenRenderEnv=lambda **kw: env),
        "openpi_client": SimpleNamespace(image_tools=SimpleNamespace(
            resize_with_pad=lambda image, h, w: image, convert_to_uint8=lambda image: image)),
        "imageio.v2": SimpleNamespace(get_writer=lambda *a, **kw: SimpleNamespace(
            append_data=lambda frame: None, close=lambda: None)),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, "imageio", SimpleNamespace(v2=modules["imageio.v2"]))
    monkeypatch.setattr(run, "configure_libero", lambda: None)
    monkeypatch.setattr(run, "load_initial_states", lambda task: [np.zeros(5)])
    monkeypatch.setattr(run, "connect_policy", lambda port: SimpleNamespace(
        get_server_metadata=lambda: {"backend": "official-molmoact2-libero-vla"},
        infer=lambda payload: {"actions": np.zeros((10, 7))}))
    checks = []

    def changed_goal(env):
        checks.append(True)
        return len(checks) == 2

    changed_goal.definition = "explicit changed goal"
    args = run.parse_args(["--suite", "libero_goal", "--output", str(tmp_path / "trial"),
                           "--max-steps", "3"])
    result = run.rollout(args, status=lambda text: None, success_check=changed_goal)
    assert result["result"] == "success" and result["steps"] == 2
    assert result["score_definition"] == changed_goal.definition
