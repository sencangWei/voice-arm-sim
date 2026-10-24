"""Contract tests use explicit doubles; they are not model-performance evidence."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np

spec = importlib.util.spec_from_file_location("vla_run", Path(__file__).resolve().parents[1] / "vla/run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


class ImageTools:
    @staticmethod
    def resize_with_pad(image, height, width):
        assert (height, width) == (224, 224)
        return image

    @staticmethod
    def convert_to_uint8(image):
        return image.astype(np.uint8)


def observation():
    return dict(agentview_image=np.arange(12, dtype=np.uint8).reshape(2, 2, 3),
        robot0_eye_in_hand_image=np.arange(12, 24, dtype=np.uint8).reshape(2, 2, 3),
        robot0_eef_pos=np.array([1., 2., 3.]), robot0_eef_quat=np.array([0., 0., 0., 1.]),
        robot0_gripper_qpos=np.array([.1, .2]))


class ContractTests(unittest.TestCase):
    def test_images_state_and_arbitrary_prompt(self):
        obs = observation()
        prompt = "Move the bowl beside the plate; then open the drawer."
        result = run.observation_payload(obs, prompt, ImageTools)
        self.assertEqual(result["prompt"], prompt)
        np.testing.assert_array_equal(result["observation/image"], obs["agentview_image"][::-1, ::-1])
        np.testing.assert_array_equal(result["observation/wrist_image"], obs["robot0_eye_in_hand_image"][::-1, ::-1])
        np.testing.assert_allclose(result["observation/state"], [1, 2, 3, 0, 0, 0, .1, .2])
        np.testing.assert_array_equal(obs["robot0_eef_quat"], [0, 0, 0, 1])

    def test_axis_angle_and_bad_state(self):
        obs = observation()
        obs["robot0_eef_quat"] = np.array([0., 0., 1., 0.])
        self.assertAlmostEqual(run.observation_payload(obs, "x", ImageTools)["observation/state"][5], np.pi)
        obs["robot0_eef_pos"][0] = np.nan
        with self.assertRaises(ValueError):
            run.observation_payload(obs, "x", ImageTools)

    def test_rejects_invalid_action_chunks(self):
        for actions in (np.zeros(7), np.zeros((5, 8)), np.zeros((4, 7)),
                        np.full((5, 7), np.nan), np.full((5, 7), np.inf)):
            with self.subTest(shape=actions.shape), self.assertRaises(ValueError):
                run.validate_actions(actions, 5)
        actions = np.arange(70).reshape(10, 7)
        np.testing.assert_array_equal(run.validate_actions(actions, 5), actions)

    def test_cli_is_bounded_and_raw(self):
        args = run.parse_args(["--output", "new", "--prompt", "arbitrary words"])
        self.assertEqual(args.prompt, "arbitrary words")
        self.assertEqual((args.port, args.max_steps, args.replan_steps), (8766, 220, 5))
        with self.assertRaises(SystemExit):
            run.parse_args(["--output", "new", "--max-steps", "0"])

    def test_rollout_executes_model_prefix_and_records_evidence(self):
        executed, requests, frames = [], [], []
        actions = np.arange(35, dtype=float).reshape(5, 7) / 100
        task = types.SimpleNamespace(language="task predicate language", problem_folder="folder", bddl_file="task.bddl")
        class Suite:
            n_tasks = 1
            def get_task(self, index): return task
            def get_task_init_states(self, index): return [np.array([1., 2., 3.])]
        class Env:
            def __init__(self, **kwargs): pass
            def seed(self, seed): pass
            def reset(self): pass
            def set_init_state(self, state): return observation()
            def step(self, action):
                executed.append(action)
                return observation(), 0, len(executed) == 13, {}
            def close(self): pass
        class Client:
            def get_server_metadata(self): return {"backend": "official-pi05-libero-vla", "checkpoint": "explicit-test-double"}
            def infer(self, payload):
                requests.append(payload)
                return {"actions": actions}
        class Writer:
            def append_data(self, frame): frames.append(frame)
            def close(self): pass
        libero = types.ModuleType("libero.libero")
        libero.benchmark = types.SimpleNamespace(get_benchmark_dict=lambda: {"libero_spatial": Suite})
        libero.get_libero_path = lambda key: "/test"
        envs = types.ModuleType("libero.libero.envs")
        envs.OffScreenRenderEnv = Env
        clientmodule = types.ModuleType("openpi_client")
        clientmodule.image_tools = ImageTools
        imageio = types.ModuleType("imageio.v2")
        imageio.get_writer = lambda *args, **kwargs: Writer()
        imageioparent = types.ModuleType("imageio")
        imageioparent.v2 = imageio
        with tempfile.TemporaryDirectory() as directory:
            args = run.parse_args(["--output", str(Path(directory) / "episode"), "--max-steps", "3", "--prompt", "free raw instruction"])
            with patch.dict(sys.modules, {"libero": types.ModuleType("libero"), "libero.libero": libero,
                    "libero.libero.envs": envs, "openpi_client": clientmodule,
                    "imageio": imageioparent, "imageio.v2": imageio}), \
                    patch.object(run, "configure_libero"), patch.object(run, "load_initial_states", return_value=[np.array([1., 2., 3.])]), patch.object(run, "connect_policy", return_value=Client()):
                result = run.rollout(args, status=lambda text: None)
            self.assertEqual(result["result"], "success")
            self.assertEqual(len(executed), 13)
            np.testing.assert_array_equal(executed[:10], np.tile([0.] * 6 + [-1.], (10, 1)))
            np.testing.assert_array_equal(executed[10:], actions[:3])
            self.assertEqual(requests[0]["prompt"], "free raw instruction")
            report = json.loads((Path(args.output) / "report.json").read_text())
            self.assertEqual(report["server_metadata"]["checkpoint"], "explicit-test-double")
            self.assertEqual(report["steps"], 3)
            self.assertEqual(len(report["initial_state_sha256"]), 64)
            np.testing.assert_array_equal(np.load(Path(args.output) / "actions_0000.npy"), actions)
            np.testing.assert_array_equal(np.load(Path(args.output) / "executed_actions.npy"), actions[:3])
            self.assertEqual(frames[0].shape, (2, 4, 3))
            self.assertEqual(len((Path(args.output) / "inference.jsonl").read_text().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
