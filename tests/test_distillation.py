import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from vla.teacher_data import TASKS, digest
from vla.train_distillation import DistillationDataset, freeze, load_episode


class DistillationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.names = []
        for task_name, task in TASKS.items():
            for initial in (0, 1, 8):
                name = f"{task_name}-{initial}"
                self.names.append(name)
                path = self.root / name
                path.mkdir()
                n = 12
                np.savez_compressed(path / "frames.npz", state=np.full((n, 8), initial, dtype=np.float32),
                    action=np.repeat(np.arange(n, dtype=np.float32)[:, None], 7, axis=1),
                    image=np.zeros((n, 224, 224, 3), dtype=np.uint8),
                    wrist_image=np.zeros((n, 224, 224, 3), dtype=np.uint8))
                meta = dict(schema="simulation-teacher-v1", success=True, human_actions_only=False,
                    training_demonstration=True, task_name=task_name, task=task, initial_state_id=initial,
                    initial_state_sha256=f"scene-{initial}", frames=n, frames_sha256=digest(path / "frames.npz"))
                (path / "metadata.json").write_text(json.dumps(meta))
        self.review = self.root / "review.json"
        self.review.write_text(json.dumps(dict(reviewer="agent_visual_review", accepted=self.names)))

    def test_balance_holdout_exclusion_and_terminal_window(self):
        manifest = freeze(self.root, self.review, self.root / "manifest.json")
        data = DistillationDataset(manifest)
        counts = {task: 0 for task in TASKS.values()}
        for i in range(len(data)):
            sample = data[i]
            counts[sample["prompt"]] += 1
            self.assertLess(sample["state"][0], 8)
            actions = sample["actions"][:, 0]
            self.assertTrue(np.all(np.diff(actions) >= 0))
            self.assertLessEqual(actions[-1], 11)
        self.assertEqual(len(set(counts.values())), 1)
        np.testing.assert_array_equal(data[3 * 11]["actions"][:, 0], np.full(10, 11))

    def test_corruption_and_human_relabeling_rejected(self):
        path = self.root / self.names[0]
        meta_path = path / "metadata.json"
        meta = json.loads(meta_path.read_text())
        meta["human_actions_only"] = True
        meta_path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "provenance"):
            load_episode(path)
        meta["human_actions_only"] = False
        meta["frames_sha256"] = "corrupted"
        meta_path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "content changed"):
            load_episode(path)

    def test_review_and_each_task_holdout_required(self):
        self.review.write_text(json.dumps(dict(reviewer="agent_visual_review", accepted=self.names[:-1])))
        with self.assertRaisesRegex(ValueError, "Insufficient holdout"):
            freeze(self.root, self.review, self.root / "bad.json")

    def test_terminal_settle_trim_keeps_raw_and_windows_inside_boundary(self):
        path = self.root / self.names[0]
        trace = [{"phase": "retreat" if i < 7 else "settle"} for i in range(12)]
        (path / "trace.json").write_text(json.dumps(trace))
        manifest = freeze(self.root, self.review, self.root / "trim.json")
        self.assertEqual(manifest["clips"][0]["training_stop"], 10)
        data = DistillationDataset(manifest)
        np.testing.assert_array_equal(data[3 * 9]["actions"][:, 0], np.full(10, 9))
        self.assertEqual(load_episode(path)[0]["frames"], 12)
        (path / "trace.json").write_text("[]")
        with self.assertRaisesRegex(ValueError, "trace changed"):
            DistillationDataset(manifest)


def test_chinese_evaluation_executes_translated_request(tmp_path, monkeypatch):
    from vla import evaluate_distillation as evaluation

    requests, executed = [], []

    def translate(text):
        requests.append(text)
        return "Please put the wine bottle into the bowl."

    def rollout(args, **kwargs):
        Path(args.output).mkdir()
        executed.append((args.prompt, args.replan_steps))
        return {"result": "failure", "steps": 360}

    monkeypatch.setattr(evaluation, "translate_prompt", translate)
    monkeypatch.setattr(evaluation, "rollout", rollout)
    output = tmp_path / "chinese"
    result = evaluation.evaluate(output, [0, 1], tasks=["bottle-in-bowl"], chinese=True, replan_steps=1)
    assert requests == ["帮我把酒瓶放进碗里"]
    assert executed == [("Please put the wine bottle into the bowl.", 1)] * 2
    assert result[0]["replan_steps"] == 1
    assert result[0]["user_instruction"] == requests[0]
    assert json.loads((output / "instructions.json").read_text())["bottle-in-bowl"]["model_prompt"] == executed[0][0]


if __name__ == "__main__":
    unittest.main()
