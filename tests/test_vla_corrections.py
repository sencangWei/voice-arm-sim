import json

import numpy as np
import pytest

from vla.corrections import CorrectionRecorder, load_frames
from vla.train_corrections import CorrectionDataset, prepare_dataset, validate_manifest


def clip(root, name, split="train", scenario="a", approved=True, action=.2):
    recorder = CorrectionRecorder(root / f"human_{name}", dict(task="put bowl on plate", split=split,
        scenario_key=scenario, source_scene="scene/test"))
    for step in range(10):
        recorder.append({"observation/state": np.full(8, step, dtype=float),
            "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8)},
            np.full(7, action), np.full(8, step+1, dtype=float), False)
    recorder.finish(approved)
    return recorder.path


def test_frame_alignment_and_tamper(tmp_path):
    path = clip(tmp_path, "one")
    meta, frames = load_frames(path)
    assert meta["frame_count"] == 10
    assert meta["official_success_metric"] is False
    np.testing.assert_array_equal(frames[3]["state"] + 1, frames[3]["post_state"])
    (path / "frame-000003.npz").write_bytes(b"modified")
    with pytest.raises(ValueError, match="校验"):
        load_frames(path)


def test_split_guard_and_drafts(tmp_path):
    clip(tmp_path, "a", scenario="same")
    clip(tmp_path, "b", scenario="b")
    clip(tmp_path, "h", split="holdout", scenario="same")
    with pytest.raises(ValueError, match="同时"):
        prepare_dataset(tmp_path, tmp_path / "manifest")


def test_windows_never_include_holdout_or_cross_clips(tmp_path):
    clip(tmp_path, "a", scenario="a", action=.1)
    clip(tmp_path, "b", scenario="b", action=.2)
    clip(tmp_path, "h", split="holdout", scenario="h", action=.9)
    clip(tmp_path, "draft", approved=False, action=-.9)
    path = prepare_dataset(tmp_path, tmp_path / "prepared")
    manifest = validate_manifest(path)
    assert manifest["train_frames"] == 20 and manifest["holdout_frames"] == 10
    dataset = CorrectionDataset(manifest)
    assert len(dataset) == 20
    np.testing.assert_allclose(dataset[9]["actions"], .1)
    np.testing.assert_allclose(dataset[10]["actions"], .2)
    metadata = tmp_path / "human_a" / "metadata.json"
    value = json.loads(metadata.read_text())
    value["task"] = "different task"
    metadata.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="已变化"):
        validate_manifest(path)


def test_empty_data_not_ready(tmp_path):
    with pytest.raises(ValueError, match="数据不足"):
        prepare_dataset(tmp_path, tmp_path / "prepared")
