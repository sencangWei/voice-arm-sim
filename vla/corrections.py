"""Task-independent human correction records, adapted from the earlier course.

Approval is a human review decision, never an invented LIBERO success score.
Only actual control steps are recorded; GUI idle time is not a sample.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

MAX_FRAMES = 5000
KEYS = {"state", "action", "image", "wrist_image", "post_state", "done"}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def validate_frame(frame):
    if set(frame) != KEYS:
        raise ValueError("纠正帧字段不完整")
    for key, shape in (("state", (8,)), ("action", (7,)), ("post_state", (8,))):
        array = np.asarray(frame[key])
        if array.shape != shape or array.dtype.kind not in "fi" or not np.isfinite(array).all():
            raise ValueError(f"{key} 必须为有限数值，形状 {shape}")
    if np.max(np.abs(frame["action"])) > 1.000001:
        raise ValueError("人工动作须在 [-1,1] 内")
    for key in ("image", "wrist_image"):
        if frame[key].shape != (224, 224, 3) or frame[key].dtype != np.uint8:
            raise ValueError("相机须为 224×224 RGB uint8，沿用模型预处理")
    if np.asarray(frame["done"]).shape != () or np.asarray(frame["done"]).dtype != np.bool_:
        raise ValueError("done 须为布尔标量")


def load_frames(path):
    path = Path(path)
    meta = json.loads((path / "metadata.json").read_text())
    if (meta.get("schema") != "vla-human-v1" or not meta.get("task", "").strip()
            or not meta.get("scenario_key") or meta.get("split") not in ("train", "holdout")
            or meta.get("status") not in ("pending", "draft", "approved")
            or not meta.get("human_actions_only") or meta.get("control_dt") != .05):
        raise ValueError("纠正元数据契约不匹配")
    files = meta["files"]
    if not 0 <= len(files) <= MAX_FRAMES or meta["frame_count"] != len(files):
        raise ValueError("纠正帧数与清单不符")
    frames = []
    for index, item in enumerate(files):
        if item["path"] != f"frame-{index:06d}.npz":
            raise ValueError("纠正帧顺序不连续")
        source = path / item["path"]
        if source.is_symlink() or digest(source) != item["sha256"]:
            raise ValueError("纠正帧校验失败")
        with np.load(source, allow_pickle=False) as data:
            frame = {key: data[key].copy() for key in data.files}
        validate_frame(frame)
        frames.append(frame)
    if meta["status"] == "approved" and (len(frames) < 10 or meta.get("approval") != "manual_user_review"):
        raise ValueError("合格片段缺少人工审核或不足 10 个动作")
    return meta, frames


class CorrectionRecorder:
    def __init__(self, path, metadata):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        self.meta = dict(metadata, schema="vla-human-v1", status="pending", frame_count=0,
                         files=[], approval=None, human_actions_only=True, control_dt=.05,
                         idle_time_excluded=True, official_success_metric=False)
        self.active = True
        self._save()
        load_frames(self.path)

    def _save(self):
        write_json(self.path / "metadata.json", self.meta)

    def append(self, payload, action, post_state, done):
        if not self.active or self.meta["frame_count"] >= MAX_FRAMES:
            raise ValueError("录制已结束或达到 5000 步上限")
        frame = {"state": np.asarray(payload["observation/state"]), "action": np.asarray(action),
                 "image": payload["observation/image"], "wrist_image": payload["observation/wrist_image"],
                 "post_state": np.asarray(post_state), "done": np.asarray(done, dtype=bool)}
        validate_frame(frame)
        name = f"frame-{self.meta['frame_count']:06d}.npz"
        with (self.path / name).open("xb") as stream:
            np.savez_compressed(stream, **frame)
        self.meta["files"].append({"path": name, "sha256": digest(self.path / name)})
        self.meta["frame_count"] += 1
        self._save()

    def finish(self, approved=False):
        if not self.active:
            raise ValueError("片段已经保存")
        if approved and self.meta["frame_count"] < 10:
            raise ValueError("不足 10 个动作，请继续录制或存草稿")
        self.meta.update(status="approved" if approved else "draft",
                         approval="manual_user_review" if approved else None)
        self._save()
        self.active = False
        return self.path
