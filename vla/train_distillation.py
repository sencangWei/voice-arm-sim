"""LoRA distillation of explicitly synthetic, matched-scene demonstrations.

Inference stays in vla.serve: images + state + language -> pi05 actions. This
module never makes the simulator-geometry teacher available to that server.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from vla.teacher_data import TASKS, digest
from vla.train_corrections import ASSET, NAME, ROOT, SOURCE, REVISION, make_config


def load_episode(path):
    path = Path(path)
    meta = json.loads((path / "metadata.json").read_text())
    if (meta.get("schema") != "simulation-teacher-v1" or not meta.get("success")
            or meta.get("human_actions_only") is not False or not meta.get("training_demonstration")
            or meta.get("task") != TASKS.get(meta.get("task_name"))):
        raise ValueError("Invalid synthetic demonstration provenance")
    if digest(path / "frames.npz") != meta["frames_sha256"]:
        raise ValueError("Demonstration content changed")
    with np.load(path / "frames.npz", allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in ("state", "action", "image", "wrist_image")}
    n = meta["frames"]
    if (n < 10 or arrays["state"].shape != (n, 8) or arrays["action"].shape != (n, 7)
            or not np.isfinite(arrays["state"]).all() or not np.isfinite(arrays["action"]).all()
            or any(arrays[k].shape != (n, 224, 224, 3) or arrays[k].dtype != np.uint8
                   for k in ("image", "wrist_image"))):
        raise ValueError("Demonstration frame contract violated")
    return meta, arrays


def freeze(root, review, output):
    root, output = Path(root).resolve(), Path(output)
    reviewed = json.loads(Path(review).read_text())
    if reviewed.get("reviewer") != "agent_visual_review" or not reviewed.get("accepted"):
        raise ValueError("An explicit visual review of teacher videos is required")
    records = []
    for name in reviewed["accepted"]:
        path = (root / name).resolve()
        if path.parent != root or path.is_symlink():
            raise ValueError("Demonstration path outside selected dataset")
        meta, _ = load_episode(path)
        stop = meta["frames"]
        trace_path = path / "trace.json"
        trace_sha256 = None
        if trace_path.exists():
            trace = json.loads(trace_path.read_text())
            if len(trace) != meta["frames"]:
                raise ValueError("Trace/frame count mismatch")
            # Preserve the full release and retreat. Long terminal physics
            # verification is evidence, not a demonstration of useful motion.
            settle = next((i for i, row in enumerate(trace) if row["phase"] == "settle"), None)
            if settle is not None:
                stop = min(stop, settle + 3)
            trace_sha256 = digest(trace_path)
        records.append(dict(path=name, metadata_sha256=digest(path / "metadata.json"),
            task=meta["task"], task_name=meta["task_name"], initial_state_id=meta["initial_state_id"],
            initial_state_sha256=meta["initial_state_sha256"], frames=meta["frames"],
            training_stop=stop, trace_sha256=trace_sha256,
            split="train" if meta["initial_state_id"] < 8 else "holdout"))
    if len({r["path"] for r in records}) != len(records):
        raise ValueError("Duplicate demonstration")
    for task in TASKS:
        for split, minimum in (("train", 2), ("holdout", 1)):
            if sum(r["task_name"] == task and r["split"] == split for r in records) < minimum:
                raise ValueError(f"Insufficient {split} demonstrations for {task}")
    train_states = {r["initial_state_sha256"] for r in records if r["split"] == "train"}
    if train_states & {r["initial_state_sha256"] for r in records if r["split"] == "holdout"}:
        raise ValueError("Train/holdout initial scene leakage")
    manifest = dict(schema="counterfactual-distillation-v1", root=str(root), clips=records,
        horizon=10, seed=42, review_sha256=digest(review), ready=True,
        inference="pure pi05, no teacher, no object poses or task-specific routing",
        sampling="equal task probability; uniform control frames within task; shuffle seed 42",
        terminal_filter="keep release and retreat; at most 3 terminal settle frames; raw evidence retained",
        tail="repeat final action inside same episode", human_demonstrations=False,
        limitation="task-specific synthetic distillation; not proof of zero-shot general understanding")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as f:
        f.write(json.dumps(manifest, indent=2) + "\n")
    return manifest


class DistillationDataset:
    """Only train clips, balanced by task. Cache RGB in RAM, not repeated disk IO."""
    def __init__(self, manifest):
        if manifest.get("schema") != "counterfactual-distillation-v1" or not manifest.get("ready"):
            raise ValueError("Unfrozen distillation dataset")
        self.episodes, self.task_indices = [], {t: [] for t in TASKS}
        root = Path(manifest["root"]).resolve()
        holdout = {r["initial_state_sha256"] for r in manifest["clips"] if r["split"] == "holdout"}
        for record in manifest["clips"]:
            if record["split"] != "train":
                continue
            path = (root / record["path"]).resolve()
            if path.parent != root or digest(path / "metadata.json") != record["metadata_sha256"]:
                raise ValueError("Dataset identity changed")
            meta, arrays = load_episode(path)
            if record["trace_sha256"] is not None and digest(path / "trace.json") != record["trace_sha256"]:
                raise ValueError("Demonstration trace changed")
            stop = record["training_stop"]
            if not 10 <= stop <= meta["frames"]:
                raise ValueError("Invalid training boundary")
            if meta["initial_state_sha256"] in holdout:
                raise ValueError("Train/holdout leakage")
            if any(meta[k] != record[k] for k in ("task", "task_name", "frames", "initial_state_sha256")):
                raise ValueError("Episode does not match frozen manifest")
            episode_id = len(self.episodes)
            self.episodes.append((meta, arrays, stop))
            self.task_indices[meta["task_name"]].extend((episode_id, i) for i in range(stop))
        if any(not indices for indices in self.task_indices.values()):
            raise ValueError("All three tasks must be represented")
        self.tasks = list(TASKS)
        self.per_task = max(map(len, self.task_indices.values()))

    def __len__(self):
        return len(self.tasks) * self.per_task

    def __getitem__(self, index):
        indices = self.task_indices[self.tasks[index % len(self.tasks)]]
        episode, frame = indices[(index // len(self.tasks)) % len(indices)]
        meta, arrays, stop = self.episodes[episode]
        window = np.minimum(np.arange(frame, frame + 10), stop - 1)
        return dict(state=arrays["state"][frame], image=arrays["image"][frame],
            wrist_image=arrays["wrist_image"][frame], actions=arrays["action"][window], prompt=meta["task"])


def train(args):
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if json.loads((SOURCE / "SNAPSHOT.json").read_text())["openpi"] != REVISION:
        raise ValueError("Pinned source mismatch")
    manifest = json.loads(args.manifest.read_text())
    dataset = DistillationDataset(manifest)
    cfg = make_config(args.parent, args.experiment, args.steps, 1, args.learning_rate)
    if cfg.checkpoint_dir.exists():
        raise FileExistsError(cfg.checkpoint_dir)
    from openpi.training import data_loader
    sample = data_loader.transform_dataset(dataset, cfg.data.create(cfg.assets_dirs, cfg.model))[0]
    if sample["actions"].shape != (10, 32):
        raise ValueError("Official action transform mismatch")
    record = dict(config_name=NAME, steps=args.steps, batch_size=1, learning_rate=args.learning_rate,
        seed=42, parent=str(args.parent.resolve()), manifest=str(args.manifest.resolve()),
        manifest_sha256=digest(args.manifest), openpi_revision=REVISION,
        code_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        source_sha256={name: digest(ROOT / "vla" / name) for name in
            ("teacher_data.py", "train_distillation.py", "train_corrections.py")},
        synthetic_teacher=True, training_kind="pi05 LoRA behavioral distillation",
        tasks=list(TASKS.values()), sampled_windows=len(dataset), normalization="unchanged official parent",
        trainable="LoRA only", optimizer_resumed=False, dry_run=args.dry_run)
    launch = args.manifest.parent / f"{args.experiment}-{'dry-run' if args.dry_run else 'launch'}.json"
    with launch.open("x") as f:
        f.write(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)
    if args.dry_run:
        return
    import jax
    if not jax.devices("gpu"):
        raise RuntimeError("GPU required")
    spec = importlib.util.spec_from_file_location("distillation_upstream_train", SOURCE / "scripts/train.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    original = data_loader.create_torch_dataset
    def create(data_config, action_horizon, model_config):
        if data_config.repo_id != ASSET or action_horizon != 10:
            raise ValueError("Training data contract mismatch")
        return dataset
    data_loader.create_torch_dataset = create
    try:
        trainer.main(cfg)
        (cfg.checkpoint_dir / "launch.json").write_text(json.dumps(record, indent=2) + "\n")
    finally:
        data_loader.create_torch_dataset = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze_parser = sub.add_parser("freeze")
    freeze_parser.add_argument("--root", type=Path, required=True)
    freeze_parser.add_argument("--review", type=Path, required=True)
    freeze_parser.add_argument("--output", type=Path, required=True)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--manifest", type=Path, required=True)
    train_parser.add_argument("--parent", type=Path, default=ROOT / "models/pi05_libero")
    train_parser.add_argument("--experiment", required=True)
    train_parser.add_argument("--steps", type=int, default=2000)
    train_parser.add_argument("--learning-rate", type=float, default=5e-5)
    train_parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.root, args.review, args.output)
    else:
        train(args)


if __name__ == "__main__":
    main()
