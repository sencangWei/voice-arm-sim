"""Independent, manifest-locked π0.5 LoRA training for reviewed corrections."""
import argparse
import dataclasses
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

from vla.corrections import digest, load_frames, write_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".local/openpi"
REVISION = "15a9616a00943ada6c20a0f158e3adb39df2ccac"
NAME = "voice_pi05_lora"
ASSET = "physical-intelligence/libero"


def prepare_dataset(corrections_root, output):
    corrections_root, output = Path(corrections_root).resolve(), Path(output)
    records = []
    for path in sorted(corrections_root.glob("human_*/metadata.json")):
        meta = json.loads(path.read_text())
        if meta.get("status") != "approved":
            continue
        if not path.resolve().is_relative_to(corrections_root) or path.is_symlink():
            raise ValueError("数据必须在本项目纠正目录内")
        meta, frames = load_frames(path.parent)
        records.append(dict(path=str(path.parent.resolve()), metadata_sha256=digest(path),
            split=meta["split"], scenario_key=meta["scenario_key"], task=meta["task"], frames=len(frames)))
    validate_selection(records)
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema="vla-corrections-dataset-v1", ready=True, corrections_root=str(corrections_root),
        clips=records, horizon=10, tail="repeat last action within each clip; never cross clips",
        sampling="uniform over approved training control frames", public_replay=False,
        train_frames=sum(item["frames"] for item in records if item["split"] == "train"),
        holdout_frames=sum(item["frames"] for item in records if item["split"] == "holdout"),
        preprocessing="recorded official rotate180/resize-pad224 RGB; state8; LIBERO delta action7; no extra delta",
        pauses_excluded=True, speed_resampling=False)
    path = output / "manifest.json"
    write_json(path, manifest)
    return path


def validate_selection(records):
    train = [item for item in records if item["split"] == "train"]
    holdout = [item for item in records if item["split"] == "holdout"]
    if len(train) < 2 or not holdout:
        raise ValueError(f"数据不足：合格训练 {len(train)} 条、评测 {len(holdout)} 条；至少 2 条训练＋1 条独立布局评测")
    if {item["scenario_key"] for item in train} & {item["scenario_key"] for item in holdout}:
        raise ValueError("同一场景同时进入训练和评测；请更换留出布局")
    if len({item["path"] for item in records}) != len(records):
        raise ValueError("数据清单存在重复片段")


def validate_manifest(path):
    manifest = json.loads(Path(path).read_text())
    if manifest.get("schema") != "vla-corrections-dataset-v1" or not manifest.get("ready") or manifest.get("horizon") != 10:
        raise ValueError("数据清单未就绪或版本不符")
    records = manifest["clips"]
    validate_selection(records)
    root = Path(manifest["corrections_root"]).resolve()
    for item in records:
        clip = Path(item["path"])
        if not clip.resolve().is_relative_to(root) or clip.is_symlink():
            raise ValueError("片段路径越界")
        if digest(clip / "metadata.json") != item["metadata_sha256"]:
            raise ValueError("片段元数据已变化，请重新准备数据")
        meta, frames = load_frames(clip)
        if (meta["status"] != "approved" or meta["split"] != item["split"] or len(frames) != item["frames"]
                or meta["scenario_key"] != item["scenario_key"] or meta["task"] != item["task"]):
            raise ValueError("片段身份、用途或帧数与冻结清单不符")
    for split in ("train", "holdout"):
        if manifest[f"{split}_frames"] != sum(item["frames"] for item in records if item["split"] == split):
            raise ValueError("数据清单总帧数不符")
    return manifest


class CorrectionDataset:
    """Training split only; one clip cached, windows never cross episode boundaries."""
    def __init__(self, manifest):
        self.clips = [item for item in manifest["clips"] if item["split"] == "train"]
        self.indices = [(clip, frame) for clip, item in enumerate(self.clips) for frame in range(item["frames"])]
        self.cache_index, self.cache = None, None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        clip_index, frame_index = self.indices[index]
        record = self.clips[clip_index]
        if self.cache_index != clip_index:
            _, self.cache = load_frames(record["path"])
            self.cache_index = clip_index
        frame = self.cache[frame_index]
        actions = np.stack([self.cache[min(frame_index + offset, len(self.cache) - 1)]["action"] for offset in range(10)])
        return dict(state=frame["state"].astype(np.float32), image=frame["image"],
                    wrist_image=frame["wrist_image"], actions=actions.astype(np.float32), prompt=record["task"])


def validate_parent(parent):
    parent = Path(parent).resolve()
    if not (parent / "params").is_dir() or not (parent / "assets" / ASSET / "norm_stats.json").is_file():
        raise ValueError("父模型缺少 params 或官方 LIBERO 归一化文件")
    if (parent / "manifest.json").exists():
        manifest = json.loads((parent / "manifest.json").read_text())
        if manifest.get("source") != "gs://openpi-assets/checkpoints/pi05_libero":
            raise ValueError("仅支持固定的官方 LIBERO 父模型")
        for item in manifest["objects"]:
            file = (parent / item["relative_path"]).resolve()
            if not file.is_relative_to(parent) or file.stat().st_size != int(item["size"]):
                raise ValueError("父模型文件清单不完整")
    else:
        record = json.loads((parent.parent / "launch.json").read_text())
        if record.get("config_name") != NAME or not parent.name.isdigit() or not (parent / "_CHECKPOINT_METADATA").exists():
            raise ValueError("仅支持本项目已完成的 LoRA checkpoint")
    return parent


def make_config(parent, experiment, steps, batch, lr):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", experiment):
        raise ValueError("实验名称仅支持字母、数字、下划线、短横线")
    if not 1 <= steps <= 100000 or batch not in (1, 2, 4) or not math.isfinite(lr) or not 0 < lr <= .001:
        raise ValueError("更新次数、批大小或学习率超出界限")
    parent = validate_parent(parent)
    sys.path[:0] = [str(SOURCE / "src"), str(SOURCE / "packages/openpi-client/src")]
    from flax import nnx
    from openpi.models.pi0_config import Pi0Config
    from openpi.shared.nnx_utils import PathRegex
    from openpi.training import config, optimizer, weight_loaders
    model = Pi0Config(pi05=True, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
                      action_horizon=10, discrete_state_input=False)
    return dataclasses.replace(config.get_config("pi05_libero"), name=NAME, exp_name=experiment,
        project_name="voice-arm-corrections", model=model,
        freeze_filter=nnx.All(nnx.Param, nnx.Not(PathRegex(".*lora.*"))), ema_decay=None,
        data=config.LeRobotLiberoDataConfig(repo_id=ASSET, assets=config.AssetsConfig(
            assets_dir=str(parent / "assets"), asset_id=ASSET),
            base_config=config.DataConfig(prompt_from_task=False), extra_delta_transform=False),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(parent / "params")),
        batch_size=batch, num_workers=0, seed=42, num_train_steps=steps,
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=min(20, steps), peak_lr=lr,
            decay_steps=max(steps, 21), decay_lr=lr), optimizer=optimizer.AdamW(clip_gradient_norm=1.),
        checkpoint_base_dir=str(ROOT / "checkpoints"), assets_base_dir=str(ROOT / "assets"),
        save_interval=steps, keep_period=None, log_interval=10, wandb_enabled=False, resume=False, overwrite=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--parent", type=Path, default=ROOT / "models/pi05_libero")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if json.loads((SOURCE / "SNAPSHOT.json").read_text())["openpi"] != REVISION:
        raise ValueError("源码版本不匹配")
    manifest = validate_manifest(args.manifest)
    cfg = make_config(args.parent, args.experiment, args.steps, args.batch_size, args.learning_rate)
    if cfg.checkpoint_dir.exists():
        raise ValueError("实验已存在；请选择新名称，禁止覆盖")
    dataset = CorrectionDataset(manifest)
    from openpi.training import data_loader
    sample = data_loader.transform_dataset(dataset, cfg.data.create(cfg.assets_dirs, cfg.model))[0]
    if sample["actions"].shape != (10, cfg.model.action_dim):
        raise ValueError("经过官方转换后的动作形状错误")
    record = dict(config_name=NAME, code_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        openpi_revision=REVISION, manifest=str(args.manifest.resolve()), manifest_sha256=digest(args.manifest),
        parent=str(args.parent.resolve()), parent_norm_sha256=digest(args.parent / "assets" / ASSET / "norm_stats.json"),
        source_sha256={name: digest(ROOT / "vla" / name) for name in ("train_corrections.py", "corrections.py", "workbench.py")},
        checkpoint_dir=str(cfg.checkpoint_dir), seed=cfg.seed, steps=args.steps, batch_size=args.batch_size,
        learning_rate=args.learning_rate, train_frames=len(dataset), holdout_frames=manifest["holdout_frames"],
        trainable="LoRA only", horizon=10, normalization="unchanged official parent stats; no holdout fitting",
        public_replay=False, dry_run=args.dry_run, optimizer_resumed=False,
        sample_action_shape=list(sample["actions"].shape), python=sys.version)
    if args.dry_run:
        write_json(args.manifest.parent / f"{args.experiment}-dry-run.json", record)
        print(json.dumps(dict(status="READY", **record), ensure_ascii=False, indent=2))
        return 0
    import jax
    if not jax.devices("gpu"):
        raise ValueError("训练需要 GPU")
    spec = importlib.util.spec_from_file_location("voice_upstream_train", SOURCE / "scripts/train.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    original = data_loader.create_torch_dataset
    def create(data_config, action_horizon, model_config):
        if data_config.repo_id != ASSET or action_horizon != 10:
            raise ValueError("训练数据契约错误")
        return dataset
    data_loader.create_torch_dataset = create
    record_path = args.manifest.parent / f"{args.experiment}-launch.json"
    write_json(record_path, record)
    try:
        trainer.main(cfg)
        write_json(cfg.checkpoint_dir / "launch.json", record)
    finally:
        data_loader.create_torch_dataset = original
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
