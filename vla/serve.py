"""Serve the official multitask pi05 LIBERO policy on localhost."""

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".local/openpi"
REVISION = "15a9616a00943ada6c20a0f158e3adb39df2ccac"



class ProbePolicy:
    """Optional fixed diffusion noise for a controlled language ablation only."""

    def __init__(self, policy, horizon, action_dim):
        self.policy, self.horizon, self.action_dim = policy, horizon, action_dim

    def infer(self, obs):
        import numpy as np
        obs = dict(obs)
        seed = obs.pop("_probe_noise_seed", None)
        if seed is None:
            return self.policy.infer(obs)
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("probe seed must be a uint32 integer")
        noise = np.random.default_rng(seed).standard_normal(
            (self.horizon, self.action_dim)).astype(np.float32)
        return self.policy.infer(obs, noise=noise)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/pi05_libero")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    snapshot = json.loads((SOURCE / "SNAPSHOT.json").read_text())
    if snapshot["openpi"] != REVISION:
        raise ValueError("Unexpected openpi source revision")
    from vla.train_corrections import validate_parent
    args.checkpoint = validate_parent(args.checkpoint)
    local_lora = not (args.checkpoint / "manifest.json").exists()
    manifest_path = args.checkpoint.parent / "launch.json" if local_lora else args.checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    # Prepend isolated pinned source; do not import the old learning project's code.
    sys.path[:0] = [str(SOURCE / "src"), str(SOURCE / "packages/openpi-client/src")]
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.50")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    import jax
    from openpi.policies import policy_config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from openpi.training.config import get_config

    devices = jax.devices("gpu")
    logging.info("Actual inference devices: %s", devices)
    if local_lora:
        from vla.train_corrections import make_config
        config = make_config(args.checkpoint, "serve", manifest["steps"], manifest["batch_size"], manifest["learning_rate"])
    else:
        config = get_config("pi05_libero")
    policy = policy_config.create_trained_policy(config, args.checkpoint)
    metadata = {
        "backend": "local-pi05-libero-lora" if local_lora else "official-pi05-libero-vla",
        "action_source": "pi05.sample_actions",
        "openpi_revision": REVISION,
        "libero_revision": snapshot["libero"],
        "checkpoint_source": str(args.checkpoint) if local_lora else manifest["source"],
        "checkpoint_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "action_horizon": config.model.action_horizon,
        "action_dim": 7,
        "default_policy_rng_seed": 0,
        "training_this_run": False,
    }
    logging.info("Policy loaded: %s", json.dumps(metadata))
    WebsocketPolicyServer(ProbePolicy(policy, config.model.action_horizon, config.model.action_dim), host="127.0.0.1", port=args.port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
