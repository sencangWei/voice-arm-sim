"""Isolated official MolmoAct2 LIBERO inference, using the existing policy wire format."""
import argparse
import asyncio
import hashlib
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = "allenai/MolmoAct2-LIBERO"
REVISION = "0d24a92bd1faf321ef497c3bbd5681af97c65aa2"
sys.path.insert(0, str(ROOT / ".local/openpi/packages/openpi-client/src"))


def validate_observation(obs):
    images = [np.asarray(obs[key]) for key in
              ("observation/image", "observation/wrist_image")]
    if any(im.dtype != np.uint8 or im.shape != (256, 256, 3) for im in images):
        raise ValueError("MolmoAct2 requires two ordered 256x256 uint8 RGB images")
    state = np.asarray(obs["observation/state"], dtype=np.float32)
    prompt = obs["prompt"]
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError("state must be finite shape (8,)")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 2000:
        raise ValueError("prompt must be nonempty text, <=2000 characters")
    return images, state, prompt


def vision_inputs_on_device(module, args):
    # Upstream moves pixels inside vision.forward, but not patch-index tensors.
    # Keep both on the CPU vision device before pooling indexes the features.
    return tuple(value.to(module.device) for value in args)


class MolmoPolicy:
    def __init__(self, checkpoint, placement):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        torch.set_num_threads(4)
        self.processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True,
                                                       local_files_only=True)
        # An Accelerate mixed device_map uses CPU as weight storage and still
        # executes offloaded layers on GPU. Explicit moves keep vision compute
        # on CPU too, leaving room alongside the user's existing GPU services.
        device_map = {"": "cuda:0" if placement == "cuda" else "cpu"}
        self.model = AutoModelForImageTextToText.from_pretrained(
            checkpoint, trust_remote_code=True, local_files_only=True,
            dtype=torch.bfloat16, device_map=device_map).eval()
        if placement == "vision-cpu":
            self.model.model.transformer.to("cuda:0")
            self.model.model.action_expert.to("cuda:0")
            self.model.model.vision_backbone.register_forward_pre_hook(vision_inputs_on_device)
        self.generator = torch.Generator(device="cuda").manual_seed(0)
        self.metadata = {
            "backend": "official-molmoact2-libero-vla",
            "action_source": "MolmoAct2.predict_action.continuous",
            "checkpoint_source": REPO, "checkpoint_revision": REVISION,
            "checkpoint_manifest_sha256": hashlib.sha256(
                (checkpoint / "verified-manifest.json").read_bytes()).hexdigest(),
            "precision": "bfloat16", "placement": placement,
            "image_size": 256, "action_horizon": 10, "action_dim": 7,
            "flow_steps": 10, "cuda_graph": False, "training_this_run": False,
            "default_policy_rng_seed": 0,
            "rng_policy": "advances per request; optional fixed _probe_noise_seed",
        }

    def infer(self, obs):
        images, state, prompt = validate_observation(obs)
        seed = obs.get("_probe_noise_seed")
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**32):
            raise ValueError("probe seed must be uint32")
        torch = self.torch
        generator = (self.generator if seed is None else
                     torch.Generator(device="cuda").manual_seed(seed))
        tick = time.monotonic()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = self.model.predict_action(
                processor=self.processor, images=images, state=state, task=prompt,
                norm_tag="libero", inference_action_mode="continuous",
                enable_depth_reasoning=False, enable_cuda_graph=False,
                num_steps=10, n_action_steps=10, generator=generator)
        actions = output.actions.detach().float().cpu().numpy()
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        if actions.shape != (10, 7) or not np.isfinite(actions).all():
            raise ValueError(f"Invalid official action output: {actions.shape}")
        return {"actions": actions, "server_timing": {
            "infer_ms": 1000 * (time.monotonic() - tick)}}


async def serve(policy, port):
    from openpi_client import msgpack_numpy
    from websockets.asyncio.server import serve as websocket_serve
    from websockets.exceptions import ConnectionClosed

    async def handler(ws):
        await ws.send(msgpack_numpy.packb(policy.metadata))
        try:
            async for request in ws:
                try:
                    response = policy.infer(msgpack_numpy.unpackb(request))
                    await ws.send(msgpack_numpy.packb(response))
                except Exception as error:
                    logging.exception("Inference failed")
                    await ws.send(f"{type(error).__name__}: {error}")
                    await ws.close(code=1011)
                    break
        except ConnectionClosed:
            pass

    async with websocket_serve(handler, "127.0.0.1", port, compression=None,
                               max_size=4 * 1024 * 1024, ping_timeout=None):
        logging.info("MolmoAct2 ready at 127.0.0.1:%s: %s", port, policy.metadata)
        await asyncio.Future()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/molmoact2/checkpoint")
    parser.add_argument("--port", type=int, default=8777)
    parser.add_argument("--placement", choices=("vision-cpu", "cuda"), default="vision-cpu")
    args = parser.parse_args()
    manifest = json.loads((args.checkpoint / "verified-manifest.json").read_text())
    if manifest["repo"] != REPO or manifest["revision"] != REVISION:
        raise ValueError("Expected pinned verified official MolmoAct2 weights")
    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve(MolmoPolicy(args.checkpoint, args.placement), args.port))


if __name__ == "__main__":
    main()
