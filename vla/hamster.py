"""Isolated HAMSTER path probe. This module never sends robot actions.

Uses pinned official VILA projector/conversation code and official HF weights.
The CPU BF16 adapter replaces VILA's hard-coded CUDA loader, not its model.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
VILA_REVISION = "a5a380d6d09762d6f3fd0443aac6b475fba84f7e"


def instruction(command):
    # Literal prompt from HAMSTER_beta/gradio_server_example.py at 526a37f.
    return (
        f"\nIn the image, please execute the command described in <quest>{command}</quest>.\n"
        "Provide a sequence of points denoting the trajectory of a robot gripper to achieve the goal.\n"
        "Format your answer as a list of tuples enclosed by <ans> and </ans> tags. For example:\n"
        "<ans>[(0.25, 0.32), (0.32, 0.17), (0.13, 0.24), <action>Open Gripper</action>, "
        "(0.74, 0.21), <action>Close Gripper</action>, ...]</ans>\n"
        "The tuple denotes point x and y location of the end effector in the image. "
        "The action tags indicate gripper actions.\n"
        "Coordinates should be floats between 0 and 1, representing relative positions.\n"
        "Remember to provide points between <ans> and </ans> tags and think step by step."
    )


def parse_path(text):
    """Reject malformed/unbounded output; never execute text or invent a path."""
    answers = re.findall(r"<ans>(.*?)</ans>", text, re.DOTALL)
    if len(answers) != 1:
        raise ValueError("Expected exactly one complete <ans> path")
    body = answers[0]
    for action in ("Open Gripper", "Close Gripper"):
        body = body.replace(f"<action>{action}</action>", repr(action))
    try:
        items = ast.literal_eval(body)
    except (ValueError, SyntaxError) as exc:
        raise ValueError("Path is not a literal list of points/actions") from exc
    if not isinstance(items, list) or not 2 <= len(items) <= 256:
        raise ValueError("Expected 2..256 path entries")
    points = []
    for item in items:
        if isinstance(item, str):
            if item not in ("Open Gripper", "Close Gripper") or not points:
                raise ValueError("Unknown gripper action or action before first point")
            points[-1]["action"] = item
        else:
            if (not isinstance(item, tuple) or len(item) != 2
                    or any(type(x) not in (int, float) or not math.isfinite(x)
                           or not 0 <= x <= 1 for x in item)):
                raise ValueError("Point must contain two finite coordinates in [0,1]")
            points.append({"x": float(item[0]), "y": float(item[1]), "action": None})
    if len(points) < 2:
        raise ValueError("Expected at least two points")
    return points


def load_upstream(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def embed_prompt(prompt, tokenizer, image_features, embedding):
    """One image sentinel replaced by learned image embeddings, BOS only once."""
    import torch

    if prompt.count("<image>") != 1:
        raise ValueError("This probe requires exactly one image")
    left, right = (tokenizer(part).input_ids for part in prompt.split("<image>"))
    if left and left[0] == tokenizer.bos_token_id:
        right = right[1:]
    before = embedding(torch.tensor([left], dtype=torch.long))
    after = embedding(torch.tensor([right], dtype=torch.long))
    return torch.cat((before, image_features, after), dim=1)


def overlay(image, points, output):
    from PIL import ImageDraw

    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    pixels = [(round(p["x"] * (image.width - 1)), round(p["y"] * (image.height - 1)))
              for p in points]
    draw.line(pixels, fill="yellow", width=3)
    for i, ((x, y), point) in enumerate(zip(pixels, points)):
        color = {"Open Gripper": "cyan", "Close Gripper": "red", None: "yellow"}[point["action"]]
        draw.ellipse((x-5, y-5, x+5, y+5), fill=color, outline="black")
        draw.text((x+6, y), str(i), fill="black", stroke_width=1, stroke_fill="white")
    canvas.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--commands", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {args.output}")
    manifest = json.loads((ROOT / "evidence/hamster/official_manifest.json").read_text())
    verified = json.loads((ROOT / "evidence/hamster/verified-checkpoint.json").read_text())
    if (verified["revision"] != manifest["revision"]
            or len(verified["files"]) != len(manifest["files"])):
        raise ValueError("Run scripts/prepare_hamster.py --verify-only first")
    vila = ROOT / ".local/hamster-vila"
    revision = subprocess.check_output(["git", "-C", str(vila), "rev-parse", "HEAD"], text=True).strip()
    if revision != VILA_REVISION:
        raise ValueError("Incorrect VILA revision")
    dirty = subprocess.check_output(["git", "-C", str(vila), "status", "--porcelain"], text=True)
    if dirty.strip():
        raise ValueError("Upstream VILA must be clean")
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    from transformers import (
        AutoTokenizer,
        LlamaForCausalLM,
        SiglipImageProcessor,
        SiglipVisionModel,
    )

    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    checkpoint = ROOT / "models/hamster" / manifest["subfolder"]
    config = json.loads((checkpoint / "config.json").read_text())
    if (config["mm_vision_select_layer"] != -2 or config["mm_vision_select_feature"] != "cls_patch"
            or config["image_aspect_ratio"] != "resize" or config["s2"]):
        raise ValueError("Unsupported checkpoint vision contract")
    args.output.mkdir(parents=True, exist_ok=True)
    commands = json.loads(args.commands.read_text())
    image = Image.open(args.image).convert("RGB")
    image.save(args.output / "input.png")
    metadata = {"model": manifest["repo"], "revision": manifest["revision"], "vila_revision": revision,
                    "device": "cpu", "precision": "bfloat16", "transformers": "4.37.2", "seed": 42,
                    "image_sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
                    "adapter": "standard HF vision/LLM + pinned VILA projector/conversation",
                    "deviation": "CPU BF16 instead of official server CUDA FP16; use_cache=True",
                    "max_new_tokens": args.max_new_tokens, "threads": args.threads,
                    "physical_robot_execution": False}
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    conversation = load_upstream("hamster_conversation", vila / "llava/conversation.py")
    projector_code = load_upstream("hamster_projector", vila / "llava/model/multimodal_projector/base_projector.py")
    print("Loading official vision/projector on CPU", flush=True)
    vision = SiglipVisionModel.from_pretrained(checkpoint / "vision_tower", torch_dtype=torch.bfloat16,
                                               low_cpu_mem_usage=True, local_files_only=True).eval()
    processor = SiglipImageProcessor.from_pretrained(checkpoint / "vision_tower", local_files_only=True)
    projector = projector_code.MultimodalProjector(
        projector_code.MultimodalProjectorConfig("mlp_downsample"),
        SimpleNamespace(mm_hidden_size=config["mm_hidden_size"], hidden_size=config["hidden_size"]),
    ).to(dtype=torch.bfloat16).eval()
    projector.load_state_dict(load_file(checkpoint / "mm_projector/model.safetensors"), strict=True)
    with torch.inference_mode():
        # Match VILA process_image: PIL resize first, then SigLIP preprocess.
        resized = image.resize((processor.size["height"], processor.size["width"]))
        pixels = processor.preprocess(resized, return_tensors="pt")["pixel_values"].to(torch.bfloat16)
        features = vision(pixels, output_hidden_states=True).hidden_states[-2]
        image_features = projector(features)
    if tuple(image_features.shape) != (1, 196, 5120):
        raise ValueError(f"Unexpected projected image shape: {image_features.shape}")
    del vision, projector, features, pixels
    print("Loading official 13B LLM on CPU (existing GPU services preserved)", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint / "llm", use_fast=False, local_files_only=True)
    llm = LlamaForCausalLM.from_pretrained(checkpoint / "llm", torch_dtype=torch.bfloat16,
                                          low_cpu_mem_usage=True, local_files_only=True).eval()
    for index, command in enumerate(commands):
        conv = conversation.conv_templates["vicuna_v1"].copy()
        conv.append_message(conv.roles[0], "<image>" + instruction(command["english"]))
        prompt = conv.get_prompt()  # Exactly the official server: no extra assistant prefix.
        prefix = args.output / f"command-{index:02d}"
        prefix.with_suffix(".prompt.txt").write_text(prompt)
        print(f"Inference {index}: {command['english']}", flush=True)
        start = time.monotonic()
        with torch.inference_mode():
            embeddings = embed_prompt(prompt, tokenizer, image_features, llm.get_input_embeddings())
            ids = llm.generate(inputs_embeds=embeddings, attention_mask=torch.ones(embeddings.shape[:2], dtype=torch.long),
                               do_sample=False, num_beams=1, max_new_tokens=args.max_new_tokens,
                               use_cache=True, pad_token_id=0, bos_token_id=1, eos_token_id=2)
        text = tokenizer.batch_decode(ids, skip_special_tokens=True)[0].strip()
        prefix.with_suffix(".raw.txt").write_text(text + "\n")
        result = {"command": command, "duration_s": time.monotonic()-start,
                      "generated_ids": ids.tolist(), "raw": text, "physical_execution": False,
                      "semantic_success": None}
        try:
            result["points"] = parse_path(text)
            result["syntax_valid"] = True
            overlay(image, result["points"], prefix.with_suffix(".overlay.png"))
        except ValueError as exc:
            result.update(syntax_valid=False, error=str(exc))
        prefix.with_suffix(".json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
