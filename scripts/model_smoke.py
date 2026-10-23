"""Exercise the real localhost models; records responses, never substitutes detections."""

import argparse
import base64
import io
import json
import sys
import time
import urllib.request
from pathlib import Path

import requests
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/models-smoke"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image", type=Path, default=ROOT / "runs/scene-final/overhead.png"
    )
    parser.add_argument("--audio", type=Path)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    url = "http://127.0.0.1:8911"
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        try:
            health = requests.get(url + "/health", timeout=5).json()
            if health.get("error"):
                raise RuntimeError(health["error"])
            if health.get("ready"):
                break
        except requests.RequestException:
            pass
        time.sleep(2)
    else:
        raise RuntimeError("Model service did not become ready in 600 seconds")

    audio = args.audio or OUT / "official_asr_zh.wav"
    audio_source = (
        "user-provided"
        if args.audio
        else (
            "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav"
        )
    )
    if not audio.exists():
        with urllib.request.urlopen(audio_source, timeout=60) as response:
            audio.write_bytes(response.read())
    results = {
        "health": health,
        "audio_source": audio_source,
        "image": str(args.image),
        "tests": [],
    }

    def record(name, endpoint, payload=None, data=None):
        started = time.monotonic()
        response = requests.post(url + endpoint, json=payload, data=data, timeout=180)
        result = {
            "name": name,
            "endpoint": endpoint,
            "status": response.status_code,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "response": response.json(),
        }
        results["tests"].append(result)
        (OUT / "results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n"
        )
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return result

    record("official_chinese_asr", "/transcribe", data=audio.read_bytes())
    image = Image.open(args.image).convert("RGB")
    encoded = io.BytesIO()
    image.save(encoded, format="JPEG", quality=95)
    encoded = base64.b64encode(encoded.getvalue()).decode()
    cases = [
        ("red_cube", "/ground", "把红色方块放进蓝色托盘"),
        ("green_apple", "/ground", "把绿色苹果放进蓝色托盘"),
        ("yellow_banana", "/ground", "把黄色香蕉放进蓝色托盘"),
        ("blue_tray", "/locate", "蓝色托盘"),
        ("absent_target", "/ground", "把紫色恐龙玩具放进蓝色托盘"),
    ]
    for name, endpoint, text in cases:
        result = record(
            name,
            endpoint,
            {"image": encoded, "target" if endpoint == "/locate" else "text": text},
        )
        bbox = result["response"].get("bbox_2d", [])
        if len(bbox) == 4:
            overlay = image.copy()
            draw = ImageDraw.Draw(overlay)
            box = [
                bbox[i] * (image.width if i % 2 == 0 else image.height) / 1000
                for i in range(4)
            ]
            draw.rectangle(box, outline="magenta", width=3)
            draw.text((box[0], max(0, box[1] - 14)), name, fill="magenta")
            overlay.save(OUT / f"{name}.png")
    record("invalid_json", "/ground", payload=[])
    record("oversize_body", "/transcribe", data=b"x" * (8 * 1024 * 1024 + 1))
    failures = [x["name"] for x in results["tests"][:6] if x["status"] != 200]
    tests = {x["name"]: x for x in results["tests"]}
    if not tests["official_chinese_asr"]["response"].get("text", "").strip():
        failures.append("empty_asr")
    for name in ("red_cube", "green_apple", "yellow_banana", "blue_tray"):
        if len(tests[name]["response"].get("bbox_2d", [])) != 4:
            failures.append(name + "_empty")
    if tests["absent_target"]["response"].get("bbox_2d") != []:
        failures.append("absent_target_hallucination")
    for name, status in (("invalid_json", 400), ("oversize_body", 413)):
        if tests[name]["status"] != status:
            failures.append(name)
    results["failures"] = failures
    results["validation_boundary"] = (
        "Real pretrained inference; semantic box placement requires image inspection."
    )
    (OUT / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n"
    )
    sys.exit(bool(failures))


if __name__ == "__main__":
    main()
