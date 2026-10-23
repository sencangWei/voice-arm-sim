"""Generate a local synthetic Mandarin command; optionally test unchanged ASR."""

import argparse
import hashlib
import json
import urllib.request
import wave
from pathlib import Path

from piper import PiperVoice

ROOT = Path(__file__).resolve().parents[1]
REVISION = "5041a7a49ae5b3a0f30e6837769eb03fdbb231bd"
SOURCE = "https://huggingface.co/rhasspy/piper-voices"
VOICE = "zh_CN-huayan-medium"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default="把苹果放进蓝色托盘。")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/neural-command.wav")
    parser.add_argument("--transcribe", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError(f"Refusing to overwrite evidence: {args.output}")
    directory = ROOT / "models/piper-zh_CN-huayan-medium"
    directory.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in (VOICE + ".onnx", VOICE + ".onnx.json", "MODEL_CARD"):
        path = directory / name
        url = f"{SOURCE}/resolve/{REVISION}/zh/zh_CN/huayan/medium/{name}"
        if not path.exists():
            with (
                urllib.request.urlopen(url, timeout=120) as source,
                path.open("wb") as out,
            ):
                while chunk := source.read(1024 * 1024):
                    out.write(chunk)
        with path.open("rb") as source:
            hashes[name] = hashlib.file_digest(source, "sha256").hexdigest()
    voice = PiperVoice.load(str(directory / (VOICE + ".onnx")), use_cuda=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(args.output), "wb") as audio:
        voice.synthesize_wav(args.text, audio)
    with wave.open(str(args.output), "rb") as audio:
        duration = audio.getnframes() / audio.getframerate()
        rate = audio.getframerate()
    result = {
        "source": SOURCE,
        "revision": REVISION,
        "voice": VOICE,
        "engine": "piper-tts==1.4.2 (CPU ONNX)",
        "engine_license": "GPL-3.0",
        "voice_dataset_license": "Unknown according to upstream MODEL_CARD",
        "text": args.text,
        "audio": str(args.output),
        "sample_rate": rate,
        "duration_seconds": duration,
        "model_sha256": hashes,
        "boundary": "Synthetic Mandarin test audio; human microphone input remains unverified.",
    }
    if args.transcribe:
        request = urllib.request.Request(
            "http://127.0.0.1:8911/transcribe",
            data=args.output.read_bytes(),
            headers={"Content-Type": "audio/wav"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            result["asr_status"] = response.status
            result["asr_response"] = json.load(response)
    args.output.with_suffix(".json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
