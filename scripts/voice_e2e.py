"""Real local WAV -> ASR -> perception -> physical grasp acceptance and video.

Requires an already-running 8910 app. Do not operate the UI concurrently.
"""
import argparse
from io import BytesIO
import json
from pathlib import Path
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageOps


BASE = "http://127.0.0.1:8910"
TERMINAL = {"done", "failed", "clarify", "stopped", "error"}


def request(path, payload=None, *, audio=False, mode="vlm", timeout=10):
    headers = {}
    if audio:
        body = payload
        headers.update({"Content-Type": "audio/wav", "X-Vision-Mode": mode})
    elif payload is not None:
        body = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    else:
        body = None
    req = Request(BASE + path, data=body, headers=headers)
    try:
        with urlopen(req, timeout=timeout) as response:
            raw = response.read()
            return response.status, raw if path.startswith("/api/frame") else json.loads(raw)
    except HTTPError as exc:
        raw = exc.read()
        try:
            result = json.loads(raw)
        except (ValueError, UnicodeError):
            result = {"error": raw.decode(errors="replace")}
        return exc.code, result


def get_state():
    status, state = request("/api/state")
    if status != 200:
        raise RuntimeError(f"状态读取失败：{status} {state}")
    return state


def pair_frame():
    views = []
    for view in ("scene", "top"):
        status, raw = request("/api/frame?view=" + view, timeout=3)
        if status != 200:
            raise RuntimeError(f"{view} 画面不可用：{status}")
        with Image.open(BytesIO(raw)) as image:
            image = ImageOps.pad(image.convert("RGB"), (480, 480))
            views.append(np.asarray(image))
    return np.concatenate(views, axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--mode", choices=("color", "vlm"), default="vlm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if not 0 <= args.seed < 2**31:
        parser.error("seed 必须为 0 到 2^31-1")
    audio = args.audio.read_bytes()
    if len(audio) > 8*1024*1024 or audio[:4] != b"RIFF" or audio[8:12] != b"WAVE":
        parser.error("audio 必须是小于等于 8 MiB 的 WAV 文件")
    output = args.output or Path("runs") / f"voice-e2e-{time.time_ns()}"
    output.mkdir(parents=True, exist_ok=False)
    report = {"audio": str(args.audio.resolve()), "mode": args.mode, "seed": args.seed,
              "started_unix": time.time(), "recognized": "", "text": "",
              "response": None, "status": None, "success": False,
              "video": "demo.mp4", "video_fps": 10, "video_frames": 0,
              "frame_errors": [], "state_errors": []}
    trace = []
    response_done = threading.Event()
    response_data = {}
    writer = None
    submitted = False
    started = time.monotonic()

    def submit():
        try:
            status, result = request("/api/audio", audio, audio=True, mode=args.mode, timeout=180)
            response_data.update(status=status, response=result)
        except Exception as exc:
            response_data.update(status=None, response={"error": str(exc)})
        finally:
            response_done.set()

    try:
        initial = get_state()
        if initial.get("busy"):
            raise RuntimeError("仿真正在执行其它任务；请在空闲时运行验收")
        status, result = request("/api/reset", {"seed": args.seed, "mode": args.mode})
        if status != 202:
            raise RuntimeError(f"布局重置失败：{status} {result}")
        reset_deadline = time.monotonic() + 20
        while True:
            state = get_state()
            if not state.get("busy"):
                if state.get("last_result", {}).get("seed") != args.seed:
                    raise RuntimeError("未确认指定布局重置成功：" + str(state))
                break
            if time.monotonic() > reset_deadline:
                raise RuntimeError("等待布局重置超时")
            time.sleep(.1)
        writer = imageio.get_writer(output / "demo.mp4", fps=10)
        started = time.monotonic()
        submitted = True
        threading.Thread(target=submit, name="audio-submit", daemon=True).start()
        deadline = started + 180
        next_frame = started
        while time.monotonic() < deadline:
            # Read completion before state so an old reset snapshot cannot be
            # mistaken for the submitted command's terminal state.
            responded = response_done.is_set()
            state = None
            elapsed = time.monotonic() - started
            try:
                state = get_state()
                trace.append({"elapsed_s": round(elapsed, 3), "state": state})
                report["final_state"] = state
            except Exception as exc:
                report["state_errors"].append({"elapsed_s": round(elapsed, 3), "error": str(exc)})
            try:
                writer.append_data(pair_frame())
                report["video_frames"] += 1
            except Exception as exc:
                report["frame_errors"].append({"elapsed_s": round(elapsed, 3), "error": str(exc)})
            if responded:
                report.update(response_data)
                response = response_data.get("response", {})
                text = response.get("text", "") if isinstance(response, dict) else ""
                report.update(recognized=text, text=text)
                if response_data.get("status") not in (200, 202):
                    report["error"] = "音频请求未被接受"
                    break
                if state and not state.get("busy") and state.get("phase") in TERMINAL:
                    report["success"] = (state.get("phase") == "done"
                                         and state.get("last_result", {}).get("success") is True)
                    if not report["success"]:
                        report["error"] = state.get("message", "抓取未成功")
                    break
            next_frame += .1
            # Do not issue a burst of catch-up requests after slow network/render.
            next_frame = max(next_frame, time.monotonic())
            time.sleep(max(0, min(next_frame, deadline)-time.monotonic()))
        else:
            report["error"] = "音频与抓取验收超过 180 秒"
            report["timed_out"] = True
            stop_status, stop_response = request("/api/stop", {})
            report["timeout_stop"] = {"status": stop_status, "response": stop_response}
        if response_done.is_set():
            report.update(response_data)
            response = response_data.get("response", {})
            if isinstance(response, dict):
                report["recognized"] = report["text"] = response.get("text", "")
        if not report["video_frames"]:
            report["success"] = False
            report["error"] = "没有获取到真实双视角视频帧"
    except Exception as exc:
        report["error"] = str(exc)
        report["success"] = False
        if submitted:
            try:
                status, result = request("/api/stop", {})
                report["error_stop"] = {"status": status, "response": result}
            except Exception as stop_exc:
                report["stop_error"] = str(stop_exc)
    finally:
        if writer is not None:
            try:
                writer.close()
            except Exception as exc:
                report["success"] = False
                report["video_error"] = str(exc)
        report["elapsed_s"] = round(time.monotonic()-started, 3)
        report["finished_unix"] = time.time()
        report["validation_boundary"] = "真实服务与物理仿真结果；非训练结果，非泛化成功率。录像记录成功读取的真实帧，帧错误另计。"
        (output / "trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n")
        (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"success": report["success"], "text": report["text"],
                          "phase": report.get("final_state", {}).get("phase"),
                          "result": str(output / "result.json")}, ensure_ascii=False))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
