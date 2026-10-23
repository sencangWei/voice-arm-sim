"""Repeatable localhost API acceptance, simulation only. Leaves seed 42 idle."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8910")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / "runs/api-smoke/result.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"url": args.url, "started_unix": time.time(), "checks": {}, "passed": False}

    def request(path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        req = Request(args.url + path, data=data, headers={"Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=10) as res:
                body = res.read()
                return res.status, body if path.startswith("/api/frame") else json.loads(body)
        except HTTPError as exc:
            return exc.code, json.load(exc)

    def state():
        code, data = request("/api/state")
        assert code == 200, data
        return data

    def wait(predicate, timeout=70):
        until = time.monotonic() + timeout
        while time.monotonic() < until:
            data = state()
            if predicate(data):
                return data
            time.sleep(.1)
        raise AssertionError("等待状态超时：" + json.dumps(data, ensure_ascii=False))

    def post(path, data):
        code, result = request(path, data)
        assert 200 <= code < 300, (code, result)
        return result

    def reset(seed):
        post("/api/reset", {"seed": seed, "mode": "color"})
        result = wait(lambda s: not s["busy"])
        assert result["last_result"]["seed"] == seed, result
        return result

    try:
        wait(lambda s: not s["busy"], 120)
        reset(42)
        first_code, first = request("/api/frame?view=top")
        assert first_code == 200
        reset(43)
        second_code, second = request("/api/frame?view=top")
        assert second_code == 200 and first != second, "随机种子未改变俯视画面"
        report["checks"]["reset_changes_layout"] = {"seeds": [42, 43], "top_jpeg_sha256": [hashlib.sha256(x).hexdigest() for x in (first, second)]}
        reset(42)
        post("/api/command", {"text": "把紫色独角兽放进蓝色托盘", "mode": "color"})
        unknown = wait(lambda s: not s["busy"])
        assert unknown["steps"] == 0 and unknown["phase"] == "clarify", unknown
        report["checks"]["unknown_zero_motion"] = unknown

        post("/api/command", {"text": "把红色方块放进蓝色托盘", "mode": "color"})
        wait(lambda s: s["busy"] and s["steps"] >= 5)
        code, duplicate = request("/api/command", {"text": "把红色方块放进蓝色托盘", "mode": "color"})
        assert code in (400, 409, 429), (code, duplicate)
        report["checks"]["busy_rejected"] = {"status": code, "body": duplicate}
        post("/api/pause", {})
        paused = wait(lambda s: s["phase"] == "paused")
        time.sleep(.5)
        still = state()
        assert still["phase"] == "paused" and still["steps"] == paused["steps"], (paused, still)
        post("/api/pause", {})
        resumed = wait(lambda s: s["steps"] > paused["steps"])
        report["checks"]["pause_resume"] = {"paused_steps": paused["steps"], "after_half_second": still["steps"], "resumed_steps": resumed["steps"]}
        post("/api/stop", {})
        stopped = wait(lambda s: not s["busy"])
        assert stopped["phase"] == "stopped", stopped
        report["checks"]["stop"] = stopped

        reset(42)
        post("/api/command", {"text": "把红色方块放进蓝色托盘", "mode": "color"})
        completed = wait(lambda s: not s["busy"], 120)
        assert completed["last_result"].get("success") is True, completed
        assert "颜色" in completed["detection_source"], completed
        report["checks"]["cube_complete"] = completed
        report["passed"] = True
    except Exception as exc:
        report["error"] = str(exc)
        raise
    finally:
        try:
            if state()["busy"]:
                post("/api/stop", {})
                wait(lambda s: not s["busy"], 125)
            final = reset(42)
            report["final_state"] = final
        except Exception as exc:
            report["cleanup_error"] = str(exc)
            report["passed"] = False
        report["finished_unix"] = time.time()
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({"passed": report["passed"], "checks": list(report["checks"]), "result": str(output)}, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
