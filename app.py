"""Native desktop or local web control; simulation belongs to one worker."""
import argparse
import base64
import json
import logging
import math
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
LOG = logging.getLogger("desktop")


class Lab:
    def __init__(self, model_url, native=False):
        self.model_url = model_url.rstrip("/")
        self.native = native
        self.closed = threading.Event()
        self.viewer = None
        self.lock = threading.RLock()
        self.jobs = queue.Queue()
        self.stop = threading.Event()
        self.pause = threading.Event()
        self.generation = 0
        self.vision_ready = False
        self.frames = {}
        self.state = dict(phase="loading", message="正在启动仿真", busy=True, steps=0, mode="auto", detection_source="颜色 RGB-D 基线",
                          last_result=None, display="native" if native else "web",
                          models={"asr": "未连接", "vision": "颜色 RGB-D 基线（非通用识别）"}, events=[])

    def update(self, **values):
        with self.lock:
            self.state.update(values)

    def event(self, message):
        with self.lock:
            self.state["events"] = (self.state["events"] + [message])[-40:]

    def snapshot(self):
        with self.lock:
            return json.loads(json.dumps(self.state))

    def reserve(self, phase, mode=None):
        with self.lock:
            if self.state["busy"]:
                raise ValueError("当前任务尚未结束，请等待或先停止")
            if self.state["phase"] == "error":
                raise ValueError("仿真未就绪，请检查服务日志")
            mode = self.state["mode"] if mode is None else mode
            if mode not in ("auto", "color", "vlm"):
                raise ValueError("视觉模式须为 auto、color 或 vlm")
            self.state["mode"] = mode
            self.stop.clear()
            self.pause.clear()
            self.generation += 1
            self.state.update(busy=True, phase=phase, steps=0, last_result=None)
            return self.generation

    def model_health(self):
        health = {}
        try:
            with urlopen(self.model_url + "/health", timeout=.5) as res:
                health = json.load(res)
            asr = "本地 Qwen ASR 已就绪" if health.get("asr") else "语音模型未就绪"
        except Exception:
            asr = "语音服务未连接"
        with self.lock:
            self.state["models"]["asr"] = asr
            self.vision_ready = bool(health.get("vision"))
            self.state["models"]["vision"] = "本地 Qwen VLM 已就绪" if self.vision_ready else "VLM 未就绪；可使用颜色 RGB-D 基线"

    def model_request(self, endpoint, payload):
        request = Request(self.model_url + endpoint, data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=120) as response:
                return json.load(response)
        except Exception as exc:
            # Keep the model's original error body when the HTTP service provides one.
            detail = str(exc)
            if hasattr(exc, "read"):
                detail += ": " + exc.read(4096).decode(errors="replace")
            raise ValueError("VLM 定位失败：" + detail) from exc

    def visual_hooks(self, env, text, cv2):
        from perception import localize, validate_command, parse_command

        def resolve(command):
            lowered = validate_command(command)
            if any(word in lowered for word in ("真机", "撞", "攻击", "砸", "打人", "碰人", "刺", "桌外", "真人", "real robot")):
                raise ValueError("仅支持仿真桌面内单个目标放进托盘或从盘子取出，请重新描述")
            if any(word in lowered for word in ("分别", "依次", "所有", "全部", "以及", "和", " and ")):
                raise ValueError("请一次指定一个要抓取的物体")
            # Known assets can check colour identity inside a VLM box; unknown
            # names still use the general, ambiguity-checked RGB-D path.
            try:
                return parse_command(command)
            except ValueError:
                return command

        def locate(label):
            if self.stop.is_set():
                raise InterruptedError("用户停止")
            rgb, depth = env.rgbd("overhead", size=480)
            ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if not ok:
                raise ValueError("VLM 输入图像编码失败")
            payload = {"image": base64.b64encode(encoded).decode()}
            if label == "tray":
                payload["target"] = "蓝色托盘"
                result = self.model_request("/locate", payload)
            else:
                payload["text"] = text
                result = self.model_request("/ground", payload)
            if self.stop.is_set():
                raise InterruptedError("用户停止")
            box = result.get("bbox_2d")
            if not box:
                raise ValueError("VLM 未找到明确目标，请澄清物体或重置布局")
            if (not isinstance(box, list) or len(box) != 4
                    or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1000 for v in box)
                    or not box[0] < box[2] or not box[1] < box[3]):
                raise ValueError("VLM 返回了无效目标框")
            found = str(result.get("label", "" )).lower()
            if label != "tray" and any(word in found for word in ("tray", "托盘", "盘子")):
                raise ValueError("VLM 将放置容器识别为抓取目标，请澄清物体")
            pixels = [int(box[0]*480/1000), int(box[1]*480/1000),
                      min(480, math.ceil(box[2]*480/1000)), min(480, math.ceil(box[3]*480/1000))]
            self.event("VLM 检测：" + (result.get("label") or label))
            return localize(rgb, depth, *env.calibration("overhead", size=480), label, bbox=pixels)
        return resolve, locate

    def run(self):
        env = None
        viewer = None
        try:
            import cv2
            from scene import DesktopScene
            from controller import Controller

            env = DesktopScene()
            env.reset()

            if self.native:
                import mujoco.viewer

                def keypress(key):
                    if key == 32 and self.snapshot()["busy"]:
                        self.pause.clear() if self.pause.is_set() else self.pause.set()
                    elif key == ord("S"):
                        self.stop.set()
                        self.pause.clear()

                viewer = mujoco.viewer.launch_passive(
                    env.sim.model._model, env.sim.data._data, key_callback=keypress,
                    show_left_ui=False, show_right_ui=False,
                )
                self.viewer = viewer
                with viewer.lock():
                    viewer.cam.lookat[:] = [0, 0, .85]
                    viewer.cam.distance = 1.8
                    viewer.cam.azimuth = 140
                    viewer.cam.elevation = -32
                    viewer.opt.geomgroup[0] = 0
                    viewer.opt.geomgroup[1] = 1

            def render():
                if viewer is not None:
                    if not viewer.is_running():
                        self.closed.set()
                        self.stop.set()
                        raise InterruptedError("MuJoCo 窗口已关闭")
                    viewer.sync()
                    with viewer.lock():
                        camera = dict(azimuth=float(viewer.cam.azimuth),
                                      elevation=float(viewer.cam.elevation),
                                      distance=float(viewer.cam.distance),
                                      lookat=viewer.cam.lookat.tolist())
                    self.update(camera=camera)
                    return
                images = {}
                for view, camera in (("scene", "agentview"), ("top", "overhead"), ("wrist", "robot0_eye_in_hand")):
                    rgb, _ = env.rgbd(camera, size=480)
                    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
                    if not ok:
                        raise RuntimeError("JPEG 编码失败")
                    images[view] = encoded.tobytes()
                with self.lock:
                    self.frames = images

            next_step = time.monotonic()

            def callback(obs, phase, steps):
                nonlocal next_step
                if self.stop.is_set() or self.closed.is_set():
                    raise InterruptedError("用户停止")
                self.update(phase="paused" if self.pause.is_set() else phase, steps=int(steps))
                paused = self.pause.is_set()
                while self.pause.is_set():
                    if viewer is not None:
                        render()
                    if self.stop.wait(.05):
                        raise InterruptedError("用户停止")
                if paused:
                    next_step = time.monotonic()
                next_step = max(next_step + .05, time.monotonic())
                if self.stop.wait(max(0, next_step-time.monotonic())):
                    raise InterruptedError("用户停止")
                if self.pause.is_set():
                    return callback(obs, phase, steps)
                self.update(phase=phase)
                if viewer is not None or steps % 3 == 0:
                    render()

            render()
            self.update(busy=False, phase="idle", message="仿真就绪，请输入指令")
            self.event("桌面仿真已就绪；当前使用颜色 RGB-D 定位")
            last_health = 0
            while not self.closed.is_set():
                if viewer is not None:
                    render()
                if time.monotonic() - last_health > 5:
                    self.model_health()
                    last_health = time.monotonic()
                try:
                    kind, payload, token = self.jobs.get(timeout=.03 if viewer is not None else .2)
                except queue.Empty:
                    continue
                started = time.monotonic()
                try:
                    if self.stop.is_set() or token != self.generation:
                        raise InterruptedError("用户停止")
                    if kind == "reset":
                        env.seed = payload
                        env.reset()
                        result = {"message": "已生成新的随机布局", "seed": payload}
                    else:
                        text, mode = payload["text"], payload["mode"]
                        self.model_health()
                        use_vlm = mode == "vlm" or (mode == "auto" and self.vision_ready)
                        if use_vlm and not self.vision_ready:
                            raise ValueError("VLM 未就绪，请等待模型或明确选择颜色基线")
                        source = "Qwen VLM + RGB-D" if use_vlm else "颜色 RGB-D 基线（非通用识别）"
                        self.update(phase="planning", message=text, detection_source=source)
                        self.event("收到指令：" + text + "；定位来源：" + source)
                        resolve, locate = self.visual_hooks(env, text, cv2) if use_vlm else (None, None)
                        next_step = time.monotonic()
                        result = Controller(env, step_callback=callback, locate=locate, resolve=resolve).run(text)
                        if not isinstance(result, dict):
                            result = {"message": str(result)}
                    result["elapsed_s"] = round(time.monotonic() - started, 2)
                    render()
                    self.update(phase="failed" if result.get("success") is False else "done",
                                message=result.get("message", "执行结束"), last_result=result)
                except InterruptedError:
                    if not self.closed.is_set():
                        render()
                    self.update(phase="stopped", message="已停止", last_result={"message": "用户停止", "elapsed_s": round(time.monotonic()-started, 2)})
                except ValueError as exc:
                    self.update(phase="clarify", message=str(exc), last_result={"message": str(exc)})
                    self.event(str(exc))
                except Exception as exc:
                    LOG.exception("Simulation command failed")
                    self.update(phase="failed", message=str(exc), last_result={"message": str(exc)})
                finally:
                    self.pause.clear()
                    self.update(busy=False)
                    self.jobs.task_done()
        except InterruptedError:
            pass
        except Exception as exc:
            LOG.exception("Simulation worker failed")
            self.update(busy=False, phase="error", message="仿真启动失败：" + str(exc))
        finally:
            if self.native:
                self.closed.set()
            if viewer is not None:
                viewer.close()
            if env is not None:
                env.close()


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(35)

    def respond(self, status, data, mime="application/json; charset=utf-8"):
        if not isinstance(data, bytes):
            data = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self'; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(data)

    def local_request(self):
        # Host check also prevents DNS rebinding. Browser POSTs must be same-origin.
        allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        if self.headers.get("Host") not in allowed:
            return False
        origin = self.headers.get("Origin")
        if origin and origin not in {"http://" + host for host in allowed}:
            return False
        return self.headers.get("Sec-Fetch-Site", "same-origin") not in ("cross-site", "same-site")

    def do_GET(self):
        if not self.local_request():
            return self.respond(403, {"error": "仅允许本机同源访问"})
        parsed = urlparse(self.path)
        if parsed.path == "/":
            if self.server.lab.native:
                return self.respond(200, {"message": "请使用 MuJoCo 原生窗口和中文桌面面板", "display": "native"})
            return self.respond(200, (ROOT / "web/index.html").read_bytes(), "text/html; charset=utf-8")
        if parsed.path == "/api/state":
            return self.respond(200, self.server.lab.snapshot())
        if parsed.path == "/api/frame":
            view = parse_qs(parsed.query).get("view", ["scene"])[0]
            with self.server.lab.lock:
                frame = self.server.lab.frames.get(view)
            return self.respond(200, frame, "image/jpeg") if frame else self.respond(503, {"error": "画面尚未就绪"})
        self.respond(404, {"error": "接口不存在"})

    def do_POST(self):
        if not self.local_request():
            return self.respond(403, {"error": "仅允许本机同源访问"})
        lab = self.server.lab
        path = urlparse(self.path).path
        if path not in ("/api/command", "/api/reset", "/api/pause", "/api/stop", "/api/audio", "/api/mode"):
            return self.respond(404, {"error": "接口不存在"})
        token = None
        try:
            length = int(self.headers.get("Content-Length", "0"))
            limit = 8*1024*1024 if path == "/api/audio" else 4096
            if not 0 < length <= limit or self.headers.get("Transfer-Encoding"):
                return self.respond(413, {"error": "请求大小不符合要求"})
            expected = "audio/wav" if path == "/api/audio" else "application/json"
            if self.headers.get("Content-Type", "").split(";")[0].strip() != expected:
                return self.respond(415, {"error": "请求类型不符合要求"})
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError("请求不完整")
            data = {} if path == "/api/audio" else json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("请求须为 JSON 对象")
            if path == "/api/mode":
                with lab.lock:
                    if lab.state["busy"]:
                        raise ValueError("任务运行时不能切换视觉模式")
                    if data.get("mode") not in ("auto", "color", "vlm"):
                        raise ValueError("视觉模式须为 auto、color 或 vlm")
                    lab.state["mode"] = data["mode"]
                return self.respond(200, {"mode": data["mode"], "message": "视觉模式已切换"})
            if path == "/api/stop":
                with lab.lock:
                    lab.stop.set()
                    lab.pause.clear()
                    lab.generation += 1
                return self.respond(200, {"message": "已请求停止"})
            if path == "/api/pause":
                with lab.lock:
                    if not lab.state["busy"]:
                        raise ValueError("当前没有运行中的任务")
                    if lab.pause.is_set():
                        lab.pause.clear()
                    else:
                        lab.pause.set()
                return self.respond(200, {"message": "已切换暂停状态"})
            if path == "/api/reset":
                seed = data.get("seed")
                if type(seed) is not int or not 0 <= seed < 2**31:
                    raise ValueError("seed 应为非负整数")
                token = lab.reserve("resetting", data.get("mode"))
                lab.jobs.put(("reset", seed, token))
                return self.respond(202, {"message": "正在重置布局"})
            if path == "/api/command":
                text = data.get("text")
                if not isinstance(text, str) or not text.strip() or len(text) > 300:
                    raise ValueError("请提供 1 至 300 字的指令")
                token = lab.reserve("planning", data.get("mode"))
            else:
                if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
                    raise ValueError("需要 WAV 音频")
                token = lab.reserve("transcribing", self.headers.get("X-Vision-Mode"))
                lab.update(message="正在识别语音")
                request = Request(lab.model_url + "/transcribe", data=raw, headers={"Content-Type": "audio/wav"})
                with urlopen(request, timeout=120) as response:
                    text = json.load(response).get("text")
                if not isinstance(text, str) or not text.strip() or len(text) > 300:
                    raise ValueError("没有识别出有效指令，请重新录音")
            with lab.lock:
                if token != lab.generation or lab.stop.is_set():
                    lab.update(busy=False, phase="stopped", message="已取消指令")
                    return self.respond(409, {"error": "指令已停止"})
                lab.jobs.put(("command", {"text": text.strip(), "mode": lab.state["mode"]}, token))
            self.respond(202, {"text": text.strip(), "message": "指令已加入执行队列"})
        except (ValueError, UnicodeError) as exc:
            if token is not None:
                lab.update(busy=False, phase="clarify", message=str(exc))
            self.respond(400, {"error": str(exc)})
        except Exception as exc:
            LOG.exception("Request failed")
            if token is not None:
                lab.update(busy=False, phase="failed", message="语音服务失败，请使用文字指令")
            self.respond(502, {"error": str(exc)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8910)
    parser.add_argument("--model-url", default="http://127.0.0.1:8911")
    parser.add_argument("--native", action="store_true", help="MuJoCo 原生视角与中文桌面面板")
    args = parser.parse_args()
    endpoint = urlparse(args.model_url)
    if endpoint.scheme != "http" or endpoint.hostname not in ("127.0.0.1", "localhost"):
        parser.error("model-url must be a local HTTP address")
    logging.basicConfig(level=logging.INFO)
    if args.native:
        # robosuite's Linux GLX setting selects GLFW offscreen contexts; EGL is
        # kept for the optional headless web and evaluation entry points.
        os.environ["MUJOCO_GL"] = "glx"
    lab = Lab(args.model_url, native=args.native)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    server.lab = lab
    worker = threading.Thread(target=lab.run, name="simulation", daemon=True)
    worker.start()
    panel = None
    if args.native:
        panel = subprocess.Popen(["/usr/bin/python3", str(ROOT / "desktop_panel.py"),
                                  "--url", f"http://127.0.0.1:{args.port}"])

        def supervise():
            while not lab.closed.wait(.1):
                if panel.poll() is not None or (lab.viewer is not None and not lab.viewer.is_running()):
                    break
            lab.closed.set()
            lab.stop.set()
            lab.pause.clear()
            server.shutdown()

        threading.Thread(target=supervise, name="desktop-lifecycle", daemon=True).start()
    LOG.info("%s simulation control on localhost:%s", "Native MuJoCo" if args.native else "Web", args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        lab.stop.set()
        lab.pause.clear()
        lab.closed.set()
        server.server_close()
        worker.join(timeout=5)
        if panel is not None and panel.poll() is None:
            panel.terminate()


if __name__ == "__main__":
    main()
