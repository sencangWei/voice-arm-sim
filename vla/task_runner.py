"""Whole-request gating and verified leaf execution in the same MuJoCo scene."""
import base64
import hashlib
import json
import time

from vla.semantic_grasp import image_bytes
from vla.task_plan import parse_plan
from vla.visual_control import VisualControl


class TaskRunner(VisualControl):
    def __init__(self, *args, on_plan=lambda plan, step: None, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_plan = on_plan
        self.report.update(backend="Qwen3-VL whole-request plan + verified RGB-D/OSC skills",
                           completed_steps=0, steps=[], control_steps=0)

    def make_plan(self, text, port, failure=None):
        rgb, _ = self.capture()
        raw = image_bytes(rgb)
        name = "failure" if failure else "plan"
        (self.output / f"{name}-input.png").write_bytes(raw)
        payload = dict(text=text, image=base64.b64encode(raw).decode())
        if failure:
            payload["failure"] = failure
        response = self.request(port, "plan", payload)
        (self.output / f"{name}-model.json").write_text(json.dumps(response, ensure_ascii=False, indent=2))
        if response.get("image_sha256") != hashlib.sha256(raw).hexdigest():
            raise ValueError("Plan response image does not match current observation")
        return parse_plan(json.dumps(response["plan"], ensure_ascii=False))

    def failure_feedback(self, text, port, step, report):
        # One advisory model roundtrip. Never automatically retry a physical
        # action with an uncertain held object or an unverified placement.
        self.status("本步未核验通过，已停止；模型正在看失败现场，分析下一步建议")
        failure = dict(step=step, completed_steps=self.report["completed_steps"],
                       result=report.get("result"), exception=report.get("exception"),
                       completion=report.get("completion"))
        try:
            advice = self.make_plan(text, port, failure)
            self.report["failure_feedback"] = dict(plan=advice, executed=False)
        except InterruptedError:
            raise
        except Exception as exc:
            self.report["failure_feedback"] = dict(error=str(exc), executed=False)

    def run(self, text, task_port=8911):
        started = time.monotonic()
        self.report.update(chinese=text, result="planning")
        try:
            self.status("模型正在理解完整指令并检查全部步骤；此时不发送动作")
            plan = self.make_plan(text, task_port)
            self.report["plan"] = plan
            self.on_plan(plan, 0)
            if self.stop():
                raise InterruptedError("用户已暂停；未发送后续动作")
            if plan["status"] != "ready":
                self.report.update(result=plan["status"], completion={"reason": plan["reason"]})
                return self.report
            total = len(plan["steps"])
            for number, step in enumerate(plan["steps"], 1):
                if self.stop():
                    raise InterruptedError("用户已暂停；未发送后续动作")
                self.on_plan(plan, number)
                prefix = f"第 {number}/{total} 步："
                self.status(prefix + step["instruction"])
                if self.stop():
                    raise InterruptedError("用户已暂停；未发送后续动作")
                leaf = VisualControl(self.env, self.obs, None, self.output / f"step-{number:02d}",
                    advance=self.advance, stop=self.stop, heartbeat=self.heartbeat, on_task=self.on_task,
                    status=lambda message, p=prefix: self.status(p + message))
                try:
                    leaf.run(step["instruction"], task_port=task_port,
                             destination_kind=step["destination_kind"])
                except InterruptedError:
                    raise
                except Exception as exc:
                    leaf.report.update(result="exception", semantic_success=None,
                                       exception=f"{type(exc).__name__}: {exc}")
                finally:
                    self.obs = leaf.obs
                    self.trace.extend(leaf.trace)
                    self.report["inference_calls"] += leaf.report["inference_calls"]
                    self.report["steps"].append(dict(number=number, instruction=step["instruction"],
                                                   output=str(leaf.output), report=leaf.report))
                success = leaf.report.get("semantic_success")
                if success is not True:
                    self.report.update(result="step_failed" if success is False else "step_unconfirmed",
                        semantic_success=success, completion=leaf.report.get("completion") or
                        {"reason": leaf.report.get("exception", "本步无法确认完成，未执行后续步骤")})
                    self.failure_feedback(text, task_port, number, leaf.report)
                    return self.report
                self.report["completed_steps"] += 1
            self.report.update(result="visual_completed", semantic_success=True,
                completion={"reason": f"全部 {total} 步均逐步通过视觉核对"})
            return self.report
        except Exception as exc:
            self.report.update(result="interrupted" if isinstance(exc, InterruptedError) else "exception",
                               exception=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            self.report.update(control_steps=len(self.trace), wall_duration_s=time.monotonic() - started)
            (self.output / "result.json").write_text(json.dumps(self.report, ensure_ascii=False, indent=2))
