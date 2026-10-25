"""Whole-request gating and verified leaf execution in the same MuJoCo scene."""
import base64
import hashlib
import json
import time

import numpy as np

from vla.semantic_grasp import image_bytes
from vla.task_plan import parse_plan
from vla.recovery import parse_recovery
from vla.drawer_control import DrawerControl
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

    def recovery_decision(self, text, port, number, step, previous, attempt):
        self.status('模型正在看失败现场，选择并执行恢复动作')
        rgb, _ = self.capture()
        raw = image_bytes(rgb)
        name = f'recovery-{number:02d}-{attempt}'
        (self.output / f'{name}-input.png').write_bytes(raw)
        failure = dict(step=number, instruction=step, completed_steps=self.report['completed_steps'],
            result=previous.get('result'), exception=previous.get('exception'),
            completion=previous.get('completion'), hold_check=previous.get('hold_check'),
            drawer_depth_check=previous.get('drawer_depth_check'),
            grasp_context=bool(previous.get('grasp_context')),
            motion_completed=previous.get('motion_completed', False))
        allowed = ({'retry_drawer', 'reverify', 'stop'} if step['skill'] != 'pick_place'
                   else {'retry_grasp', 'reverify', 'stop'})
        if previous.get('grasp_context'):
            allowed.add('resume_place')
            gap = float(np.abs(self.obs.get('robot0_gripper_qpos', [0., 0.])).sum())
            failure['current_gripper_gap'] = gap
            if not previous.get('motion_completed') and .003 < gap < .075:
                allowed.discard('retry_grasp')
                failure['possibly_holding_object'] = True
        failure['allowed_actions'] = sorted(allowed)
        response = self.request(port, 'recovery', dict(text=text, failure=failure,
            image=base64.b64encode(raw).decode()))
        (self.output / f'{name}-model.json').write_text(json.dumps(response, ensure_ascii=False, indent=2))
        if response.get('image_sha256') != hashlib.sha256(raw).hexdigest():
            raise ValueError('Recovery image does not match current observation')
        decision = parse_recovery(json.dumps(response['recovery'], ensure_ascii=False))
        if decision['action'] not in allowed:
            raise ValueError('恢复动作不适用于原始技能')
        if decision['action'] == 'resume_place' and not previous.get('grasp_context'):
            raise ValueError('没有持有上下文，拒绝继续放置')
        entry = dict(step=number, attempt=attempt, decision=decision, executed=False)
        self.report['recoveries'].append(entry)
        return entry

    def execute_leaf(self, step, number, port, prefix, recovery=None, previous=None):
        action = recovery['decision']['action'] if recovery else None
        cls = DrawerControl if step['skill'] != 'pick_place' and action != 'reverify' else VisualControl
        suffix = f"-recovery-{recovery['attempt']}" if recovery else ''
        leaf = cls(self.env, self.obs, None, self.output / f'step-{number:02d}{suffix}',
            advance=self.advance, stop=self.stop, heartbeat=self.heartbeat, on_task=self.on_task,
            status=lambda message: self.status(prefix + message))
        try:
            if self.stop():
                raise InterruptedError('用户已暂停')
            if recovery:
                recovery['attempted'] = True
            if action in ('retry_grasp', 'resume_place', 'reverify'):
                leaf.recover(step['instruction'], action, previous, task_port=port,
                             destination_kind=step['destination_kind'])
            elif step['skill'] != 'pick_place':
                leaf.run(step['instruction'], task_port=port, skill=step['skill'])
            else:
                leaf.run(step['instruction'], task_port=port, destination_kind=step['destination_kind'])
        except InterruptedError:
            raise
        except Exception as exc:
            leaf.report.update(result='exception', semantic_success=None,
                               exception=f'{type(exc).__name__}: {exc}')
        finally:
            if recovery:
                recovery.update(executed=bool(leaf.trace) or action == 'reverify',
                                control_steps=len(leaf.trace), result=leaf.report.get('result'))
            self.obs = leaf.obs
            self.trace.extend(leaf.trace)
            self.report['inference_calls'] += leaf.report['inference_calls']
            self.report['steps'].append(dict(number=number, instruction=step['instruction'],
                recovery=action, output=str(leaf.output), report=leaf.report))
        return leaf.report

    def run(self, text, task_port=8911):
        started = time.monotonic()
        self.report.update(chinese=text, result="planning", recoveries=[])
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
                previous = self.execute_leaf(step, number, task_port, prefix)
                for attempt in range(1, 3):
                    if previous.get('semantic_success') is True:
                        break
                    if self.stop():
                        raise InterruptedError('用户已暂停')
                    try:
                        recovery = self.recovery_decision(text, task_port, number, step, previous, attempt)
                        if recovery['decision']['action'] == 'stop':
                            break
                        previous = self.execute_leaf(step, number, task_port, prefix, recovery, previous)
                    except InterruptedError:
                        raise
                    except Exception as exc:
                        self.report['recovery_error'] = str(exc)
                        break
                success = previous.get('semantic_success')
                if success is not True:
                    self.report.update(result="step_failed" if success is False else "step_unconfirmed",
                        semantic_success=success, completion=previous.get("completion") or
                        {"reason": previous.get("exception", "恢复后仍未确认完成，保留现场")})
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
