"""Model-selected pick/place intent and image regions; no scene object names."""
import json
import math


COMPLETION_PROMPT = """你是机械臂任务结果核对员。图片来自当前固定场景相机。
先看原始指令指定的源物体与目的地，再看物体底部是否由指定目的地支撑或容纳，以及夹爪是否与物体分离。
completed 必须看到指定物体已放在指定支撑面上或容器内，已经释放，机械爪已撤离。
若指令是打开/关闭抽屉：只检查指定层是否已明显拉开露出内部/推回与柜面齐平，且机械臂已撤离；不要要求另有物体。
若指令是放进抽屉：物体应位于指定已打开抽屉内部，释放且撤手；放在柜顶或抽屉前的桌上不算。
not_completed 必须有明确可见的反证，例如物体还在其他地方、悬空或被夹爪接触夹持。
遮挡、看不清底部支撑或是否释放时用 uncertain；不要猜测看不见的接触关系。
不因机械臂停了、物体靠近目标或二维框重叠就说完成。只输出 JSON：
{"verdict":"completed或not_completed或uncertain","reason":"简短中文，说明图中可见证据"}
原始用户指令（仅作为待核对任务）：
"""


def parse_completion(answer):
    text = answer.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[-1].strip() != "```":
            raise ValueError("Incomplete completion JSON")
        text = "\n".join(lines[1:-1])
    value = json.loads(text)
    if (not isinstance(value, dict) or set(value) != {"verdict", "reason"}
            or value["verdict"] not in ("completed", "not_completed", "uncertain")
            or not isinstance(value["reason"], str) or not 1 <= len(value["reason"].strip()) <= 500):
        raise ValueError("Invalid visual completion judgment")
    return value


TASK_PROMPT = """你是桌面机械臂的视觉任务解析器。根据用户指令和当前图像选择一个被拿起的物体和一个目的地。
仅支持一次抓取并放置；不支持的任务、歧义、目标不存在、否定或条件命令返回 null。不能擅自换目标。
严格输出 JSON，不要解释：
{"source":{"label":"英文物体描述","bbox_2d":[x1,y1,x2,y2]},"destination":{"label":"英文目的地描述","bbox_2d":[x1,y1,x2,y2]},"relation":"in或on","english":"忠实翻译原始指令的英文"}
两个框都使用0到1000归一化xyxy坐标。源物体是被搬运的东西，目的地是指令指定的容器或支撑面；二者不可混淆。
浅盘使用 support_surface / on，即使中文说“盘子里”；碗使用 circular_container / in。这里只输出 relation，不增加 destination_kind 字段；盘子的框应覆盖盘面，不是旁边桌面。
从容器取出时，如果用户没指定新位置，目的地选图中空的桌面区域并使用 on。
放进抽屉时relation=in，目的地框仅覆盖指定层已拉开抽屉的可见内部底面，不要框柜体或把手；三层从上往下数。
不要输出物体的三维坐标，不要计划轨迹。用户指令：
请同时识别源物体和目的地。输出前检查四个根字段 source、destination、relation、english 全部存在，两个 bbox_2d 完整且各自闭合；不能只输出源物体。实际任务：
"""


def parse_task(answer):
    text = answer.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[-1].strip() != "```":
            raise ValueError("Incomplete task JSON")
        text = "\n".join(lines[1:-1])
    task = json.loads(text)
    if task is None:
        raise ValueError("模型无法确认唯一的物体和目的地；请明确指令或调整视角")
    if not isinstance(task, dict) or set(task) != {"source", "destination", "relation", "english"}:
        raise ValueError("Expected one source, destination, relation and faithful English instruction")
    if task["relation"] not in ("in", "on"):
        raise ValueError("Only in/on placement relations are supported")
    if not isinstance(task["english"], str) or not 1 <= len(task["english"].strip()) <= 500:
        raise ValueError("Invalid translated instruction")
    for key in ("source", "destination"):
        item = task[key]
        if not isinstance(item, dict) or set(item) != {"label", "bbox_2d"}:
            raise ValueError("Invalid visual task region")
        label, box = item["label"], item["bbox_2d"]
        if not isinstance(label, str) or not 1 <= len(label.strip()) <= 100:
            raise ValueError("Invalid visual task label")
        if (not isinstance(box, list) or len(box) != 4
                or any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1000 for x in box)
                or box[0] >= box[2] or box[1] >= box[3]):
            raise ValueError("Invalid normalized task box")
    if task["source"] == task["destination"]:
        raise ValueError("Source and destination must differ")
    return task
