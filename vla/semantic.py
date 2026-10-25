"""Model-selected pick/place intent and image regions; no scene object names."""
import json
import math


TASK_PROMPT = """你是桌面机械臂的视觉任务解析器。根据用户指令和当前图像选择一个被拿起的物体和一个目的地。
仅支持一次抓取并放置；不支持的任务、歧义、目标不存在、否定或条件命令返回 null。不能擅自换目标。
严格输出 JSON，不要解释：
{"source":{"label":"英文物体描述","bbox_2d":[x1,y1,x2,y2]},"destination":{"label":"英文目的地描述","bbox_2d":[x1,y1,x2,y2]},"relation":"in或on","english":"忠实翻译原始指令的英文"}
两个框都使用0到1000归一化xyxy坐标。源物体是被搬运的东西，目的地是指令指定的容器或支撑面；二者不可混淆。
从容器取出时，如果用户没指定新位置，目的地选图中空的桌面区域并使用 on。
不要输出物体的三维坐标，不要计划轨迹。用户指令：
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
