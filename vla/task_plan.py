"""Bounded whole-request contract; this is not an arbitrary code/tool executor."""
import json


PLAN_PROMPT = """把中文指令分成按顺序执行的步骤，只输出 JSON。你负责语言分解，不负责判断当前物体能否抓到。
技能表：pick_place = 拿起物体并放到指定位置；unsupported = 技能表之外的动作。
目的地类型：support_surface = 桌面、柜顶、灶台等支撑表面；circular_container = 碗、盘子等开放圆形容器；unsupported = 抽屉内部、封闭柜内等。
开关抽屉/柜门、倾倒、切割、递给人都不支持。不能删掉这些步骤。
整体 status：全部步骤都是抓放且目的地支持，则 ready；含不支持步骤则 unsupported；没有说清对象或目的地名称才 clarify。
不要询问初始位置或可抓取性，也不要询问柜顶、桌面、灶台是不是支撑面；实际定位由执行器在各步开始时完成。
每步只包含一个动作，代词必须换成对应物体名，保留原文目的地。最多8步。
字段固定为 status, reason, missing_skills, steps。每步字段固定为 skill, instruction, destination_kind。
skill 只能是 pick_place 或 unsupported，不能填 ready 或 clarify。
示例输入：先把苹果放到桌面，再把它放进碗里
示例输出：{"status":"ready","reason":"两步抓放","missing_skills":[],"steps":[{"skill":"pick_place","instruction":"把苹果放到桌面","destination_kind":"support_surface"},{"skill":"pick_place","instruction":"把苹果放进碗里","destination_kind":"circular_container"}]}
示例输入：打开抽屉，把积木放进去，然后关上抽屉
示例输出：{"status":"unsupported","reason":"缺少抽屉操作技能","missing_skills":["打开抽屉","放进抽屉","关闭抽屉"],"steps":[{"skill":"unsupported","instruction":"打开抽屉","destination_kind":"unsupported"},{"skill":"unsupported","instruction":"把积木放进抽屉","destination_kind":"unsupported"},{"skill":"unsupported","instruction":"关闭抽屉","destination_kind":"unsupported"}]}
示例输入：把这个放过去
示例输出：{"status":"clarify","reason":"请说明物体和目的地名称","missing_skills":[],"steps":[]}
若附有执行失败日志，reason 给出针对失败的建议，仍保留原指令的步骤，不能改成另一任务。不输出代码、坐标或 Markdown。
现在处理这条用户指令：
"""


def parse_plan(answer):
    value = json.loads(answer)
    if not isinstance(value, dict) or set(value) != {"status", "reason", "missing_skills", "steps"}:
        raise ValueError("任务计划字段不完整")
    status, steps, missing = value["status"], value["steps"], value["missing_skills"]
    if status not in ("ready", "clarify", "unsupported"):
        raise ValueError("无效的计划状态")
    if not isinstance(value["reason"], str) or not 1 <= len(value["reason"].strip()) <= 500:
        raise ValueError("计划必须说明原因")
    if (not isinstance(missing, list) or len(missing) > 16
            or any(not isinstance(s, str) or not 1 <= len(s.strip()) <= 100 for s in missing)):
        raise ValueError("无效的缺失技能列表")
    if not isinstance(steps, list) or len(steps) > 8 or (status == "ready" and not steps):
        raise ValueError("计划须包含1～8个可执行步骤")
    for step in steps:
        if not isinstance(step, dict) or set(step) != {"skill", "instruction", "destination_kind"}:
            raise ValueError("无效的任务步骤字段")
        if step["skill"] not in ("pick_place", "unsupported"):
            raise ValueError("未知技能")
        if step["destination_kind"] not in ("support_surface", "circular_container", "unsupported"):
            raise ValueError("未知目的地类型")
        if not isinstance(step["instruction"], str) or not 1 <= len(step["instruction"].strip()) <= 500:
            raise ValueError("步骤指令不能为空")
        if status == "ready" and (step["skill"] != "pick_place" or step["destination_kind"] == "unsupported"):
            raise ValueError("ready 计划含有不支持的技能")
    if status == "ready" and missing or status == "unsupported" and not missing:
        raise ValueError("计划状态与缺失技能矛盾")
    return value
