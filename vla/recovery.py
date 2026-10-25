"""Model choices for bounded, observable recovery; never arbitrary code."""
import json


RECOVERY_PROMPT = """你是机械臂恢复规划员。查看刚拍的失败现场和执行日志，为原始步骤选择一个实际动作，只输出JSON。
动作：retry_grasp=物体未被夹住/掉落，张爪撤离后重新观察并抓取；resume_place=物体仍稳稳夹住，保留夹持重新定位目的地继续放置；reverify=已完成动作但视觉遮挡/核验不确定，只重新观察核验；retry_drawer=抽屉动作未成功，重新定位把手再开/关；stop=无法安全恢复。
必须依据当前画面选择，不要换目标或重复已完成步骤。没有grasp_context时不能resume_place；开关抽屉只选retry_drawer/reverify/stop。
日志中的allowed_actions是本次允许的动作集合，必须仅从该集合选择。possibly_holding_object=true时保留夹持，若目的地仍可见通常选resume_place继续放置，不能重新抓取导致空中松手。
物体已释放在错误支撑面上时选retry_grasp；仍被持有时不得retry_grasp导致空中松手。不输出坐标、程序、额外字段。
格式：{"action":"动作名","reason":"中文说明现场证据和下一步"}
原始步骤及失败日志：
"""


def parse_recovery(answer):
    value = json.loads(answer)
    if (not isinstance(value, dict) or set(value) != {"action", "reason"}
            or value["action"] not in ("retry_grasp", "resume_place", "reverify", "retry_drawer", "stop")
            or not isinstance(value["reason"], str) or not 1 <= len(value["reason"].strip()) <= 500):
        raise ValueError("Invalid recovery decision")
    return value
