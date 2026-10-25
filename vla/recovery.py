"""Scene-grounded recovery planning using executable skills, never arbitrary code."""
import json

from vla.task_plan import parse_plan


RECOVERY_PROMPT = """你是机械臂恢复规划员。比较图1（上次行动前）和图2（当前现场），结合历史日志规划接下来一个实际动作，只输出JSON。
动作：retry_grasp=物体未被夹住/掉落，张爪撤离后重新观察并抓取；resume_place=物体仍稳稳夹住，保留夹持重新定位目的地继续放置；reverify=已完成动作但视觉遮挡/核验不确定，只重新观察核验；retry_drawer=抽屉动作未成功，重新定位把手再开/关；stop=无法安全恢复。
prepare=原步骤缺少前置条件时，先执行一个必要子目标。step使用技能表 pick_place/open_drawer/close_drawer 和目的地 support_surface/circular_container/drawer_interior，字段 skill,instruction,destination_kind。只选择为原目标必要的动作，不能随意挪其他物体或换最终目的地。不要重复已成功的前置步骤。
prepare不能与日志instruction中的当前目标相同；前置步骤核验成功后，系统自动继续既有原目标，不需要再次prepare原任务。历史中continue_original表示此自动续接，其他动作均由模型选择。
先检查原目标的目的地是否可用：如果要放进的抽屉仍关着，即使物体尚未夹起，也应prepare打开对应抽屉，不能只重复抓取。目的地已准备好而抓取失败，才retry_grasp；它包含重新定位、抓取和放置的完整原步骤，即使之前尚未抓过也可使用。
例如要放进抽屉但抽屉还关着，且手中没有物体，选择prepare打开同一层抽屉；打开后再继续原来的放置目标。关闭抽屉时如果原物体突出，可prepare将原物体重新摆进原抽屉，之后继续关闭。
必须依据当前画面选择，不要换目标或重复已完成步骤。没有grasp_context时不能resume_place；开关抽屉的直接重试用retry_drawer，缺少前置条件时才用prepare。
日志中的allowed_actions是本次允许的动作集合，必须仅从该集合选择。possibly_holding_object=true时保留夹持，若目的地仍可见通常选resume_place继续放置，不能重新抓取导致空中松手。
物体已释放在错误支撑面上时选retry_grasp；仍被持有时不得retry_grasp导致空中松手。当前一步尚未成功时不能跳到后续目标。无法看清可以reverify；相同失败反复发生应改变方案或stop，并解释缺少的能力。
progress判断上次行动是否让物体状态/目标条件更接近完成：progress/no_progress/uncertain。只看到机械臂移动、时间过去、像素抖动不算进展；要说明具体物体或抽屉的变化。没有证据用uncertain。这个字段不代表任务成功，完成仍需单独核验。
不输出坐标、程序、额外字段。字段固定 action,reason,progress,step。非prepare动作step必须null。
格式：{"action":"动作名","reason":"中文说明观察到的变化、失败原因和下一步","progress":"no_progress","step":null}
原始步骤及失败日志：
"""


def parse_recovery(answer):
    value = json.loads(answer)
    if (not isinstance(value, dict) or set(value) not in ({"action", "reason"}, {"action", "reason", "progress", "step"})
            or value["action"] not in ("retry_grasp", "resume_place", "reverify", "retry_drawer", "prepare", "stop")
            or not isinstance(value["reason"], str) or not 1 <= len(value["reason"].strip()) <= 500):
        raise ValueError("Invalid recovery decision")
    if 'progress' in value and value['progress'] not in ('progress', 'no_progress', 'uncertain'):
        raise ValueError('Invalid recovery progress')
    if value['action'] == 'prepare':
        if 'step' not in value:
            raise ValueError('Preparation requires an executable step')
        parse_plan(json.dumps(dict(status='ready', reason=value['reason'], missing_skills=[], steps=[value['step']])))
    elif value.get('step') is not None:
        raise ValueError('Only preparation may introduce a step')
    return value
