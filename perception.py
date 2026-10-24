"""Camera-only RGB-D baseline. Explicit colour vocabulary; not an open-vocabulary model."""
from dataclasses import dataclass
import re

import cv2
import numpy as np
from scipy.spatial import cKDTree


TAKE_OUT = re.compile(
    r"(?:请)?(?:帮我)?把(?P<target>[^，,。！!]+?)从(?:蓝色)?(?:的)?(?:托盘|盘子)"
    r"(?:里面|里|中|内|上面|上)?(?:取出来|拿出来|取出|拿出)"
    r"(?:[，,]?(?:放到|放在|放回)(?:桌面|桌子|桌)(?:的空位|空位|上面|上)?)?"
    r"[。！!，,\s]*$"
)


@dataclass
class Detection:
    label: str
    xyz: np.ndarray
    bbox: list
    pixels: int
    height: float
    yaw: float = 0.


def point_cloud(depth, intrinsic, camera_to_world):
    v, u = np.indices(depth.shape)
    rays = np.stack(((u - intrinsic[0, 2]) / intrinsic[0, 0],
                     (v - intrinsic[1, 2]) / intrinsic[1, 1], np.ones_like(u)), axis=-1)
    return (rays * depth[..., None]) @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]


def free_table_position(depth, intrinsic, camera_to_world, source_xy):
    """Select an observed clear table patch, never a simulator object pose."""
    cloud = point_cloud(depth, intrinsic, camera_to_world).reshape(-1, 3)
    cloud = cloud[np.isfinite(cloud).all(axis=1)]
    table = cloud[(cloud[:, 2] > .795) & (cloud[:, 2] < .805)]
    obstacles = cloud[(cloud[:, 2] >= .805) & (cloud[:, 2] < 1.3)]
    if not len(table) or not len(obstacles):
        raise ValueError("相机未确认桌面空位，请移开遮挡后重试")
    visible, occupied = cKDTree(table[:, :2]), cKDTree(obstacles[:, :2])
    candidates = np.array([(x, y) for x in np.arange(-.24, .241, .06)
                           for y in np.arange(-.24, .241, .06)])
    candidates = candidates[np.argsort(np.linalg.norm(candidates - source_xy, axis=1))]
    # Entire 16 cm square must be observed as table, with 11 cm clearance to
    # raised geometry (tray walls, other fruit and arm). This covers the banana.
    offsets = np.array([(x, y) for x in np.linspace(-.08, .08, 5)
                        for y in np.linspace(-.08, .08, 5)])
    for xy in candidates:
        if (occupied.query(xy)[0] > .11
                and visible.query(xy + offsets)[0].max() < .009):
            return np.r_[xy, .8]
    raise ValueError("桌面没有足够大的可见空位，未开始抓取；请清理桌面")


def localize(rgb, depth, intrinsic, camera_to_world, label, bbox=None):
    """If bbox is provided by VLM, use depth foreground within it. Otherwise use HSV."""
    cloud = point_cloud(depth, intrinsic, camera_to_world)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    if label == "cube":
        mask = ((hsv[..., 0] < 10) | (hsv[..., 0] > 170)) & (hsv[..., 1] > 110)
    elif label == "apple":
        mask = (hsv[..., 0] > 32) & (hsv[..., 0] < 85) & (hsv[..., 1] > 100)
    elif label == "banana":
        mask = (hsv[..., 0] > 15) & (hsv[..., 0] < 35) & (hsv[..., 1] > 100)
    elif label == "tray":
        mask = (hsv[..., 0] > 95) & (hsv[..., 0] < 130) & (hsv[..., 1] > 100)
    elif bbox is None:
        raise ValueError("未知目标，请指定可见物体；颜色基线仅支持红方块、绿苹果、黄香蕉、蓝托盘")
    else:
        mask = np.ones(depth.shape, bool)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        region = np.zeros(depth.shape, bool)
        region[y1:y2, x1:x2] = True
        # Known targets retain colour identity inside the language model box.
        mask &= region
    mask &= (cloud[..., 2] > .815) & (cloud[..., 2] < .91)
    mask &= (np.abs(cloud[..., 0]) < .37) & (np.abs(cloud[..., 1]) < .37)
    count, components, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if count < 2 or stats[1:, cv2.CC_STAT_AREA].max() < 30:
        raise ValueError(f"相机里没有清晰找到 {label}，请检查遮挡或重置布局")
    if bbox is not None and label not in ("cube", "apple", "banana", "tray"):
        areas = np.sort(stats[1:, cv2.CC_STAT_AREA])
        if len(areas) > 1 and areas[-2] > max(30, areas[-1] * .15):
            raise ValueError("目标框包含多个前景，无法确定抓取点，请重新描述")
    index = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    points = cloud[components == index]
    xyz = np.median(points, axis=0)
    # Surface points define object top. Grasp centre is chosen from measured visible height.
    top = float(np.percentile(points[:, 2], 80))
    height = max(0., top - .8)
    xyz[2] = top
    x, y, w, h, area = stats[index]
    return Detection(label, xyz, [int(x), int(y), int(x+w), int(y+h)], int(area), height)


def validate_command(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("请说出要抓取的物体和放置位置")
    lowered = text.lower().strip()
    if any(word in lowered for word in ("不要", "别", "取消", "停止", "暂停", "不能", "禁止", "不想", "不抓", "don't", "do not", "stop")):
        raise ValueError("这条包含否定或停止，请用停止按钮；执行前请重新说出明确的肯定指令")
    if any(word in lowered for word in ("如果", "等我", "再说", "假如", "等到")) or re.search(r"\b(?:if|when|unless|wait)\b", lowered):
        raise ValueError("暂不执行带条件或等待要求的指令，请准备好后再下达肯定指令")
    source = re.split(r"从|放|装|\b(?:into|in|on)\b", lowered, maxsplit=1)[0]
    known_targets = (("方块", "积木", "cube"), ("苹果", "apple"), ("香蕉", "banana"))
    if sum(any(word in source for word in group) for group in known_targets) > 1 or re.search(r"或者|或|、|以及|和|与|\b(?:and|or)\b", source):
        raise ValueError("请一次指定一个明确目标，不要使用并列或选择指令")
    chinese = re.search(r"(?:放进|放入|放到|放在|放至|装进|装入)(?:蓝色)?(?:的)?(?:托盘|盘子)(?:里|中|内|里面|上|上面)?[。！!，,\s]*$", lowered)
    english = re.search(r"\bput\b.+\b(?:in|into|on)\s+(?:the\s+)?(?:blue\s+)?tray[.!\s]*$", lowered)
    take_out = TAKE_OUT.fullmatch(lowered)
    if take_out and any(word in take_out["target"] for word in ("放", "装", "从", "取", "拿", "再", "然后")):
        raise ValueError("请一次只给出一个取出动作，不要附加其它动作或目的地")
    if not take_out and ("从" in lowered or not (chinese or english)):
        raise ValueError("请明确放进托盘，或从盘子取出放到桌面，例如：把香蕉从盘子取出来")
    return lowered


def command_action(text):
    return "take_out" if TAKE_OUT.fullmatch(validate_command(text)) else "put_in"


def parse_command(text):
    lowered = validate_command(text)
    labels = {"cube": ("方块", "积木", "cube"), "apple": ("苹果", "apple"),
              "banana": ("香蕉", "banana")}
    found = [name for name, words in labels.items() if any(w in lowered for w in words)]
    if len(found) != 1:
        raise ValueError("请一次指定一个物体：红色方块、绿色苹果或黄色香蕉")
    # Colour baseline cannot fulfil arbitrary attributes or unknown classes.
    source = re.split(r"从|放|装|\b(?:into|in|on)\b", lowered, maxsplit=1)[0]
    allowed = {"cube": ("红", "red"), "apple": ("绿", "green"), "banana": ("黄", "yellow")}[found[0]]
    colours = ("红", "绿", "黄", "蓝", "紫", "黑", "白", "橙", "red", "green", "yellow", "blue", "purple", "black", "white", "orange")
    if any(c in source and c not in allowed for c in colours):
        raise ValueError("颜色基线没有符合该颜色的目标，请切换 VLM 或明确使用桌面上可见物体")
    return found[0]
