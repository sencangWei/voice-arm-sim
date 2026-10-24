import numpy as np
import pytest

from controller import Controller
from perception import command_action, free_table_position, localize, parse_command, point_cloud, validate_command


@pytest.mark.parametrize("text", [
    "不要把苹果放进蓝色托盘", "把香蕉从盘子取出来放进抽屉",
    "暂停，把苹果放进托盘", "把苹果和香蕉放进托盘", "把蓝莓放进托盘",
    "把紫色苹果放进蓝色托盘", "放下", "", "碗在托盘上吗？",
])
def test_ambiguous_commands_never_step(text):
    class Env:
        def _get_observations(self, **kwargs):
            return {}
        def step(self, action):
            pytest.fail("Invalid command moved the robot")
    with pytest.raises(ValueError):
        Controller(Env()).run(text)


def test_command_and_projection_contract():
    assert parse_command("把苹果放进蓝色托盘") == "apple"
    assert parse_command("put the cube into the blue tray") == "cube"
    assert validate_command("把蓝莓放进蓝色托盘")
    k = np.array([[100,0,1],[0,100,1],[0,0,1]])
    cloud = point_cloud(np.ones((3,3)), k, np.eye(4))
    np.testing.assert_allclose(cloud[1,1], [0,0,1])
    np.testing.assert_allclose(cloud[2,2], [.01,.01,1])


@pytest.mark.parametrize("text,label", [
    ("把香蕉从盘子里面取出来", "banana"),
    ("把香蕉从盘子取出来", "banana"),
    ("把苹果从蓝色托盘拿出来放在桌上", "apple"),
    ("把红色方块从托盘里拿出来放到桌面上", "cube"),
])
def test_take_out_commands(text, label):
    assert parse_command(text) == label
    assert command_action(text) == "take_out"


@pytest.mark.parametrize("text", [
    "不要把香蕉从盘子取出来", "把香蕉和苹果从盘子取出来",
    "把香蕉从盘子取出来放在苹果上", "把香蕉从盘子取出来放进托盘",
    "如果可以把香蕉从盘子取出来", "把香蕉从桌上取出来",
    "把香蕉放进抽屉再从盘子取出来", "把香蕉从抽屉拿到盘子再从盘子取出来",
])
def test_invalid_take_out_commands(text):
    with pytest.raises(ValueError):
        validate_command(text)


def test_free_destination_avoids_camera_obstacles_and_requires_visible_table():
    depth = np.ones((100, 100))
    depth[55:70, 55:70] = .95
    k = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]])
    camera = np.diag([1., -1., -1., 1.])
    camera[2, 3] = 1.8
    xyz = free_table_position(depth, k, camera, np.array([.1, -.1]))
    cloud = point_cloud(depth, k, camera)
    obstacle_xy = cloud[depth == .95, :2]
    assert np.linalg.norm(obstacle_xy - xyz[:2], axis=1).min() > .11
    assert xyz[2] == .8
    with pytest.raises(ValueError, match="空位"):
        free_table_position(np.full_like(depth, .95), k, camera, np.zeros(2))


@pytest.mark.parametrize("text", ["把苹果、香蕉放进托盘", "把苹果或香蕉放进托盘", "如果我同意，就把苹果放进托盘", "等我回来把苹果放进托盘", "if I agree put the apple in the tray"])
def test_shared_validation_rejects_ambiguity_before_vlm(text):
    with pytest.raises(ValueError):
        validate_command(text)


def test_vlm_box_does_not_replace_cube_with_larger_blue_tray():
    rgb = np.full((80,80,3), 220, np.uint8)
    rgb[15:25,15:25] = [255,0,0]
    rgb[35:55,35:55] = [0,0,255]
    depth = np.ones((80,80))
    depth[15:25,15:25] = .95
    depth[35:55,35:55] = .95
    k = np.array([[200,0,40],[0,200,40],[0,0,1]])
    camera = np.diag([1.,-1.,-1.,1.])
    camera[2,3] = 1.8
    detection = localize(rgb, depth, k, camera, "cube", bbox=[10,10,60,60])
    assert detection.bbox == [15,15,25,25]
    with pytest.raises(ValueError, match="多个前景"):
        localize(rgb, depth, k, camera, "unseen-object", bbox=[10,10,60,60])
