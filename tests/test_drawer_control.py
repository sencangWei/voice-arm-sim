import json

import numpy as np
from scipy.spatial.transform import Rotation
import pytest

from vla.drawer_control import parse_drawer, handle_geometry, drawer_depth_observation, drawer_depth_check
from vla.task_plan import parse_plan


def test_drawer_plan_keeps_level_and_all_three_real_skills():
    steps = [dict(skill=s, instruction=t, destination_kind='drawer_interior') for s, t in (
        ('open_drawer', '打开第三层抽屉'), ('pick_place', '把奶酪放进第三层抽屉'),
        ('close_drawer', '关闭第三层抽屉'))]
    plan = dict(status='ready', reason='三步', missing_skills=[], steps=steps)
    assert parse_plan(json.dumps(plan)) == plan


def test_invalid_drawer_region_fails_before_contact():
    with pytest.raises(ValueError):
        parse_drawer(json.dumps(dict(action='open', level='bottom', handle_bbox_2d=[1, 1, 0, 2],
                                    cabinet_bbox_2d=[0, 0, 1000, 1000])))


def drawer_cloud(position):
    x, z = np.meshgrid(np.linspace(-.04, .04, 60), np.linspace(.994, 1.006, 12))
    bar = np.stack([x, np.full_like(x, position), z], axis=-1)
    x, z = np.meshgrid(np.linspace(-.06, .06, 60), np.linspace(1.07, 1.17, 12))
    cabinet = np.stack([x, np.full_like(x, -.025), z], axis=-1)
    return np.concatenate([bar, cabinet])


def test_depth_vetoes_no_motion_wrong_direction_and_partial_close():
    geometry = dict(handle=[0., 0., 1.], outward_normal=[0., 1., 0.])
    closed = dict(geometry=geometry, initial=drawer_depth_observation(drawer_cloud(0.), geometry))
    assert drawer_depth_check(drawer_cloud(0.), closed, 'open')['accepted'] is False
    assert drawer_depth_check(drawer_cloud(-.1), closed, 'open')['accepted'] is False
    assert drawer_depth_check(drawer_cloud(.15), closed, 'open')['accepted'] is True
    opened = dict(geometry=geometry, initial=drawer_depth_observation(drawer_cloud(.15), geometry))
    assert drawer_depth_check(drawer_cloud(.07), opened, 'close')['accepted'] is False
    assert drawer_depth_check(drawer_cloud(0.), opened, 'close')['accepted'] is True
    assert drawer_depth_check(None, opened, 'close')['accepted'] is None


def test_handle_normal_comes_from_depth_not_fixed_world_axis():
    x, z = np.meshgrid(np.linspace(-.1, .1, 100), np.linspace(.9, 1.12, 100))
    cloud = np.stack([x, np.zeros_like(x), z], axis=-1)
    cloud[70:77, 30:70, 1] = .025
    task = dict(handle_bbox_2d=[300, 700, 700, 770], cabinet_bbox_2d=[0, 0, 1000, 1000])
    point, normal, _ = handle_geometry(cloud, task)
    assert np.allclose(normal, [0, 1, 0], atol=.01)
    assert point[1] == pytest.approx(.025)


def test_open_handle_bar_excludes_table_and_uses_observed_slide_axis():
    x, y = np.meshgrid(np.linspace(-.05, .05, 50), np.linspace(0, .015, 20))
    bar = np.stack([x, y, np.full_like(x, .955)], axis=-1)
    # The open front is outside the cabinet box; a nearby table is visible.
    cloud = np.concatenate([bar, bar.copy()], axis=1)
    cloud[:, 50:, 1] -= .2
    cloud[:, 50:, 2] = np.linspace(.92, 1.12, 20)[:, None]
    cloud[:5, :50, 2] = np.linspace(.9, .925, 5)[:, None]
    angle = .7
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    cloud[:, :, :2] = cloud[:, :, :2] @ rotation.T
    task = dict(action='close', handle_bbox_2d=[0, 0, 500, 1000],
                cabinet_bbox_2d=[500, 0, 1000, 1000])
    point, normal, geometry = handle_geometry(cloud, task)
    assert point[2] == pytest.approx(.955)
    assert np.allclose(normal[:2], rotation @ [0, 1], atol=.01)
    assert geometry['plane_points'] == 0
def test_push_contact_stops_motion_for_visual_verification(tmp_path):
    from vla.drawer_control import DrawerControl
    obs = dict(robot0_eef_pos=np.array([0., 0., 1.]),
               robot0_eef_quat=Rotation.identity().as_quat())
    controller = DrawerControl(None, obs, None, tmp_path/'run', status=lambda s: None)
    calls = []
    def step(action, phase):
        calls.append(action)
        controller.obs['robot0_eef_pos'][1] = max(-.13, -.01*len(calls))
    controller.step = step
    controller.pose_move(np.array([0., -.16, 1.]), np.eye(3), 1., 'push', stop_on_contact=True)
    assert len(calls) < 40
    assert controller.report['contact_stop']['travel_m'] >= .12
    assert controller.report['semantic_success'] is None
