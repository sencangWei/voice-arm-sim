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


def test_close_handle_uses_supported_bar_when_background_height_is_more_common():
    # The model box includes a broad plate/table patch below the metal bar.
    # Picking just the largest height bin rejects this otherwise visible handle.
    x, y = np.meshgrid(np.linspace(-.06, .06, 60), np.linspace(-.04, .04, 30))
    background = np.stack([x, y, np.full_like(x, .91)], axis=-1)
    background[10:, :, 1] = np.linspace(-.05, .05, 20)[:, None]
    x, y = np.meshgrid(np.linspace(-.05, .05, 60), np.linspace(0, .01, 10))
    background[:10] = np.stack([x, y, np.full_like(x, .955)], axis=-1)
    cabinet = background.copy()
    cabinet[:, :, 1] -= .2
    cabinet[:, :, 2] = np.linspace(.92, 1.12, 30)[:, None]
    cloud = np.concatenate([background, cabinet], axis=1)
    task = dict(action='close', handle_bbox_2d=[0, 0, 500, 1000],
                cabinet_bbox_2d=[500, 0, 1000, 1000])
    point, normal, geometry = handle_geometry(cloud, task)
    assert point[2] == pytest.approx(.955)
    assert np.allclose(normal, [0, 1, 0], atol=.01)


def test_close_handle_from_retained_panel_failure():
    from pathlib import Path
    cloud = np.load(Path(__file__).parent / 'fixtures/drawer_handle_background.npz')['cloud']
    task = dict(action='close', handle_bbox_2d=[458, 609, 508, 720],
                cabinet_bbox_2d=[512, 355, 1000, 823])
    point, normal, geometry = handle_geometry(cloud, task)
    assert .948 < point[2] < .960
    assert np.isfinite(normal).all()


@pytest.mark.parametrize("action", ["open", "close"])
def test_handle_still_rejects_planar_patch_without_bar(action):
    x, y = np.meshgrid(np.linspace(-.06, .06, 60), np.linspace(-.04, .04, 30))
    cloud = np.stack([x, y, np.full_like(x, .91)], axis=-1)
    task = dict(action=action, level='middle', handle_bbox_2d=[0, 0, 1000, 1000],
                cabinet_bbox_2d=[0, 0, 1000, 1000])
    from vla.drawer_control import ordered_handle_geometry
    with pytest.raises(ValueError, match='细长横杆|三层把手'):
        (ordered_handle_geometry if action == 'open' else handle_geometry)(cloud, task)


def test_top_drawer_can_reference_closed_front_below_handle():
    geometry = dict(handle=[0.,0.,1.], outward_normal=[0.,1.,0.])
    closed_cloud = drawer_cloud(0.).copy()
    closed_cloud[...,2] = 2. - closed_cloud[...,2]
    opened_cloud = drawer_cloud(.15).copy()
    opened_cloud[...,2] = 2. - opened_cloud[...,2]
    baseline = dict(geometry=geometry, initial=drawer_depth_observation(closed_cloud, geometry))
    assert drawer_depth_check(opened_cloud, baseline, 'open')['accepted'] is True


@pytest.mark.parametrize('level, height', [('top', 1.095), ('middle', 1.024), ('bottom', .955)])
def test_handle_layer_is_ranked_from_retained_depth_even_when_model_points_at_top(level, height):
    from pathlib import Path
    from vla.drawer_control import ordered_handle_geometry
    cloud = np.load(Path(__file__).parent / 'fixtures/drawer_three_handles.npz')['cloud']
    task = dict(action='open', level=level, handle_bbox_2d=[680, 500, 730, 570],
                cabinet_bbox_2d=[680, 345, 1000, 820])
    point, normal, geometry = ordered_handle_geometry(cloud, task)
    assert abs(point[2] - height) < .006
    assert len(geometry['observed_handle_heights']) == 3
    assert geometry['selected_level'] == level


def test_handle_layer_rejects_incomplete_depth_instead_of_guessing():
    from vla.drawer_control import ordered_handle_geometry
    x, z = np.meshgrid(np.linspace(-.1, .1, 100), np.linspace(.9, 1.12, 100))
    cloud = np.stack([x, np.zeros_like(x), z], axis=-1)
    cloud[70:77, 30:70, 1] = .025
    task = dict(action='open', level='middle', handle_bbox_2d=[300, 700, 700, 770],
                cabinet_bbox_2d=[0, 0, 1000, 1000])
    with pytest.raises(ValueError, match='三层'):
        ordered_handle_geometry(cloud, task)


def test_height_ranking_preserves_localization_already_on_requested_bar():
    from pathlib import Path
    from vla.drawer_control import ordered_handle_geometry
    cloud = np.load(Path(__file__).parent / 'fixtures/drawer_three_handles.npz')['cloud']
    task = dict(action='open', level='bottom', handle_bbox_2d=[675, 620, 730, 710],
                cabinet_bbox_2d=[698, 350, 1000, 820])
    seed, normal, _ = handle_geometry(cloud, task)
    point, ranked_normal, _ = ordered_handle_geometry(cloud, task)
    assert np.array_equal(point, seed)
    assert np.array_equal(ranked_normal, normal)


def test_reacquire_open_handle_from_current_depth_and_verified_prior():
    from pathlib import Path
    from vla.drawer_control import reacquire_open_handle
    data = np.load(Path(__file__).parent / 'fixtures/drawer_open_middle.npz')
    context = json.loads(str(data['context']))
    point, normal, geometry = reacquire_open_handle(data['cloud'], context, 'middle')
    assert .014 < point[1] < .035
    assert 1.012 < point[2] < 1.032
    assert normal[1] > .98
    assert geometry['current_bar_points'] >= 40
    with pytest.raises(ValueError):
        reacquire_open_handle(data['cloud'], context, 'bottom')
    with pytest.raises(ValueError):
        reacquire_open_handle(np.zeros_like(data['cloud']), context, 'middle')
