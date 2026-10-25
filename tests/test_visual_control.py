"""Geometry contract tests; physical grasp success is checked in real rollouts."""
import numpy as np
import pytest

from vla.visual_control import VisualControl, source_support_height
from vla.semantic_grasp import circular_container_point, nearby_support_point, region_point


def test_top_grasp_reserves_palm_clearance_and_uses_elevated_source(tmp_path):
    obs = dict(robot0_eef_pos=np.array([-.3, .2, 1.20]))
    skill = VisualControl(None, obs, None, tmp_path / 'run')
    skill.report['source'] = dict(tabletop=.90)
    # Visible elevated object, not an assumed object on the global table.
    cloud = np.zeros((20, 20, 3))
    cloud[..., 0], cloud[..., 1] = -.30, .20
    cloud[..., 2] = np.linspace(.94, 1.10, 400).reshape(20, 20)
    moves = []
    def move(target, grip, phase):
        moves.append((np.asarray(target), grip))
        skill.obs['robot0_eef_pos'] = np.asarray(target)
    skill.oriented_move = move
    skill.step = lambda *args: None
    skill.confirm_hold = lambda: None
    min_z, grasp_eef, offset = skill.grasp({}, None, np.ones((20, 20), bool), cloud)
    target = moves[2][0]
    assert target[2] + .03 > cloud[..., 2].max() + .01
    assert np.allclose(target, grasp_eef)
    assert np.allclose(offset, 0)
    assert skill.release_height(min_z, 1.12, .90) == pytest.approx(min_z + 1.12 - skill.source_base + .012)
    assert skill.release_height(min_z, 1.12, .90) < min_z + 1.12 - .90
    assert skill.policy_metadata()['learned_action_policy'] is False


def test_orientation_timeout_is_reported_instead_of_advancing(tmp_path):
    obs = dict(robot0_eef_pos=np.array([0., 0., 1.2]), robot0_eef_quat=np.array([1., 0., 0., 0.]))
    skill = VisualControl(None, obs, None, tmp_path / 'run', status=lambda message: None)
    actions = []
    skill.step = lambda action, phase: actions.append(action)
    with pytest.raises(TimeoutError, match='未收敛'):
        skill.oriented_move(np.array([.3, 0., 1.2]), -1., 'test', limit=2)
    assert len(actions) == 2
    assert all(a[-1] == -1. for a in actions)


def test_thin_box_grasp_reaches_its_sides_instead_of_closing_above(tmp_path):
    skill = VisualControl(None, dict(robot0_eef_pos=np.array([0., .2, 1.2])), None, tmp_path/'run')
    skill.report['source'] = dict(tabletop=.901)
    x, y = np.meshgrid(np.linspace(-.02, .02, 20), np.linspace(.12, .16, 20))
    cloud = np.stack([x, y, np.full_like(x, .920)], axis=-1)
    skill.oriented_move = lambda target, *args: skill.obs.update(robot0_eef_pos=np.asarray(target))
    skill.step = lambda *args: None
    skill.confirm_hold = lambda: None
    skill.grasp({}, None, np.ones((20, 20), bool), cloud)
    tcp_z = skill.report['geometric_grasp']['target'][2]
    assert .901 < tcp_z < .920
    assert tcp_z-.009 < .919  # fingertip intersects the object's side height
    margin = skill.report['geometric_grasp']['support_margin']
    assert .035 <= margin < .05
    # A thin box fits on a small plate. A universal 6 cm inset wrongly
    # rejects every point on this measured, partly occluded support.
    px, py = np.meshgrid(np.linspace(-.065, .065, 100), np.linspace(-.055, .055, 100))
    plate = np.stack([px, py, np.full_like(px, .908)], axis=-1)
    mask = (px/.065)**2 + (py/.055)**2 <= 1
    point = nearby_support_point(plate, mask, [0., .2], margin=margin)
    assert np.linalg.norm(point[:2]) < .03


def test_drawer_landing_prefers_exposed_near_edge_with_margin(tmp_path):
    skill = VisualControl(None, dict(robot0_eef_pos=np.array([0., .2, 1.2])), None, tmp_path/'run')
    x, y = np.meshgrid(np.linspace(-.09, .09, 100), np.linspace(-.14, .01, 100))
    cloud = np.stack([x, y, np.full_like(x, .924)], axis=-1)
    point, _, _ = skill.landing_point(cloud, [0, 0, 1000, 1000], 'in', 'drawer_interior')
    assert -.04 < point[1] <= -.025
    assert abs(point[0]) < .01


def test_drawer_landing_reserves_the_held_objects_observed_footprint(tmp_path):
    skill = VisualControl(None, dict(robot0_eef_pos=np.array([0., .2, 1.2])), None, tmp_path/'run')
    skill.report['geometric_grasp'] = dict(support_margin=.055)
    x, y = np.meshgrid(np.linspace(-.09, .09, 100), np.linspace(-.14, .01, 100))
    cloud = np.stack([x, y, np.full_like(x, .924)], axis=-1)
    point, _, metadata = skill.landing_point(cloud, [0, 0, 1000, 1000], 'in', 'drawer_interior')
    assert point[1] <= -.045
    assert metadata['edge_margin'] == pytest.approx(.055)


def test_drawer_place_turns_palm_and_keeps_original_offset_on_recovery(tmp_path):
    from scipy.spatial.transform import Rotation
    start = np.diag([1., -1., -1.])
    skill = VisualControl(None, dict(robot0_eef_pos=np.array([0., .2, 1.2]),
        robot0_eef_quat=Rotation.from_matrix(start).as_quat()), None, tmp_path/'run')
    skill.report['geometric_grasp'] = dict(rotation=start.tolist())
    moves = []
    def move(target, grip, phase, desired=None):
        moves.append(desired)
        skill.obs['robot0_eef_quat'] = Rotation.from_matrix(desired).as_quat()
    skill.oriented_move = move
    metadata = dict(floor_width_axis=[1., 0., 0.])
    first = skill.align_for_place(metadata, np.array([.01, .02]))
    again = skill.align_for_place(metadata, np.array([.01, .02]))
    assert np.allclose(first, again)
    assert np.allclose(moves[0][:, 1], [1., 0., 0.])
    assert np.allclose(first, [-.02, .01])
    assert np.linalg.det(moves[0]) == pytest.approx(1.)


@pytest.mark.parametrize('height', [.90, 1.12])
def test_occluded_source_bottom_uses_local_elevated_support(height):
    y, x = np.mgrid[-.15:.151:.005, -.15:.151:.005]
    cloud = np.stack([x, y, np.full(x.shape, height)], axis=-1)
    visible = np.c_[np.zeros(100), np.zeros(100), np.linspace(height+.025, height+.16, 100)]
    assert source_support_height(cloud, visible) == pytest.approx(height)


def test_source_bottom_does_not_snap_to_distant_table_below_elevated_object():
    y, x = np.mgrid[-.15:.151:.005, -.15:.151:.005]
    cloud = np.stack([x, y, np.full(x.shape, .90)], axis=-1)
    visible = np.c_[np.zeros(100), np.zeros(100), np.linspace(1.14, 1.25, 100)]
    assert source_support_height(cloud, visible) == pytest.approx(np.quantile(visible[:, 2], .05))


@pytest.mark.parametrize('tilt', [0., .004])
def test_round_rim_center_is_independent_of_front_surface_bias(tilt):
    angles = np.linspace(0, 2 * np.pi, 800, endpoint=False)
    rim = np.c_[-.1 + .055 * np.cos(angles), .02 + .055 * np.sin(angles), .95 + tilt * np.sin(angles)]
    front = np.tile([-.13, .02, .915], (1600, 1))
    cloud = np.vstack([rim, front])[None]
    assert np.median(cloud[..., 0]) < -.12
    target, meta = circular_container_point(cloud, np.ones(cloud.shape[:2], bool))
    assert np.allclose(target, [-.1, .02, .915])
    assert meta['rim_radius'] == pytest.approx(.055)


def test_flat_destination_is_not_assumed_to_be_an_open_container():
    cloud = np.ones((20, 20, 3))
    with pytest.raises(ValueError, match='圆沿不足'):
        circular_container_point(cloud, np.ones((20, 20), bool))


def test_wide_round_source_is_grasped_at_rim_and_keeps_center_offset(tmp_path):
    angles = np.linspace(0, 2 * np.pi, 800, endpoint=False)
    rim = np.c_[-.1 + .055 * np.cos(angles), .02 + .055 * np.sin(angles), np.full(800, .95)]
    front = np.tile([-.13, .02, .915], (1600, 1))
    cloud = np.vstack([rim, front])[None]
    skill = VisualControl(None, dict(robot0_eef_pos=np.array([-.3, .2, 1.2])), None, tmp_path / 'run')
    skill.report['source'] = dict(tabletop=.90)
    moves = []
    def move(target, grip, phase):
        moves.append(np.asarray(target))
        skill.obs['robot0_eef_pos'] = np.asarray(target)
    skill.oriented_move = move
    skill.step = lambda *args: None
    skill.confirm_hold = lambda: None
    _, eef, offset = skill.grasp({}, None, np.ones(cloud.shape[:2], bool), cloud)
    assert np.allclose(moves[2][:2], [-.1, .075])
    assert np.allclose(eef[:2] + offset, [-.1, .02])
    assert skill.report['geometric_grasp']['mode'] == 'round_rim'
    assert skill.report['geometric_grasp']['support_margin'] == pytest.approx(.038)


def test_plate_landing_uses_connected_shallow_rim_without_table_pixels():
    y, x = np.mgrid[-.05:.051:.001, -.05:.051:.001]
    radius = np.hypot(x, y)
    z = np.full(x.shape, .90)
    z[radius <= .038] = .910
    rim = (radius > .030) & (radius <= .038)
    z[rim] += (radius[rim] - .030)
    cloud = np.stack([x, y, z], axis=-1)
    _, mask, metadata = region_point(cloud, [100, 100, 900, 900], allow_table=True)
    assert metadata['support_pixels'] > metadata['pixels']
    assert np.all(z[mask] > .90)
    target = nearby_support_point(cloud, mask, [0., 0.], margin=.034)
    assert np.linalg.norm(target[:2]) < .003


def test_support_landing_avoids_obstacle_inside_the_convex_table_boundary():
    y, x = np.mgrid[-.2:.201:.004, -.2:.201:.004]
    z = np.full(x.shape, .90)
    # A raised stove occupies a hole inside the selected table support. Its
    # low base is at table height; the convex hull alone fills that hole.
    stove = (np.abs(x) < .06) & (np.abs(y) < .06)
    z[stove] = .925
    cloud = np.stack([x, y, z], axis=-1)
    mask = ~stove
    target = nearby_support_point(cloud, mask, [0., 0.], margin=.04)
    obstacle_xy = cloud[stove, :2]
    assert np.min(np.linalg.norm(obstacle_xy - target[:2], axis=1)) >= .04


def test_visual_controller_defaults_to_gpu_task_service(tmp_path, monkeypatch):
    from vla.semantic_grasp import VisualSkill
    monkeypatch.setattr(VisualSkill, 'run', lambda self, text, task_port: (text, task_port))
    skill = VisualControl(None, {}, None, tmp_path / 'run')
    assert skill.run('把碗放到盘子上') == ('把碗放到盘子上', 8911)


def test_failed_recovery_retreat_preserves_error_and_trace(tmp_path, monkeypatch):
    import json
    import imageio.v2 as imageio
    from types import SimpleNamespace
    skill = VisualControl(None, dict(robot0_eef_pos=np.array([0., 0., 1.]),
        robot0_gripper_qpos=np.zeros(2)), None, tmp_path/'run')
    monkeypatch.setattr(imageio, 'get_writer', lambda *a, **k: SimpleNamespace(close=lambda: None))
    def fail(*args):
        skill.trace.append(dict(phase='retreat'))
        raise TimeoutError('retreat blocked')
    skill.oriented_move = fail
    with pytest.raises(TimeoutError):
        skill.recover('放到盘子', 'retry_grasp', {})
    report = json.loads((skill.output/'result.json').read_text())
    assert report['result'] == 'exception'
    assert report['control_steps'] == 1
    assert 'retreat blocked' in report['exception']
    assert len(json.loads((skill.output/'trace.json').read_text())) == 1


@pytest.mark.parametrize('released', [False, True])
def test_reverify_keeps_holding_context_for_the_next_recovery(tmp_path, released):
    skill = VisualControl(None, dict(robot0_gripper_qpos=np.array([.012, -.012])),
                          None, tmp_path/'run')
    previous = dict(grasp_context={'source_base': .9},
                    geometric_grasp={'support_margin': .06}, motion_completed=released)
    skill.capture = lambda: (np.zeros((8, 8, 3), dtype=np.uint8), None)
    skill.verify_completion = lambda *args: skill.report.update(semantic_success=None)
    skill.recover('放到柜顶', 'reverify', previous)
    assert skill.report['grasp_context'] == previous['grasp_context']
    assert skill.report['geometric_grasp'] == previous['geometric_grasp']
    assert skill.report['motion_completed'] is released


def test_narrow_source_on_wide_touching_support_uses_top_grasp(tmp_path):
    angles = np.linspace(0, 2 * np.pi, 800, endpoint=False)
    base = np.c_[.06 * np.cos(angles), .06 * np.sin(angles), np.full(800, .92)]
    top = np.c_[.015 * np.cos(angles), .015 * np.sin(angles), np.full(800, 1.10)]
    cloud = np.vstack([base, top])[None]
    skill = VisualControl(None, dict(robot0_eef_pos=np.array([-.3, .2, 1.2])), None, tmp_path / 'run')
    skill.report['source'] = dict(tabletop=.90)
    skill.oriented_move = lambda target, *args: skill.obs.update(robot0_eef_pos=np.asarray(target))
    skill.step = lambda *args: None
    skill.confirm_hold = lambda: None
    skill.grasp({}, None, np.ones(cloud.shape[:2], bool), cloud)
    assert skill.report['geometric_grasp']['mode'] == 'top_parallel'
