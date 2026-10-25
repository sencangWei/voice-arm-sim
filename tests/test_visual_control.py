"""Geometry contract tests; physical grasp success is checked in real rollouts."""
import numpy as np
import pytest

from vla.visual_control import VisualControl
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


def test_visual_controller_defaults_to_gpu_task_service(tmp_path, monkeypatch):
    from vla.semantic_grasp import VisualSkill
    monkeypatch.setattr(VisualSkill, 'run', lambda self, text, task_port: (text, task_port))
    skill = VisualControl(None, {}, None, tmp_path / 'run')
    assert skill.run('把碗放到盘子上') == ('把碗放到盘子上', 8911)


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
