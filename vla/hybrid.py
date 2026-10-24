"""Experimental HAMSTER target + pi05 grasp + RGB-D/OSC transfer, simulation only.

This is NOT HAMSTER's learned low-level policy or an end-to-end VLA. Cached
plans are accepted only for their exact image. No simulator object pose enters
the controller; object truth is used separately by the relation evaluator.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from vla.hamster import parse_path
from vla.run import (
    configure_libero,
    connect_policy,
    load_initial_states,
    observation_payload,
    validate_actions,
)


def execution_plan(plan, unscored=False):
    points = parse_path(plan['raw'])
    closes = [i for i, point in enumerate(points) if point['action'] == 'Close Gripper']
    opens = [i for i, point in enumerate(points) if point['action'] == 'Open Gripper']
    if len(closes) != 1 or len(opens) != 1 or closes[0] >= opens[0]:
        raise ValueError('Only one close followed by one open is supported')
    prompt = plan['command']['english']
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 2000:
        raise ValueError('Invalid model instruction')
    if not unscored and prompt not in ('put the wine bottle in the bowl', 'put the bowl on the plate'):
        raise ValueError('Custom instructions require --unscored and visual result review')
    return prompt, points[closes[0]], points[opens[0]]


def world_cloud(depth, intrinsic, extrinsic):
    """Depth is top-left upright; planner RGB is additionally mirrored in X."""
    height, width = depth.shape
    y, x = np.mgrid[:height, :width]
    pixels = np.stack((width - 1 - x, y, np.ones_like(x)), axis=-1)
    camera = (pixels @ np.linalg.inv(intrinsic).T) * depth[:, ::-1, None]
    return camera @ extrinsic[:3, :3].T + extrinsic[:3, 3]


def locate_surface(cloud, point):
    """Select a depth-continuous tabletop component near a model pixel.

    This geometric heuristic assumes separated tabletop objects, not learned
    segmentation. Robust visible bounds approximate the object center; partial
    views, touching objects, tall objects and non-table scenes remain limitations.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    height, width = cloud.shape[:2]
    u, v = round(point['x'] * (width - 1)), round(point['y'] * (height - 1))
    z = cloud[..., 2]
    values, counts = np.unique(np.round(z[(z > .7) & (z < 1.05)], 3), return_counts=True)
    if not len(values):
        raise ValueError('No visible tabletop plane')
    tabletop = float(values[np.argmax(counts)])
    y, x = np.mgrid[:height, :width]
    mask = (z > tabletop + .007) & (z < tabletop + .30)
    indices = np.full((height, width), -1, dtype=int)
    indices[mask] = np.arange(mask.sum())
    edges = []
    for dy, dx in ((0, 1), (1, 0)):
        a = (slice(0, height-dy), slice(0, width-dx))
        b = (slice(dy, height), slice(dx, width))
        adjacent = mask[a] & mask[b] & (np.linalg.norm(cloud[a]-cloud[b], axis=-1) < .02)
        edges.append((indices[a][adjacent], indices[b][adjacent]))
    rows = np.concatenate([edge[0] for edge in edges])
    cols = np.concatenate([edge[1] for edge in edges])
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(int(mask.sum()),)*2)
    count, components = connected_components(graph, directed=False)
    labels = np.full((height, width), -1, dtype=int)
    labels[mask] = components
    candidates = []
    for label in range(count):
        region = labels == label
        if region.sum() >= 60:
            distance = int(np.min((x[region] - u)**2 + (y[region] - v)**2))
            candidates.append((distance, label))
    if not candidates or min(candidates)[0] > (width * .03)**2:
        raise ValueError('No supported surface region near model release point')
    region = labels == min(candidates)[1]
    visible = cloud[region]
    low, high = np.quantile(visible, [.01, .99], axis=0)
    target = np.r_[(low[:2]+high[:2])/2, np.median(visible[:, 2])]
    return target, region, dict(raw_point=cloud[v, u].tolist(), refined_point=target.tolist(),
                               tabletop=tabletop, pixels=int(region.sum()),
                               method='depth-connected component visible XY bounds midpoint; scene-limited')


def servo_action(eef, target, gripper):
    target = np.asarray(target, dtype=float)
    if (not np.isfinite(target).all() or np.any(np.abs(target[:2]) > .45)
            or not .90 <= target[2] <= 1.35):
        raise ValueError('Transfer target outside declared simulation workspace')
    return np.r_[np.clip((target - eef) / .05, -.45, .45), np.zeros(3), gripper]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--viewer', action='store_true')
    parser.add_argument('--unscored', action='store_true',
                        help='Accept a fresh instruction; no automatic semantic success claim')
    parser.add_argument('--policy-only', action='store_true',
                        help='Matched official pi05 control; disables the hybrid handover')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite nonempty output: {args.output}')
    plan = json.loads(args.plan.read_text())
    prompt, source_pixel, release_pixel = execution_plan(plan, args.unscored)
    expected = json.loads((args.plan.parent / 'run.json').read_text())['image_sha256']
    os.environ.setdefault('MUJOCO_GL', 'glfw' if args.viewer else 'egl')
    configure_libero()
    import imageio.v2 as imageio
    from PIL import Image
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    from robosuite.utils.camera_utils import (
        get_camera_extrinsic_matrix,
        get_camera_intrinsic_matrix,
        get_real_depth_map,
    )

    from vla.compare import BottleInBowlCheck

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'plan.json').write_text(json.dumps(plan, indent=2))
    task = benchmark.get_benchmark_dict()['libero_goal']().get_task(8)
    env = OffScreenRenderEnv(
        bddl_file_name=str(Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file),
        camera_heights=512, camera_widths=512, camera_depths=True)
    writer = viewer = None
    trace = []
    relation = BottleInBowlCheck() if not args.unscored and 'wine bottle' in prompt else None
    report = dict(scheme=('official pi05 policy-only matched control' if args.policy_only else
                         'experimental HAMSTER target + pi05 grasp + RGB-D/OSC transfer'),
                  learned_low_level=args.policy_only, end_to_end_vla=args.policy_only, prompt=prompt,
                  seed=42, suite='libero_goal', task_id=8, initial_state=0,
                  diffusion_noise='explicit uint32 seed 42 + control_step//5 per inference',
                  max_controls=420, replan_steps=5, success=None if args.unscored else False,
                  semantic_scoring=not args.unscored, result='not_started',
                  ground_truth_controller_inputs=False,
                  controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest())
    started = time.monotonic()
    try:
        env.seed(42)
        env.reset()
        obs = env.set_init_state(load_initial_states(task)[0])
        for _ in range(10):
            obs = env.step([0.] * 6 + [-1.])[0]
        rgb = obs['agentview_image'][::-1, ::-1]
        Image.fromarray(rgb).save(args.output / 'input.png')
        digest = hashlib.sha256((args.output / 'input.png').read_bytes()).hexdigest()
        report['image_sha256'] = digest
        if digest != expected:
            raise ValueError('Cached HAMSTER plan image does not match current scene')
        sim = env.sim
        intrinsic = get_camera_intrinsic_matrix(sim, 'agentview', 512, 512)
        extrinsic = get_camera_extrinsic_matrix(sim, 'agentview')
        depth = get_real_depth_map(sim, obs['agentview_depth'][::-1, :, 0])
        cloud = world_cloud(depth, intrinsic, extrinsic)
        destination, region, metadata = locate_surface(cloud, release_pixel)
        source, _, source_metadata = locate_surface(cloud, source_pixel)
        metadata['source_surface_point'] = source.tolist()
        metadata['source_refinement'] = source_metadata
        report['destination'] = metadata
        np.savez_compressed(args.output / 'calibration.npz', depth=depth, intrinsic=intrinsic,
                            extrinsic=extrinsic, destination_mask=region)
        overlay = rgb.copy()
        overlay[region] = (overlay[region] // 2 + np.array([0, 100, 0])).astype('uint8')
        Image.fromarray(overlay).save(args.output / 'destination.png')
        client = connect_policy(args.port)
        report['server'] = client.get_server_metadata()
        if report['server'].get('backend') != 'official-pi05-libero-vla':
            raise ValueError('Probe requires verified official pi05 server')
        writer = imageio.get_writer(args.output / 'rollout.mp4', fps=20)
        if args.viewer:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(sim.model._model, sim.data._data)
            viewer.opt.geomgroup[0] = 0
            viewer.opt.geomgroup[1] = 1
            viewer.cam.lookat[:] = [0, 0, 1]
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.4, 225, -30
        phase, queue, grasp_z, min_z = 'vla_grasp', [], None, None
        grasp_eef, destination_eef = None, None
        phase_count, handover_step = 0, None
        target = None
        writer.append_data(rgb)
        for step in range(420):
            if viewer is not None and not viewer.is_running():
                report['result'] = 'viewer_closed'
                break
            eef = np.asarray(obs['robot0_eef_pos'])
            gap = float(np.diff(obs['robot0_gripper_qpos'])[0] * -1)
            if phase == 'vla_grasp':
                if not queue:
                    payload = observation_payload(obs, prompt, image_tools)
                    payload['_probe_noise_seed'] = 42 + step // 5
                    response = client.infer(payload)
                    actions = validate_actions(response['actions'], 5)
                    np.save(args.output / f'prediction-{step:03}.npy', actions)
                    queue.extend(actions[:5])
                action = np.asarray(queue.pop(0))
                if action[-1] > .5:
                    if min_z is None or eef[2] < min_z:
                        min_z, grasp_eef = float(eef[2]), eef.copy()
                # A bowl wall is much thinner than a bottle. This is a grasp
                # candidate from proprioception, not proof of object contact.
                if (min_z is not None and eef[2] > min_z + .07
                        and .003 < gap < .075 and action[-1] > .5 and not args.policy_only):
                    grasp_z = min_z
                    # Gripping a rim displaces the object from the TCP. Estimate
                    # this offset from initial RGB-D and the measured close pose.
                    offset = source[:2] - grasp_eef[:2]
                    if np.linalg.norm(offset) > .10:
                        raise RuntimeError('Model grasp is too far from visual source point')
                    destination_eef = destination[:2] - offset
                    report['grasp_xy_offset_estimate'] = offset.tolist()
                    handover_step = step
                    phase, phase_count = 'lift', 0
                    target = np.r_[eef[:2], max(1.20, float(eef[2]))]
                if step >= 179 and phase == 'vla_grasp' and not args.policy_only:
                    raise RuntimeError('No measurable lifted grasp within 180 controls')
            if phase != 'vla_grasp':
                action = servo_action(eef, target, -1. if phase in ('release', 'retreat', 'settle') else 1.)
            obs = env.step(action.tolist())[0]
            phase_count += 1
            eef_after = np.asarray(obs['robot0_eef_pos'])
            success = None if args.unscored else (bool(relation(env)) if relation else bool(env.check_success()))
            trace.append(dict(step=step, phase=phase, source='pi05' if phase == 'vla_grasp' else 'OSC',
                              action=action.tolist(), eef=eef_after.tolist(),
                              gripper=obs['robot0_gripper_qpos'].tolist(), success=success))
            writer.append_data(obs['agentview_image'][::-1, ::-1])
            if step % 20 == 0:
                print(json.dumps(trace[-1]), flush=True)
            if viewer is not None:
                viewer.sync()
                time.sleep(.05)
            reached = target is not None and np.linalg.norm(eef_after - target) < .008
            if phase in ('lift', 'transfer', 'lower'):
                if phase_count > 65:
                    raise RuntimeError(f'{phase}: servo did not converge')
                if reached:
                    if phase == 'lift':
                        phase, target = 'transfer', np.r_[destination_eef, target[2]]
                    elif phase == 'transfer':
                        release_z = grasp_z + (destination[2] - metadata['tabletop']) + .018
                        phase, target = 'lower', np.r_[destination_eef, release_z]
                    else:
                        phase = 'release'
                    phase_count = 0
            elif phase == 'release' and phase_count >= 15:
                phase, phase_count = 'retreat', 0
                target = np.r_[eef_after[:2], 1.20]
            elif phase == 'retreat' and reached:
                phase, phase_count = 'settle', 0
            elif phase == 'settle' and phase_count >= 30:
                report['success'] = success
                report['result'] = ('motion_completed_unscored' if args.unscored else
                                    'success' if success else 'relation_failed')
                break
        else:
            if args.policy_only:
                report['success'], report['result'] = success, 'success' if success else 'relation_failed'
            else:
                report['result'] = 'control_limit'
        report['grasp_min_eef_z'], report['handover_step'] = grasp_z, handover_step
        Image.fromarray(obs['agentview_image'][::-1, ::-1]).save(args.output / 'final.png')
    except KeyboardInterrupt:
        report['result'] = 'interrupted'
        raise
    except Exception as exc:
        report['result'], report['exception'] = 'exception', f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if writer:
            writer.close()
        if viewer:
            viewer.close()
        if 'client' in locals():
            client._ws.close()
        env.close()
        report.update(control_steps=len(trace), wall_duration_s=time.monotonic() - started)
        (args.output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
        (args.output / 'trace.json').write_text(json.dumps(trace, indent=2) + '\n')
        if relation:
            (args.output / 'relation.json').write_text(json.dumps(relation.history, indent=2) + '\n')
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
