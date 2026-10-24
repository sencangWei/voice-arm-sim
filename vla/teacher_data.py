"""Offline simulation demonstrations, NOT an inference controller.

The bottle-in-bowl teacher uses pi05 for grasping and simulator geometry for
transport. Other tasks replay the official learned policy. All sources remain
explicit; none of these recordings are human corrections or zero-shot results.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from vla.run import (configure_libero, connect_policy, load_initial_states,
                     observation_payload, validate_actions)

TASKS = {
    "bottle-in-bowl": "put the wine bottle in the bowl",
    "bottle-on-rack": "put the wine bottle on the rack",
    "bowl-on-plate": "put the bowl on the plate",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect(output, task_name, initial_id, port=8766):
    from vla.compare import BottleInBowlCheck
    from vla.language_guidance import BottleOnRackCheck, GuidedPolicy
    from vla.hybrid import servo_action
    configure_libero()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    import imageio.v2 as imageio

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    task = benchmark.get_benchmark_dict()["libero_goal"]().get_task(8)
    initial = np.asarray(load_initial_states(task)[initial_id])
    prompt = TASKS[task_name]
    env = OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path("bddl_files")) /
        task.problem_folder / task.bddl_file), camera_heights=256, camera_widths=256)
    check = BottleInBowlCheck() if task_name == "bottle-in-bowl" else (
        BottleOnRackCheck() if task_name == "bottle-on-rack" else lambda e: bool(e.check_success()))
    frames, trace = [], []
    report = dict(schema="simulation-teacher-v1", task=prompt, task_name=task_name,
        initial_state_id=initial_id, initial_state_sha256=hashlib.sha256(initial.tobytes()).hexdigest(),
        seed=42, source="pi05 grasp + simulator-geometry transport" if task_name == "bottle-in-bowl"
        else "official pi05 + CAG-TF" if task_name == "bottle-on-rack" else "official pi05 policy",
        human_actions_only=False, training_demonstration=True,
        policy_evaluation=False, success=False, code_sha256=digest(__file__))
    client = writer = None
    try:
        env.seed(42)
        env.reset()
        obs = env.set_init_state(initial)
        for _ in range(10):
            obs = env.step([0.] * 6 + [-1.])[0]
        client = connect_policy(port)
        report["server"] = client.get_server_metadata()
        if report["server"].get("backend") != "official-pi05-libero-vla":
            raise ValueError("Teacher requires the official checkpoint")
        if task_name == "bottle-on-rack":
            client = GuidedPolicy(client, scale=1.5, seed=42)
            report["server"] = client.get_server_metadata()
        writer = imageio.get_writer(output / "teacher.mp4", fps=20)
        sim_env = env.env
        bottle = sim_env.get_object("wine_bottle_1")
        bowl = np.asarray(sim_env.sim.data.body_xpos[sim_env.obj_body_id["akita_black_bowl_1"]]).copy()
        phase, queue, phase_count = "learned", [], 0
        min_z = None
        success_streak = 0
        target = transfer_xy = None
        release_z = None
        for step in range(360):
            payload = observation_payload(obs, prompt, image_tools)
            writer.append_data(np.concatenate((payload["observation/image"], payload["observation/wrist_image"]), axis=1))
            eef = np.asarray(obs["robot0_eef_pos"])
            if phase == "learned":
                if not queue:
                    response = client.infer({**payload, "_probe_noise_seed": 42 + step // 5})
                    queue.extend(validate_actions(response["actions"], 5)[:5])
                action = np.asarray(queue.pop(0))
                if action[-1] > .5:
                    min_z = min(float(eef[2]), min_z) if min_z is not None else float(eef[2])
                held = sim_env.check_contact(sim_env.robots[0].gripper.contact_geoms, bottle)
                if task_name == "bottle-in-bowl" and held and min_z is not None and eef[2] > min_z + .07:
                    bottle_pos = sim_env.sim.data.body_xpos[sim_env.obj_body_id["wine_bottle_1"]].copy()
                    transfer_xy = bowl[:2] - (bottle_pos[:2] - eef[:2])
                    # Offline expert only: match the released object's bottom
                    # to the bowl interior, preserving the measured TCP offset.
                    bottom = bottle_pos[2] + bottle.bottom_offset[2]
                    release_z = float(eef[2] + bowl[2] + .012 - bottom)
                    phase, phase_count = "lift", 0
                    target = np.r_[eef[:2], 1.22]
            if phase != "learned":
                action = servo_action(eef, target, -1. if phase in ("release", "retreat", "settle") else 1.)
            action = np.asarray(action, dtype=np.float32)
            frames.append(dict(state=np.asarray(payload["observation/state"], dtype=np.float32),
                image=payload["observation/image"], wrist_image=payload["observation/wrist_image"], action=action))
            obs = env.step(action.tolist())[0]
            phase_count += 1
            success = bool(check(env))
            if task_name == "bowl-on-plate":
                source = sim_env.get_object("akita_black_bowl_1")
                success = success and not sim_env.check_contact(sim_env.robots[0].gripper.contact_geoms, source)
            success_streak = success_streak + 1 if success else 0
            trace.append(dict(step=step, phase=phase, success=success))
            if success_streak >= 5 and task_name != "bottle-in-bowl":
                report["success"] = True
                break
            reached = target is not None and np.linalg.norm(np.asarray(obs["robot0_eef_pos"]) - target) < .009
            if phase in ("lift", "transfer", "lower"):
                if phase_count > 70:
                    break
                if reached:
                    if phase == "lift":
                        phase, target = "transfer", np.r_[transfer_xy, 1.22]
                    elif phase == "transfer":
                        phase, target = "lower", np.r_[transfer_xy, release_z]
                    else:
                        phase = "release"
                    phase_count = 0
            elif phase == "release" and phase_count >= 12:
                phase, target, phase_count = "retreat", np.r_[transfer_xy, 1.22], 0
            elif phase == "retreat" and reached:
                phase, phase_count = "settle", 0
            elif phase == "settle" and phase_count >= 30:
                report["success"] = success_streak >= 5
                break
        payload = observation_payload(obs, prompt, image_tools)
        writer.append_data(np.concatenate((payload["observation/image"], payload["observation/wrist_image"]), axis=1))
    except Exception as exc:
        report["exception"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if writer is not None:
            writer.close()
        if client is not None:
            client._ws.close()
        env.close()
        if frames:
            np.savez_compressed(output / "frames.npz", **{
                key: np.stack([f[key] for f in frames]) for key in frames[0]})
            report["frames_sha256"] = digest(output / "frames.npz")
        report["frames"] = len(frames)
        (output / "metadata.json").write_text(json.dumps(report, indent=2) + "\n")
        (output / "trace.json").write_text(json.dumps(trace, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--initial-states", nargs="+", type=int, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    for initial in args.initial_states:
        for task in args.tasks:
            collect(args.output / f"{task}-init{initial:03}", task, initial, args.port)


if __name__ == "__main__":
    main()
