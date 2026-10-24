"""Controlled same-observation/same-noise language ablation; no robot motion."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from vla.run import configure_libero, load_initial_states, observation_payload, connect_policy, validate_actions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MUJOCO_GL", "egl")
    configure_libero()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    import imageio.v2 as imageio
    task = benchmark.get_benchmark_dict()["libero_goal"]().get_task(8)
    env = client = None
    try:
        env = OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path("bddl_files")) /
            task.problem_folder / task.bddl_file), camera_heights=256, camera_widths=256)
        env.seed(7)
        env.reset()
        obs = env.set_init_state(load_initial_states(task)[0])
        for _ in range(10):
            obs, _, _, _ = env.step([0.] * 6 + [-1.])
        payload = observation_payload(obs, task.language, image_tools)
        np.savez(args.output / "observation.npz", **{k.replace("/", "_"): v for k,v in payload.items() if k != "prompt"})
        imageio.imwrite(args.output / "observation.png", np.concatenate((payload["observation/image"], payload["observation/wrist_image"]), axis=1))
        client = connect_policy(args.port)
        prompts = ["put the bowl on the plate", "put the bowl on the plate",
                   "put the bowl on the stove", "open the middle drawer of the cabinet"]
        results = []
        for index, prompt in enumerate(prompts):
            result = client.infer({**payload, "prompt": prompt, "_probe_noise_seed": 42})
            actions = validate_actions(result["actions"], 1)
            np.save(args.output / f"actions_{index}.npy", actions)
            results.append(actions)
        repeat = float(np.max(np.abs(results[0] - results[1])))
        changed = [float(np.max(np.abs(results[0] - a))) for a in results[2:]]
        report = dict(prompts=prompts, noise_seed=42, scene="libero_goal/task8/init0",
            observation_sha256=hashlib.sha256((args.output / "observation.npz").read_bytes()).hexdigest(),
            repeated_prompt_max_abs_difference=repeat, changed_prompt_max_abs_differences=changed,
            server_metadata=client.get_server_metadata(),
            interpretation="Action sensitivity only; this probe does not prove task completion or open-world generalization.")
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        if repeat > 1e-5 or any(d <= 1e-4 for d in changed):
            raise RuntimeError("Language ablation failed deterministic/sensitivity check")
    finally:
        if client is not None:
            client._ws.close()
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
