"""Capture the fixed real MuJoCo observation for the HAMSTER language probe."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from vla.run import configure_libero, load_initial_states


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {args.output}")
    os.environ["MUJOCO_GL"] = "egl"
    configure_libero()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from PIL import Image

    task = benchmark.get_benchmark_dict()["libero_goal"]().get_task(8)
    env = OffScreenRenderEnv(
        bddl_file_name=str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
        camera_heights=512, camera_widths=512,
    )
    try:
        env.seed(42)
        env.reset()
        obs = env.set_init_state(load_initial_states(task)[0])
        for _ in range(10):
            obs = env.step([0.] * 6 + [-1.])[0]
        args.output.mkdir(parents=True, exist_ok=True)
        image = args.output / "scene-init0.png"
        Image.fromarray(obs["agentview_image"][::-1, ::-1]).save(image)
        metadata = {"task": "libero_goal/task8", "initial_state": 0, "seed": 42,
                    "settle_steps": 10, "image_rotation": 180,
                    "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                    "task_language": task.language}
        (args.output / "scene.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps(metadata), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
