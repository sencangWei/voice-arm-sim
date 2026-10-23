"""Small reproducible physical regression matrix; simulation truth scores only."""
import argparse
import json
from pathlib import Path

import numpy as np
from controller import Controller
from scene import DesktopScene


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="runs/regression.json")
    args = p.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    env = DesktopScene()
    results = []
    try:
        for seed in [42,7,99]:
            for label, name in [("cube","红色方块"),("apple","苹果"),("banana","香蕉")]:
                env.seed = seed
                env.reset()
                record = {"seed":seed, "target":label}
                try:
                    record.update(Controller(env).run(f"把{name}放进蓝色托盘"))
                except Exception as exc:
                    record.update(success=False, exception=str(exc))
                obj, tray = env.evaluation_truth(label), env.evaluation_truth("tray")
                record["truth_in_tray"] = bool(np.all(np.abs(obj[:2]-tray[:2]) < [.065,.055]) and .81<obj[2]<.89)
                results.append(record)
                output.write_text(json.dumps(results, ensure_ascii=False, indent=2))
                print(seed,label,record["success"],record["truth_in_tray"],flush=True)
    finally:
        env.close()
    assert all(r["success"] and r["truth_in_tray"] for r in results)


if __name__ == "__main__":
    main()
