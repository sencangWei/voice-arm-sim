"""Real models, two commands on one retained scene; no resets between commands."""
import base64
import hashlib
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from vla.run import parse_args  # noqa: E402
from vla.workbench import Workbench  # noqa: E402
from vla.semantic_grasp import post  # noqa: E402
HERE = Path(__file__).resolve().parent
args = parse_args(["--panel", "--suite", "libero_goal", "--task-id", "8", "--seed", "42",
    "--prompt", "把碗放到盘子上", "--port", "8766", "--output", str(ROOT/"runs/task-completion-20261025/sequence")])
with patch("vla.workbench.threading.Thread.start"):
    lab = Workbench(args)
records = []
try:
    lab._load(8, 0, args.prompt, 8766)
    env = lab.env
    original_update = lab._update
    def update(**values):
        original_update(**values)
        if "status" in values:
            print(values["status"], flush=True)
    lab._update = update
    for text in ("把碗放到盘子上", "把酒瓶放到架子上"):
        before = hashlib.sha256(env.sim.get_state().flatten().tobytes()).hexdigest()
        try:
            lab._command("semantic", (text,))
            error = None
        except Exception as exc:
            error = repr(exc)
        assert lab.env is env and not lab.running and not lab.semantic_active and not lab.plan
        state = env.sim.get_state().flatten().copy()
        time.sleep(1)
        assert np.array_equal(state, env.sim.get_state().flatten())
        result = dict(lab.scene["last_instruction_result"], exception_observed=error,
            same_environment=True, reset_between_commands=False, idle_state_unchanged=True,
            start_state_sha256=before, end_state_sha256=hashlib.sha256(state.tobytes()).hexdigest())
        records.append(result)
        (HERE/"sequence.json").write_text(json.dumps(records, ensure_ascii=False, indent=2))
        print(json.dumps(result, ensure_ascii=False), flush=True)
finally:
    lab._close_scene()
# Reuse the real first trial's current captured cameras to test a wrong destination.
source = ROOT/"runs/task-completion-20261025/bowl-plate"
images = [base64.b64encode((source/name).read_bytes()).decode() for name in
          ("completion-agent.png", "completion-wrist.png")]
negative = post(8912, "verify", {"text": "把碗放到架子上", "images": images})
(HERE/"wrong-destination.json").write_text(json.dumps(negative, ensure_ascii=False, indent=2))
assert negative["judgment"]["verdict"] == "not_completed", negative
