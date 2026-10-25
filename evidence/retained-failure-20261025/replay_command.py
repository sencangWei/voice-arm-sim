import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from vla.run import configure_libero
from vla.semantic_grasp import VisualSkill,image_bytes
from vla.run import connect_policy
configure_libero()
from libero.libero import benchmark,get_libero_path
from libero.libero.envs import OffScreenRenderEnv
source=Path(__file__).resolve().parent/"scene-source"
import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--output",type=Path,required=True)
parser.add_argument("--replay-only",action="store_true")
args=parser.parse_args()
here=args.output
here.mkdir(parents=True,exist_ok=False)
meta=json.loads((source/"report.json").read_text())
task=benchmark.get_benchmark_dict()[meta["suite"]]().get_task(meta["task_id"])
env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path("bddl_files"))/task.problem_folder/task.bddl_file),camera_heights=256,camera_widths=256,ignore_done=True)
try:
 env.seed(meta["seed"]);env.reset()
 obs=env.set_init_state(np.load(source/"initial_state.npy"))
 for _ in range(10):obs=env.step([0.]*6+[-1.])[0]
 for action in np.load(source/"executed_actions.npy"):obs=env.step(action.tolist())[0]
 np.save(here/"retained-state.npy",env.sim.get_state().flatten())
 skill=VisualSkill(env,obs,None,here/"cabinet-command")
 rgb,_=skill.capture();raw=image_bytes(rgb)
 (here/"cabinet-command"/"current.png").write_bytes(raw)
 old=(Path(__file__).resolve().parent/"cabinet-execution"/"input.png").read_bytes()
 check=dict(replayed_actions=meta["steps"],same_image_bytes=raw==old,sha256=hashlib.sha256(raw).hexdigest(),original_sha256=hashlib.sha256(old).hexdigest(),original_gui_untouched=True)
 print(json.dumps(check),flush=True)
 (here/"cabinet-command"/"check.json").write_text(json.dumps(check,indent=2))
 if args.replay_only:sys.exit(0)
 client=connect_policy(8766)
 command=VisualSkill(env,obs,client,here/"cabinet-command"/"execute")
 try:print(json.dumps(command.run("把酒瓶放柜子上"),ensure_ascii=False),flush=True)
 except Exception as exc:print(type(exc).__name__,str(exc),flush=True)
finally:env.close()
