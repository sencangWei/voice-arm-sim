import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from vla.run import configure_libero,connect_policy,observation_payload,validate_actions
from vla.semantic_grasp import image_bytes
configure_libero()
from libero.libero import benchmark,get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
import imageio.v2 as imageio
source=Path(__file__).resolve().parent/"scene-source"
meta=json.loads((source/"report.json").read_text())
task=benchmark.get_benchmark_dict()[meta["suite"]]().get_task(meta["task_id"])
import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--output",type=Path,required=True)
args=parser.parse_args()
client=connect_policy(8766)
for name,prompt in [("source-only","pick up the wine bottle"),("official-rack-language",task.language)]:
 output=args.output/name
 output.mkdir(parents=True,exist_ok=False)
 env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path("bddl_files"))/task.problem_folder/task.bddl_file),camera_heights=256,camera_widths=256,ignore_done=True)
 writer=None
 trace=[]
 try:
  env.seed(meta["seed"]);env.reset()
  obs=env.set_init_state(np.load(source/"initial_state.npy"))
  for _ in range(10):obs=env.step([0.]*6+[-1.])[0]
  for action in np.load(source/"executed_actions.npy"):obs=env.step(action.tolist())[0]
  report=dict(diagnostic_only=True,original_user_goal="把酒瓶放柜子上",probe_prompt=prompt,policy=client.get_server_metadata(),replayed_actions=210,noise_seed_rule="42 + control_step // 5",result="no_lifted_grasp_candidate",actual_task_success=None)
  raw=image_bytes(np.ascontiguousarray(env.sim.render(width=512,height=512,camera_name="agentview")[::-1,::-1]))
  (output/"input.png").write_bytes(raw)
  report["input_sha256"]=hashlib.sha256(raw).hexdigest()
  writer=imageio.get_writer(output/"demo.mp4",fps=20)
  queue=[];min_z=None
  for step in range(180):
   if not queue:
    payload=observation_payload(obs,prompt,image_tools)
    payload["_probe_noise_seed"]=42+step//5
    actions=validate_actions(client.infer(payload)["actions"],5)
    queue.extend(actions[:5])
   action=np.asarray(queue.pop(0))
   eef=np.asarray(obs["robot0_eef_pos"])
   gap=float(np.abs(obs["robot0_gripper_qpos"]).sum())
   if action[-1]>.5 and (min_z is None or eef[2]<min_z):min_z=float(eef[2])
   if min_z is not None and eef[2]>min_z+.07 and .003<gap<.075 and action[-1]>.5:
    report["result"]="lifted_grasp_candidate_requires_video_review"
    break
   obs=env.step(action.tolist())[0]
   trace.append(dict(step=step,eef=obs["robot0_eef_pos"].tolist(),gripper=obs["robot0_gripper_qpos"].tolist(),action=action.tolist()))
   writer.append_data(np.ascontiguousarray(obs["agentview_image"][::-1,::-1]))
  report["control_steps"]=len(trace)
  report["minimum_closed_eef_z"]=min_z
  (output/"final.png").write_bytes(image_bytes(np.ascontiguousarray(env.sim.render(width=512,height=512,camera_name="agentview")[::-1,::-1])))
  (output/"result.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
  (output/"trace.json").write_text(json.dumps(trace,indent=2))
  print(json.dumps(report,ensure_ascii=False),flush=True)
 finally:
  if writer:writer.close()
  env.close()
