import os,sys,json,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
os.environ["MUJOCO_GL"]="egl"
receipt=Path(".local/visual-semantic-20261025/panel-verification.json")
for _ in range(600):
    if receipt.exists(): break
    time.sleep(1)
else: raise TimeoutError("GUI acceptance pending")
from vla.run import configure_libero,load_initial_states,connect_policy
configure_libero()
from libero.libero import benchmark,get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from vla.semantic_grasp import VisualSkill
from vla.compare import BottleInBowlCheck
task=benchmark.get_benchmark_dict()["libero_goal"]().get_task(8)
env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path("bddl_files"))/task.problem_folder/task.bddl_file),camera_heights=512,camera_widths=512)
client=None
try:
    env.seed(42);env.reset();obs=env.set_init_state(load_initial_states(task)[0])
    for _ in range(10): obs=env.step([0.]*6+[-1.])[0]
    client=connect_policy(8766)
    base=Path("runs/semantic-20261025/sequence")
    for index,text in enumerate(["把酒瓶放进碗里","把酒瓶从碗里拿出来放到空的桌面上"]):
        out=base/str(index)
        skill=VisualSkill(env,obs,client,out)
        skill.report.update(sequence_index=index,reset_between_commands=False,seed=42,initial_state=0)
        try:
            report=skill.run(text)
            print(json.dumps(report,ensure_ascii=False),flush=True)
            obs=skill.obs
            if index==0:
                check=BottleInBowlCheck()
                success=False
                for _ in range(5): success=check(env)
                (out/"evaluation.json").write_text(json.dumps({"success":success,"definition":check.definition,"note":"five evaluations at settled end state, no controller input"},indent=2))
                if not success: break
        except Exception:
            traceback.print_exc();break
finally:
    if client: client._ws.close()
    env.close()
