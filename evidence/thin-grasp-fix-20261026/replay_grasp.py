"""Replay grasp A/B with fixed physics state and matching recorded source box."""
import importlib.util, inspect, json, os, sys, textwrap, time
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT)); os.environ.setdefault('MUJOCO_GL','egl')
from vla.visual_control import VisualControl
from vla.semantic_grasp import region_point,image_bytes
import vla.visual_control as module
spec=importlib.util.spec_from_file_location('probe',ROOT/'evidence/recovery-drawer-20261025/probe.py'); probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
import argparse
parser=argparse.ArgumentParser(description='Same-state diagnostic; uses recorded matching-state VLM box, not a fresh VLM call')
parser.add_argument('mode', choices=['baseline','topband'])
parser.add_argument('--output', required=True, type=Path)
args=parser.parse_args(); mode=args.mode
if mode=='baseline':
    code=textwrap.dedent(inspect.getsource(VisualControl.grasp))
    old="threshold = (top - .003 if top - self.report['source']['tabletop'] < .04\n                 else np.quantile(visible[:, 2], .65))"
    assert old in code, 'Grasp implementation changed; update the diagnostic before replay'
    code=code.replace(old, 'threshold = np.quantile(visible[:, 2], .65)', 1)
    scope=dict(module.__dict__);exec(code,scope);VisualControl.grasp=scope['grasp']
assert not args.output.exists(), 'Use a fresh output directory'
env,obs=probe.scene(); out=args.output
try:
    obs=env.set_init_state(np.load(ROOT/'evidence/combination-fix-20261026/runs/final-v12/drawer-chain/command-02/start-state.npy'))
    skill=VisualControl(env,obs,None,out,status=print);skill.started=time.monotonic()
    rgb,cloud=skill.capture();(out/'input.png').write_bytes(image_bytes(rgb))
    task=json.loads((ROOT/'evidence/combination-fix-20261026/runs/final-v12/drawer-chain/command-02/step-01/result.json').read_text())['task']
    source,mask,meta=region_point(cloud,task['source']['bbox_2d']);skill.report['source']=meta
    before=env.sim.data.get_body_xpos('cream_cheese_1_main').copy()
    skill.writer=imageio.get_writer(out/'demo.mp4',fps=20)
    try:
        skill.grasp(task,source,mask,cloud); skill.confirm_hold(); result='lifted'
    except Exception as exc:
        result=f'{type(exc).__name__}: {exc}'
    after=env.sim.data.get_body_xpos('cream_cheese_1_main').copy()
    held=bool(env.env.check_contact(env.robots[0].gripper.contact_geoms,env.env.get_object('cream_cheese_1')))
    record=dict(mode=mode,result=result,physical_lift=float(after[2]-before[2]),physical_contact=held,source_before=before.tolist(),source_after=after.tolist(),report=skill.report)
    (out/'result.json').write_text(json.dumps(record,indent=2));(out/'trace.json').write_text(json.dumps(skill.trace,indent=2));(out/'final.png').write_bytes(image_bytes(skill.capture()[0])); print('RECEIPT',json.dumps(record),flush=True)
finally:
    skill.writer.close();env.close()
