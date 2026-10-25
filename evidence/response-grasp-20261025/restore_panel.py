"""Reload updated code while preserving the reported failed scene. Optional GUI QA."""
import argparse
import hashlib
import json
from pathlib import Path
import queue
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from vla.panel import ChinesePanel  # noqa: E402
from vla.run import parse_args  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--validate', action='store_true')
parser.add_argument('--followup-scene', type=Path)
options = parser.parse_args()
qa = options.validate
meta = json.loads((HERE / 'before/report.json').read_text())
args = parse_args(['--panel', '--viewer', '--suite', meta['suite'], '--task-id', str(meta['task_id']),
    '--initial-state', str(meta['initial_state']), '--seed', str(meta['seed']),
    '--replan-steps', str(meta['replan_steps']), '--port', str(meta['policy_port']),
    '--prompt', '把碗放到盘子上', '--output', str(ROOT / 'runs/response-grasp-20261025/panel')])
panel = ChinesePanel(args)
panel.root.title('中文机械臂 · 视觉定位与连续抓放')
panel.root.geometry('930x1370+10+20')
load, command = panel.lab._load, panel.lab._command
pending = True
finished = queue.Queue()
records = []
commands = ['把碗放到盘子上', '把酒瓶放到灶台上', '把酒瓶放到碗里面']
receipt = dict(result='PENDING', no_reset_between_commands=True, commands=records)


def save_receipt():
    target = ROOT / '.local/response-grasp-20261025/gui-live.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')


def restore_once(*values):
    global pending
    load(*values)
    if not pending:
        return
    lab = panel.lab
    # Replay real recorded controls to restore contact/controller state too.
    for action in np.load(HERE / 'before/executed_actions.npy', allow_pickle=False):
        lab.obs = lab.env.step(action.tolist())[0]
    expected = json.loads((HERE / 'before/command/trace.json').read_text())[-1]['eef']
    assert np.allclose(lab.obs['robot0_eef_pos'], expected, atol=1e-9)
    if options.followup_scene:
        for action in np.load(options.followup_scene / 'executed_actions.npy', allow_pickle=False):
            lab.obs = lab.env.step(action.tolist())[0]
    actual = lab.env.sim.get_state().flatten()
    lab.grip = -1. if np.abs(lab.obs['robot0_gripper_qpos']).sum() >= .06 else 1.
    lab.scene['restored_from'] = str(HERE / 'before')
    lab.scene['initial_settle_steps'] = 0
    lab.scene['initial_state_sha256'] = hashlib.sha256(actual.tobytes()).hexdigest()
    identity = {key: lab.scene[key] for key in ('suite', 'task_id', 'initial_state', 'initial_state_sha256')}
    lab.scene['scenario_key'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    np.save(lab.output / 'initial_state.npy', actual)
    lab._publish()
    lab._save_scene('reported_failure_restored')
    lab._update(status='已保留现场，当前暂停。输入中文后点击“执行当前指令”。')
    receipt.update(restoration='PASS', eef_exact_match=True, scene=str(lab.output),
        automatic_control_steps=len(lab.actions), source_failure_steps=len(np.load(HERE / 'before/executed_actions.npy')))
    save_receipt()
    pending = False
    finished.put('restored')


def observed_command(name, *values):
    try:
        command(name, *values)
    finally:
        if name == 'semantic' and qa and len(records) < 3:
            record = dict(panel.lab.scene['last_instruction_result'])
            result = json.loads((Path(record['output']) / 'result.json').read_text())
            record.update(first_control_wall_s=result.get('first_control_wall_s'),
                          wall_duration_s=result['wall_duration_s'])
            records.append(record)
            receipt['result'] = ('PASS' if len(records) == 3 and all(
                r['semantic_success'] is True for r in records) else 'INCOMPLETE')
            save_receipt()
            print(json.dumps(record, ensure_ascii=False), flush=True)
            finished.put('command')


def poll_validation():
    try:
        finished.get_nowait()
    except queue.Empty:
        panel.root.after(150, poll_validation)
        return
    if len(records) < 3 and (not records or records[-1]['semantic_success'] is True):
        text = commands[len(records)]
        panel.prompt.set(text)
        panel.send('semantic', text)
        panel.root.after(150, poll_validation)


panel.lab._load = restore_once
panel.lab._command = observed_command
panel.load()
panel.prompt.set(commands[0])
if qa:
    panel.root.after(150, poll_validation)
panel.root.mainloop()
