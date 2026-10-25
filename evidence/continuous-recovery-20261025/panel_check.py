"""Execute implicit-prerequisite recovery through the Chinese native panel button."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from vla.panel import ChinesePanel
from vla.run import parse_args

HERE = Path(__file__).resolve().parent
output = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'runs/continuous-recovery/panel'
receipt_path = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / 'panel-check.json'
if receipt_path.exists():
    raise FileExistsError(receipt_path)
args = parse_args(['--panel', '--viewer', '--suite', 'libero_goal', '--task-id', '9',
    '--seed', '7', '--port', '8766', '--output', str(output),
    '--prompt', '把蓝色奶酪盒放进柜子第三层抽屉里面，然后关闭第三层抽屉'])
panel = ChinesePanel(args)
panel.root.title('中文机械臂 · 连续任务与自动恢复')
panel.root.geometry('950x1370+10+20')
started = time.monotonic()
sent = False


def check():
    global sent
    state = panel.lab.state()
    if state.get('ready') and not sent:
        page = panel.book.winfo_children()[0]
        buttons = [child for frame in page.winfo_children() for child in frame.winfo_children()
                   if 'text' in child.keys() and child.cget('text') == '执行当前指令']
        assert len(buttons) == 1
        buttons[0].invoke()
        sent = True
    record = getattr(panel.lab, 'scene', {}).get('last_instruction_result', {})
    if sent and record.get('result') not in (None, 'in_progress'):
        receipt = dict(status=state['status'], task_plan=state.get('task_plan'),
            result=record, automatic_motion_steps=len(panel.lab.actions),
            ready=state.get('ready'), running=state.get('running'))
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
        print(json.dumps(receipt, ensure_ascii=False), flush=True)
        assert record['semantic_success'] is True and record['completed_steps'] == 2
        return
    if time.monotonic() - started > 300:
        raise RuntimeError('Panel acceptance timeout: ' + str(state))
    panel.root.after(250, check)


panel.load()
panel.root.after(250, check)
panel.root.mainloop()
