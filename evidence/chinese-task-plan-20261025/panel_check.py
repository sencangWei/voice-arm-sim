"""Launch an isolated paused panel and check whole-request rejection through its UI."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from vla.panel import ChinesePanel
from vla.run import parse_args

HERE = Path(__file__).resolve().parent
args = parse_args(["--panel", "--viewer", "--suite", "libero_goal", "--task-id", "9",
    "--seed", "7", "--port", "8766", "--output", str(ROOT / "runs/chinese-task-plan-20261025/panel"),
    "--prompt", "把酒瓶放到柜子顶上"])
panel = ChinesePanel(args)
panel.root.title("中文机械臂 · 完整指令与逐步核验")
panel.root.geometry("950x1370+10+20")
started = time.monotonic()
sent = False


def check():
    global sent
    state = panel.lab.state()
    if state.get("ready") and not sent:
        panel.prompt.set("打开柜子第三层，把奶油奶酪放进去，之后合上第三层")
        panel.send("semantic", panel.prompt.get())
        sent = True
    if sent and "未执行：缺少技能" in state.get("status", ""):
        panel.root.update_idletasks()
        widgets = panel.book.winfo_children()[0].winfo_children()
        label = next(w for w in widgets if "textvariable" in w.keys()
                     and str(w.cget("textvariable")) == str(panel.task_plan_text))
        visible = label.winfo_rooty() + label.winfo_height() <= panel.book.winfo_rooty() + panel.book.winfo_height()
        debug = next(w for w in panel.book.winfo_children() if panel.book.tab(w, "text") == "VLA 调试")
        panel.book.select(debug)
        panel.root.update_idletasks()
        debug_visible = all(w.winfo_rooty() + w.winfo_height() <= panel.book.winfo_rooty() + panel.book.winfo_height()
                            for w in debug.winfo_children())
        panel.book.select(0)
        receipt = dict(status=state["status"], task_plan=state.get("task_plan"),
                       task_plan_visible=visible, vla_debug_controls_visible=debug_visible,
                       automatic_motion_steps=len(panel.lab.actions))
        (HERE / "panel-check.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
        print(json.dumps(receipt, ensure_ascii=False), flush=True)
        panel.prompt.set("先把酒瓶放在柜子顶上，再把同一个酒瓶放到灶台上")
        return
    if time.monotonic() - started > 120:
        raise RuntimeError("Panel acceptance timeout: " + str(state))
    panel.root.after(250, check)


panel.load()
panel.root.after(250, check)
panel.root.mainloop()
