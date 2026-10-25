import json, sys, time
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from PIL import ImageGrab
from vla.run import parse_args
from vla.panel import ChinesePanel
args=parse_args(["--panel","--viewer","--suite","libero_goal","--task-id","8","--initial-state","0","--prompt","把酒瓶放到盘子上","--port","8766","--output","runs/semantic-20261025/panel-v3"])
p=ChinesePanel(args)
p.root.title("中文视觉抓放 · 模型看图选择目标（测试中）")
p.root.geometry("930x1370+10+20")
p.load()
started=time.monotonic(); stage="loading"; button=None; control_start=None
def descendants(widget):
    for child in widget.winfo_children():
        yield child
        yield from descendants(child)
def monitor():
    global stage,button,control_start
    s=p.lab.state()
    try:
        if stage=="loading" and s["ready"]:
            button=next(w for w in descendants(p.root) if w.winfo_class()=="TButton" and w.cget("text")=="视觉理解抓放")
            p.root.update_idletasks()
            assert button.winfo_ismapped()
            assert button.winfo_rooty()+button.winfo_height()<p.root.winfo_rooty()+p.root.winfo_height()
            import subprocess, os
            wins=subprocess.check_output(["xdotool","search","--pid",str(os.getpid()),"--name","MuJoCo"],text=True).split()
            assert wins, "No owned MuJoCo window"
            for win in wins:
                subprocess.run(["xdotool","windowsize",win,"1510","1300"],check=True)
                subprocess.run(["xdotool","windowmove",win,"960","40"],check=True)
            p.root.lift()
            control_start=len(p.lab.actions)
            button.invoke()
            stage="executing"
        elif stage=="executing" and s["steps"]>control_start and not s["running"] and not p.lab.semantic_active:
            children=list(p.lab.output.glob("semantic_*/result.json"))
            if not children:
                raise RuntimeError("No model result")
            result=json.loads(children[-1].read_text())
            assert result["result"]=="motion_completed_unscored",result
            assert result["task"]["relation"]=="on"
            assert result["control_steps"]>0 and result["inference_calls"]>0
            p.root.title("中文视觉抓放 · 已暂停，可自行试用")
            receipt={k:v for k,v in s.items() if k!="images"}
            receipt.update(status="PASS: real model command completed, semantic success requires video",semantic_output=str(children[-1].parent),button_bounds=[button.winfo_rootx(),button.winfo_rooty(),button.winfo_width(),button.winfo_height()])
            Path(".local/visual-semantic-20261025/panel-verification.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
            p.root.after(500,lambda:ImageGrab.grab(xdisplay=":0").save(".local/visual-semantic-20261025/panel-desktop.png"))
            print(json.dumps(receipt,ensure_ascii=False),flush=True)
            stage="finished"
        if stage not in ("finished","failed") and (time.monotonic()-started>420 or "操作失败" in s["status"]):
            raise RuntimeError(s["status"])
    except Exception as exc:
        p.send("pause")
        Path(".local/visual-semantic-20261025/panel-error.txt").write_text(repr(exc))
        print("PANEL FAILED",repr(exc),flush=True)
        stage="failed"
    p.root.after(150,monitor)
p.root.after(150,monitor)
p.root.mainloop()
