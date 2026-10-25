import hashlib, json, os, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from vla.run import voice_prompt
root=Path.cwd()
audio=root/'evidence/hamster/physical-hybrid/hybrid-voice-command.wav'
output=root/'runs/semantic-20261025/voice-bottle-bowl'
started=time.monotonic()
chinese, english=voice_prompt(audio.read_bytes())
receipt=dict(audio=str(audio.relative_to(root)),audio_sha256=hashlib.sha256(audio.read_bytes()).hexdigest(),chinese=chinese,english=english,asr_translation_s=time.monotonic()-started,source='existing locally synthesized Chinese WAV, NOT a live microphone')
Path('.local/visual-semantic-20261025/voice-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
print(json.dumps(receipt,ensure_ascii=False),flush=True)
assert '酒' in chinese and '碗' in chinese
result=subprocess.run([sys.executable,'-u','-m','vla.semantic_grasp','--prompt',chinese,'--output',str(output),'--initial-state','0','--score','bottle-in-bowl'],env={**os.environ,'MUJOCO_GL':'egl'})
raise SystemExit(result.returncode)
