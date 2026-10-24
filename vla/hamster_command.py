"""Fresh Chinese text/recorded speech -> actual HAMSTER -> isolated hybrid trial.

Fixed LIBERO goal8/init0 scene, single grasp/place only. No object/action keyword
table and no automatic success scoring for a custom instruction. Review the video.
The planner runs on CPU; existing GPU services and user scenes stay untouched.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from vla.run import ROOT, translate_prompt, voice_prompt


def save_report(output, report):
    temporary = output / 'workflow.tmp'
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(output / 'workflow.json')


def run_stage(command, environment, timeout):
    child = subprocess.Popen(command, cwd=ROOT, env=environment)
    try:
        code = child.wait(timeout=timeout)
        if code:
            raise subprocess.CalledProcessError(code, command)
    finally:
        if child.poll() is None:
            child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--prompt')
    source.add_argument('--audio', type=Path, help='Local WAV; actual ASR, not a text surrogate')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--viewer', action='store_true')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite nonempty output: {args.output}')
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(result='started', training=False, fixed_scene='libero_goal/8/init0/seed42',
                  semantic_success=None, stages=[])
    started = time.monotonic()
    environment = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       MUJOCO_GL='egl', OMP_NUM_THREADS='8')
    sim_python = str(ROOT / '.venv-vla-sim/bin/python')
    try:
        print('正在识别／翻译中文指令（本地模型）', flush=True)
        original, english = (voice_prompt(args.audio.read_bytes()) if args.audio else
                             (args.prompt, translate_prompt(args.prompt)))
        report.update(original=original, english=english, input_mode='audio' if args.audio else 'text')
        commands = args.output / 'commands.json'
        commands.write_text(json.dumps([dict(chinese=original, english=english)], ensure_ascii=False, indent=2))
        stages = [
            ('capture', [sim_python, '-u', '-m', 'vla.capture_hamster_scene',
                         '--output', str(args.output / 'scene')], 120),
            ('planner', [str(ROOT / '.venv-hamster/bin/python'), '-u', '-m', 'vla.hamster',
                         '--image', str(args.output / 'scene/scene-init0.png'),
                         '--commands', str(commands), '--output', str(args.output / 'planner')], 1800),
            ('execution', [sim_python, '-u', '-m', 'vla.hybrid', '--unscored',
                           '--plan', str(args.output / 'planner/command-00.json'),
                           '--output', str(args.output / 'execution'), '--port', str(args.port)]
                          + (['--viewer'] if args.viewer else []), 300),
        ]
        for name, command, timeout in stages:
            print('阶段：' + name, flush=True)
            report['stages'].append(dict(stage=name, command=command, state='running'))
            save_report(args.output, report)
            stage_env = dict(environment, MUJOCO_GL='glfw' if name == 'execution' and args.viewer else 'egl')
            run_stage(command, stage_env, timeout)
            report['stages'][-1]['state'] = 'finished'
        execution = json.loads((args.output / 'execution/result.json').read_text())
        report['result'] = execution['result']
    except KeyboardInterrupt:
        report['result'] = 'interrupted'
        raise
    except Exception as exc:
        report.update(result='exception', exception=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['wall_duration_s'] = time.monotonic() - started
        save_report(args.output, report)
        print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
