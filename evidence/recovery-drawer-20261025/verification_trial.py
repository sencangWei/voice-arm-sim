"""Check actual VLM judgments on open and closed frames, including negatives."""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from vla.visual_control import VisualControl
from vla.drawer_control import drawer_depth_observation


source, output = map(Path, sys.argv[1:3])
results = []
for action, frame, expected in (
    ('open', 'step-01/input.png', False),
    ('open', 'step-01/final.png', True),
    ('close', 'step-03/input.png', False),
    ('close', 'step-03/final.png', True),
):
    number = '01' if action == 'open' else '03'
    metadata = json.loads((source / f'step-{number}/result.json').read_text())
    rgb = np.asarray(Image.open(source / frame).convert('RGB'))
    state = (0 if frame == 'step-01/input.png' else 1 if frame == 'step-01/final.png'
             else 2 if frame == 'step-03/input.png' else 3)
    cloud = np.load(source / f'depth-{int(number)}-{state}/cloud.npz')['cloud']
    initial_cloud = np.load(source / f'step-{number}/geometry.npz')['cloud']
    obs = dict(robot0_gripper_qpos=np.array([.04, -.04]),
               robot0_eye_in_hand_image=np.zeros_like(rgb))
    skill = VisualControl(None, obs, None, output / f'{action}-{expected}', status=print)
    skill.capture = lambda: (rgb, cloud)
    skill.verification_box = metadata['verification_box']
    skill.report['verification_action'] = action
    skill.report['drawer_verification'] = dict(geometry=metadata['geometry'],
        initial=drawer_depth_observation(initial_cloud, metadata['geometry']))
    skill.verify_completion(('打开' if action == 'open' else '关闭') + '柜子第三层抽屉', 8911)
    result = dict(action=action, frame=frame, expected=expected,
                  actual=skill.report['semantic_success'], judgment=skill.report['completion'],
                  depth=skill.report['drawer_depth_check'], model=skill.report['model_completion'])
    results.append(result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
(output / 'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
assert all(r['actual'] is r['expected'] for r in results), results
