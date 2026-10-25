import base64
import hashlib
import io

import numpy as np
from PIL import Image

from vla.semantic_grasp import VisualSkill


def test_completion_model_receives_current_global_and_wrist_in_one_image(tmp_path):
    global_rgb = np.full((32, 32, 3), [200, 0, 0], dtype=np.uint8)
    wrist_rgb = np.full((16, 16, 3), [0, 200, 0], dtype=np.uint8)
    obs = dict(robot0_gripper_qpos=np.array([.04, -.04]), robot0_eye_in_hand_image=wrist_rgb)
    skill = VisualSkill(None, obs, None, tmp_path/'run', status=lambda _: None)
    skill.capture = lambda: (global_rgb, None)
    requests = []
    def request(port, endpoint, payload):
        requests.append(payload)
        raws = [base64.b64decode(value) for value in payload['images']]
        return dict(image_sha256=[hashlib.sha256(raw).hexdigest() for raw in raws],
                    judgment=dict(verdict='uncertain', reason='仍看不清，不能确认'))
    skill.request = request
    skill.verify_completion('把盒子放在灶台上', 8911)
    first = np.asarray(Image.open(io.BytesIO(base64.b64decode(requests[0]['images'][0]))))
    assert first.shape == (32, 64, 3)
    assert np.all(first[:, :32] == global_rgb)
    assert np.all(first[:, 32:] == [0, 200, 0])
    assert '同一时刻' in requests[0]['text'] and '腕部' in requests[0]['text']
    assert skill.report['semantic_success'] is None
