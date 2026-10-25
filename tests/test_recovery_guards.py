import base64
import hashlib
import numpy as np
import pytest
from vla.task_runner import TaskRunner

@pytest.mark.parametrize("previous, permitted", [
    ({"result":"exception", "exception":"TimeoutError: 下降未收敛"}, False),
    ({"motion_completed":True, "completion":{"verdict":"uncertain"}}, True),
    ({"motion_completed":True, "drawer_depth_check":{"accepted":False}}, False),
])
def test_reverify_requires_finished_motion(tmp_path, previous, permitted):
    runner=TaskRunner(None, {}, None, tmp_path/"run", status=lambda s: None)
    runner.report["recoveries"]=[]
    runner.capture=lambda: (np.zeros((8,8,3),np.uint8), None)
    def request(port, endpoint, payload):
        assert ("reverify" in payload["failure"]["allowed_actions"]) is permitted
        return dict(image_sha256=hashlib.sha256(base64.b64decode(payload["image"])).hexdigest(),
                    recovery=dict(action="stop",reason="保留现场"))
    runner.request=request
    runner.recovery_decision("抓放",8911,1,dict(skill="pick_place",instruction="抓放",destination_kind="support_surface"),previous,1)
