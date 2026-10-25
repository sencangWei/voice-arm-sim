"""Regression for malformed task JSON seen in retained-scene commands."""
import base64
from contextlib import nullcontext
import io
import json
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

import model_service


BROKEN = '{"source":{"label":"wine bottle","bbox_2d":[248,256,300,416]},{"destination":{"label":"cabinet","bbox_2d":[678,351,1000,820]},"relation":"on","english":"Place the wine bottle on the cabinet."}'
VALID = BROKEN.replace('},{"destination"', '},"destination"')


def mock_vision(monkeypatch, answers):
    calls = []
    answers = iter(answers)
    class Inputs(dict):
        input_ids = np.zeros((1, 1), dtype=int)
        def to(self, device):
            return self
    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            calls.append(list(messages))
            return Inputs()
        def batch_decode(self, *args, **kwargs):
            return [next(answers)]
    model = SimpleNamespace(device="cpu", generate=lambda **kw: np.zeros((1, 2)))
    monkeypatch.setattr(model_service, "MODELS", {"vision": model, "processor": Processor()})
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    stream = io.BytesIO()
    Image.new("RGB", (8, 8)).save(stream, format="PNG")
    payload = {"text": "把酒瓶放柜子上", "image": base64.b64encode(stream.getvalue()).decode()}
    return payload, calls


def test_destination_localization_accepts_visible_regions_without_inventing_boxes(monkeypatch):
    detection = dict(bbox_2d=[515, 535, 735, 730], label='抽屉内部')
    payload, calls = mock_vision(monkeypatch, [json.dumps([detection]), '[]'])
    payload['target'] = '柜子第三层抽屉内部'
    assert model_service.ground(payload, locate=True) == detection
    prompt = calls[0][0]['content'][-1]['text']
    assert payload['target'] in prompt and '物体或区域' in prompt
    assert '当前可见' in prompt
    assert model_service.ground(payload, locate=True) == dict(bbox_2d=[], label=None)


def test_whole_request_plan_uses_dedicated_prompt_and_binds_image(monkeypatch):
    import hashlib
    plan = dict(status="unsupported", reason="缺少开关抽屉技能", missing_skills=["open_drawer"],
                steps=[dict(skill="unsupported", instruction="打开第三层抽屉", destination_kind="unsupported")])
    payload, calls = mock_vision(monkeypatch, [json.dumps(plan, ensure_ascii=False)])
    payload["text"] = "打开第三层抽屉"
    response = model_service.visual_task(payload, plan=True)
    assert response["plan"] == plan
    assert response["image_sha256"] == hashlib.sha256(base64.b64decode(payload["image"])).hexdigest()
    from vla.task_plan import PLAN_PROMPT
    assert calls[0][0]["content"][-1]["text"] == PLAN_PROMPT + payload["text"]
    assert "一次抓取并放置" not in calls[0][0]["content"][-1]["text"]
    assert len(calls[0][0]["content"]) == 1
    assert response["model_input_image_sha256"] == []


def test_observed_stove_refusal_returns_plan_instead_of_service_error(monkeypatch):
    plan = dict(status="unsupported", reason="打开炉灶不属于技能表内动作",
                missing_skills=[], steps=[])
    payload, calls = mock_vision(monkeypatch, [json.dumps(plan, ensure_ascii=False)])
    payload["text"] = "打开炉灶"
    response = model_service.visual_task(payload, plan=True)
    assert response["plan"] == plan
    assert len(calls) == 1


def test_failure_log_is_supplied_to_model_without_replacing_original_task(monkeypatch):
    plan = dict(status="clarify", reason="请先核对是否仍持有酒瓶", missing_skills=[], steps=[])
    payload, calls = mock_vision(monkeypatch, [json.dumps(plan, ensure_ascii=False)])
    payload["failure"] = {"step": 1, "completion": {"reason": "瓶子未放到柜顶"}}
    response = model_service.visual_task(payload, plan=True)
    text = calls[0][0]["content"][-1]["text"]
    assert payload["text"] in text and "瓶子未放到柜顶" in text
    assert response["plan"] == plan
    assert calls[0][0]["content"][0]["type"] == "image"
    assert response["model_input_image_sha256"] == [response["image_sha256"]]


def test_observed_extra_brace_preserves_every_model_selected_value(monkeypatch):
    payload, calls = mock_vision(monkeypatch, [BROKEN])
    response = model_service.visual_task(payload)
    assert response["task"] == json.loads(VALID)
    assert len(calls) == 1
    assert response["raw"] == BROKEN
    repair = response["serialization_repair"]
    assert repair["operation"] == "remove_extra_open_brace_before_destination"
    offset = repair["offset"]
    assert BROKEN[:offset] + BROKEN[offset+1:] == repair["normalized"] == VALID


@pytest.mark.parametrize("answer", ["{", VALID[:-1], BROKEN.replace('"destination"', '"other"')])
def test_unknown_syntax_fails_closed_without_guessing(monkeypatch, answer):
    payload, calls = mock_vision(monkeypatch, [answer, answer])
    with pytest.raises(json.JSONDecodeError):
        model_service.visual_task(payload)
    assert len(calls) == 2


def test_invalid_serialization_reasks_once_from_same_image(monkeypatch):
    broken = VALID[:-1]
    payload, calls = mock_vision(monkeypatch, [broken, VALID])
    response = model_service.visual_task(payload)
    assert response["task"] == json.loads(VALID)
    assert len(calls) == 2
    assert calls[0][0]["content"][0]["image"] is calls[1][0]["content"][0]["image"]
    assert response["serialization_retry"]["raw"] == broken
    assert "合法 JSON" in calls[1][0]["content"][-1]["text"]


def test_observed_missing_closures_preserve_all_model_values(monkeypatch):
    broken = '{"source":{"label":"酒瓶","bbox_2d":[272,310,330,481],"destination":{"label":"碗","bbox_2d":[438,505,572,612],"relation":"in","english":"Place the wine bottle into the bowl."}}'
    payload, calls = mock_vision(monkeypatch, [broken])
    response = model_service.visual_task(payload)
    assert len(calls) == 1
    assert response["raw"] == broken
    assert response["task"] == {
        "source": {"label": "酒瓶", "bbox_2d": [272, 310, 330, 481]},
        "destination": {"label": "碗", "bbox_2d": [438, 505, 572, 612]},
        "relation": "in", "english": "Place the wine bottle into the bowl."}
    repair = response["serialization_repair"]
    assert repair["operation"] == "close_source_and_destination_objects"
    assert json.loads(repair["normalized"]) == response["task"]


@pytest.mark.parametrize("answer", ["null", '{"actions":[1,2]}', BROKEN.replace('"on"', '"under"'), BROKEN.replace("248,256,300,416", "300,256,248,416")])
def test_ambiguity_and_invalid_task_schema_are_not_repaired_into_actions(monkeypatch, answer):
    payload, calls = mock_vision(monkeypatch, [answer])
    with pytest.raises(ValueError):
        model_service.visual_task(payload)
    assert len(calls) == 1


def test_valid_output_uses_one_inference(monkeypatch):
    payload, calls = mock_vision(monkeypatch, [VALID])
    assert model_service.visual_task(payload)["task"] == json.loads(VALID)
    assert len(calls) == 1


def test_leaf_prompt_reiterates_complete_schema_immediately_before_instruction(monkeypatch):
    # Recorded retained-scene requests omitted destination or the translation.
    payload, calls = mock_vision(monkeypatch, [VALID])
    model_service.visual_task(payload)
    prompt = calls[0][0]['content'][-1]['text']
    assert prompt.endswith('实际任务：\n' + payload['text'])
    assert '四个根字段 source、destination、relation、english 全部存在' in prompt
    assert '不能只输出源物体' in prompt


def test_observed_destination_wrapper_after_source_only_retry(monkeypatch):
    partial = '{"source":{"label":"blue cheese box","bbox_2d":[286,582,351,663]}}'
    broken = '{"source":{"label":"蓝色奶酪盒","bbox_2d": [286, 581, 351, 663]},{"destination":{"label":"柜子第三层抽屉","bbox_2d": [505, 357, 741, 798]}},"relation":"in","english":"Put the blue cheese box into the third layer drawer of the cabinet."}'
    payload, calls = mock_vision(monkeypatch, [partial, broken])
    response = model_service.visual_task(payload)
    assert len(calls) == 2
    assert response['task'] == {
        'source': {'label': '蓝色奶酪盒', 'bbox_2d': [286, 581, 351, 663]},
        'destination': {'label': '柜子第三层抽屉', 'bbox_2d': [505, 357, 741, 798]},
        'relation': 'in',
        'english': 'Put the blue cheese box into the third layer drawer of the cabinet.'}
    assert response['serialization_retry']['raw'] == partial
    assert response['serialization_repair']['operation'] == 'unwrap_destination_object'


def test_missing_destination_reasks_model_without_filling_values(monkeypatch):
    partial = json.dumps({'source': json.loads(VALID)['source']})
    payload, calls = mock_vision(monkeypatch, [partial, VALID])
    response = model_service.visual_task(payload)
    assert response['task'] == json.loads(VALID)
    assert len(calls) == 2
    assert response['serialization_retry']['raw'] == partial
    assert calls[0][0]['content'][0]['image'] is calls[1][0]['content'][0]['image']
    reminder = calls[1][0]['content'][-1]['text']
    assert '缺少 destination, english, relation' in reminder
    assert '不能猜坐标' in reminder


def test_incomplete_schema_stops_after_second_invalid_answer(monkeypatch):
    partial = json.dumps({'source': json.loads(VALID)['source']})
    payload, calls = mock_vision(monkeypatch, [partial, partial])
    with pytest.raises(ValueError):
        model_service.visual_task(payload)
    assert len(calls) == 2


def test_completion_uses_global_view_and_archives_request_hashes(monkeypatch):
    answer = '{"verdict":"completed","reason":"物体已放下，夹爪已撤开"}'
    payload, calls = mock_vision(monkeypatch, [answer])
    payload["images"] = [payload.pop("image")] * 2
    result = model_service.visual_task(payload, verify=True)
    content = calls[0][0]["content"]
    assert len([item for item in content if item["type"] == "image"]) == 1
    assert len(result["image_sha256"]) == 2
    assert result["model_input_image_sha256"] == result["image_sha256"][:1]
    assert result["judgment"]["verdict"] == "completed"


def test_recovery_compares_previous_and_current_images_with_history(monkeypatch):
    import hashlib
    answer = dict(action='retry_grasp', reason='物体仍在桌上，首次空抓',
                  progress='no_progress', step=None)
    payload, calls = mock_vision(monkeypatch, [json.dumps(answer, ensure_ascii=False)])
    payload['before_image'] = payload['image']
    stream = io.BytesIO()
    Image.new('RGB', (8, 8), 'red').save(stream, format='PNG')
    payload['image'] = base64.b64encode(stream.getvalue()).decode()
    payload['failure'] = dict(history=[dict(attempt=1, result='empty_grasp')])
    result = model_service.visual_task(payload, recovery=True)
    content = calls[0][0]['content']
    assert len([item for item in content if item['type'] == 'image']) == 2
    hashes = [hashlib.sha256(base64.b64decode(payload[key])).hexdigest()
              for key in ('before_image', 'image')]
    assert result['model_input_image_sha256'] == hashes
    assert result['before_image_sha256'] == hashes[0]
    assert result['image_sha256'] == hashes[1]
    assert 'empty_grasp' in content[-1]['text']
    assert result['recovery'] == answer
