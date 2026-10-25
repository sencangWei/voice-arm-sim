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
    payload, calls = mock_vision(monkeypatch, [answer])
    with pytest.raises(json.JSONDecodeError):
        model_service.visual_task(payload)
    assert len(calls) == 1


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
