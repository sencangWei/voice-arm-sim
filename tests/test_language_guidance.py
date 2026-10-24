import numpy as np
import pytest

from vla.language_guidance import GuidedPolicy
from vla.serve import MaskPrompt, ProbePolicy


class Model:
    def __init__(self):
        self.calls = []

    def get_server_metadata(self):
        return {"backend": "official-pi05-libero-vla", "openpi_revision":
                "15a9616a00943ada6c20a0f158e3adb39df2ccac"}

    def infer(self, obs):
        self.calls.append(dict(obs))
        return {"actions": np.full((10, 7), .2 if obs["prompt"] else -.1)}


def test_same_observation_and_noise_only_language_changes():
    model = Model()
    policy = GuidedPolicy(model, scale=1.5, seed=42)
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    obs = {"prompt": "move this unknown object there", "observation/image": image}
    response = policy.infer(obs)
    np.testing.assert_allclose(response["actions"], .35)
    assert len(model.calls) == 2
    assert model.calls[0]["prompt"] == obs["prompt"]
    assert model.calls[1]["prompt"] == ""
    assert model.calls[0]["_probe_noise_seed"] == model.calls[1]["_probe_noise_seed"] == 42
    assert model.calls[0]["observation/image"] is model.calls[1]["observation/image"] is image
    assert "_probe_noise_seed" not in obs
    policy.infer(obs)
    assert model.calls[2]["_probe_noise_seed"] == 43


def test_baseline_uses_matched_seed_without_extra_prediction():
    model = Model()
    response = GuidedPolicy(model, scale=1, seed=42).infer({"prompt": "put it there"})
    assert len(model.calls) == 1
    np.testing.assert_allclose(response["actions"], .2)
    assert response["guidance_noise_seed"] == 42


def test_empty_prompt_override_and_bad_branch_do_not_get_executed():
    model = Model()
    policy = GuidedPolicy(model, scale=1.5, seed=42)
    with pytest.raises(ValueError):
        policy.infer({"prompt": ""})
    assert not model.calls
    original = model.infer
    def broken(obs):
        result = original(obs)
        if not obs["prompt"]:
            result["actions"][0, 0] = np.nan
        return result
    model.infer = broken
    with pytest.raises(ValueError):
        policy.infer({"prompt": "move"})


def test_pinned_affine_output_contract_required():
    model = Model()
    model.get_server_metadata = lambda: {"backend": "official-molmoact2-libero-vla"}
    with pytest.raises(ValueError):
        GuidedPolicy(model, scale=1.5, seed=42)


def test_masked_branch_shares_weights_without_mutating_conditional_transform():
    transformed = {"tokenized_prompt": np.arange(3), "tokenized_prompt_mask": np.ones(3, bool),
                   "state": np.arange(8)}
    class Policy:
        _input_transform = staticmethod(lambda obs: transformed)
        def infer(self, obs, *, noise=None):
            return self._input_transform(obs), noise
    original = Policy()
    wrapper = ProbePolicy(original, 10, 32)
    masked, noise = wrapper.infer({"_probe_mask_prompt": True, "_probe_noise_seed": 42})
    assert not masked["tokenized_prompt"].any()
    assert not masked["tokenized_prompt_mask"].any()
    assert masked["state"] is transformed["state"]
    conditional, other_noise = wrapper.infer({"_probe_noise_seed": 42})
    np.testing.assert_array_equal(conditional["tokenized_prompt"], np.arange(3))
    np.testing.assert_array_equal(noise, other_noise)
    with pytest.raises(ValueError):
        wrapper.infer({"_probe_mask_prompt": "yes"})


def test_masked_guidance_requires_explicit_server_capability():
    model = Model()
    with pytest.raises(ValueError):
        GuidedPolicy(model, unconditional="mask")
    metadata = model.get_server_metadata()
    model.get_server_metadata = lambda: {**metadata, "prompt_masking": "zero-token-and-mask-v1"}
    GuidedPolicy(model, unconditional="mask").infer({"prompt": "put it there"})
    assert "_probe_mask_prompt" not in model.calls[0]
    assert model.calls[1]["_probe_mask_prompt"] is True
    assert model.calls[1]["prompt"] == "put it there"


def test_mask_transform_keeps_original_tokens_unchanged():
    tokens = np.arange(8)
    source = {"tokenized_prompt": tokens, "tokenized_prompt_mask": np.ones(8, bool)}
    output = MaskPrompt(lambda obs: source)({})
    assert output is not source
    np.testing.assert_array_equal(tokens, np.arange(8))
