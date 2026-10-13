"""Policy contract checks use the genuine hash-verified checkpoint, never driving."""
from pathlib import Path

import numpy as np
import pytest

from sdv.driving.policy import (LEGACY_SHA256, PolicyError, RoachPolicy,
                                _load_checkpoint, _normalise_state_dict, validate_observation)

CHECKPOINT = Path(__file__).resolve().parents[1] / "roach/log/ckpt_11833344.pth"


@pytest.fixture(scope="module")
def policy():
    torch = pytest.importorskip("torch")
    pytest.importorskip("gym")
    if not CHECKPOINT.is_file():
        pytest.skip("Original checkpoint is not installed")
    torch.set_num_threads(2)
    return RoachPolicy(CHECKPOINT)


def observation():
    return {"birdview": np.zeros((15, 192, 192), np.uint8),
            "state": np.array([0, 0, 0, 1, 0, 0], np.float32)}


def test_original_checkpoint_inference_is_deterministic(policy):
    a = policy.predict(observation())
    b = policy.predict(observation())
    assert a["raw_action"] == b["raw_action"]
    assert len(a["raw_action"]) == 2
    assert np.isfinite(a["raw_action"]).all()
    assert a["inference_ms"] > 0
    assert policy.metadata["checkpoint_sha256"] == LEGACY_SHA256
    assert policy.metadata["original_checkpoint_verified"] is True


@pytest.mark.parametrize("mutation", ["wrong_shape", "nonfinite_state", "nonfinite_birdview", "pixel_range", "missing"])
def test_bad_observation_rejected_before_forward(policy, monkeypatch, mutation):
    obs = observation()
    if mutation == "wrong_shape":
        obs["birdview"] = np.zeros((15, 384, 384), np.uint8)
    elif mutation == "nonfinite_state":
        obs["state"][0] = np.nan
    elif mutation == "nonfinite_birdview":
        obs["birdview"] = obs["birdview"].astype(np.float32)
        obs["birdview"][0, 0, 0] = np.inf
    elif mutation == "pixel_range":
        obs["birdview"] = np.full((15, 192, 192), 256, np.float32)
    else:
        del obs["birdview"]
    monkeypatch.setattr(policy._model, "forward", lambda *a, **kw: pytest.fail("invalid observation reached model"))
    with pytest.raises(PolicyError):
        policy.predict(obs)


def test_seven_return_values_and_nonfinite_action_rejected(policy, monkeypatch):
    monkeypatch.setattr(policy._model, "forward", lambda *a, **kw: (None,) * 6)
    with pytest.raises(PolicyError, match="7 outputs"):
        policy.predict(observation())
    monkeypatch.setattr(policy._model, "forward", lambda *a, **kw: (np.array([[np.nan, 0]]), None, None, None, None, None, None))
    with pytest.raises(PolicyError, match="nonfinite action"):
        policy.predict(observation())


def test_missing_checkpoint_no_random_fallback(tmp_path):
    pytest.importorskip("torch")
    with pytest.raises(PolicyError, match="not found"):
        RoachPolicy(tmp_path / "missing.pth")


def test_checksum_mismatch_rejected_before_deserialization(tmp_path):
    pytest.importorskip("torch")
    path = tmp_path / "incorrect.pth"
    path.write_bytes(b"not a model")
    with pytest.raises(PolicyError, match="SHA256 mismatch"):
        _load_checkpoint(path, expected_sha256=LEGACY_SHA256)


def test_legacy_conversion_does_not_corrupt_current_layout():
    original = {f"features_extractor.cnn.{i}.weight": i for i in [0, 2, 4, 6, 8, 10]}
    converted, layout = _normalise_state_dict(original)
    assert layout == "sequential_to_modulelist"
    assert [converted[f"features_extractor.cnn.{i}.weight"] for i in range(6)] == [0, 2, 4, 6, 8, 10]
    unchanged, layout = _normalise_state_dict(converted)
    assert layout == "modulelist"
    assert unchanged == converted


def test_same_outputs_after_tensor_only_checkpoint_roundtrip(policy, tmp_path):
    torch = pytest.importorskip("torch")
    target = tmp_path / "tensor_only.pth"
    torch.save({"policy_state_dict": policy._model.state_dict(), "policy_init_kwargs": {
        "observation_space": {"birdview_shape": [15, 192, 192], "state_shape": [6]},
        "action_space": {"low": [-1, -1], "high": [1, 1]},
        "policy_head_arch": [256, 256], "value_head_arch": [256, 256],
        "features_extractor_kwargs": {"states_neurons": [256, 256]},
        "distribution_kwargs": {"dist_init": None}}}, target)
    reloaded = RoachPolicy(target)
    assert reloaded.metadata["parameter_layout"] == "modulelist"
    assert reloaded.predict(observation())["raw_action"] == policy.predict(observation())["raw_action"]
