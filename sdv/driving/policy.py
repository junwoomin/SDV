"""Verified LRS Roach inference, independent of CARLA actor or frame ownership.

Only a loaded checkpoint can create a ready policy. The historical checkpoint
receives its narrow metadata allowlist after its complete SHA256 is verified.
No path uses unrestricted pickle, random-policy fallback or dynamic entrypoints.
"""
from __future__ import annotations

import argparse
import codecs
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

LRS_COMMIT = "6eaf6c38450efc540f5fb1c86ed2980dd96a59bc"
LEGACY_SHA256 = "1fecec0c8a206a9ac07a6a9b0be77a5b7e95a073b20d9dfc2a7397354355094a"
DEFAULT_CHECKPOINT = "roach/log/ckpt_11833344.pth"
OBSERVATION_SHAPE = {"birdview": [15, 192, 192], "state": [6]}


class PolicyError(RuntimeError):
    """The caller must apply a fresh full-brake command when this is raised."""


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise PolicyError("Roach requires the Python 3.10+ environment in requirements/roach.txt") from exc
    if not hasattr(torch.serialization, "safe_globals"):
        raise PolicyError("Roach requires PyTorch >=2.6 with restricted weights_only checkpoint loading")
    return torch


def _legacy_safe_globals():
    import gym
    # The historical pickle refers to the old path, including on NumPy 2.
    try:
        from numpy._core.multiarray import _reconstruct
    except ImportError:
        from numpy.core.multiarray import _reconstruct
    from numpy.random._pickle import __randomstate_ctor
    allowed = [gym.spaces.Dict, gym.spaces.Box, np.dtype, np.ndarray,
               (_reconstruct, "numpy.core.multiarray._reconstruct"), codecs.encode,
               __randomstate_ctor, np.random.RandomState]
    allowed += [type(np.dtype(t)) for t in ("float32", "float64", "uint32", "uint8", "int64", "int32", "bool")]
    return allowed


def _load_checkpoint(path: Path, expected_sha256: str | None = None):
    torch = _torch()
    if not path.is_file():
        raise PolicyError(f"Roach checkpoint not found: {path}. Random-weight driving is disabled.")
    if not 0 < path.stat().st_size <= 512 * 1024 * 1024:
        raise PolicyError("Roach checkpoint must be a nonempty file no larger than 512 MiB")
    # Hash and deserialize the same open file, so pathname replacement cannot
    # change which file receives the historical metadata allowlist.
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _stream_sha256(stream)
        stream.seek(0)
        if expected_sha256 and digest != expected_sha256.lower():
            raise PolicyError(f"Checkpoint SHA256 mismatch: expected {expected_sha256}, got {digest}")
        try:
            if digest == LEGACY_SHA256:
                with torch.serialization.safe_globals(_legacy_safe_globals()):
                    checkpoint = torch.load(stream, map_location="cpu", weights_only=True)
            else:
                checkpoint = torch.load(stream, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise PolicyError(f"Restricted checkpoint load failed ({type(exc).__name__}); the file was not accepted. No unsafe-pickle fallback is enabled.") from exc
    if not isinstance(checkpoint, Mapping) or "policy_state_dict" not in checkpoint or "policy_init_kwargs" not in checkpoint:
        raise PolicyError("Checkpoint must contain policy_state_dict and policy_init_kwargs")
    return checkpoint, digest


def _stream_sha256(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _normalise_state_dict(state_dict):
    """Map old Sequential convolution indices without overwriting live keys."""
    if not isinstance(state_dict, Mapping):
        raise PolicyError("policy_state_dict must be a mapping")
    prefix = "features_extractor.cnn."
    indexes = {int(k.split(".")[2]) for k in state_dict if k.startswith(prefix) and k.endswith(".weight")}
    if indexes == {0, 2, 4, 6, 8, 10}:
        converted = {}
        for key, value in state_dict.items():
            if key.startswith(prefix):
                parts = key.split(".")
                parts[2] = str(int(parts[2]) // 2)
                key = ".".join(parts)
            if key in converted:
                raise PolicyError("Duplicate convolution parameter after checkpoint conversion")
            converted[key] = value
        return converted, "sequential_to_modulelist"
    if indexes == {0, 1, 2, 3, 4, 5}:
        return dict(state_dict), "modulelist"
    raise PolicyError(f"Unsupported Roach CNN parameter layout: {sorted(indexes)}")


def _spaces(kwargs):
    """Validate architecture metadata. Never import checkpoint entrypoints."""
    import gym
    if not isinstance(kwargs, Mapping):
        raise PolicyError("policy_init_kwargs must be a mapping")
    obs = kwargs.get("observation_space")
    action = kwargs.get("action_space")
    # A tensor-only export can express spaces as plain dictionaries.
    if isinstance(obs, Mapping) and "birdview_shape" in obs:
        if list(obs["birdview_shape"]) != [15, 192, 192] or list(obs.get("state_shape", [])) != [6]:
            raise PolicyError("Checkpoint observation shape must be birdview(15,192,192), state(6)")
        obs = gym.spaces.Dict({"birdview": gym.spaces.Box(0, 255, (15, 192, 192), dtype=np.uint8),
                               "state": gym.spaces.Box(-np.inf, np.inf, (6,), dtype=np.float32)})
    if isinstance(action, Mapping):
        action = gym.spaces.Box(low=np.asarray(action["low"], np.float32), high=np.asarray(action["high"], np.float32), dtype=np.float32)
    try:
        valid = tuple(obs["birdview"].shape) == (15, 192, 192) and tuple(obs["state"].shape) == (6,)
        valid &= np.all(obs["birdview"].low == 0) and np.all(obs["birdview"].high == 255)
        valid &= tuple(action.shape) == (2,) and np.all(action.low == -1) and np.all(action.high == 1)
    except (AttributeError, KeyError, TypeError) as exc:
        raise PolicyError("Checkpoint observation/action space metadata is missing or malformed") from exc
    if not valid:
        raise PolicyError("Checkpoint requires incompatible observation/action spaces; expected (15,192,192)+(6) -> acceleration/steer [-1,1]")
    for key, suffix in (("features_extractor_entry_point", ":XtMaCNN"), ("distribution_entry_point", ":BetaDistribution")):
        if key in kwargs and not str(kwargs[key]).endswith(suffix):
            raise PolicyError(f"Unsupported checkpoint architecture: {key}={kwargs[key]}")
    return obs, action


def validate_observation(observation: Mapping[str, Any]):
    if not isinstance(observation, Mapping) or not {"birdview", "state"}.issubset(observation):
        raise PolicyError("Observation must contain birdview and state")
    output = {}
    for name, shape in (("birdview", (15, 192, 192)), ("state", (6,))):
        value = np.asarray(observation[name])
        if value.shape == shape:
            value = value[None, ...]
        if value.shape != (1,) + shape:
            raise PolicyError(f"{name} shape {value.shape} is invalid; expected {shape} or {(1,) + shape}")
        if value.dtype.kind not in "uif" or not np.isfinite(value).all():
            raise PolicyError(f"{name} must contain finite numeric values")
        if name == "birdview" and (np.min(value) < 0 or np.max(value) > 255):
            raise PolicyError("birdview values must be in [0,255]; policy performs /255 normalization")
        output[name] = np.ascontiguousarray(value, dtype=np.float32)
    return output


class RoachPolicy:
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, device="cpu", deterministic=True,
                 observation_profile="lrs_loop", expected_sha256=None):
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.device = str(device)
        self.deterministic = bool(deterministic)
        self.observation_profile = str(observation_profile)
        if self.observation_profile != "lrs_loop":
            raise PolicyError("Only the explicitly implemented lrs_loop observation profile is supported")
        self._model = None
        self._status = "loading"
        torch = _torch()
        if self.device not in {"cpu", "cuda"}:
            raise PolicyError("Device must be cpu or cuda")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise PolicyError("CUDA was requested but is unavailable; choose CPU explicitly")
        saved, self.checkpoint_sha256 = _load_checkpoint(self.checkpoint, expected_sha256)
        kwargs = saved["policy_init_kwargs"]
        obs_space, action_space = _spaces(kwargs)
        from .vendor.roach.models.ppo_policy import PpoPolicy
        try:
            model = PpoPolicy(observation_space=obs_space, action_space=action_space,
                              policy_head_arch=kwargs.get("policy_head_arch", [256, 256]),
                              value_head_arch=kwargs.get("value_head_arch", [256, 256]),
                              features_extractor_kwargs=kwargs.get("features_extractor_kwargs", {}),
                              distribution_kwargs=kwargs.get("distribution_kwargs", {}), device="cpu")
            state_dict, self.parameter_layout = _normalise_state_dict(saved["policy_state_dict"])
            for key, value in state_dict.items():
                if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
                    raise PolicyError(f"Checkpoint parameter is not a finite tensor: {key}")
            model.load_state_dict(state_dict, strict=True)
            model.to(self.device)
            model.device = self.device
            model.action_dist.device = self.device
            model.eval()
        except PolicyError:
            raise
        except Exception as exc:
            raise PolicyError(f"Checkpoint architecture/weights mismatch: {exc}") from exc
        self._model = model
        self._status = "ready"

    @property
    def metadata(self):
        return {"policy_name": "LRS Roach PPO", "status": self._status,
                "checkpoint": str(self.checkpoint), "checkpoint_sha256": self.checkpoint_sha256,
                "original_checkpoint_verified": self.checkpoint_sha256 == LEGACY_SHA256,
                "device": self.device, "deterministic": self.deterministic,
                "observation_shape": dict(OBSERVATION_SHAPE), "observation_profile": self.observation_profile,
                "state_fields": ["throttle", "steer", "brake", "lrs_categorical_gear", "ego_velocity_x_m_s", "ego_velocity_y_m_s"],
                "gear_contract": "LRS P=0,R=1,N=2,D=3; CARLA forward automatic gears map to D",
                "policy_ppm": 5, "lrs_commit": LRS_COMMIT,
                "checkpoint_training_observation_profile": "not recorded in checkpoint; must validate CARLA transfer",
                "parameter_layout": self.parameter_layout}

    def predict(self, observation):
        if self._model is None or self._status != "ready":
            raise PolicyError("Roach policy is not ready")
        arrays = validate_observation(observation)
        torch = _torch()
        started = time.perf_counter()
        try:
            tensors = {k: torch.as_tensor(v, device=self.device) for k, v in arrays.items()}
            with torch.inference_mode():
                result = self._model.forward(tensors, deterministic=self.deterministic, clip_action=False)
            # LRS PpoPolicy.forward returns SEVEN objects; do not use the broken
            # legacy RlBirdviewAgent.run_step six-target unpacking path.
            if not isinstance(result, tuple) or len(result) != 7:
                raise PolicyError("LRS policy forward must return 7 outputs")
            actions, values, log_probs, mu, sigma, features, cnn_features = result
            action = np.asarray(actions, dtype=np.float64)
            if action.shape != (1, 2) or not np.isfinite(action).all():
                raise PolicyError("Policy returned malformed or nonfinite action")
            if not np.isfinite(np.asarray(values)).all():
                raise PolicyError("Policy returned nonfinite value")
            if np.any(np.abs(action) > 1.0 + 1e-5):
                raise PolicyError("Policy action exceeded acceleration/steering bounds")
            if self.device == "cuda":
                torch.cuda.synchronize()
        except PolicyError:
            raise
        except Exception as exc:
            raise PolicyError(f"Roach inference failed: {type(exc).__name__}: {exc}") from exc
        return {"raw_action": action[0].tolist(), "inference_ms": (time.perf_counter() - started) * 1000,
                "value": float(np.asarray(values).reshape(-1)[0]),
                "action_mu": np.asarray(mu)[0].tolist(), "action_sigma": np.asarray(sigma)[0].tolist()}


def inspect_checkpoint(checkpoint=DEFAULT_CHECKPOINT, device="cpu", deterministic=True, observation_profile="lrs_loop"):
    policy = RoachPolicy(checkpoint, device, deterministic, observation_profile)
    return policy.metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description="Strict Roach checkpoint inspection")
    parser.add_argument("command", choices=["inspect"])
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--observation-profile", default="lrs_loop")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(inspect_checkpoint(args.checkpoint, args.device, observation_profile=args.observation_profile)))
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc), "checkpoint": args.checkpoint}))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
