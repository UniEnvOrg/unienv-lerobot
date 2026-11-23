"""Small LeRobot policies for the tests: a random-weight ACT and a scripted chunk policy (no downloads)."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Any, List, Optional

import numpy as np
import pytest
import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from unienv_interface.backends.numpy import NumpyComputeBackend as B
from unienv_interface.space import BoxSpace, DictSpace


def pytest_addoption(parser):
    parser.addoption("--run-gpu", action="store_true", help="run tests that need a GPU and download models")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-gpu"):
        return
    skip = pytest.mark.skip(reason="needs --run-gpu")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


STATE, ENV, ACTION_DIM = 4, 3, 2
ACT_INPUTS = {
    "observation.state": PolicyFeature(FeatureType.STATE, (STATE,)),
    "observation.environment_state": PolicyFeature(FeatureType.ENV, (ENV,)),
}
ACT_OUTPUTS = {"action": PolicyFeature(FeatureType.ACTION, (ACTION_DIM,))}


def tiny_act(chunk_size: int = 8, n_action_steps: int = 8, seed: int = 0):
    """A randomly initialized ACT on (state, env state) and its processors."""
    from lerobot.policies.factory import get_policy_class, make_policy_config, make_pre_post_processors

    config = make_policy_config(
        "act", device="cpu", input_features=dict(ACT_INPUTS), output_features=dict(ACT_OUTPUTS),
        chunk_size=chunk_size, n_action_steps=n_action_steps, dim_model=32, n_heads=2, dim_feedforward=64,
        n_encoder_layers=1, n_decoder_layers=1, use_vae=False, pretrained_backbone_weights=None,
    )
    torch.manual_seed(seed)
    policy = get_policy_class("act")(config)
    stats = {key: {"mean": torch.full(f.shape, 0.1), "std": torch.full(f.shape, 2.0)}
             for key, f in {**ACT_INPUTS, **ACT_OUTPUTS}.items()}
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=stats)
    return policy, preprocessor, postprocessor


def act_spaces(batch_size: Optional[int] = None):
    shape = () if batch_size is None else (batch_size,)
    observation_space = DictSpace(B, {
        "state": BoxSpace(B, -5.0, 5.0, np.float32, shape=shape + (STATE,)),
        "env": BoxSpace(B, -5.0, 5.0, np.float32, shape=shape + (ENV,)),
    })
    action_space = BoxSpace(B, -10.0, 10.0, np.float32, shape=shape + (ACTION_DIM,))
    return observation_space, action_space


ACT_KEYS = {"observation.state": "state", "observation.environment_state": "env"}


def act_observation(value: float, batch_size: Optional[int] = None):
    shape = () if batch_size is None else (batch_size,)
    return {"state": np.full(shape + (STATE,), value, np.float32), "env": np.full(shape + (ENV,), -value, np.float32)}


def identity_processors():
    return (PolicyProcessorPipeline(steps=[]),
            PolicyProcessorPipeline(steps=[], to_transition=policy_action_to_transition,
                                    to_output=transition_to_policy_action))


class ScriptedChunkPolicy:
    """A chunk policy whose actions say where they come from: ``[request index, step in the chunk]``.

    Records every ``predict_action_chunk`` call (the observation value, RTC delay and prefix);
    ``sleep`` makes predictions take that long (wall clock).
    """

    def __init__(self, chunk_size: int = 10, rtc: bool = True, sleep: float = 0.0, batch_size: int = 1):
        self.config = SimpleNamespace(
            input_features={"observation.state": PolicyFeature(FeatureType.STATE, (1,))},
            output_features={"action": PolicyFeature(FeatureType.ACTION, (2,))},
            device="cpu", rtc_config=None, chunk_size=chunk_size,
        )
        self.chunk_size, self.rtc, self.sleep = chunk_size, rtc, sleep
        self.calls: List[dict] = []
        self.resets = self.drops = 0
        self.threads: List[str] = []

    def eval(self):
        return self

    def reset(self):
        self.resets += 1

    def drop_queued_actions(self):
        self.drops += 1

    def supports_rtc(self) -> bool:
        return self.rtc

    def init_rtc_processor(self):
        self.rtc_initialized = True

    def chunk_inference_spec(self):
        return None

    def select_action(self, batch):
        raise AssertionError("not used in async mode")

    def predict_action_chunk(self, batch, inference_delay=None, prev_chunk_left_over=None, execution_horizon=None):
        self.threads.append(threading.current_thread().name)
        if self.sleep:
            time.sleep(self.sleep)
        index = len(self.calls)
        state = batch["observation.state"]
        self.calls.append({"state": state[:, 0].clone(), "delay": inference_delay, "horizon": execution_horizon,
                           "prefix": None if prev_chunk_left_over is None else prev_chunk_left_over.clone(),
                           "grad": torch.is_grad_enabled(), "inference_mode": torch.is_inference_mode_enabled()})
        steps = torch.arange(self.chunk_size, dtype=torch.float32)
        chunk = torch.stack([torch.full_like(steps, float(index)), steps], dim=-1)
        return chunk.expand(state.shape[0], -1, -1).clone()


def scripted_spaces(batch_size: Optional[int] = None):
    shape = () if batch_size is None else (batch_size,)
    return (DictSpace(B, {"x": BoxSpace(B, -1e6, 1e6, np.float32, shape=shape + (1,))}),
            BoxSpace(B, -1e6, 1e6, np.float32, shape=shape + (2,)))


def scripted_observation(value: float, batch_size: Optional[int] = None) -> Any:
    shape = () if batch_size is None else (batch_size,)
    return {"x": np.full(shape + (1,), value, np.float32)}
