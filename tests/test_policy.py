"""LeRobotPolicy in sync mode, on a random-weight ACT: LeRobot's loop, UniEnv's protocol."""

import numpy as np
import pytest
import torch

from lerobot.processor import TokenizerProcessorStep, policy_output_to_transition, transition_to_policy_action
from unienv_interface.backends.pytorch import PyTorchComputeBackend as T
from unienv_interface.remote import RemoteServer
from unienv_interface.space import BoxSpace, DictSpace
from unienv_lerobot import LeRobotPolicy

from conftest import ACT_KEYS, act_observation, act_spaces, tiny_act


def make(batch_size=None, **kwargs):
    policy, pre, post = tiny_act()
    observation_space, action_space = act_spaces(batch_size)
    kwargs.setdefault("observation_keys", ACT_KEYS)
    return LeRobotPolicy(policy, pre, post, observation_space=observation_space, action_space=action_space,
                         batch_size=batch_size, **kwargs)


def lerobot_loop_actions(observations):
    """What LeRobot's own evaluation loop does with the same policy and observations."""
    policy, pre, post = tiny_act()
    policy.reset()
    actions = []
    for observation in observations:
        batch = {"observation.state": torch.from_numpy(observation["state"])[None],
                 "observation.environment_state": torch.from_numpy(observation["env"])[None], "task": [""]}
        with torch.inference_mode():
            output = policy.select_action(pre(batch))
            actions.append(post(transition_to_policy_action(policy_output_to_transition(output)))[0].numpy())
    return actions


def test_actions_match_lerobots_own_loop():
    observations = [act_observation(0.1 * t) for t in range(20)]
    policy = make()
    policy.reset(None, observations[0])
    actions = []
    for t in range(20):
        action = policy.get_action()
        assert isinstance(action, np.ndarray) and action.dtype == np.float32 and action.shape == (2,)
        actions.append(action)
        if t + 1 < 20:
            policy.observe_transition(action, observations[t + 1], 0.05)
    np.testing.assert_allclose(np.stack(actions), np.stack(lerobot_loop_actions(observations)), rtol=1e-5, atol=1e-6)


def test_one_prediction_per_step():
    policy = make()
    calls = []
    inner = policy.lerobot_policy.select_action
    policy.lerobot_policy.select_action = lambda batch: calls.append(1) or inner(batch)
    policy.reset(None, act_observation(0.0))
    first = policy.get_action()
    assert policy.get_action() is first and len(calls) == 1  # get_action does not advance the policy
    policy.observe_transition(first, act_observation(1.0), 0.05)
    policy.get_action()
    assert len(calls) == 2
    with pytest.raises(RuntimeError, match="reset"):
        make().get_action()


def test_batched_masked_resets():
    policy = make(batch_size=3)
    lerobot = policy.lerobot_policy
    policy.reset(None, act_observation(0.0, 3))
    action = policy.get_action()
    assert action.shape == (3, 2)
    policy.observe_transition(action, act_observation(0.5, 3), 0.05)
    policy.get_action()
    assert lerobot.count_queued_actions() == 6  # 8-step chunk, two played
    policy.reset(None, act_observation(0.0, 3), mask=np.array([False, True, False]))
    assert lerobot.count_queued_actions() == 0  # every env gets a fresh prediction next
    policy.get_action()
    assert lerobot.count_queued_actions() == 7

    full = make(batch_size=3, masked_reset="full")
    resets = []
    full.lerobot_policy.reset = lambda: resets.append(1)
    full.reset(None, act_observation(0.0, 3))
    full.reset(None, act_observation(0.0, 3), mask=np.array([True, False, False]))
    assert len(resets) == 2


def test_action_spaces():
    observation_space, _ = act_spaces()
    torch_space = BoxSpace(T, -10.0, 10.0, torch.float64, shape=(2,))
    pi, pre, post = tiny_act()
    torch_policy = LeRobotPolicy(pi, pre, post, observation_space=observation_space, action_space=torch_space,
                                 observation_keys=ACT_KEYS)
    torch_policy.reset(None, act_observation(0.0))
    action = torch_policy.get_action()
    assert isinstance(action, torch.Tensor) and action.dtype == torch.float64 and action.shape == (2,)

    dict_space = DictSpace(T, {"arm": BoxSpace(T, -10.0, 10.0, torch.float32, shape=(1,)),
                               "gripper": BoxSpace(T, -10.0, 10.0, torch.float32, shape=(1,))})
    pi, pre, post = tiny_act()
    split = LeRobotPolicy(pi, pre, post, observation_space=observation_space, action_space=dict_space,
                          observation_keys=ACT_KEYS, action_fn=lambda a: {"arm": a[0, :1], "gripper": a[0, 1:]})
    split.reset(None, act_observation(0.0))
    assert set(split.get_action()) == {"arm", "gripper"}

    pi, pre, post = tiny_act()
    with pytest.raises(TypeError, match="pass action_fn"):
        LeRobotPolicy(pi, pre, post, observation_space=observation_space, action_space=dict_space,
                      observation_keys=ACT_KEYS)
    with pytest.raises(ValueError, match="2-dimensional actions but the action space has shape \\(3,\\)"):
        LeRobotPolicy(pi, pre, post, observation_space=observation_space, observation_keys=ACT_KEYS,
                      action_space=BoxSpace(T, -1.0, 1.0, torch.float32, shape=(3,)))


def test_options_are_checked():
    with pytest.raises(ValueError, match="apply to mode='async'"):
        make(latency=0.1)
    with pytest.raises(ValueError, match="mode must be"):
        make(mode="fast")
    with pytest.raises(ValueError, match="masked_reset must be"):
        make(masked_reset="some")
    policy = make()
    policy.preprocessor.steps.append(object.__new__(TokenizerProcessorStep))  # a VLA's tokenizer step
    policy._needs_task = True
    with pytest.raises(ValueError, match="conditioned on a language task"):
        policy.reset(None, act_observation(0.0))
    policy.reset("pick up the cube", act_observation(0.0))  # the context is the task
    assert policy._tasks == ["pick up the cube"]


def test_hosted_remotely_on_demand():
    observation_space, action_space = act_spaces()
    with RemoteServer() as server:
        server.register_factory("act", lambda: make(), release_after=0)
        client = server.connect()
        remote = client.policy("act")
        remote.reset(None, act_observation(0.0))
        action = remote.get_action()
        assert action.shape == (2,) and action.dtype == np.float32
        remote.observe_transition(action, act_observation(0.5), 0.05)
        assert remote.get_action().shape == (2,)
        remote.close()
        client.close()
