"""Asynchronous chunking and RTC: when chunks are requested, what they continue, what plays."""

import time

import numpy as np
import pytest

from unienv_lerobot import LeRobotPolicy

from conftest import (ACT_KEYS, ScriptedChunkPolicy, act_observation, act_spaces, identity_processors,
                      scripted_observation, scripted_spaces, tiny_act)


def make(policy, batch_size=None, **kwargs):
    pre, post = identity_processors()
    observation_space, action_space = scripted_spaces(batch_size)
    return LeRobotPolicy(policy, pre, post, observation_space=observation_space, action_space=action_space,
                         batch_size=batch_size, observation_keys={"observation.state": "x"}, mode="async", **kwargs)


def run(policy, steps, dt=0.01, batch_size=None, sleep=0.0):
    """Play ``steps`` steps; the observation of step t is t. Returns the executed (request, chunk step) pairs."""
    policy.reset(None, scripted_observation(0.0, batch_size))
    played = []
    for t in range(steps):
        action = policy.get_action()
        assert policy.get_action() is action
        row = action if batch_size is None else action[0]
        played.append((int(row[0]), int(row[1])))
        if sleep:
            time.sleep(sleep)
        policy.observe_transition(action, scripted_observation(t + 1.0, batch_size), dt)
    return played


def check_continuity(played):
    """Within a chunk steps follow each other; a new chunk starts where its delay says."""
    for (request, step), (next_request, next_step) in zip(played, played[1:]):
        if next_request == request:
            assert next_step == step + 1
        else:
            assert next_request == request + 1


def test_simulated_latency_is_deterministic_and_continues_the_queue():
    scripted = ScriptedChunkPolicy(chunk_size=10)
    policy = make(scripted, latency=0.03)  # 3 steps of 0.01 s
    assert policy.rtc and policy._rtc_horizon == 10
    played = run(policy, 30)
    check_continuity(played)
    # The next chunk is requested when 4 actions remain (0.04 s <= latency + one step) and lands 3 steps later:
    # it starts at its 4th action, right after the previous chunk's last executed one. Nothing starves.
    assert played[:13] == [(0, s) for s in range(9)] + [(1, s) for s in range(3, 7)]
    assert [call["state"].item() for call in scripted.calls[:3]] == [0.0, 6.0, 12.0]  # the request step's observation
    assert policy.scheduler.accepted[:3] == [(0, 10), (3, 10), (3, 10)]
    second = scripted.calls[1]
    assert second["delay"] == 3 and second["horizon"] == 10
    prefix = second["prefix"]
    assert prefix.shape == (1, 10, 2)
    np.testing.assert_array_equal(prefix[0, :4].numpy(), [[0, 6], [0, 7], [0, 8], [0, 9]])  # not yet executed
    np.testing.assert_array_equal(prefix[0, 4:].numpy(), [[0, 9]] * 6)  # padded by holding the last action
    assert scripted.calls[0]["prefix"] is None and scripted.calls[0]["delay"] is None  # first chunk: plain
    assert all(not call["inference_mode"] for call in scripted.calls)  # guided RTC needs autograd
    assert run(make(ScriptedChunkPolicy(chunk_size=10), latency=0.03), 30) == played


def test_without_rtc_chunks_are_spliced_without_a_prefix():
    scripted = ScriptedChunkPolicy(chunk_size=10, rtc=False)
    policy = make(scripted, latency=0.03)
    assert not policy.rtc
    check_continuity(run(policy, 25))
    assert all(call["prefix"] is None and call["inference_mode"] for call in scripted.calls)
    with pytest.raises(ValueError, match="does not support RTC"):
        make(ScriptedChunkPolicy(rtc=False), rtc=True, latency=0.03)
    plain = make(ScriptedChunkPolicy(chunk_size=10), rtc=False, latency=0.03)
    assert not plain.rtc


def test_latency_longer_than_a_chunk_waits_instead_of_starving():
    scripted = ScriptedChunkPolicy(chunk_size=10)
    played = run(make(scripted, latency=0.2), 40)  # 20 steps of latency, 10-action chunks
    check_continuity(played)
    assert len(played) == 40  # an action every step: an empty queue waits for the chunk in flight


def test_execution_horizon_and_batches():
    scripted = ScriptedChunkPolicy(chunk_size=10)
    policy = make(scripted, batch_size=2, latency=0.03, rtc_execution_horizon=5)
    played = run(policy, 12, batch_size=2)
    check_continuity(played)
    call = scripted.calls[1]
    assert call["horizon"] == 5 and call["prefix"].shape == (2, 5, 2) and call["state"].shape == (2,)


def test_resets_drop_the_queue_and_the_chunk_in_flight():
    scripted = ScriptedChunkPolicy(chunk_size=10)
    policy = make(scripted, batch_size=2, latency=0.05)
    policy.reset(None, scripted_observation(0.0, 2))
    for t in range(6):  # with 6 queued actions or fewer a request is in flight
        action = policy.get_action()
        policy.observe_transition(action, scripted_observation(t + 1.0, 2), 0.01)
    assert policy.scheduler._pending is not None and len(scripted.calls) == 2
    policy.reset(None, scripted_observation(100.0, 2), mask=np.array([True, False]))
    assert policy.scheduler._pending is None and policy.scheduler.remaining == 0
    action = policy.get_action()
    assert int(action[0, 0]) == 2 and int(action[0, 1]) == 0  # a fresh chunk from the new observation
    assert scripted.calls[-1]["state"].tolist() == [100.0, 100.0] and scripted.drops == 0


def test_measured_latency_runs_on_a_worker():
    scripted = ScriptedChunkPolicy(chunk_size=10, sleep=0.02)
    policy = make(scripted)  # latency=None: measured
    played = run(policy, 40, sleep=0.01)  # a real-time loop: each step takes about dt
    check_continuity(played)
    assert set(scripted.threads) and all(name.startswith("unienv-lerobot") for name in scripted.threads)
    assert 0.015 < policy.scheduler.expected_latency < 0.5
    assert len(scripted.calls) >= 4
    policy.close()


def test_dt_must_be_positive_and_modes_are_checked():
    policy = make(ScriptedChunkPolicy(), latency=0.03)
    policy.reset(None, scripted_observation(0.0))
    action = policy.get_action()
    with pytest.raises(ValueError, match="dt must be a positive"):
        policy.observe_transition(action, scripted_observation(1.0), 0.0)
    with pytest.raises(ValueError, match="latency must be"):
        make(ScriptedChunkPolicy(), latency=-1.0)


def test_async_act():
    pi, pre, post = tiny_act(chunk_size=8)
    observation_space, action_space = act_spaces()
    policy = LeRobotPolicy(pi, pre, post, observation_space=observation_space, action_space=action_space,
                           observation_keys=ACT_KEYS, mode="async", latency=0.02)
    assert not policy.rtc  # ACT has no RTC: plain asynchronous chunking
    policy.reset(None, act_observation(0.0))
    for t in range(20):
        action = policy.get_action()
        assert action.shape == (2,) and np.isfinite(action).all()
        policy.observe_transition(action, act_observation(0.05 * t), 0.01)
    assert [trim for trim, _ in policy.scheduler.accepted[1:]] == [2] * (len(policy.scheduler.accepted) - 1)
