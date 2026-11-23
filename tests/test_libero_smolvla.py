"""lerobot/smolvla_libero on unienv-libero's LIBERO (needs a CUDA GPU, LIBERO and model downloads: --run-gpu)."""

import os

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

CONTROL_DT = 1 / 20


def libero_inputs(observation):
    return {
        "observation.images.camera1": observation["agentview_rgb"][::-1, ::-1],
        "observation.images.camera2": observation["eye_in_hand_rgb"][::-1, ::-1],
        "observation.state": np.concatenate(
            [observation["ee_pos"], observation["ee_ori"], observation["gripper_states"]]).astype(np.float32),
    }


@pytest.mark.parametrize("options", [{"mode": "sync"}, {"mode": "async", "latency": 0.15}], ids=["sync", "rtc"])
def test_smolvla_solves_a_libero_spatial_task(options):
    os.environ.setdefault("MUJOCO_GL", "egl")
    unienv_libero = pytest.importorskip("unienv_libero")
    from unienv_lerobot import LeRobotPolicy

    env = unienv_libero.FromLiberoEnv("libero_spatial", 0, image_size=360, settle_steps=10, max_episode_steps=280)
    policy = LeRobotPolicy.from_pretrained("lerobot/smolvla_libero", env, device="cuda",
                                           observation_fn=libero_inputs, **options)
    assert policy.rtc == (options["mode"] == "async")
    successes = []
    for episode in range(2):
        context, observation, info = env.reset(init_state_id=episode)
        policy.reset(context, observation, seed=episode)
        success = False
        for _ in range(280):
            action = policy.get_action()
            observation, reward, terminated, truncated, info = env.step(action)
            success = bool(info.get("success"))
            if terminated or truncated or success:
                break
            policy.observe_transition(action, observation, CONTROL_DT)
        successes.append(success)
    policy.close()
    env.close()
    assert any(successes), successes
