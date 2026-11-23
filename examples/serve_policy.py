"""Serve a LeRobot policy from a GPU machine; drive it from the robot or simulator machine.

LeRobot pins its own Python (>= 3.12) and torch versions, which often differ from the robot's
or simulator's environment. Host the policy where LeRobot is installed; it is built when a client
first opens it and released a minute after the last client leaves.

    # on the GPU machine
    python examples/serve_policy.py serve --address tcp://0.0.0.0:7001

    # on the robot / simulator machine (needs only unienv)
    MUJOCO_GL=egl python examples/serve_policy.py run --address tcp://gpu-box:7001
"""

from __future__ import annotations

import argparse

import numpy as np

CONTROL_DT = 1 / 20


def serve(address: str, policy_path: str) -> None:
    from unienv_interface.remote import RemoteServer
    from unienv_libero import FromLiberoEnv  # only for the spaces; any env with the same spaces works

    from unienv_lerobot import LeRobotPolicy
    from libero_smolvla import libero_inputs

    def make_policy():
        env = FromLiberoEnv("libero_spatial", 0, image_size=360)
        try:
            return LeRobotPolicy.from_pretrained(policy_path, env, observation_fn=libero_inputs,
                                                 mode="async")  # RTC with measured latency
        finally:
            env.close()

    with RemoteServer() as server:
        server.register_factory("pi", make_policy, release_after=60)
        server.serve(address)


def run(address: str) -> None:
    from unienv_interface.remote import RemoteClient
    from unienv_libero import FromLiberoEnv

    env = FromLiberoEnv("libero_spatial", 0, image_size=360, settle_steps=10, max_episode_steps=280)
    with RemoteClient.connect(address) as client, client.policy("pi") as policy:
        context, observation, info = env.reset(init_state_id=0)
        policy.reset(context, observation)
        for _ in range(280):
            action = policy.get_action()
            observation, reward, terminated, truncated, info = env.step(np.asarray(action))
            if terminated or truncated or info.get("success"):
                break
            policy.observe_transition(action, observation, CONTROL_DT)
        print("success:", bool(info.get("success")))
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("role", choices=["serve", "run"])
    parser.add_argument("--address", default="tcp://127.0.0.1:7001")
    parser.add_argument("--policy", default="lerobot/smolvla_libero")
    args = parser.parse_args()
    serve(args.address, args.policy) if args.role == "serve" else run(args.address)
