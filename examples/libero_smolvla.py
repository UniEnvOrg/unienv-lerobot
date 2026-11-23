"""Run LeRobot's SmolVLA LIBERO checkpoint on unienv-libero's LIBERO tasks.

    pip install unienv_lerobot "unienv_libero[env]" "lerobot[smolvla]"
    MUJOCO_GL=egl python examples/libero_smolvla.py --suite libero_spatial --tasks 0-9 --episodes 2
    MUJOCO_GL=egl python examples/libero_smolvla.py --mode async --latency 0.15   # RTC, 3 steps of latency

Observations are converted as LeRobot's own LIBERO evaluation does: both cameras rotated by 180
degrees (LIBERO renders upside down) and the state built from the end-effector position, its
axis-angle orientation and the gripper joints. The instruction is the env's context.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from unienv_lerobot import LeRobotPolicy

CONTROL_DT = 1 / 20  # LIBERO's robosuite controllers run at 20 Hz
MAX_STEPS = {"libero_spatial": 280, "libero_object": 280, "libero_goal": 300, "libero_10": 520}  # as lerobot-eval


def libero_inputs(observation):
    """unienv-libero's observation as the inputs of ``lerobot/smolvla_libero``."""
    return {
        "observation.images.camera1": observation["agentview_rgb"][::-1, ::-1],
        "observation.images.camera2": observation["eye_in_hand_rgb"][::-1, ::-1],
        "observation.state": np.concatenate(
            [observation["ee_pos"], observation["ee_ori"], observation["gripper_states"]]).astype(np.float32),
    }


def parse_tasks(text: str):
    if "-" in text:
        start, end = text.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(part) for part in text.split(",")]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", default="lerobot/smolvla_libero")
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--tasks", default="0-9", help="task ids, e.g. 0-9 or 0,3,7")
    parser.add_argument("--episodes", type=int, default=2, help="episodes per task (init states 0, 1, ...)")
    parser.add_argument("--mode", choices=["sync", "async"], default="sync")
    parser.add_argument("--latency", type=float, default=None,
                        help="async: simulated inference latency in seconds (default: measured)")
    parser.add_argument("--no-rtc", action="store_true", help="async: plain chunk switching, no RTC")
    parser.add_argument("--image-size", type=int, default=360, help="camera renders, as lerobot-eval")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from unienv_libero import FromLiberoEnv

    async_options = {}
    if args.mode == "async":
        async_options = {"latency": args.latency, "rtc": False if args.no_rtc else None}
    policy = None
    results = []
    for task_id in parse_tasks(args.tasks):
        env = FromLiberoEnv(args.suite, task_id, image_size=args.image_size, settle_steps=10,
                            max_episode_steps=MAX_STEPS.get(args.suite, -1), seed=args.seed)
        if policy is None:
            policy = LeRobotPolicy.from_pretrained(args.policy, env, device=args.device,
                                                   observation_fn=libero_inputs, mode=args.mode, **async_options)
            print(policy, flush=True)
        for episode in range(args.episodes):
            started = time.perf_counter()
            context, observation, info = env.reset(init_state_id=episode)
            policy.reset(context, observation, seed=args.seed + episode)
            success, steps = False, 0
            while True:
                action = policy.get_action()
                observation, reward, terminated, truncated, info = env.step(action)
                steps += 1
                success = success or bool(info.get("success"))
                if terminated or truncated or success:
                    break
                policy.observe_transition(action, observation, CONTROL_DT)
            results.append({"task": task_id, "episode": episode, "success": success, "steps": steps,
                            "seconds": round(time.perf_counter() - started, 1)})
            print(json.dumps({**results[-1], "instruction": context}), flush=True)
        env.close()
    if policy is not None:
        policy.close()
    rate = float(np.mean([r["success"] for r in results])) if results else 0.0
    print(json.dumps({"suite": args.suite, "mode": args.mode, "latency": args.latency, "episodes": len(results),
                      "success_rate": rate}), flush=True)


if __name__ == "__main__":
    main()
