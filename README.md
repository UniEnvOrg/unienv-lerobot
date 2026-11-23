# unienv-lerobot

[LeRobot](https://github.com/huggingface/lerobot) for [UniEnv](https://github.com/UniEnvOrg/UniEnv): LeRobot
policies (ACT, Diffusion Policy, SmolVLA, pi0, pi0.5, GR00T, ...) as UniEnv policies, synchronous or
asynchronous with Real-Time Chunking.

```bash
pip install unienv_lerobot                 # LeRobot >= 0.7, < 0.8 (Python >= 3.12)
pip install "lerobot[smolvla]"             # the extras of the policies you run
```

## Usage

```python
from unienv_lerobot import LeRobotPolicy

policy = LeRobotPolicy.from_pretrained("lerobot/smolvla_libero", env, device="cuda",
                                       observation_fn=libero_inputs)   # see below
context, obs, info = env.reset()
policy.reset(context, obs)                 # the context is the instruction
while True:
    action = policy.get_action()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
    policy.observe_transition(action, obs, dt)   # dt: the step's duration in seconds
```

`LeRobotPolicy` is a UniEnv `Policy`: it runs the steps of LeRobot's own evaluation loop (the policy's
preprocessor, which normalizes and tokenizes the task, then the policy, then the postprocessor) and
converts between UniEnv data and LeRobot batches. `from_pretrained` takes a Hub repo id or a local
checkpoint and the env (for its spaces and batch size); `LeRobotPolicy(policy, preprocessor,
postprocessor, observation_space=..., action_space=...)` wraps policies you built yourself.

### Observations

The policy declares its inputs (`policy.config.input_features`, e.g. `observation.state`,
`observation.images.<camera>`). Each input comes from:

1. `observation_keys={"observation.images.front": "cameras/front"}`: a key path into the observation;
2. a top-level key with the input's name (data read with UniEnv's `LeRobotAsUniEnvDataset` has LeRobot's
   keys);
3. a leaf whose space metadata `"source"` is the input's name;
4. `observation_fn(observation) -> {input name: array}` for inputs that must be computed.

A missing input is an error that lists the observation's leaves. VLA policies can run without some cameras:
list those in `optional_features`. Images may be channel-last or channel-first, `uint8` (scaled to `[0, 1]`)
or float, and of another size than the checkpoint's (policies resize them); the channel count must match.

For LIBERO (with [unienv-libero](https://github.com/UniEnvOrg/unienv-libero)), LeRobot's checkpoints expect
both cameras rotated by 180 degrees and an 8-D state:

```python
def libero_inputs(obs):
    return {"observation.images.camera1": obs["agentview_rgb"][::-1, ::-1],
            "observation.images.camera2": obs["eye_in_hand_rgb"][::-1, ::-1],
            "observation.state": np.concatenate([obs["ee_pos"], obs["ee_ori"], obs["gripper_states"]])}
```

### Task, actions, batches

- **Task**: VLA policies need the episode's instruction. By default it is the context (a string, or a dict's
  `"task"`, `"language_instruction"` or `"instruction"`); pass `task=` as a string or a function of the
  context otherwise.
- **Actions**: for a `BoxSpace` action space of the policy's action size, actions come in the space's backend,
  device and dtype. Other action spaces need `action_fn(actions)`, which gets the `(B, A)` tensor.
- **Batched envs**: LeRobot policies keep one action queue for the whole batch. A masked `reset` updates the
  tasks of the reset envs and, by default (`masked_reset="drop_queued"`), makes every env predict afresh at
  the next step, keeping other state; policies with observation history (`n_obs_steps > 1`) keep the old
  episode's frames for the reset envs until it refills. `masked_reset="full"` resets the policy for all envs.
- **`get_action`** computes at most one action per step and returns it until the next `observe_transition`.

### Asynchronous inference and Real-Time Chunking

With `mode="async"`, the policy predicts whole action chunks ahead of time while the current chunk plays, so
a robot keeps moving while the model runs:

- The next chunk is requested when the queued actions last about as long as a prediction. Its first actions
  (those of the steps that passed while it was computed) are dropped, so it starts at the action due when it
  arrives. If the queue runs out, `get_action` waits for the chunk.
- With **RTC** (`rtc=None`: whenever the policy supports it, e.g. SmolVLA, pi0, pi0.5, GR00T), each request
  carries the queued actions of the current chunk and the expected delay, and the policy generates a chunk
  that continues them ([Real-Time Chunking](https://www.physicalintelligence.company/research/real_time_chunking),
  guided). `rtc_execution_horizon` sets the guided prefix (LeRobot's default: 10). `rtc=False` switches chunks
  without guidance (any chunking policy, e.g. ACT).
- **Time** is the env's: the `dt` passed to `observe_transition`. `latency=None` runs inference on a worker
  thread and measures it (robots, whose env time is wall time); `latency=0.15` simulates 0.15 s of inference
  in env time, deterministically (simulation, tests).

Policies with relative actions and observation history are not supported in async mode yet.

### Serving a policy remotely

LeRobot's Python and torch requirements often differ from the robot's or simulator's environment. Host the
policy with UniEnv's remote execution on the machine that has LeRobot, built on demand:

```python
server.register_factory("pi", lambda: LeRobotPolicy.from_pretrained(..., mode="async"), release_after=60)
```

and use `client.policy("pi")` on the robot side (`examples/serve_policy.py`).

## Validation

`lerobot/smolvla_libero` on LIBERO-Spatial (10 tasks x initial states 0-2, 360x360 renders, as `lerobot-eval`),
LeRobot 0.7.0:

| run | success |
|---|---|
| `lerobot-eval` (LeRobot's own loop and LIBERO env) | 18 / 30 |
| `LeRobotPolicy`, sync, on unienv-libero | 18 / 30 |
| `LeRobotPolicy`, async + RTC, 0.15 s simulated latency | 17 / 30 |
| `LeRobotPolicy`, async without RTC, 0.15 s | 17 / 30 |

The sync run and `lerobot-eval` agree on 25 of the 30 episodes; the rest is the policy's sampling noise.

## Examples

- `examples/libero_smolvla.py`: `lerobot/smolvla_libero` on LIBERO suites, sync or async (`--mode async
  --latency 0.15`), with success rates.
- `examples/serve_policy.py`: the same policy served from another machine.

## Tests

`pytest tests` runs on CPU without downloads (a random-weight ACT and a scripted chunk policy). `pytest tests
--run-gpu` also runs SmolVLA on a LIBERO task (CUDA, `unienv_libero[env]`, model downloads).

## License

Apache-2.0. LeRobot (Apache-2.0) and the policies' weights keep their own licenses.
