# Changelog

## 0.0.1a1 (unreleased)

- Initial release: `LeRobotPolicy`, a LeRobot `PreTrainedPolicy` with its pre- and postprocessors as a UniEnv
  `Policy` (`from_pretrained(repo_or_path, env)`).
- Observations map to the policy's inputs by name, space metadata `"source"`, `observation_keys` or
  `observation_fn`; images are converted to LeRobot's layout; the task comes from the episode's context.
- `mode="async"`: action chunks predicted ahead of time, timed by `observe_transition`'s `dt`, with Real-Time
  Chunking for policies that support it; measured (`latency=None`) or simulated (`latency=...`) inference
  latency.
- Examples: SmolVLA on LIBERO (`examples/libero_smolvla.py`) and serving a policy remotely.
