"""A LeRobot policy as a UniEnv :class:`~unienv_interface.policy.Policy`."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Mapping, Optional, Sequence, Union

import torch

from lerobot.policies import PreTrainedPolicy, RTCConfig
from lerobot.processor import (
    PolicyProcessorPipeline,
    RelativeActionsProcessorStep,
    TokenizerProcessorStep,
    policy_output_to_transition,
    transition_to_policy_action,
)
from lerobot.processor.relative_action_processor import bind_relative_anchor
from lerobot.utils.constants import ACTION
from unienv_interface.policy import Policy
from unienv_interface.space import Space

from .actions import ActionConverter
from .observations import ObservationMapping, TaskSource, resolve_task
from .rtc import ChunkScheduler

__all__ = ["LeRobotPolicy"]


class LeRobotPolicy(Policy):
    """Runs a LeRobot ``PreTrainedPolicy`` with its pre- and postprocessors in UniEnv's protocol.

    Each action goes through the steps of LeRobot's own evaluation loop: the observation becomes a
    LeRobot batch (:class:`~unienv_lerobot.policy.observations.ObservationMapping`, plus the
    episode's task), the preprocessor normalizes it (and tokenizes the task for VLA policies), the
    policy predicts, the postprocessor un-normalizes, and the action is converted to the env's
    action space (:class:`~unienv_lerobot.policy.actions.ActionConverter`).

    Modes:

    * ``"sync"`` (default): ``select_action`` on the step's observation, as ``lerobot-eval`` does;
      chunking policies play their own queued actions.
    * ``"async"``: predict whole chunks ahead of time while the current one plays
      (:class:`~unienv_lerobot.policy.rtc.ChunkScheduler`), timed by the ``dt`` passed to
      :meth:`observe_transition`. With ``rtc`` (default: whenever the policy supports it, e.g.
      SmolVLA, pi0, pi0.5) each chunk continues the queued actions of the previous one (Real-Time
      Chunking). ``latency=None`` runs inference on a worker thread and measures it (robots);
      ``latency=<seconds of env time>`` simulates it deterministically (simulation).

    :meth:`get_action` computes at most one action per step and returns it until the next
    :meth:`observe_transition`. A masked :meth:`reset` (batched envs) updates the task of the reset
    envs; LeRobot policies keep one action queue for the whole batch, so by default
    (``masked_reset="drop_queued"``) every env gets a fresh prediction at the next step while
    other state stays (policies with observation history, ``n_obs_steps > 1``, keep the old
    episode's frames for the reset envs until the history refills); ``masked_reset="full"``
    resets the policy for every env.

    The task (the language instruction of VLA policies) is fixed per episode: ``task`` as a string,
    a function of the context, or by default read from the context (a string, or a dict's
    ``"task"`` / ``"language_instruction"`` / ``"instruction"``).
    """

    metadata: Dict[str, Any] = {}

    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        *,
        observation_space: Space,
        action_space: Space,
        context_space: Optional[Space] = None,
        batch_size: Optional[int] = None,
        observation_keys: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
        observation_fn: Optional[Callable[[Any], Mapping[str, Any]]] = None,
        optional_features: Sequence[str] = (),
        task: TaskSource = None,
        action_fn: Optional[Callable[[torch.Tensor], Any]] = None,
        mode: Literal["sync", "async"] = "sync",
        masked_reset: Literal["drop_queued", "full"] = "drop_queued",
        rtc: Optional[bool] = None,
        rtc_execution_horizon: Optional[int] = None,
        latency: Optional[float] = None,
    ) -> None:
        if mode not in ("sync", "async"):
            raise ValueError(f"mode must be 'sync' or 'async', got {mode!r}")
        if masked_reset not in ("drop_queued", "full"):
            raise ValueError(f"masked_reset must be 'drop_queued' or 'full', got {masked_reset!r}")
        if mode == "sync" and (rtc or latency is not None or rtc_execution_horizon is not None):
            raise ValueError("rtc, rtc_execution_horizon and latency apply to mode='async'")
        self.lerobot_policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.observation_space = observation_space
        self.action_space = action_space
        self.context_space = context_space
        self.batch_size = batch_size
        self.backend = action_space.backend
        self.device = action_space.device
        self.rng = self.backend.random.random_number_generator(device=self.device)
        self.mode = mode
        self.masked_reset = masked_reset
        self._task_source = task
        self._torch_device = torch.device(str(policy.config.device or "cpu"))

        policy.eval()
        self.mapping = ObservationMapping(
            policy.config.input_features, observation_space, batch_size=batch_size, device=self._torch_device,
            observation_keys=observation_keys, observation_fn=observation_fn, optional_features=optional_features,
        )
        action_feature = (policy.config.output_features or {}).get(ACTION)
        self.converter = ActionConverter(
            action_space, action_dim=None if action_feature is None else int(action_feature.shape[-1]),
            batch_size=batch_size, action_fn=action_fn,
        )
        self._needs_task = any(isinstance(step, TokenizerProcessorStep) for step in getattr(preprocessor, "steps", ()))

        self.scheduler: Optional[ChunkScheduler] = None
        self.rtc = False
        if mode == "sync":
            bind_relative_anchor(policy, preprocessor)  # as lerobot-eval: relative chunks keep their anchor
            rtc_config = getattr(policy.config, "rtc_config", None)
            if rtc_config is not None and rtc_config.enabled:  # select_action refuses RTC-enabled configs
                policy.config.rtc_config = dataclasses.replace(rtc_config, enabled=False)
                policy.init_rtc_processor()
        else:
            policy.chunk_inference_spec()  # raises for policies that cannot predict whole chunks this way
            if any(isinstance(step, RelativeActionsProcessorStep) and step.enabled
                   for step in getattr(preprocessor, "steps", ())):
                raise NotImplementedError("mode='async' does not support relative-action policies yet; use mode='sync'")
            supported = policy.supports_rtc()
            if rtc and not supported:
                raise ValueError(f"{type(policy).__name__} does not support RTC; use rtc=False (plain async chunking)")
            self.rtc = supported if rtc is None else bool(rtc)
            if self.rtc:
                config = policy.config.rtc_config
                if config is None or not config.enabled or rtc_execution_horizon is not None:
                    config = RTCConfig(enabled=True, execution_horizon=rtc_execution_horizon or RTCConfig.execution_horizon)
                    policy.config.rtc_config = config
                    policy.init_rtc_processor()
                self._rtc_horizon = config.execution_horizon
                self._rtc_guided = config.mode == "guided"
            self.scheduler = ChunkScheduler(self._predict_chunk, rtc=self.rtc, latency=latency)

        self._observation: Any = None
        self._tasks: Optional[list] = None
        self._action: Any = None

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_pretrained(
        cls,
        pretrained: Union[str, Path],
        env: Any = None,
        *,
        device: Optional[str] = None,
        revision: Optional[str] = None,
        observation_space: Optional[Space] = None,
        action_space: Optional[Space] = None,
        context_space: Optional[Space] = None,
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> "LeRobotPolicy":
        """Load a policy (Hub repo id or local directory) and its processors.

        The spaces and batch size come from ``env`` unless given; ``device`` is where the model
        runs (default: the checkpoint's, i.e. an available accelerator). Other keyword arguments
        are those of the constructor.
        """
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        config = PreTrainedConfig.from_pretrained(pretrained, revision=revision)
        if device is not None:
            config.device = device
        policy = get_policy_class(config.type).from_pretrained(pretrained, config=config, revision=revision)
        preprocessor, postprocessor = make_pre_post_processors(
            config, pretrained_path=str(pretrained), pretrained_revision=revision,
            preprocessor_overrides={"device_processor": {"device": str(config.device)}},
        )
        if env is not None:
            observation_space = observation_space or env.observation_space
            action_space = action_space or env.action_space
            context_space = context_space or getattr(env, "context_space", None)
            batch_size = batch_size if batch_size is not None else env.batch_size
        if observation_space is None or action_space is None:
            raise ValueError("Pass env, or observation_space and action_space")
        return cls(policy, preprocessor, postprocessor, observation_space=observation_space,
                   action_space=action_space, context_space=context_space, batch_size=batch_size, **kwargs)

    # ------------------------------------------------------------------ protocol

    def reset(self, context: Any, observation: Any, *, seed: Optional[int] = None, mask: Any = None,
              **kwargs: Any) -> None:
        tasks = resolve_task(self._task_source, context, self.batch_size)
        if self._needs_task and not all(tasks):
            raise ValueError("This policy is conditioned on a language task, but none was found: pass task= "
                             "(a string or a function of the context), or give the env's instruction as its context")
        self._tasks = tasks
        self._observation = observation
        self._action = None
        if mask is None or self.masked_reset == "full":
            if seed is not None and mask is None:
                torch.manual_seed(seed)  # LeRobot policies sample from torch's global generator
            self.lerobot_policy.reset()
            self.preprocessor.reset()
            self.postprocessor.reset()
            if self.scheduler is not None:
                self.scheduler.reset()
        elif self.scheduler is not None:
            self.scheduler.reset()  # also drops a chunk in flight, computed for the old episodes
        else:
            self.lerobot_policy.drop_queued_actions()

    def get_action(self, **kwargs: Any) -> Any:
        if self._tasks is None:
            raise RuntimeError("reset() the policy before asking for actions")
        if self._action is None:
            if self.scheduler is None:
                actions = self._select_action(self._make_batch())
            else:
                actions = self.scheduler.action(self._make_batch)
            self._action = self.converter(actions)
        return self._action

    def observe_transition(self, action_taken: Any, new_observation: Any, dt: float) -> None:
        if self._tasks is None:
            raise RuntimeError("reset() the policy before observing transitions")
        self._observation = new_observation
        self._action = None
        if self.scheduler is not None:
            self.scheduler.advance(float(dt), self._make_batch)

    def close(self) -> None:
        if self.scheduler is not None:
            self.scheduler.close()

    # ------------------------------------------------------------------ inference

    def _make_batch(self) -> Dict[str, Any]:
        return self.mapping.batch(self._observation, self._tasks)

    @torch.inference_mode()
    def _select_action(self, batch: Dict[str, Any]) -> torch.Tensor:
        prepared = self.preprocessor(batch)
        output = self.lerobot_policy.select_action(prepared)
        return self.postprocessor(transition_to_policy_action(policy_output_to_transition(output)))

    def _predict_chunk(self, batch: Dict[str, Any], prefix: Optional[torch.Tensor], delay: int):
        """``(model chunk, postprocessed chunk)``, both ``(B, T, ...)``; runs on the scheduler's worker."""
        guided = self.rtc and self._rtc_guided
        # Guided RTC differentiates through the denoiser, which inference mode forbids.
        with torch.inference_mode(not guided), torch.no_grad():
            prepared = self.preprocessor(batch)
            if self.rtc and prefix is not None:
                prefix = prefix.to(self._torch_device)
                horizon = self._rtc_horizon
                if prefix.shape[1] < horizon:  # hold the last action (zero would decode to the dataset mean)
                    prefix = torch.cat([prefix, prefix[:, -1:].expand(-1, horizon - prefix.shape[1], -1)], dim=1)
                output = self.lerobot_policy.predict_action_chunk(
                    prepared, inference_delay=min(delay, horizon), prev_chunk_left_over=prefix[:, :horizon],
                    execution_horizon=horizon,
                )
            else:
                output = self.lerobot_policy.predict_action_chunk(prepared)
            chunk = transition_to_policy_action(policy_output_to_transition(output))
            if chunk.ndim != 3:
                raise ValueError(f"predict_action_chunk returned shape {tuple(chunk.shape)}; expected (B, T, A)")
            model = chunk.detach().clone()
            canonical = self.postprocessor(chunk)
        return model, canonical.detach()

    def __str__(self) -> str:
        return f"<LeRobotPolicy {type(self.lerobot_policy).__name__} mode={self.mode}{' rtc' if self.rtc else ''}>"
