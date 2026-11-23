"""Asynchronous action chunking, with Real-Time Chunking (RTC) for policies that support it.

The scheduler plays the current action chunk one action per env step and requests the next chunk
early enough that it arrives before the current one runs out. Time is the env's: the ``dt`` of
each step (:meth:`ChunkScheduler.advance`), so the same code runs a robot in real time and a
simulator deterministically.

* A request takes the observation of the step it starts at. The steps executed until its chunk
  arrives (the *delay*) are counted, and the new chunk replaces the queue from the action that
  is due when it arrives: its first ``delay`` actions are dropped, since the robot already
  executed the previous chunk's actions for those steps.
* With RTC, the request also carries the queued, not yet executed actions of the current chunk
  (in the model's coordinates) and the expected delay, so the policy generates a chunk that
  continues them (``predict_action_chunk(..., inference_delay=..., prev_chunk_left_over=...)``)
  instead of possibly switching strategy halfway.
* Latency is either measured (``latency=None``: inference runs on a worker thread and its
  wall-clock turnaround is tracked; for robots, whose env time is wall time) or given in seconds
  of env time (``latency=...``): the chunk is computed when requested and takes effect once that
  much env time has passed, which makes RTC deterministic and testable in simulation.
* If the queue runs out before the next chunk arrives, :meth:`action` waits for it.
"""

from __future__ import annotations

import math
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch

__all__ = ["ChunkScheduler", "Predict"]

#: ``predict(batch, prefix, delay) -> (model_chunk, canonical_chunk)``, both ``(B, T, ...)``:
#: the chunk in the model's coordinates (for RTC prefixes) and postprocessed for the env.
#: ``prefix`` is ``None`` or the not yet executed actions in model coordinates, ``(B, n, A_model)``.
Predict = Callable[[dict, Optional[torch.Tensor], int], Tuple[torch.Tensor, torch.Tensor]]

_EMA = 0.3  # weight of the newest measured turnaround


@dataclass
class _Request:
    steps: int  # steps executed when the request started
    started_at: float  # env time when it started
    generation: int
    future: Optional[Future] = None  # measured latency: running on the worker
    result: Optional[Tuple[torch.Tensor, torch.Tensor]] = None  # simulated latency: computed at once
    wall_start: float = 0.0
    wall_end: Optional[float] = None


class ChunkScheduler:
    """Plays action chunks and schedules the next prediction ahead of time (see the module docs)."""

    def __init__(self, predict: Predict, *, rtc: bool, latency: Optional[float] = None) -> None:
        if latency is not None and not latency >= 0:
            raise ValueError(f"latency must be >= 0 seconds of env time or None (measured), got {latency!r}")
        self._predict = predict
        self.rtc = rtc
        self.latency = latency
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="unienv-lerobot") \
            if latency is None else None
        self._turnaround: Optional[float] = latency  # expected latency, seconds of env time
        self._dt: Optional[float] = None
        self._generation = 0
        self.reset()

    # ------------------------------------------------------------------ state

    def reset(self) -> None:
        """Drop the queue and any prediction in flight (its result is ignored)."""
        self._generation += 1
        self._model: Optional[torch.Tensor] = None  # (B, T, A_model)
        self._canonical: Optional[torch.Tensor] = None  # (B, T, A)
        self._offset = 0  # index of the action due at the current step
        self._current: Optional[torch.Tensor] = None  # the action handed out for the current step
        self._steps = 0
        self._time = 0.0
        self._pending: Optional[_Request] = None
        #: (trimmed actions, chunk length) of the chunks accepted so far, for inspection
        self.accepted: list = []

    @property
    def remaining(self) -> int:
        """Actions queued for this step and the following ones."""
        return 0 if self._canonical is None else self._canonical.shape[1] - self._offset

    @property
    def expected_latency(self) -> Optional[float]:
        """Seconds of env time a prediction is expected to take (given, or measured so far)."""
        return self._turnaround

    # ------------------------------------------------------------------ protocol

    def action(self, make_batch: Callable[[], dict]) -> torch.Tensor:
        """The action for the current step, ``(B, A)``; the same until :meth:`advance`."""
        if self._current is not None:
            return self._current
        if self.remaining == 0 and self._pending is not None:
            self._accept(self._pending, wait=True)  # starved: wait for the chunk in flight
        if self.remaining == 0:
            # First chunk of an episode, or a stale one was discarded: predict now and wait.
            request = self._start(make_batch, with_prefix=False)
            self._accept(request, wait=True)
            if self.remaining == 0:
                raise RuntimeError("The policy returned an empty action chunk")
        self._current = self._canonical[:, self._offset]
        return self._current

    def advance(self, dt: float, make_batch: Callable[[], dict]) -> None:
        """One env step of ``dt`` seconds passed; the new observation is what ``make_batch`` builds."""
        if not dt > 0:
            raise ValueError(f"dt must be a positive number of seconds, got {dt!r}")
        if self.remaining > 0:
            self._offset += 1  # this step's action was executed
        self._current = None
        self._steps += 1
        self._time += dt
        self._dt = dt
        pending = self._pending
        if pending is not None and self._arrived(pending):
            self._accept(pending, wait=False)
        if self._pending is None and self._due():
            self._start(make_batch, with_prefix=self.rtc)

    def close(self) -> None:
        self._generation += 1
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)

    # ------------------------------------------------------------------ scheduling

    def _due(self) -> bool:
        """Whether to request the next chunk now: the queue lasts about as long as a prediction."""
        remaining = self.remaining
        if remaining == 0:
            return False  # the next action() predicts and waits anyway
        if self._turnaround is None or self._dt is None:
            return False
        return remaining * self._dt <= self._turnaround + self._dt + 1e-12

    def _expected_delay(self) -> int:
        if not self._turnaround or not self._dt:
            return 0
        return min(math.ceil(self._turnaround / self._dt - 1e-9), self.remaining)

    def _start(self, make_batch: Callable[[], dict], *, with_prefix: bool) -> _Request:
        prefix = self._model[:, self._offset:] if with_prefix and self.remaining > 0 else None
        delay = self._expected_delay() if prefix is not None else 0
        batch = make_batch()
        request = _Request(self._steps, self._time, self._generation)
        if self._executor is None:
            request.result = self._predict(batch, prefix, delay)
        else:
            request.wall_start = time.perf_counter()
            request.future = self._executor.submit(self._predict, batch, prefix, delay)
            request.future.add_done_callback(lambda _: setattr(request, "wall_end", time.perf_counter()))
        self._pending = request
        return request

    def _arrived(self, request: _Request) -> bool:
        if request.future is not None:
            return request.future.done()
        return self._time - request.started_at >= (self.latency or 0.0) - 1e-12

    def _accept(self, request: _Request, *, wait: bool) -> None:
        self._pending = None
        if request.future is not None:
            model, canonical = request.future.result()
            turnaround = (request.wall_end or time.perf_counter()) - request.wall_start
            self._turnaround = turnaround if self._turnaround is None else \
                (1 - _EMA) * self._turnaround + _EMA * turnaround
        else:
            model, canonical = request.result
        if request.generation != self._generation or canonical is None:
            return  # reset since it was requested
        trim = self._steps - request.steps  # actions executed while it was computed
        length = canonical.shape[1]
        self.accepted.append((trim, length))
        if trim >= length:
            return  # entirely in the past; the next action() predicts again
        self._model, self._canonical, self._offset = model, canonical, trim
