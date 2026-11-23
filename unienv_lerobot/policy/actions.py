"""LeRobot action tensors to actions of a UniEnv action space."""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch

from unienv_interface.space import BoxSpace, Space

__all__ = ["ActionConverter"]


class ActionConverter:
    """Turns the postprocessed ``(B, A)`` action tensor into an action of ``action_space``.

    A ``BoxSpace`` whose last dimension is ``action_dim`` gets the tensor in its own backend, device
    and dtype (unbatched policies drop the batch dimension). Any other space, or a different
    layout, needs ``action_fn(tensor) -> action``, which receives the ``(B, A)`` tensor (``B = 1``
    for unbatched policies) and returns the action as the env takes it.
    """

    def __init__(
        self,
        action_space: Space,
        *,
        action_dim: Optional[int],
        batch_size: Optional[int],
        action_fn: Optional[Callable[[torch.Tensor], Any]] = None,
    ) -> None:
        self.action_space = action_space
        self.batch_size = batch_size
        self.action_fn = action_fn
        if action_fn is not None:
            return
        if not isinstance(action_space, BoxSpace):
            raise TypeError(f"The action space is a {type(action_space).__name__}; LeRobot policies output one "
                            "action vector, so pass action_fn to build the env's action from it")
        shape = tuple(action_space.shape)
        item = shape[1:] if batch_size is not None else shape
        if len(item) != 1 or (action_dim is not None and item[0] != action_dim):
            raise ValueError(f"The policy outputs {action_dim}-dimensional actions but the action space has shape "
                             f"{shape}; pass action_fn to map one to the other")

    def __call__(self, actions: torch.Tensor) -> Any:
        if self.action_fn is not None:
            return self.action_fn(actions)
        space = self.action_space
        backend = space.backend
        if self.batch_size is None:
            actions = actions[0]
        if backend.simplified_name == "pytorch":
            return actions.to(device=space.device, dtype=space.dtype)
        array = actions.detach().to("cpu", torch.float32).numpy()
        return backend.from_numpy(array, dtype=space.dtype, device=space.device)
