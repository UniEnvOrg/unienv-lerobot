"""UniEnv observations and contexts to LeRobot policy batches.

A LeRobot policy declares its inputs in ``policy.config.input_features``: names such as
``observation.state`` or ``observation.images.<camera>``, each with a type (``VISUAL``,
``STATE``, ``ENV``) and a shape (images channel-first, ``(C, H, W)``). :class:`ObservationMapping`
finds the source of every input in a UniEnv observation once, checks its shape, and then turns
observations into the batch LeRobot's processors expect: float32 tensors with a leading batch
dimension, images channel-first in ``[0, 1]``, plus ``"task"``, a list of strings.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from lerobot.configs import FeatureType, PolicyFeature
from unienv_interface.space import DictSpace, Space

__all__ = ["ObservationMapping", "TaskSource", "resolve_task"]

Path = Tuple[str, ...]
#: The task: a string, a function of the episode's context, or ``None`` to read it from the context
TaskSource = Union[None, str, Callable[[Any], Union[str, Sequence[str]]]]
#: Context keys a task is read from by default
_TASK_KEYS = ("task", "language_instruction", "instruction")
_INPUT_TYPES = (FeatureType.VISUAL, FeatureType.STATE, FeatureType.ENV)


def _leaves(space: Space, path: Path = ()) -> Dict[Path, Space]:
    """Every non-dict subspace of ``space`` by its key path (``()`` for a non-dict observation)."""
    if isinstance(space, DictSpace):
        found: Dict[Path, Space] = {}
        for key, subspace in space.spaces.items():
            found.update(_leaves(subspace, path + (key,)))
        return found
    return {path: space}


def _parse_path(path: Union[str, Sequence[str]]) -> Path:
    if isinstance(path, str):
        return tuple(part for part in path.split("/") if part)
    return tuple(path)


def _show(path: Path) -> str:
    return "/".join(path) if path else "<the observation>"


def _get(data: Any, path: Path) -> Any:
    for key in path:
        data = data[key]
    return data


def _as_tensor(value: Any, backend: Any) -> torch.Tensor:
    """``value`` as a torch tensor, sharing memory where possible (callers copy)."""
    if isinstance(value, torch.Tensor):
        return value
    if backend is not None and backend.is_backendarray(value):
        value = backend.to_numpy(value)
    array = np.asarray(value)
    if any(stride < 0 for stride in array.strides):
        array = array.copy()  # flipped views (e.g. image[::-1, ::-1]); torch cannot wrap negative strides
    return torch.as_tensor(array)


@dataclasses.dataclass(frozen=True)
class _Source:
    """Where one LeRobot input comes from and how to convert it."""

    name: str
    feature: PolicyFeature
    path: Optional[Path]  # None: produced by observation_fn
    channels_last: Optional[bool]  # VISUAL inputs from the observation; None: decided by shape


class ObservationMapping:
    """Builds LeRobot batches from UniEnv observations for one policy.

    Each input in ``input_features`` comes from, in order of preference:

    1. ``observation_keys[name]``: a key path into the observation (``"agentview_rgb"``,
       ``"robot/joint_pos"`` or a tuple of keys; ``""`` for a non-dict observation);
    2. a top-level key named exactly like the input (e.g. data read with
       ``LeRobotAsUniEnvDataset``, whose keys are LeRobot's);
    3. a leaf whose space metadata ``"source"`` is the input's name;
    4. ``observation_fn(observation)``, a mapping from input names to arrays, for inputs that
       must be computed (e.g. a state vector assembled from several leaves).

    An input with no source is an error, except SmolVLA's ``empty_camera_*`` placeholders and the
    names in ``optional_features`` (cameras a VLA policy may run without). Images may be
    channel-last (``H, W, C``, as UniEnv's image metadata usually says) or channel-first; ``uint8``
    images are scaled to ``[0, 1]``.
    """

    def __init__(
        self,
        input_features: Mapping[str, PolicyFeature],
        observation_space: Space,
        *,
        batch_size: Optional[int],
        device: Union[str, torch.device] = "cpu",
        observation_keys: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
        observation_fn: Optional[Callable[[Any], Mapping[str, Any]]] = None,
        optional_features: Sequence[str] = (),
    ) -> None:
        self.observation_space = observation_space
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.observation_fn = observation_fn
        self._backend = getattr(observation_space, "backend", None)
        leaves = _leaves(observation_space)
        observation_keys = dict(observation_keys or {})
        unknown = set(observation_keys) - set(input_features)
        if unknown:
            raise KeyError(f"observation_keys names inputs the policy does not have: {sorted(unknown)}; "
                           f"its inputs are {sorted(input_features)}")
        by_source = {}
        for path, leaf in leaves.items():
            source = (getattr(leaf, "metadata", None) or {}).get("source")
            if isinstance(source, str):
                by_source.setdefault(source, path)

        self.sources: List[_Source] = []
        missing: List[str] = []
        for name, feature in input_features.items():
            if feature.type not in _INPUT_TYPES:
                continue
            if name in observation_keys:
                path = _parse_path(observation_keys[name])
                if path not in leaves:
                    raise KeyError(f"observation_keys[{name!r}] = {_show(path)!r} is not a leaf of the observation "
                                   f"space; its leaves are {sorted(_show(p) for p in leaves)}")
            elif (name,) in leaves:
                path = (name,)
            elif name in by_source:
                path = by_source[name]
            else:
                path = None
            if path is None:
                if observation_fn is not None:
                    self.sources.append(_Source(name, feature, None, None))
                elif name.rsplit(".", 1)[-1].startswith("empty_camera_") or name in optional_features:
                    continue
                else:
                    missing.append(name)
                continue
            self.sources.append(_Source(name, feature, path, self._check_leaf(name, feature, path, leaves[path])))
        if missing:
            listing = ", ".join(f"{_show(p)} {tuple(getattr(s, 'shape', ()) or ())}" for p, s in leaves.items())
            details = ", ".join(f"{n!r} ({input_features[n].type.value}, {tuple(input_features[n].shape)})"
                                for n in missing)
            raise KeyError(
                f"The policy's inputs {details} have no source in the observation (leaves: {listing}). Map them "
                f"with observation_keys={{name: 'key/path'}}, compute them with observation_fn, or list cameras "
                f"the policy may run without in optional_features."
            )

    def _check_leaf(self, name: str, feature: PolicyFeature, path: Path, leaf: Space) -> Optional[bool]:
        """Check the leaf's shape against ``feature``; for images, whether the leaf is channel-last.

        Images must have the feature's channel count, but may have another size: LeRobot policies
        resize their camera inputs (e.g. SmolVLA checkpoints evaluated on 360x360 renders of a
        256x256 dataset).
        """
        shape = tuple(getattr(leaf, "shape", None) or ())
        if self.batch_size is not None:
            shape = shape[1:]
        expected = tuple(feature.shape)
        if feature.type is not FeatureType.VISUAL:
            if shape != expected:
                raise ValueError(f"Input {name!r} expects shape {expected}; {_show(path)!r} has {shape}")
            return None
        channels = expected[0]
        layout = str((getattr(leaf, "metadata", None) or {}).get("layout", "")).upper()
        if len(shape) != 3:
            raise ValueError(f"Input {name!r} expects an image with {channels} channels; {_show(path)!r} has shape {shape}")
        if layout in ("HWC", "CHW"):
            if shape[-1 if layout == "HWC" else 0] != channels:
                raise ValueError(f"Input {name!r} expects an image with {channels} channels; {_show(path)!r} "
                                 f"({layout}) has shape {shape}")
            return layout == "HWC"
        if shape == expected or (shape[0] == channels and shape[-1] != channels):
            return False
        if shape[-1] == channels:
            return True
        raise ValueError(f"Input {name!r} expects an image with {channels} channels (C, H, W or H, W, C); "
                         f"{_show(path)!r} has shape {shape}")

    # ------------------------------------------------------------------ conversion

    def _convert(self, source: _Source, value: Any) -> torch.Tensor:
        raw = _as_tensor(value, self._backend)
        tensor = raw.to(self.device, non_blocking=True)
        expected = tuple(source.feature.shape)
        batched = self.batch_size is not None
        if source.feature.type is FeatureType.VISUAL:
            channels_last = source.channels_last
            if channels_last is None:  # from observation_fn: decide by shape
                trailing = tuple(tensor.shape[-3:])
                channels_last = trailing != expected and trailing[-1] == expected[0] and trailing[0] != expected[0]
            if channels_last:
                tensor = tensor.movedim(-1, -3)
            if tensor.dtype == torch.uint8:
                tensor = tensor.to(torch.float32) / 255.0
            else:
                tensor = tensor.to(torch.float32)
        else:
            tensor = tensor.to(torch.float32)
        tensor = tensor.contiguous()
        if tensor.data_ptr() == raw.data_ptr():
            tensor = tensor.clone()  # never alias the env's (possibly live) buffers
        item_shape = tuple(tensor.shape[1:]) if batched else tuple(tensor.shape)
        if source.feature.type is FeatureType.VISUAL:
            mismatch = len(item_shape) != 3 or item_shape[0] != expected[0]
        else:
            mismatch = item_shape != expected
        if mismatch:
            raise ValueError(f"Input {source.name!r} expects shape {expected} per env, got {tuple(tensor.shape)}"
                             + (" (batched)" if batched else ""))
        return tensor if batched else tensor.unsqueeze(0)

    def batch(self, observation: Any, task: List[str]) -> Dict[str, Any]:
        """The LeRobot batch for ``observation`` (a full batch for batched policies)."""
        computed = self.observation_fn(observation) if self.observation_fn is not None else {}
        batch: Dict[str, Any] = {}
        for source in self.sources:
            if source.path is None:
                if source.name not in computed:
                    raise KeyError(f"observation_fn returned no {source.name!r} "
                                   f"(it returned {sorted(computed)}); the policy needs it")
                value = computed[source.name]
            else:
                value = _get(observation, source.path)
            batch[source.name] = self._convert(source, value)
        batch["task"] = list(task)
        return batch


def resolve_task(task: TaskSource, context: Any, batch_size: Optional[int]) -> List[str]:
    """The task string per env (one entry for an unbatched policy) for an episode's ``context``."""
    count = 1 if batch_size is None else batch_size
    value = task(context) if callable(task) else (task if task is not None else _task_from_context(context))
    if value is None:
        return [""] * count
    if isinstance(value, str):
        return [value] * count
    if isinstance(value, np.ndarray):
        value = value.reshape(-1).tolist()
    values = [str(v) for v in value]
    if len(values) != count:
        raise ValueError(f"Expected {count} task strings, got {len(values)}")
    return values


def _task_from_context(context: Any) -> Any:
    if context is None or isinstance(context, (str, list, tuple)):
        return context
    if isinstance(context, np.ndarray) and context.dtype == object:
        return context
    if isinstance(context, Mapping):
        for key in _TASK_KEYS:
            if key in context:
                return _task_from_context(context[key])
    return None
