"""UniEnv observations to LeRobot batches: finding sources, converting, tasks."""

import numpy as np
import pytest
import torch

from lerobot.configs import FeatureType, PolicyFeature
from unienv_interface.backends.numpy import NumpyComputeBackend as B
from unienv_interface.backends.pytorch import PyTorchComputeBackend as T
from unienv_interface.space import BoxSpace, DictSpace
from unienv_lerobot.policy.observations import ObservationMapping, resolve_task

IMAGE = PolicyFeature(FeatureType.VISUAL, (3, 6, 8))
STATE = PolicyFeature(FeatureType.STATE, (5,))


def image_space(shape=(6, 8, 3), dtype=np.uint8, **metadata):
    return BoxSpace(B, 0, 255, dtype, shape=shape, metadata=metadata or None)


def test_sources_by_name_metadata_and_explicit_path():
    space = DictSpace(B, {
        "observation.state": BoxSpace(B, -1, 1, np.float32, shape=(5,)),  # LeRobot's own key
        "cams": DictSpace(B, {"front": image_space(source="observation.images.front"),
                              "wrist": image_space(layout="HWC")}),
    })
    features = {"observation.state": STATE, "observation.images.front": IMAGE, "observation.images.wrist": IMAGE,
                "action": PolicyFeature(FeatureType.ACTION, (2,))}
    mapping = ObservationMapping(features, space, batch_size=None,
                                 observation_keys={"observation.images.wrist": "cams/wrist"})
    assert {s.name: s.path for s in mapping.sources} == {
        "observation.state": ("observation.state",), "observation.images.front": ("cams", "front"),
        "observation.images.wrist": ("cams", "wrist"),
    }  # actions are not inputs

    front = np.zeros((6, 8, 3), np.uint8)
    front[1, 2] = [255, 0, 51]
    observation = {"observation.state": np.arange(5, dtype=np.float64),
                   "cams": {"front": front, "wrist": np.full((6, 8, 3), 255, np.uint8)}}
    batch = mapping.batch(observation, ["stack"])
    assert batch["task"] == ["stack"]
    assert batch["observation.state"].dtype == torch.float32 and batch["observation.state"].shape == (1, 5)
    image = batch["observation.images.front"]
    assert image.shape == (1, 3, 6, 8) and image.dtype == torch.float32
    np.testing.assert_allclose(image[0, :, 1, 2].numpy(), [1.0, 0.0, 0.2], rtol=1e-6)
    assert torch.all(batch["observation.images.wrist"] == 1.0)


def test_channel_first_float_images_and_batches():
    space = DictSpace(B, {"img": BoxSpace(B, 0.0, 1.0, np.float32, shape=(2, 3, 6, 8)),
                          "s": BoxSpace(B, -1, 1, np.float32, shape=(2, 5))})
    mapping = ObservationMapping({"observation.images.cam": IMAGE, "observation.state": STATE}, space, batch_size=2,
                                 observation_keys={"observation.images.cam": "img", "observation.state": "s"})
    image = np.random.default_rng(0).random((2, 3, 6, 8)).astype(np.float32)
    batch = mapping.batch({"img": image, "s": np.zeros((2, 5), np.float32)}, ["a", "b"])
    assert batch["observation.images.cam"].shape == (2, 3, 6, 8)
    np.testing.assert_array_equal(batch["observation.images.cam"].numpy(), image)  # not rescaled
    assert batch["observation.state"].shape == (2, 5)


def test_batches_never_alias_the_observation():
    space = DictSpace(B, {"s": BoxSpace(B, -1, 1, np.float32, shape=(5,))})
    mapping = ObservationMapping({"observation.state": STATE}, space, batch_size=None,
                                 observation_keys={"observation.state": "s"})
    live = np.zeros(5, np.float32)
    batch = mapping.batch({"s": live}, [""])
    live += 1.0  # a simulator updating its buffer in place
    assert torch.all(batch["observation.state"] == 0.0)

    tensor_space = DictSpace(T, {"s": BoxSpace(T, -1, 1, torch.float32, shape=(5,))})
    tensor_mapping = ObservationMapping({"observation.state": STATE}, tensor_space, batch_size=None,
                                        observation_keys={"observation.state": "s"})
    live_tensor = torch.zeros(5)
    batch = tensor_mapping.batch({"s": live_tensor}, [""])
    live_tensor += 1.0
    assert torch.all(batch["observation.state"] == 0.0)


def test_missing_and_mismatched_sources_are_explained():
    space = DictSpace(B, {"rgb": image_space(), "joints": BoxSpace(B, -1, 1, np.float32, shape=(7,))})
    with pytest.raises(KeyError, match=r"'observation.images.cam' \(VISUAL, \(3, 6, 8\)\).*rgb \(6, 8, 3\)"):
        ObservationMapping({"observation.images.cam": IMAGE}, space, batch_size=None)
    with pytest.raises(ValueError, match="expects shape \\(5,\\); 'joints' has \\(7,\\)"):
        ObservationMapping({"observation.state": STATE}, space, batch_size=None,
                           observation_keys={"observation.state": "joints"})
    with pytest.raises(ValueError, match="expects an image with 1 channels"):
        ObservationMapping({"observation.images.cam": PolicyFeature(FeatureType.VISUAL, (1, 6, 8))}, space,
                           batch_size=None, observation_keys={"observation.images.cam": "rgb"})
    resized = ObservationMapping({"observation.images.cam": PolicyFeature(FeatureType.VISUAL, (3, 4, 4))}, space,
                                 batch_size=None, observation_keys={"observation.images.cam": "rgb"})
    assert resized.batch({"rgb": np.zeros((6, 8, 3), np.uint8)}, [""])["observation.images.cam"].shape == (1, 3, 6, 8)
    with pytest.raises(ValueError, match=r"\(CHW\) has shape \(6, 8, 3\)"):
        ObservationMapping({"observation.images.cam": IMAGE}, DictSpace(B, {"rgb": image_space(layout="CHW")}),
                           batch_size=None, observation_keys={"observation.images.cam": "rgb"})
    with pytest.raises(KeyError, match="not a leaf"):
        ObservationMapping({"observation.state": STATE}, space, batch_size=None,
                           observation_keys={"observation.state": "nope"})
    with pytest.raises(KeyError, match="inputs the policy does not have"):
        ObservationMapping({"observation.state": STATE}, space, batch_size=None,
                           observation_keys={"observation.typo": "joints"})


def test_optional_cameras_and_empty_camera_placeholders():
    space = DictSpace(B, {"rgb": image_space()})
    features = {"observation.images.main": IMAGE, "observation.images.wrist": IMAGE,
                "observation.images.empty_camera_0": IMAGE}
    mapping = ObservationMapping(features, space, batch_size=None, observation_keys={"observation.images.main": "rgb"},
                                 optional_features=["observation.images.wrist"])
    batch = mapping.batch({"rgb": np.zeros((6, 8, 3), np.uint8)}, [""])
    assert set(batch) == {"observation.images.main", "task"}


def test_observation_fn_computes_inputs():
    space = DictSpace(B, {"pos": BoxSpace(B, -1, 1, np.float32, shape=(3,)),
                          "grip": BoxSpace(B, -1, 1, np.float32, shape=(2,)),
                          "rgb": image_space()})

    def compute(observation):
        return {"observation.state": np.concatenate([observation["pos"], observation["grip"]]),
                "observation.images.cam": observation["rgb"][::-1, ::-1]}  # rotated by 180 degrees

    mapping = ObservationMapping({"observation.state": STATE, "observation.images.cam": IMAGE}, space,
                                 batch_size=None, observation_fn=compute)
    rgb = np.zeros((6, 8, 3), np.uint8)
    rgb[0, 0] = 255
    batch = mapping.batch({"pos": np.ones(3, np.float32), "grip": np.zeros(2, np.float32), "rgb": rgb}, [""])
    np.testing.assert_array_equal(batch["observation.state"][0].numpy(), [1, 1, 1, 0, 0])
    assert batch["observation.images.cam"][0, :, 5, 7].tolist() == [1.0, 1.0, 1.0]  # HWC detected by shape

    broken = ObservationMapping({"observation.state": STATE}, space, batch_size=None, observation_fn=lambda o: {})
    with pytest.raises(KeyError, match="observation_fn returned no 'observation.state'"):
        broken.batch({"pos": np.ones(3, np.float32)}, [""])


def test_tasks_from_the_context():
    assert resolve_task(None, "open the drawer", None) == ["open the drawer"]
    assert resolve_task(None, None, 2) == ["", ""]
    assert resolve_task(None, {"language_instruction": "wipe"}, 2) == ["wipe", "wipe"]
    assert resolve_task(None, np.array(["a", "b"], dtype=object), 2) == ["a", "b"]
    assert resolve_task("fixed", {"anything": 1}, None) == ["fixed"]
    assert resolve_task(lambda context: context["goal"], {"goal": "stack"}, None) == ["stack"]
    assert resolve_task(None, np.zeros(3), None) == [""]  # not a task
    with pytest.raises(ValueError, match="Expected 2 task strings, got 3"):
        resolve_task(None, ["a", "b", "c"], 2)
