"""Tests for `datasets.py`."""

import pytest
import numpy as np
import torch

from hypertor.core import common_ht
from hypertor.core import datasets


# ------------------------------------------------------------
#   Class `TaskGenerator` Tests
# ------------------------------------------------------------


def _make_data(
    batch_size: int,
    channels: int,
    flows_shape: int,
    num_labels: int,
) -> tuple[common_ht.DatasetConfig, dict[int, np.ndarray]]:
    assert batch_size % num_labels == 0
    repetitions = batch_size // num_labels

    # (batch_size, channels, flows_shape, flows_shape)
    images = np.zeros(
        (batch_size, channels, flows_shape, flows_shape),
        dtype=np.float32,
    )
    # [0,1,..,num_labels, ..., 0,1,..,num_labels]
    labels = np.array(list(range(num_labels))*repetitions, dtype=np.int32)
    ds = datasets.HTDataset(images, labels)

    dataset_info = common_ht.DatasetInfo(
        num_labels=num_labels,
        num_samples_per_label=repetitions,
        transpose_samples=False,
    )
    data_config = common_ht.DatasetConfig(
        dataset_name="dataset",
        ds=ds,
        dataset_info=dataset_info,
    )

    return data_config, datasets.make_numpy_data(
        ds,
        batch_size=batch_size,
        num_labels=num_labels,
        samples_per_label=repetitions,
    )


@pytest.fixture(params=[
    {
        "batch_size": 4,
        "channels": 1,
        "flows_shape": 16,
        "num_labels": 4,
    },
    {
        "batch_size": 10,
        "channels": 3,
        "flows_shape": 32,
        "num_labels": 5,
    },
    {
        "batch_size": 12,
        "channels": 8,
        "flows_shape": 64,
        "num_labels": 6,
    },
])
def data_params(request):
    return request.param


def test_make_numpy_data(data_params):
    batch_size = data_params["batch_size"]
    channels = data_params["channels"]
    flows_shape = data_params["flows_shape"]
    num_labels = data_params["num_labels"]

    _, data = _make_data(batch_size, channels, flows_shape, num_labels)

    samples_per_label = batch_size // num_labels
    assert list(data.keys()) == list(range(num_labels))
    for label in range(num_labels):
        assert len(data[label]) == samples_per_label

def test_gen_get_batch(data_params):
    """Tests image and label generation in the `TaskGenerator`."""
    batch_size = data_params["batch_size"]
    channels = data_params["channels"]
    flows_shape = data_params["flows_shape"]
    num_labels = data_params["num_labels"]

    data_config, data = _make_data(batch_size, channels, flows_shape, num_labels)
    gen = datasets.TaskGenerator(data_config, data, num_labels)

    aug_config = datasets.AugmentationConfig(
        random_config=datasets.RandomizedAugmentationConfig(
            smooth_probability=0.0,
        )
    )
    images, labels, classes = gen.get_batch(
        batch_size=batch_size,
        config=aug_config,
    )

    # Checks
    assert isinstance(images, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert isinstance(classes, torch.Tensor)

    assert images.shape == (batch_size, channels, flows_shape, flows_shape)
    assert labels.shape == (batch_size,)
    assert classes.shape == (batch_size,)


# ------------------------------------------------------------
#   Class `AugmentationConfig` Tests
# ------------------------------------------------------------


@pytest.mark.parametrize(
    "config",
    [
        dict(
            smooth_probability=0.0,
            negate_probability=0.0,
        ),
        dict(
            smooth_probability=1.0,
            negate_probability=1.0,
        ),
    ],
)
def test_augmentation(config: dict):
    aug_config = datasets.AugmentationConfig(
        random_config=datasets.RandomizedAugmentationConfig(**config)
    )

    images = torch.ones((4, 1, 8, 8))
    aug_config.randomize()
    out = aug_config.process(images)

    assert out.shape == images.shape
    assert out.dtype == images.dtype
