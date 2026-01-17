"""Tests for `train_lib.py`."""

import pytest
import numpy as np

from hypertor.core import common_ht
from hypertor.core import train_lib
from hypertor.core import datasets


@pytest.mark.parametrize(
    "num_transformer_samples,num_cnn_samples,num_samples_per_label,num_labels,channels,flows_shape",
    [
        (8, 8, 8, 4, 1, 16), # num_transformer_samples < num_samples_per_label and num_cnn_samples < num_samples_per_label
        (2, 6, 10, 5, 3, 32),
        (96, 16, 128, 6, 3, 64),
    ],
)
def test_make_dataset_helper(
    num_transformer_samples: int,
    num_cnn_samples: int,
    num_samples_per_label: int,
    num_labels: int,
    channels: int,
    flows_shape: int,
):
    """Tests `make_dataset_helper` function."""
    batch_size = num_samples_per_label * num_labels
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

    model_config = common_ht.LayerwiseModelConfig(
        num_transformer_samples=num_transformer_samples,
        num_cnn_samples=num_cnn_samples,
        flows_shape=(flows_shape, flows_shape),
    )
    model_config.use_variable_proto = False
    dataset_info = common_ht.DatasetInfo(
        num_labels=num_labels,
        num_samples_per_label=num_samples_per_label,
        transpose_samples=False,
    )
    data_config = common_ht.DatasetConfig(
        dataset_name="dataset",
        ds=ds,
        dataset_info=dataset_info,
    )

    numpy_arr = train_lib.make_numpy_array(
        data_config,
        batch_size=model_config.num_transformer_samples,
    )
    samples = train_lib.make_dataset(
        numpy_arr=numpy_arr,
        model_config=model_config,
        data_config=data_config,
        shuffle_labels=True,
    )
    outputs = {
        "train_images": samples.transformer_samples,
        "train_labels": samples.transformer_labels,
        "cnn_samples": samples.cnn_samples,
        "cnn_labels": samples.cnn_labels,
    }

    assert outputs["train_images"].shape == (num_transformer_samples, channels, flows_shape, flows_shape)
    assert outputs["train_labels"].shape == (num_transformer_samples,)
    assert outputs["cnn_samples"].shape == (num_cnn_samples, channels, flows_shape, flows_shape)
    assert outputs["cnn_labels"].shape == (num_cnn_samples,)
