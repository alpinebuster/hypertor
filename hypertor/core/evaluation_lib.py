"""Evaluation library."""

import copy
import functools
import dataclasses
import os
import time

from typing import Callable, Optional, Tuple

import numpy as np
import torch

from hypertor.core import common
from hypertor.core import common_ht
from hypertor.core import layerwise
from hypertor.core import train_lib
from hypertor.core import util

# In some evaluation scenarios we may want to run multiple "CNN batches"
# for the same meta-training Transformer batch, which means that we have
# to fix Transformer batch and update it only occasionally.
#
# This is done in `make_train_samples` that returns:
# (a) cached Transformer samples (or original tensor if `same_batch` is False);
# (b) cached Transformer labels;
TrainSamples = Tuple[torch.Tensor, torch.Tensor]

# Type returned by `evaluate_dataset`:
# (a) a dictionary mapping a task number to a list of accuracies for all
#     batches;
# (b) a list of mean accuracies for all tasks.
Accuracies = Tuple[dict[int, list[float]], list[float]]


@dataclasses.dataclass
class EvaluationConfig:
    num_task_evals: int
    num_eval_batches: int


def wait_for_new_checkpoint(
    ckpt_dir: str,
    last_ckpt: Optional[str],
    sleep_seconds: int = 10,
) -> str:
    while True:
        newest = util.latest_checkpoint(ckpt_dir)

        if not newest:
            time.sleep(sleep_seconds)
            continue
        if newest != last_ckpt:
            return newest

        time.sleep(sleep_seconds)


def dataset_with_custom_labels(
    model_config: common_ht.LayerwiseModelConfig,
    dataset_config: common_ht.DatasetConfig,
) -> tuple[
    Callable[[], "common_ht.DatasetSamples"],
    Optional[list[int]],
]:
    """Returns a dataset with a controlled label set (should be reshuffled)."""
    custom_labels = copy.copy(dataset_config.use_label_subset)
    dataset_config = dataclasses.replace(
        dataset_config, use_label_subset=lambda: custom_labels
    )

    # ALL data
    numpy_arr = train_lib.make_numpy_array(
        data_config=dataset_config,
        batch_size=model_config.num_transformer_samples,
    )
    batch_provider = lambda: train_lib.make_dataset(
        numpy_arr=numpy_arr,
        model_config=model_config,
        data_config=dataset_config,
        shuffle_labels=False,
    )
    return batch_provider, custom_labels


@torch.no_grad()
def evaluate_dataset(
    custom_labels,
    batch_provider: Callable[[], "common_ht.DatasetSamples"],
    model: layerwise.LayerwiseModel,
    eval_config: EvaluationConfig,
) -> Accuracies:
    """Runs evaluation loop for a specific dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    make_outputs_fn: Callable[
        [
            torch.Tensor,
            torch.Tensor,
            Optional[torch.Tensor],
            torch.Tensor,
            torch.Tensor,
        ],
        "common.ModelOutputs",
    ] = functools.partial(
        apply_layerwise_model,
        model=model,
    )

    test_accs = {}
    all_accs: list[float] = []

    for task_num in range(eval_config.num_task_evals):
        if custom_labels:
            np.random.shuffle(custom_labels)

        task_batch = batch_provider().to(device)
        cached_transformer_samples = task_batch.transformer_samples
        cached_transformer_labels = task_batch.transformer_labels
        cached_transformer_masks = task_batch.transformer_masks

        accs: list[float] = []
        for _ in range(eval_config.num_eval_batches):
            eval_batch = batch_provider().to(device)
            outputs: common.ModelOutputs = make_outputs_fn(
                transformer_samples=cached_transformer_samples,
                transformer_labels=cached_transformer_labels,
                transformer_masks=cached_transformer_masks,
                cnn_samples=eval_batch.cnn_samples,
                cnn_labels=eval_batch.cnn_labels,
            )
            accs.append(outputs.accuracy.item())

        test_accs[task_num] = accs
        all_accs.append(float(np.mean(accs)))

    """
    {
        task_0: [acc_1, ..., acc_16],
        task_1: [acc_1, ..., acc_16],
        ...
    },
    [
        mean_task_0,
        mean_task_1,
        ...
    ]
    """
    return test_accs, all_accs


def apply_layerwise_model(
    model: layerwise.LayerwiseModel,
    transformer_samples: torch.Tensor,
    transformer_labels: torch.Tensor,
    transformer_masks: Optional[torch.Tensor],
    cnn_samples: torch.Tensor,
    cnn_labels: torch.Tensor,
) -> common.ModelOutputs:
    """Applies a layerwise model to a dataset."""
    predictions: torch.Tensor = model(
        transformer_samples,
        transformer_labels,
        cnn_samples,
        mask=transformer_masks,
    )

    pred_labels = predictions.argmax(dim=-1)
    accuracy = (pred_labels == cnn_labels).float().mean()

    return common.ModelOutputs(predictions=pred_labels, accuracy=accuracy)
