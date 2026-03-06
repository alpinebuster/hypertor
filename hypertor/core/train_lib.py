"""Training library and binary."""

import os
import random

from typing import Optional

from absl import logging
import numpy as np
import torch
import torch.nn.functional as F

from hypertor.core import common_ht
from hypertor.core.common_ht import LayerwiseModelConfig, DatasetConfig
from hypertor.core import datasets
from hypertor.core import util


def _make_augmentation_config(data_config: DatasetConfig, num_labels: int):
    """Returns dataset augmentation configuration."""
    random_config = datasets.RandomizedAugmentationConfig(
        smooth_probability=data_config.smooth_probability,
        negate_probability=data_config.negate_probability,
    )
    if data_config.per_label_augmentation:
        return datasets.AugmentationConfig(
            children=[
                datasets.AugmentationConfig(random_config=random_config)
                for _ in range(num_labels)
            ]
        )
    else:
        return datasets.AugmentationConfig(random_config=random_config)


def _get_cached_npy_filename_prefix(data_config: DatasetConfig) -> tuple[str, str, str]:
    assert data_config.tcp_aug_config is not None
    sim_flags = {
        "nagle": data_config.tcp_aug_config.simulate_nagle,
        "shift": data_config.tcp_aug_config.simulate_packet_shift,
        "loss": data_config.tcp_aug_config.simulate_packet_loss,
        # "dup": data_config.tcp_aug_config.simulate_packet_duplication,
    }
    enabled = [name for name, flag in sim_flags.items() if flag]

    if len(enabled) == 0:
        sim_name = "none"
    elif len(enabled) == 1:
        sim_name = enabled[0]
    else:
        sim_name = "all"

    prefix_ds_name = ""
    if data_config.dataset_name == "cicdarknet":
        prefix_ds_name = "cic"
    elif data_config.dataset_name == "tcub":
        prefix_ds_name = "tcub"
    elif data_config.dataset_name == "safesurf2025":
        prefix_ds_name = "safesurf2025"
    else:
        raise RuntimeError("Wrong dataset")
    prefix = (
        f"{prefix_ds_name}_flows{data_config.tcp_aug_config.num_flows}"
        f"_sim{sim_name}"
        f"_len{data_config.tcp_aug_config.target_seq_len}"
    )
    simnone_prefix = (
        f"{prefix_ds_name}_flows{data_config.tcp_aug_config.num_flows}"
        f"_simnone_len{data_config.tcp_aug_config.target_seq_len}"
    )
    return simnone_prefix, prefix, sim_name

def merge_npy_files(data_config: DatasetConfig):
    """
    Merge files for seeds 2020–2026 based on the current ratio folder and current configuration.

    If sim == 'none', only need the file with seed 2026.
    """
    assert data_config.tcp_aug_config is not None

    simnone_prefix, prefix, sim_name = _get_cached_npy_filename_prefix(data_config)
    ratio_path = os.path.join(
        data_config.cache_path,
        f"ratio_{data_config.darknet_ratio}"
    )

    if os.path.exists(ratio_path):
        if sim_name == "none":
            return

        """ Episode augmentation configuration
        #
        # For each episode: all (7) classes use the same augmentations for support and query.
        # 
        # Example:
        #    Episode 1: Support → weak augmentation,   Query → strong augmentation
        #    Episode 2: Support → strong augmentation, Query → weak augmentation
        #    Episode 3: Support → augmentation_1,      Query → augmentation_2
        #    ...
        # 
        """
        arrays = []
        simnone_file_path = os.path.join(
            ratio_path,
            # `seed` only used in simulation, so we fix it to 2026 here!
            f"{simnone_prefix}_seed2026_5d.npy",
        )
        logging.info(
            f"{util.get_ddp_msg()}"
            f"Loading {simnone_file_path}"
        )
        if os.path.exists(simnone_file_path):
            simnone_arr: np.ndarray = np.load(simnone_file_path)
            if simnone_arr.ndim != 5:
                raise ValueError(f"Expected 5D array, got shape {simnone_arr.shape}")
            arrays.append(simnone_arr)

        simulation_seeds = data_config.tcp_aug_config.simulation_seeds
        for seed in range(simulation_seeds[0], simulation_seeds[1]+1):
            file_path = os.path.join(
                ratio_path,
                f"{prefix}_seed{seed}_5d.npy",
            )

            if not os.path.exists(file_path):
                logging.info(
                    f"{util.get_ddp_msg()}"
                    f"[WARNING] Missing {file_path}"
                )
                continue

            logging.info(
                f"{util.get_ddp_msg()}"
                f"Loading {file_path}"
            )
            if os.path.exists(file_path):
                arr: np.ndarray = np.load(file_path)
                if arr.ndim != 5:
                    raise ValueError(f"Expected 5D array, got shape {arr.shape}")
                if (
                    data_config.tcp_aug_config.simulation_samples_perclass is not None
                    and data_config.tcp_aug_config.simulation_samples_perclass>0
                ):
                    arrays.append(arr[:, :data_config.tcp_aug_config.simulation_samples_perclass, :, :, :])
                else:
                    arrays.append(arr)
            else:
                continue

        if len(arrays) == 0:
            raise RuntimeError("No matching npy files found.")
        merged = np.concatenate(arrays, axis=1)
        save_path = os.path.join(
            ratio_path,
            f"{prefix}_merged_5d.npy"
        )
        np.save(save_path, merged)
        logging.info(
            f"{util.get_ddp_msg()}"
            f"[MERGED] Saved → {save_path}"
        )

def _load_cache(data_config: DatasetConfig) -> Optional[dict[int, np.ndarray]]:
    """Loads cached dataset from a saved NumPy array."""
    ratio_path = os.path.join(
        data_config.cache_path,
        f"ratio_{data_config.darknet_ratio}"
    )
    logging.info(
        f"{util.get_ddp_msg()}"
        f'Looking for cache in "{ratio_path}..."'
    )
    _, prefix, sim_name = _get_cached_npy_filename_prefix(data_config)

    if sim_name == "none":
        _npy_path = os.path.join(
            ratio_path,
            f"{prefix}_seed2026_5d.npy"
        )
        logging.info(
            f"{util.get_ddp_msg()}"
            f"Loading not simulated data with seed 2026 {_npy_path}..."
        )
    else:
        _npy_path = os.path.join(
            ratio_path,
            f"{prefix}_merged_5d.npy"
        )
        if not os.path.exists(_npy_path):
            merge_npy_files(data_config)

        logging.info(
            f"{util.get_ddp_msg()}"
            f"Loading merged {_npy_path}..."
        )
    arr: np.ndarray = np.load(_npy_path)
    num_classes = arr.shape[0]
    assert data_config.tcp_aug_config is not None

    if (
        data_config.tcp_aug_config.num_pkt_features is not None
        and data_config.tcp_aug_config.num_pkt_features>0
    ):
        data: dict[int, np.ndarray] = {
            label: arr[label][:, :, :data_config.tcp_aug_config.num_pkt_features, :]
            for label in range(num_classes)
        }
    else:
        data = {
            label: arr[label]
            for label in range(num_classes)
        }

    if data is None:
        logging.info(
            f"{util.get_ddp_msg()}"
            f"No cache files for {data_config.dataset_name} found. Falling back "
            "to torch dataset."
        )
        return None
    else:
        return data

def make_numpy_array(
    data_config: DatasetConfig,
    batch_size: int,
) -> dict[int, np.ndarray]:
    """Makes a NumPy array for given dataset configuration.
    
    Returns:
       Dictionary mapping labels to tensors containing all samples (samples_per_label, C, H, W).

       Dict[int, np.ndarray]:
         int -> all_classes
         np.ndarray -> [all_samples, C, H, W]
    """
    output = None

    ds = data_config.ds
    if ds is None:
        output = _load_cache(data_config)
        if output is None:
            raise ValueError(f"Dataset {data_config.dataset_name} not found and no cache available.")

    dataset_info = data_config.dataset_info
    if dataset_info is None:
        dataset_info = datasets.get_dataset_info(data_config.dataset_name)

    if output is None and ds is not None:
        assert dataset_info.num_samples_per_label is not None
        output = datasets.make_numpy_data(
            ds=ds,
            batch_size=batch_size,
            num_labels=dataset_info.num_labels,
            samples_per_label=dataset_info.num_samples_per_label,
            transpose=dataset_info.transpose_samples,
        )
    assert output is not None
    if data_config.shuffle_labels_seed > 0:
        keys = list(output.keys())
        orig_keys = keys[:]
        random.seed(data_config.shuffle_labels_seed)
        random.shuffle(keys)
        output = {orig_keys[i]: output[keys[i]] for i in range(len(keys))}

    return output


def _resize(imgs: torch.Tensor, flows_shape: tuple[int, int]) -> torch.Tensor:
    if imgs.dim() == 2: # HW
        imgs = imgs.unsqueeze(0).unsqueeze(0)
    elif imgs.dim() == 3:
        if imgs.shape[-1] <= 4: # HWC (TF Style)
            imgs = imgs.permute(2, 0, 1).unsqueeze(0)
        else: # CHW
            imgs = imgs.unsqueeze(0)
    elif imgs.dim() == 4:
        if imgs.shape[-1] <= 4:  # BHWC
            imgs = imgs.permute(0, 3, 1, 2)

    return F.interpolate(
        imgs,
        size=flows_shape,
        mode="bilinear",
        align_corners=False,
    )


def _make_dataset_helper_unbalanced(
    numpy_arr: dict[int, np.ndarray],
    batch_size: int,
    flows_shape: tuple[int, int],
    num_labels: int,
    data_config: DatasetConfig,
    always_same_labels=False,
) -> datasets.SupervisedSample:
    """Helper function for creating a dataset.

       `numpy_arr = make_numpy_array(data_config, batch_size)`
    """
    config = _make_augmentation_config(data_config=data_config, num_labels=num_labels)

    gen = datasets.TaskGenerator(
        data_config,
        numpy_arr,
        num_labels=num_labels,
        use_label_subset=data_config.use_label_subset,
        always_same_labels=always_same_labels,
    )
    config.randomize()
    samples, labels, classes = gen.get_batch(
        batch_size=batch_size,
        config=config,
        num_unlabeled_per_class=data_config.num_unlabeled_per_class,
    )

    if flows_shape is not None and samples[0].shape != flows_shape:
        samples = _resize(samples, flows_shape)

    sample: datasets.SupervisedSample = (samples, labels, classes)
    return sample


def _make_dataset_helper_balanced(
    numpy_arr: dict[int, np.ndarray],
    batch_sizes: list[int],
    num_unlabeled_per_class: list[int],
    flows_shape,
    num_labels: int,
    data_config: DatasetConfig,
    always_same_labels=False,
) -> list[datasets.SupervisedSample]:
    """Helper function for creating a balanced dataset.

       `numpy_arr = make_numpy_array(data_config, batch_sizes[0])`
    """
    config = _make_augmentation_config(data_config=data_config, num_labels=num_labels)

    gen = datasets.TaskGenerator(
        data_config,
        numpy_arr,
        num_labels=num_labels,
        use_label_subset=data_config.use_label_subset,
        always_same_labels=always_same_labels,
    )
    config.randomize()
    samples_labels = gen.get_batches(
        batch_sizes=batch_sizes,
        config=config,
        num_unlabeled_per_class=num_unlabeled_per_class,
    )
    output: list[datasets.SupervisedSample] = []
    for samples, labels, classes in samples_labels:
        if flows_shape is not None and samples[0][0].shape != flows_shape:
            samples = _resize(samples, flows_shape)

        output.append((samples, labels, classes))

    return output


def _get_class_bounds(data_config: DatasetConfig):
    if data_config.use_label_subset is None or callable(data_config.use_label_subset):
        return None, None
    return min(data_config.use_label_subset), max(data_config.use_label_subset)


def _make_dataset_unbalanced(
    numpy_arr: dict[int, np.ndarray],
    model_config: LayerwiseModelConfig,
    data_config: DatasetConfig,
    shuffle_labels=True,
):
    """Creates data for Transformer and CNN.

    Arguments:
      model_config: Model configuration.
      data_config: Dataset configuration.
      shuffle_labels: True if should subsample random labels from
          `data_config.use_label_subset` for each new mini-dataset.

    Returns:
      `DatasetSamples` structure containing Transformer and CNN samples.
    """
    batch_size = model_config.num_transformer_samples
    batch_size += model_config.num_cnn_samples

    assert len(model_config.flows_shape) == 2
    sample = _make_dataset_helper_unbalanced(
        numpy_arr=numpy_arr,
        batch_size=batch_size,
        flows_shape=model_config.flows_shape,
        num_labels=model_config.num_labels,
        data_config=data_config,
        always_same_labels=not shuffle_labels,
    )
    samples, labels, classes = sample

    num_transformer_samples = model_config.num_transformer_samples
    transformer_samples = samples[:num_transformer_samples]
    if len(transformer_samples.shape) == 3:
        transformer_samples = transformer_samples.unsqueeze(dim=-3)
    cnn_samples = samples[num_transformer_samples:]
    if len(cnn_samples.shape) == 3:
        cnn_samples = cnn_samples.unsqueeze(dim=-3)

    real_class_min, real_class_max = _get_class_bounds(data_config)

    return common_ht.DatasetSamples(
        transformer_samples=transformer_samples,
        transformer_labels=labels[:num_transformer_samples],
        transformer_real_classes=classes[:num_transformer_samples],
        cnn_samples=cnn_samples,
        cnn_labels=labels[num_transformer_samples:],
        cnn_real_classes=classes[num_transformer_samples:],
        real_class_min=real_class_min,
        real_class_max=real_class_max,
    )


def _make_dataset_balanced(
    numpy_arr: dict[int, np.ndarray],
    model_config: LayerwiseModelConfig,
    data_config: DatasetConfig,
    shuffle_labels: bool = True,
):
    """Creates data for Transformer and CNN.

    Arguments:
      model_config: Model configuration.
      data_config: Dataset configuration.
      shuffle_labels: True if should subsample random labels from
          `data_config.use_label_subset` for each new mini-dataset.

    Returns:
      `DatasetSamples` structure containing Transformer and CNN samples.
    """
    batch_sizes = [model_config.num_transformer_samples, model_config.num_cnn_samples]
    # Removing labels only from the Transformer batch.
    num_unlabeled_per_class = [data_config.num_unlabeled_per_class, 0]

    batches = _make_dataset_helper_balanced(
        numpy_arr=numpy_arr,
        batch_sizes=batch_sizes,
        num_unlabeled_per_class=num_unlabeled_per_class,
        flows_shape=model_config.flows_shape,
        num_labels=model_config.num_labels,
        data_config=data_config,
        always_same_labels=not shuffle_labels,
    )

    transformer_samples, transformer_labels, transformer_classes = batches[0]
    cnn_samples, cnn_labels, cnn_classes = batches[1]

    # BCHW
    if len(transformer_samples.shape) == 3:
        transformer_samples = transformer_samples.unsqueeze(dim=-3)
    if len(cnn_samples.shape) == 3:
        cnn_samples = cnn_samples.unsqueeze(dim=-3)

    real_class_min, real_class_max = _get_class_bounds(data_config)

    return common_ht.DatasetSamples(
        transformer_samples=transformer_samples,
        transformer_labels=transformer_labels,
        transformer_real_classes=transformer_classes,
        cnn_samples=cnn_samples,
        cnn_labels=cnn_labels,
        cnn_real_classes=cnn_classes,
        real_class_min=real_class_min,
        real_class_max=real_class_max,
    )


def make_dataset(
    numpy_arr: dict[int, np.ndarray],
    model_config: LayerwiseModelConfig,
    data_config: DatasetConfig,
    **kwargs,
):
    """Makes dataset given dataset and model configuration.
    
    1) Balanced:
       The number of samples for each category in each batch is the same (or as much as possible the same), and within the batch, they are clearly organized by "category".

       e.g.
          Transformer: [0,1,2,3,4, ...]
          CNN: [0,1,2,3,4, ...]

    2) Unbalanced:
       In a batch, only the total number of samples is guaranteed, and the occurrence frequency of each category is not guaranteed to be consistent.

       e.g.
          Transformer: [0, 0, 0, 1, 1, 3]
          CNN: [2, 2, 2, 4, 4]
    """
    if data_config.balanced_batches:
        # The two batches are "generated separately".
        dataset_maker = _make_dataset_balanced
    else:
        dataset_maker = _make_dataset_unbalanced

    return dataset_maker(
        numpy_arr=numpy_arr,
        model_config=model_config,
        data_config=data_config,
        **kwargs,
    )
