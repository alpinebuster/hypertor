"""Meta training binary.

# N way K shots
for each Episode:
    Sample N Classes

    for each Class -> Sample K support samples + Q query samples:
        Compute prototype from support samples
        Classify query
        Compute loss and update
"""

import os
import functools
from typing import cast, Optional

from absl import app, flags, logging

import torch
import torch.distributed as dist

from hypertor import common_flags
from hypertor import eval_model_flags  # pylint:disable=unused-import
from hypertor import variable_proto_flags  # pylint:disable=unused-import

from hypertor.core import common
from hypertor.core import common_ht
from hypertor.core import layerwise
from hypertor.core import layerwise_defs  # pylint:disable=unused-import
from hypertor.core import train_lib
from hypertor.core import util
from hypertor.variable_proto import common_vp
from hypertor.variable_proto.receptive_field import compute_proto_layer_rf_info

FLAGS = flags.FLAGS


# ------------------------------------------------------------
#   Configurations
# ------------------------------------------------------------


def make_variable_proto_config():
    return common_vp.ProtoTrainConfig(
        warm_epochs=FLAGS.warmup_steps,
        push_freq=FLAGS.push_freq,
        push_start=FLAGS.push_start,
        prototypes=FLAGS.prototypes,
        prototype_size=FLAGS.prototype_size,
        prototype_activation_func=FLAGS.prototype_activation_func,
        coef_cross_entropy=FLAGS.coef_cross_entropy,
        coef_cluster_cost=FLAGS.coef_cluster_cost,
        coef_separation_cost=FLAGS.coef_separation_cost,
        coef_reg_protos=FLAGS.coef_reg_protos,
        coef_reg_last=FLAGS.coef_reg_last,
    )


def make_train_config():
    return common.TrainConfig(
        train_steps=FLAGS.train_steps,
        steps_between_saves=FLAGS.steps_between_saves,
    )


def make_optimizer_config():
    return common.OptimizerConfig(
        learning_rate=FLAGS.learning_rate,
        learning_momentum=FLAGS.learning_momentum,
        lr_decay_steps=FLAGS.learning_rate_decay_steps,
        lr_decay_rate=FLAGS.learning_rate_decay_rate,
    )


def _common_model_config():
    """Returns common ModelConfig parameters."""
    return {
        "num_transformer_samples": FLAGS.samples_transformer,
        "num_cnn_samples": FLAGS.samples_cnn,
        "num_labels": FLAGS.num_labels,
        "image_size": util.normalize_image_size(FLAGS.image_size),
        "cnn_model_name": FLAGS.cnn_model_name,
        "embedding_dim": FLAGS.embedding_dim,
        "cnn_dropout_rate": FLAGS.cnn_dropout_rate,
        "use_decoder": FLAGS.use_decoder,
        "add_trainable_weights": FLAGS.add_trainable_weights,
        "var_reg_weight": FLAGS.weight_variation_regularization,
        "transformer_activation": FLAGS.transformer_activation,
        "transformer_nonlinearity": FLAGS.transformer_nonlinearity,
        "cnn_activation": FLAGS.cnn_activation,
        "default_num_channels": FLAGS.default_num_channels,
        "shared_fe_dropout": FLAGS.shared_fe_dropout,
        "fe_dropout": FLAGS.fe_dropout,
    }

def make_layerwise_model_config():
    """Makes 'layerwise' model config."""
    if not FLAGS.num_layerwise_features:
        num_features = None
    else:
        num_features = int(FLAGS.num_layerwise_features)

    if FLAGS.lw_weight_allocation == "spatial":
        weight_allocation = common_ht.WeightAllocation.SPATIAL
    elif FLAGS.lw_weight_allocation == "output":
        weight_allocation = common_ht.WeightAllocation.OUTPUT_CHANNEL
    else:
        raise ValueError(
            f"Unknown `lw_weight_allocation` flag value "
            f'"{FLAGS.lw_weight_allocation}"'
        )

    return common_ht.LayerwiseModelConfig(
        feature_layers=2,
        query_key_dim_frac=FLAGS.lw_key_query_dim,
        value_dim_frac=FLAGS.lw_value_dim,
        internal_dim_frac=FLAGS.lw_inner_dim,
        num_layers=FLAGS.num_layers,
        heads=FLAGS.heads,
        kernel_size=common_flags.KERNEL_SIZE.value,
        stride=common_flags.STRIDE.value,
        dropout_rate=FLAGS.dropout_rate,
        num_features=num_features,
        nonlinear_feature=FLAGS.lw_use_nonlinear_feature,
        weight_allocation=weight_allocation,
        generate_bn=FLAGS.lw_generate_bn,
        generate_bias=FLAGS.lw_generate_bias,
        generate_last_layer_bias=FLAGS.lw_generate_last_layer_bias,
        shared_feature_extractor=FLAGS.shared_feature_extractor,
        shared_input_dim=FLAGS.shared_input_dim,
        shared_features_dim=FLAGS.shared_features_dim,
        separate_bn_vars=FLAGS.separate_evaluation_bn_vars,
        shared_feature_extractor_padding=FLAGS.shared_feature_extractor_padding,
        label_smoothing=FLAGS.label_smoothing,
        generator=FLAGS.layerwise_generator,
        train_heads=FLAGS.warmup_steps > 0,
        max_prob_remove_unlabeled=FLAGS.max_prob_remove_unlabeled,
        max_prob_remove_labeled=FLAGS.max_prob_remove_labeled,
        number_of_trained_cnn_layers=(common_flags.NUMBER_OF_TRAINED_CNN_LAYERS.value),
        skip_last_nonlinearity=FLAGS.transformer_skip_last_nonlinearity,
        l2_reg_weight=FLAGS.l2_reg_weight,
        logits_feature_extractor=FLAGS.logits_feature_extractor,
        shared_head_weight=common_flags.SHARED_HEAD_WEIGHT.value,
        use_variable_proto=FLAGS.use_variable_proto,
        use_prior_proto=FLAGS.use_prior_proto,
        use_l_mask=FLAGS.use_l_mask,
        use_class_specific=FLAGS.use_class_specific,
        **_common_model_config(),
    )


def make_dataset_config(dataset_spec=""):
    if not dataset_spec:
        dataset_spec = FLAGS.train_dataset
    dataset, label_set = util.parse_dataset_spec(dataset_spec)
    if label_set is None:
        label_set = list(range(FLAGS.use_labels))
    return common_ht.DatasetConfig(
        dataset_name=dataset,
        use_label_subset=label_set,
        ds_split="train",
        data_dir=FLAGS.data_dir,
        rotation_probability=FLAGS.rotation_probability,
        smooth_probability=FLAGS.smooth_probability,
        contrast_probability=FLAGS.contrast_probability,
        resize_probability=FLAGS.resize_probability,
        negate_probability=FLAGS.negate_probability,
        roll_probability=FLAGS.roll_probability,
        angle_range=FLAGS.angle_range,
        rotate_by_90=FLAGS.random_rotate_by_90,
        per_label_augmentation=FLAGS.per_label_augmentation,
        cache_path=FLAGS.data_numpy_dir,
        balanced_batches=FLAGS.balanced_batches,
        shuffle_labels_seed=FLAGS.shuffle_labels_seed,
        apply_image_augmentations=FLAGS.apply_image_augmentations,
        augment_individually=FLAGS.augment_images_individually,
        num_unlabeled_per_class=FLAGS.unlabeled_samples_per_class,
    )


def _default(new: float, default: float):
    return new if new >= 0 else default

def make_test_dataset_config(dataset_spec=""):
    if not dataset_spec:
        dataset_spec = FLAGS.test_dataset
    dataset, label_set = util.parse_dataset_spec(dataset_spec)
    if label_set is None:
        raise ValueError("Test dataset should specify a set of labels.")
    return common_ht.DatasetConfig(
        dataset_name=dataset,
        use_label_subset=label_set,
        ds_split=FLAGS.test_split,
        data_dir=FLAGS.data_dir,
        rotation_probability=_default(
            FLAGS.test_rotation_probability, FLAGS.rotation_probability
        ),
        smooth_probability=_default(
            FLAGS.test_smooth_probability, FLAGS.smooth_probability
        ),
        contrast_probability=_default(
            FLAGS.test_contrast_probability, FLAGS.contrast_probability
        ),
        resize_probability=_default(
            FLAGS.test_resize_probability, FLAGS.resize_probability
        ),
        negate_probability=_default(
            FLAGS.test_negate_probability, FLAGS.negate_probability
        ),
        roll_probability=_default(FLAGS.test_roll_probability, FLAGS.roll_probability),
        angle_range=_default(FLAGS.test_angle_range, FLAGS.angle_range),
        rotate_by_90=FLAGS.test_random_rotate_by_90,
        per_label_augmentation=FLAGS.test_per_label_augmentation,
        balanced_batches=FLAGS.balanced_batches,
        shuffle_labels_seed=FLAGS.shuffle_labels_seed,
        cache_path=FLAGS.data_numpy_dir,
        apply_image_augmentations=False,
        num_unlabeled_per_class=FLAGS.unlabeled_samples_per_class,
    )


# ------------------------------------------------------------
#   Model
# ------------------------------------------------------------


def create_layerwise_model(
    model_config: common_ht.LayerwiseModelConfig,
    dataset: common_ht.DatasetSamples,
    optim_config: common.OptimizerConfig,
):
    """Creates a hierarchical Transformer-CNN model."""
    logging.info(f"{util.get_ddp_msg()}Building the model...")

    model = layerwise.build_model(
        model_config.cnn_model_name,
        model_config=model_config,
        dataset=dataset,
    )
    # Run `setup()` to create all layers / transformer / BN, etc.
    # Use `torch.no_grad()` to avoid constructing unnecessary gradient graphs
    with torch.no_grad():
        model.setup(
            dataset.transformer_images,
            dataset.transformer_labels,
            mask=dataset.transformer_masks,
            mask_random_samples=True,
            enable_fe_dropout=True,
            only_shared_feature=False,
        )
    optimizer, scheduler = util.make_optimizer(optim_config, model)

    return common.TrainState(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_dir=FLAGS.train_log_dir,
    )


def create_shared_feature_model(
    model_config: common_ht.LayerwiseModelConfig,
    dataset: common_ht.DatasetSamples,
    optim_config: common.OptimizerConfig,
):
    """Creates an image feature extractor model for pre-training."""
    logging.info(f"{util.get_ddp_msg()}Building the model...")

    model = layerwise.build_model(
        model_config.cnn_model_name,
        model_config=model_config,
        dataset=dataset,
    )
    with torch.no_grad():
        model.setup(
            dataset.transformer_images,
            dataset.transformer_labels,
            dataset.transformer_masks,
            mask_random_samples=True,
            enable_fe_dropout=True,
            only_shared_feature=True,
        )
    optimizer, scheduler = util.make_optimizer(optim_config, model)

    return common.TrainState(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_dir=FLAGS.train_log_dir,
    )


# ------------------------------------------------------------
#   Model Loading
# ------------------------------------------------------------


def restore_shared_features(
    model: "layerwise.LayerwiseModel",
    checkpoint: Optional[str] = None,
) -> Optional["layerwise.LayerwiseModel"]:
    """Restores shared feature extractor variables from a checkpoint."""
    checkpoint = checkpoint or common_flags.RESTORE_SHARED_FEATURES_FROM.value
    if not checkpoint:
        return None

    # The parameter names in the current model
    model_state: dict[str, torch.Tensor] = model.state_dict()
    # Get shared features / head
    shared_keys = [
        k for k in model_state.keys()
        if k.startswith("shared_feature_extractor.") or k.startswith("shared_head.")
    ]    
    if not shared_keys:
        return None

    loaded_vars = util.load_variables(
        checkpoint,
        var_list=shared_keys,
        map_location="cpu",
    )
    # `strict=False` → Load only what exists
    model.load_state_dict(loaded_vars, strict=False)

    return model

def restore_variable_proto(
    model_config: common_ht.LayerwiseModelConfig,
    checkpoint: Optional[str] = None,
) -> Optional["common_vp.VariableProtoState"]:
    """Restores variable prototype variables from a checkpoint."""
    if not model_config.use_variable_proto:
        return None
    ckpt_path = util.latest_checkpoint(checkpoint or FLAGS.train_log_dir)
    if not ckpt_path:
        return None
    data = torch.load(ckpt_path, map_location="cpu")

    return common_vp.VariableProtoState(
        step_updated=int(data.get("step_updated", 0)),
        prototype_shape=tuple(data.get("prototype_shape", (0, 0, 0, 0))),
        prototype_class_identity=data.get(
            "prototype_class_identity", torch.zeros(0, 0)
        ),
        prototype_ids_per_class=data.get("prototype_ids_per_class", {}),
    )


# ------------------------------------------------------------
#   Meta Train Entrance
# ------------------------------------------------------------


def train(
    train_config: common.TrainConfig,
    optimizer_config: common.OptimizerConfig,
    dataset_config: common_ht.DatasetConfig,
    test_dataset_config: common_ht.DatasetConfig,
    model_config: common_ht.LayerwiseModelConfig,
    variable_proto_config: common_vp.ProtoTrainConfig,
):
    """Main function training the model."""
    logging.info(f"{util.get_ddp_msg()}Creating the dataset...")

    # ALL data
    numpy_arr = train_lib.make_numpy_array(
        data_config=dataset_config,
        batch_size=model_config.num_transformer_samples,
    )
    # Generating different episodes at initialization & per step
    train_batch_provider = lambda: train_lib.make_dataset(
        numpy_arr=numpy_arr,
        model_config=model_config,
        data_config=dataset_config,
    )
    test_batch_provider = lambda: train_lib.make_dataset(
        numpy_arr=numpy_arr,
        model_config=model_config,
        data_config=test_dataset_config,
    )

    # Variable prototype
    vp_state = restore_variable_proto(model_config)
    if vp_state:
        step_updated = vp_state.step_updated
        prototype_shape = vp_state.prototype_shape
        prototype_class_identity = vp_state.prototype_class_identity
        prototype_ids_per_class = vp_state.prototype_ids_per_class
    elif model_config.use_variable_proto:
        # Number of epochs already computed
        step_updated = 0
        num_classes = model_config.num_labels
        num_prototypes = num_classes * variable_proto_config.prototypes
        prototype_shape = (
            num_prototypes,
            model_config.default_num_channels,
            variable_proto_config.prototype_size[0],
            variable_proto_config.prototype_size[1],
        )
        """
        e.g. 
        num_classes = 5
        num_prototypes = 5
            ↓
        prototype_class_identity = [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ]
            ↓
        [
            [1, 0, 0, 0, 0],  # class 0
            [0, 1, 0, 0, 0],  # class 1
            [0, 0, 1, 0, 0],  # class 2
            [0, 0, 0, 1, 0],  # class 3
            [0, 0, 0, 0, 1],  # class 4
            [0, 0, 1, 0, 0],  # New prototype → class 2
        ]
        """
        prototype_class_identity = torch.zeros(num_prototypes, num_classes)
        for j in range(num_prototypes):
            prototype_class_identity[j, j // variable_proto_config.prototypes] = 1
        """
        e.g. 
        num_classes = 5
            ↓
        prototype_ids_per_class = {
            0: [0],
            1: [1],
            2: [2],
            3: [3],
            4: [4],
        }
            ↓
        prototype_ids_per_class = {
            0: [0],
            1: [1],
            2: [2, 5],  # Class 2 now have 2 prototype
            3: [3],
            4: [4],
        }
        """
        prototype_ids_per_class = {}
        for i in range(num_classes):
            prototype_ids_per_class[i] = [i]
    else:
        step_updated = 0
        prototype_shape = None
        prototype_class_identity = None
        prototype_ids_per_class = None

    # Train the model with new prototypes
    change = True
    while change:
        logging.info(f"{util.get_ddp_msg()}Start Training...")

        _args = {
            "dataset": train_batch_provider(),
            "optim_config": optimizer_config,
        }
        # For variable prototype
        if model_config.use_variable_proto:
            assert prototype_shape is not None
            assert prototype_class_identity is not None
            assert prototype_ids_per_class is not None
            model_config.prototype_shape = prototype_shape
            model_config.prototype_class_identity = prototype_class_identity
            model_config.prototype_ids_per_class = prototype_ids_per_class

        if common_flags.PRETRAIN_SHARED_FEATURE.value:
            create_model = functools.partial(
                create_shared_feature_model,
                model_config=model_config,
            )
        else:
            create_model = functools.partial(
                create_layerwise_model,
                model_config=model_config,
            )
        state = create_model(**_args)

        # This will update the values of state
        restored = common.init_training(state)
        if not restored:
            model = restore_shared_features(
                model=cast(layerwise.LayerwiseModel, state.model),
            )
            if model:
                state.model = model
        assert isinstance(state.model, layerwise.LayerwiseModel)

        if model_config.use_variable_proto:
            assert prototype_shape is not None
            assert prototype_class_identity is not None
            assert prototype_ids_per_class is not None
            if vp_state:
                state.vp_state = vp_state
            else:
                state.vp_state = common_vp.VariableProtoState(
                    step_updated=step_updated,
                    prototype_shape=prototype_shape,
                    prototype_class_identity=prototype_class_identity,
                    prototype_ids_per_class=prototype_ids_per_class,
                )

            layer_filter_sizes, layer_strides, layer_paddings = state.model.conv_info()
            proto_layer_rf_info = compute_proto_layer_rf_info(
                input_size=model_config.image_size,
                layer_filter_sizes=layer_filter_sizes,
                layer_strides=layer_strides,
                layer_paddings=layer_paddings,
                prototype_kernel_size=prototype_shape[2],
            )
            model_config.proto_layer_rf_info = proto_layer_rf_info

        (
            change,
            step_updated,
            prototype_shape,
            prototype_class_identity,
            prototype_ids_per_class,
        ) = common.train(
            train_config=train_config,
            model_config=model_config,
            variable_proto_config=variable_proto_config,
            optimizer_config=optimizer_config,
            state=state,
            batch_provider=(train_batch_provider, test_batch_provider),
        )
        del state
        torch.cuda.empty_cache()


def main(argv):
    # The command-line parameters have been parsed by absl.
    if len(argv) > 1:
        del argv

    if FLAGS.use_ddp:
        util.setup_ddp()
    if not FLAGS.use_ddp or dist.get_rank() == 0:
        logging.info(f"FLAGS: {FLAGS.flag_values_dict()}")

    gpus: str = FLAGS.gpus
    if gpus is None or gpus == "all":
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    util.print_gpu_detailed_info()

    train(
        train_config=make_train_config(),
        optimizer_config=make_optimizer_config(),
        dataset_config=make_dataset_config(),
        test_dataset_config=make_test_dataset_config(),
        model_config=make_layerwise_model_config(),
        variable_proto_config=make_variable_proto_config(),
    )


if __name__ == "__main__":
    app.run(main)
