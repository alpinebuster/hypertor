import os
import time

from typing import TYPE_CHECKING, cast, Optional, Union

from absl import logging
from scipy.stats import kurtosis
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import numpy as np
import cv2

from hypertor.core import util
from hypertor.variable_proto.receptive_field import compute_rf_prototype
from hypertor.variable_proto.util import find_high_activation_crop
import hypertor.core.layerwise as layerwise

if TYPE_CHECKING:
    from hypertor.core import common_ht
    from hypertor.variable_proto import common_vp


def _update_prototypes_on_batch(
    variable_proto_config: "common_vp.ProtoTrainConfig",
    search_batch_input,
    start_index_of_search_batch,
    model_ddp: Union["DDP", "layerwise.LayerwiseModel"],
    epoch_number,
    global_min_proto_dist, # this will be updated
    global_min_fmap_patches, # this will be updated
    global_samples_dist, # this will be updated
    proto_rf_boxes, # this will be updated
    proto_bound_boxes, # this will be updated
    class_specific=True,
    search_y=None, # required if class_specific == True
    num_classes=None, # required if class_specific == True
    preprocess_input_function=None,
    prototype_layer_stride=1,
    dir_for_saving_prototypes=None,
    prototype_img_filename_prefix=None,
    prototype_self_act_filename_prefix=None,
    prototype_activation_func_in_numpy=None,
    percentile=95,
):
    """Update each prototypes for current search batch.

    Parameters:
       search_batch_input: Episode from the data
       start_index_of_search_batch: Start index of the batch
       model_ddp: Model to process
       epoch_number: Number of epochs
       global_min_proto_dist: Minimum distance to latent patch for each prototype
       global_min_fmap_patches: Closest latent patch for each prototype
       global_samples_dist: Distances samples to prototypes
       proto_rf_boxes: Prototype receptive fields information
       proto_bound_boxes: Prototype boundaries of highly activated region
       class_specific: Class-specific processing if True
       search_y: Labels of the batch
       num_classes: Number of classes
       preprocess_input_function: Name of preprocessing function
       prototype_layer_stride: Stride used by the prototype layer on the feature maps
       dir_for_saving_prototypes: Folder to save prototypes
       prototype_img_filename_prefix: Prefix of the file to save prototypes
       prototype_self_act_filename_prefix: Prefix of the file to save prototype activation values
       prototype_activation_func_in_numpy: Name of the activation function if not 'log' or 'linear'
       percentile: Percentile of activation values
    """
    model_ddp.eval()
    unwrapped_model = util.unwrap_model(model_ddp)
    unwrapped_model = cast("layerwise.LayerwiseModel", unwrapped_model)
    proto_layer = unwrapped_model.layers[-2]
    assert isinstance(proto_layer, layerwise.VariableProtoLayer)

    if preprocess_input_function is not None:
        search_batch = preprocess_input_function(search_batch_input)
    else:
        search_batch = search_batch_input

    with torch.no_grad():
        search_batch = search_batch.cuda()
        (
            protoL_input_torch, # x: (batch_size, channel_size, Hf, Wf)
            proto_dist_torch, # distances -> (batch_size, num_prototypes, Hf-Hp+1, Wf-Wp+1)
        ) = proto_layer.push_forward()

    protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy())
    proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())

    del protoL_input_torch, proto_dist_torch

    if class_specific:
        class_to_img_index_dict = {key: [] for key in range(num_classes)} # type: ignore
        # img_y is the sample's integer label
        for img_index, img_y in enumerate(search_y): # type: ignore
            img_label = img_y.item()
            class_to_img_index_dict[img_label].append(img_index)

    prototype_shape = proto_layer.prototype_shape
    n_prototypes = prototype_shape[0]
    proto_h = prototype_shape[2]
    proto_w = prototype_shape[3]
    max_dist = prototype_shape[1]*prototype_shape[2]*prototype_shape[3]

    # In the early stage, the model is still learning the prototypes.
    # When the epoch is >= 100 in the later stage, the prototypes will be fixed and no longer increase dynamically.
    if epoch_number < variable_proto_config.push_start+variable_proto_config.push_freq*50:
        # proto_dist_ -> (batch_size, n_prototypes, Hp, Wp)
        for spl in range(0, proto_dist_.shape[0]):
            for spl_proto in range(n_prototypes):
                # One sample → One prototype → Take the feature map that most resembles the prototype
                dist_sample = np.amin(proto_dist_[spl, spl_proto, :, :])
                # global_samples_dist -> (num_total_samples, n_prototypes)
                global_samples_dist[
                    start_index_of_search_batch+spl, spl_proto
                ] = dist_sample

    for j in range(n_prototypes):
        if class_specific:
            # `target_class` is the class of the `class_specific` prototype
            target_class = torch.argmax(
                # prototype_class_identity -> (num_prototypes, num_classes)
                proto_layer.prototype_class_identity[j]
            ).item()

            # If there is not samples of the `target_class` from this batch
            # we go on to the next prototype
            if len(class_to_img_index_dict[target_class]) == 0: # type: ignore
                continue

            # `proto_dist_` -> (batch_size, n_prototypes, Hp, Wp)
            # `proto_dist_[class_to_img_index_dict[target_class]` -> (samples_of_target_class, n_prototypes, Hp, Wp)
            # `proto_dist_j` -> (samples_of_target_class, Hp, Wp)
            proto_dist_j = proto_dist_[class_to_img_index_dict[target_class]][ # type: ignore
                :, j, :, :
            ]
        else:
            # If it is not class specific, then we will search through
            # every example
            proto_dist_j = proto_dist_[:, j, :, :]

        batch_min_proto_dist_j = np.amin(proto_dist_j)
        if batch_min_proto_dist_j < global_min_proto_dist[j]:
            # `np.argmin` -> flatten to [0, 1, 2, 3, ..., samples_of_target_class x Hp x Wp] -> index of minimum distance
            # `np.unravel_index` -> restored to (samples_of_target_class, Hp, Wp) -> index of best match within batch
            batch_argmin_proto_dist_j = list(
                np.unravel_index(np.argmin(proto_dist_j, axis=None), proto_dist_j.shape)
            )
            if class_specific:
                # Change the `argmin` index from the index among samples of the target class to the index in the entire search batch
                # batch_argmin_proto_dist_j -> [global_sample_index, Hp_index, Wp_index]
                batch_argmin_proto_dist_j[0] = class_to_img_index_dict[target_class][ # type: ignore
                    batch_argmin_proto_dist_j[0]
                ]

            # Retrieve the corresponding feature map patch
            img_index_in_batch = batch_argmin_proto_dist_j[0]
            fmap_height_start_index = (
                batch_argmin_proto_dist_j[1] * prototype_layer_stride
            )
            fmap_height_end_index = fmap_height_start_index + proto_h
            fmap_width_start_index = (
                batch_argmin_proto_dist_j[2] * prototype_layer_stride
            )
            fmap_width_end_index = fmap_width_start_index + proto_w

            batch_min_fmap_patch_j = protoL_input_[
                img_index_in_batch,
                :,
                fmap_height_start_index:fmap_height_end_index,
                fmap_width_start_index:fmap_width_end_index,
            ]

            global_min_proto_dist[j] = batch_min_proto_dist_j
            global_min_fmap_patches[j] = batch_min_fmap_patch_j

            # Get the receptive field boundary of the patch that generates the representation
            protoL_rf_info = proto_layer.proto_layer_rf_info
            rf_prototype_j = compute_rf_prototype(
                search_batch.size(2),
                batch_argmin_proto_dist_j,
                protoL_rf_info,
            )

            # Get the whole sample
            original_img_j = search_batch_input[rf_prototype_j[0]]
            original_img_j = original_img_j.detach().cpu().numpy()
            original_img_j = np.transpose(original_img_j, (1, 2, 0))
            original_img_size0 = original_img_j.shape[1]
            original_img_size1 = original_img_j.shape[0]

            # Save the prototype receptive field information
            proto_rf_boxes[j, 0] = rf_prototype_j[0] + start_index_of_search_batch
            proto_rf_boxes[j, 1] = rf_prototype_j[1]
            proto_rf_boxes[j, 2] = rf_prototype_j[2]
            proto_rf_boxes[j, 3] = rf_prototype_j[3]
            proto_rf_boxes[j, 4] = rf_prototype_j[4]
            if proto_rf_boxes.shape[1] == 6 and search_y is not None:
                proto_rf_boxes[j, 5] = search_y[rf_prototype_j[0]].item()

            # Find the highly activated region of the original sample
            proto_dist_img_j = proto_dist_[img_index_in_batch, j, :, :]
            if proto_layer.prototype_activation_func == "log":
                proto_act_img_j = np.log(
                    (proto_dist_img_j + 1)
                    / (proto_dist_img_j + proto_layer.epsilon)
                )
            elif (
                proto_layer.prototype_activation_func == "linear"
            ):
                proto_act_img_j = max_dist - proto_dist_img_j
            else:
                proto_act_img_j = prototype_activation_func_in_numpy(
                    proto_dist_img_j  # type: ignore
                )
            upsampled_act_img_j = cv2.resize( # type: ignore
                proto_act_img_j,
                dsize=(original_img_size0, original_img_size1),
                interpolation=cv2.INTER_CUBIC, # type: ignore
            )
            proto_bound_j = find_high_activation_crop(
                upsampled_act_img_j, # type:ignore
                percentile=percentile,
            )

            # Save the prototype boundary (rectangular boundary of highly activated region)
            proto_bound_boxes[j, 0] = proto_rf_boxes[j, 0]
            proto_bound_boxes[j, 1] = proto_bound_j[0]
            proto_bound_boxes[j, 2] = proto_bound_j[1]
            proto_bound_boxes[j, 3] = proto_bound_j[2]
            proto_bound_boxes[j, 4] = proto_bound_j[3]
            if proto_bound_boxes.shape[1] == 6 and search_y is not None:
                proto_bound_boxes[j, 5] = search_y[rf_prototype_j[0]].item()

            if dir_for_saving_prototypes is not None:
                if prototype_img_filename_prefix is not None:
                    rescaled_act_img_j = upsampled_act_img_j - np.amin(
                        upsampled_act_img_j
                    )
                    rescaled_act_img_j = rescaled_act_img_j / np.amax(
                        rescaled_act_img_j
                    )
                    heatmap = cv2.applyColorMap( # type: ignore
                        src=np.uint8(255 * rescaled_act_img_j), # type: ignore
                        colormap=cv2.COLORMAP_JET, # type: ignore
                    ) # type: ignore
                    heatmap = np.float32(heatmap) / 255
                    heatmap = heatmap[..., ::-1] # type: ignore

                    np.save(
                        os.path.join(
                            dir_for_saving_prototypes,
                            "original_img_j" + str(j) + ".npy",
                        ),
                        original_img_j,
                    )
                    np.save(
                        os.path.join(
                            dir_for_saving_prototypes, "heatmap" + str(j) + ".npy"
                        ),
                        heatmap,
                    )

    if class_specific:
        del class_to_img_index_dict # type: ignore

def push_prototypes(
    variable_proto_config: "common_vp.ProtoTrainConfig",
    data: "common_ht.DatasetSamples", # Data must be unnormalized in [0,1]
    model_ddp: Union["DDP", "layerwise.LayerwiseModel"],
    class_specific: bool = True,
    preprocess_input_function=None, # normalize if needed
    prototype_layer_stride:int = 1,
    root_dir_for_saving_prototypes: Optional[str] = None, # if not None, prototypes will be saved here
    epoch_number: int = 0, # if not provided, prototypes saved previously will be overwritten
    prototype_img_filename_prefix: Optional[str] = None,
    prototype_self_act_filename_prefix: Optional[str] = None,
    proto_bound_boxes_filename_prefix: Optional[str] = None,
    save_prototype_class_identity: bool = True, # which class the prototype sample comes from
    prototype_activation_func_in_numpy=None,
    percentile: int = 95,
):
    """
    Push each prototype to the nearest patch in the training set
    """
    model_ddp.eval()
    logging.info(
        f"{util.get_ddp_msg()}Pushing..."
    )
    unwrapped_model = util.unwrap_model(model_ddp)
    unwrapped_model = cast("layerwise.LayerwiseModel", unwrapped_model)
    proto_layer = unwrapped_model.layers[-2]
    assert isinstance(proto_layer, layerwise.VariableProtoLayer)

    start = time.time()
    prototype_shape = proto_layer.prototype_shape
    n_prototypes = proto_layer.num_prototypes
    global_min_proto_dist = np.full(n_prototypes, np.inf)
    global_min_fmap_patches = np.zeros(
        [n_prototypes, prototype_shape[1], prototype_shape[2], prototype_shape[3]]
    )

    # Search batch size equals to global size in Query dataset
    search_batch_size = data.cnn_samples.shape[0]
    global_samples_dist = np.full((search_batch_size, n_prototypes), np.inf) # type: ignore

    """
    proto_rf_boxes and proto_bound_boxes column:
    0: index in the entire dataset
    1: height start index
    2: height end index
    3: width start index
    4: width end index
    5: (optional) class identity

    e.g.
    [
        [35, 10, 14,  8, 12, 2], # prototype 0 → From the 35th sample → matches (10:14, 8:12) → Category 2
        [89,  4,  8,  5,  9, 5], # prototype 1
        [12, 15, 19, 16, 20, 5], # prototype 2
        [ 7, 20, 24, 30, 34, 0], # prototype 3
    ]
    """
    if save_prototype_class_identity:
        proto_rf_boxes = np.full(shape=[n_prototypes, 6], fill_value=-1)
        proto_bound_boxes = np.full(shape=[n_prototypes, 6], fill_value=-1)
    else:
        proto_rf_boxes = np.full(shape=[n_prototypes, 5], fill_value=-1)
        proto_bound_boxes = np.full(shape=[n_prototypes, 5], fill_value=-1)

    if root_dir_for_saving_prototypes != None:
        if epoch_number >= 0:
            proto_epoch_dir = os.path.join(
                root_dir_for_saving_prototypes,
                f"step-{str(epoch_number)}",
            )
            os.makedirs(proto_epoch_dir, exist_ok=True)
        else:
            proto_epoch_dir = root_dir_for_saving_prototypes
    else:
        proto_epoch_dir = None

    num_classes = proto_layer.num_classes

    for push_iter in range(0, num_classes, search_batch_size):
        cnn_samples = data.cnn_samples[
            push_iter: push_iter+search_batch_size
        ]
        cnn_labels = data.cnn_labels[
            push_iter: push_iter+search_batch_size
        ]
        # `start_index_of_search_batch` keeps track of the index of the sample assigned to serve as prototype
        start_index_of_search_batch = push_iter * search_batch_size # type: ignore
        _update_prototypes_on_batch(
            variable_proto_config,
            cnn_samples,
            start_index_of_search_batch,
            model_ddp,
            epoch_number,
            global_min_proto_dist,
            global_min_fmap_patches,
            global_samples_dist,
            proto_rf_boxes,
            proto_bound_boxes,
            class_specific=class_specific,
            search_y=cnn_labels,
            num_classes=num_classes,
            preprocess_input_function=preprocess_input_function,
            prototype_layer_stride=prototype_layer_stride,
            dir_for_saving_prototypes=proto_epoch_dir,
            prototype_img_filename_prefix=prototype_img_filename_prefix,
            prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
            prototype_activation_func_in_numpy=prototype_activation_func_in_numpy,
            percentile=percentile,
        )

    if proto_epoch_dir != None and proto_bound_boxes_filename_prefix != None:
        np.save(
            os.path.join(
                proto_epoch_dir,
                proto_bound_boxes_filename_prefix
                + "-receptive_field"
                + str(epoch_number)
                + ".npy",
            ),
            proto_rf_boxes,
        )
        np.save(
            os.path.join(
                proto_epoch_dir,
                proto_bound_boxes_filename_prefix + str(epoch_number) + ".npy",
            ),
            proto_bound_boxes,
        )

    logging.info(
        f"{util.get_ddp_msg()}Executing push ..."
    )
    prototype_update = np.reshape(global_min_fmap_patches, tuple(prototype_shape))
    assert proto_layer.prototype_vectors is not None
    proto_layer.prototype_vectors.data.copy_(
        torch.tensor(prototype_update, dtype=torch.float32).cuda()
    )

    update_kurt = False
    update_prototype_shape = None
    update_prototype_class_identity = None
    update_num_prototypes_per_class = None

    # In the early stage, the model is still learning the prototypes.
    # When the epoch is >= 100 in the later stage, the prototypes will be fixed and no longer increase dynamically.
    if epoch_number < variable_proto_config.push_start+variable_proto_config.push_freq*50:
        logging.info(
            f"{util.get_ddp_msg()}Step Updated: {epoch_number}"
        )

        num_protos_class = proto_layer.prototype_ids_per_class
        output = np.full((global_samples_dist.shape[0], num_classes), np.inf)
        for i in num_protos_class:
            protos = num_protos_class[i]
            temp = global_samples_dist[:, protos]
            for j in range(0, global_samples_dist.shape[0]):
                val = np.amin(temp[j, :])
                output[j, i] = val

        # Avg across all samples [n_class]
        output_class = np.mean(output, axis=0)

        # Compute Kurtosis
        kurt = kurtosis(output_class)
        logging.info(
            f"{util.get_ddp_msg()}Kurtosis: {kurt}"
        )
        if kurt > 0:
            if epoch_number < variable_proto_config.push_start+variable_proto_config.push_freq*30:
                p_upper = np.percentile(output_class, 75)
            else:
                p_upper = np.percentile(output_class, 90)

            indices_classes = [
                ind for ind in range(len(output_class)) if output_class[ind] > p_upper
            ]
            logging.info(
                f"{util.get_ddp_msg()}Indices_classes: {indices_classes}"
            )
            for k in indices_classes:
                proto_layer.num_prototypes += 1

                prototype_ids_per_class = list(
                    proto_layer.prototype_ids_per_class[k]
                )
                prototype_ids_per_class.append(
                    int(proto_layer.num_prototypes)-1
                )
                proto_layer.prototype_ids_per_class[k] = prototype_ids_per_class

                # [num_prototypes, num_channels, prototype_size, prototype_size]
                p0, p1, p2, p3 = proto_layer.prototype_shape
                proto_layer.prototype_shape = (p0+1, p1, p2, p3)

                second_tensor = torch.zeros(1, num_classes)
                second_tensor[0, k] = 1
                proto_layer.prototype_class_identity = torch.cat(
                    (
                        proto_layer.prototype_class_identity,
                        second_tensor,
                    ),
                    dim=0,
                )

            update_kurt = True
            (
                update_prototype_shape,
                update_prototype_class_identity,
                update_num_prototypes_per_class,
            ) = (
                proto_layer.prototype_shape,
                proto_layer.prototype_class_identity,
                proto_layer.prototype_ids_per_class,
            )
            logging.info(
                f"{util.get_ddp_msg()}Number of prototypes: {proto_layer.num_prototypes}"
            )

            # Export
            np.save(
                os.path.join(proto_epoch_dir, "prototype_class_identity.npy"), # type: ignore
                update_prototype_class_identity,
            )

    end = time.time()
    logging.info(
        f"{util.get_ddp_msg()}Push Time: {end-start}, "
        f"update_kurt: {update_kurt}"
    )
    return (
        update_kurt,
        update_prototype_shape,
        update_prototype_class_identity,
        update_num_prototypes_per_class,
    )
