import math


def _compute_layer_rf_info(
    layer_filter_size,
    layer_stride,
    layer_padding,
    previous_layer_rf_info,
) -> list:
    n_in = previous_layer_rf_info[0]  # input size
    j_in = previous_layer_rf_info[1]  # receptive field jump of input layer
    r_in = previous_layer_rf_info[2]  # receptive field size of input layer
    start_in = previous_layer_rf_info[3]  # center of receptive field of input layer

    if layer_padding.lower() == "same":
        n_out = math.ceil(float(n_in) / float(layer_stride))
        if n_in % layer_stride == 0:
            pad = max(layer_filter_size - layer_stride, 0)
        else:
            pad = max(layer_filter_size - (n_in % layer_stride), 0)
        assert n_out == math.floor((n_in - layer_filter_size + pad) / layer_stride) + 1
        assert pad == (n_out - 1) * layer_stride - n_in + layer_filter_size
    elif layer_padding.lower() == "valid":
        n_out = math.ceil(float(n_in - layer_filter_size + 1) / float(layer_stride))
        pad = 0
        assert n_out == math.floor((n_in - layer_filter_size + pad) / layer_stride) + 1
        assert pad == (n_out - 1) * layer_stride - n_in + layer_filter_size
    else:
        # `layer_padding` is an int that is the amount of padding on one side
        pad = layer_padding * 2
        n_out = math.floor((n_in - layer_filter_size + pad) / layer_stride) + 1

    pL = math.floor(pad / 2)

    j_out = j_in * layer_stride
    r_out = r_in + (layer_filter_size - 1) * j_in
    start_out = start_in + ((layer_filter_size - 1) / 2 - pL) * j_in

    return [n_out, j_out, r_out, start_out]

def compute_proto_layer_rf_info(
    input_size, layer_filter_sizes, layer_strides, layer_paddings, prototype_kernel_size
) -> list:
    assert len(layer_filter_sizes) == len(layer_strides)
    assert len(layer_filter_sizes) == len(layer_paddings)

    rf_info = [input_size[0], 1, 1, 0.5]

    for i in range(len(layer_filter_sizes)):
        filter_size = layer_filter_sizes[i]
        stride_size = layer_strides[i]
        padding_size = layer_paddings[i]
        rf_info = _compute_layer_rf_info(
            layer_filter_size=filter_size,
            layer_stride=stride_size,
            layer_padding=padding_size,
            previous_layer_rf_info=rf_info,
        )

    proto_layer_rf_info = _compute_layer_rf_info(
        layer_filter_size=prototype_kernel_size,
        layer_stride=1,
        layer_padding="VALID",
        previous_layer_rf_info=rf_info,
    )
    return proto_layer_rf_info


def _compute_rf_protoL_at_spatial_location(
    size, height_index, width_index, protoL_rf_info
) -> list:
    """Compute receptive field indexes for the prototype.

    Parameters:
       size: Size of the prototype along the first dimension
       height_index: Height index of the closest patch
       width_index: Width index of the closest patch
       protoL_rf_info: Information about prototype receptive fields information

    Returns:
       receptive_field_indexes: Receptive field indexes for the prototype.
    """
    n = protoL_rf_info[0]
    j = protoL_rf_info[1]
    r = protoL_rf_info[2]
    start = protoL_rf_info[3]
    assert height_index < n
    assert width_index < n

    center_h = start + (height_index * j)
    center_w = start + (width_index * j)

    rf_start_height_index = max(int(center_h - (r / 2)), 0)
    rf_end_height_index = min(int(center_h + (r / 2)), size)

    rf_start_width_index = max(int(center_w - (r / 2)), 0)
    rf_end_width_index = min(int(center_w + (r / 2)), size)

    return [
        rf_start_height_index,
        rf_end_height_index,
        rf_start_width_index,
        rf_end_width_index,
    ]

def compute_rf_prototype(size, prototype_patch_index, protoL_rf_info) -> list:
    """Compute receptive field for the prototype.

    Parameters:
       size: Size of the prototype along the first dimension
       prototype_patch_index: Information about the closest patch
       protoL_rf_info: Information about prototype receptive fields information

    Returns:
       receptive_field: Receptive field for the prototype.
    """
    sample_index = prototype_patch_index[0]
    height_index = prototype_patch_index[1]
    width_index = prototype_patch_index[2]
    rf_indices = _compute_rf_protoL_at_spatial_location(
        size, height_index, width_index, protoL_rf_info
    )
    return [sample_index, rf_indices[0], rf_indices[1], rf_indices[2], rf_indices[3]]
