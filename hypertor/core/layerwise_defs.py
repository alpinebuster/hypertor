"""`Layerwise` convolutional neural network models."""

import functools

from hypertor.core import layerwise
from hypertor.core import common_ht

ConvLayer = layerwise.ConvLayer
LogitsLayer = layerwise.LogitsLayer
FlattenLogitsLayer = layerwise.FlattenLogitsLayer
ProtoLogitsLayer = layerwise.ProtoLogitsLayer
VariableProtoLayer = layerwise.VariableProtoLayer


def maxpool_model(
    model_config: common_ht.LayerwiseModelConfig,
    dataset: common_ht.DatasetSamples,
    num_layers=4,
    last_maxpool=2,
):
    """Creates a larger 4-layer model (similar to the one in MAML paper)."""
    conv_args = {
        "model_config": model_config,
        "maxpool_size": None if model_config.use_darkflow else 2,
        "padding": "same",
    }
    layers = [
        ConvLayer(name=f"layer_{i + 1}", head_builder=LogitsLayer, **conv_args)
        for i in range(num_layers - 1)
    ]
    # The last layer should NOT have a separate head via `head_builder`
    # We should have a real head
    conv_args["maxpool_size"] = last_maxpool
    if model_config.use_variable_proto:
        layers += [
            ConvLayer(
                name=f"layer_{num_layers}",
                head_builder=LogitsLayer,
                **conv_args,
            ),
            VariableProtoLayer(name=f"layer_{num_layers+1}", model_config=model_config),
            ProtoLogitsLayer(name=f"layer_{num_layers+2}", model_config=model_config),
        ]
    else:
        layers += [
            ConvLayer(name=f"layer_{num_layers}", **conv_args),
            FlattenLogitsLayer(name=f"layer_{num_layers+1}", model_config=model_config),
        ]
    return layerwise.LayerwiseModel(layers=layers, model_config=model_config, dataset=dataset)


def avgpool_model(
    model_config: common_ht.LayerwiseModelConfig,
    dataset: common_ht.DatasetSamples,
    num_layers=1,
):
    """Creates a basic 'layerwise' model."""
    layers = [
        ConvLayer(
            name=f"layer_{i + 1}",
            model_config=model_config,
            head_builder=LogitsLayer,
        )
        for i in range(num_layers - 1)
    ]
    # The last layer should NOT have a separate head via `head_builder`
    # We should have a real head
    if model_config.use_variable_proto:
        layers += [
            ConvLayer(
                name=f"layer_{num_layers}",
                head_builder=LogitsLayer,
                model_config=model_config,
            ),
            VariableProtoLayer(name=f"layer_{num_layers+1}", model_config=model_config),
            ProtoLogitsLayer(name=f"layer_{num_layers+2}", model_config=model_config),
        ]
    else:
        layers += [
            ConvLayer(name=f"layer_{num_layers}", model_config=model_config),
            LogitsLayer(name=f"layer_{num_layers+1}", model_config=model_config),
        ]
    return layerwise.LayerwiseModel(layers=layers, model_config=model_config, dataset=dataset)


def register():
    """Registers layerwise architectures."""
    layerwise.models["maxpool-4-layer"] = functools.partial(maxpool_model, num_layers=4)
    layerwise.models["3-layer"] = functools.partial(avgpool_model, num_layers=3)
    layerwise.models["4-layer"] = functools.partial(avgpool_model, num_layers=4)


register()
