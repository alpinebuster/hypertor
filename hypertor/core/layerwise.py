"""Core implementations of "layerwise" models."""

import dataclasses
import functools
import math
import re

from typing import cast, Callable, Optional, Union, Sequence, \
    Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hypertor.core import common_ht
from hypertor.variable_proto import common_vp
from hypertor.core.common_ht import LayerwiseModelConfig
from hypertor.core import feature_extractors
from hypertor.core import transformer
from hypertor.core import util

GAMMA_SCALE = 1.0
BETA_SCALE = 1.0
GAMMA_BIAS = 0.0
BETA_BIAS = 0.0

models: dict[str, Callable[..., "LayerwiseModel"]] = {}
HeadBuilder = Callable[..., "BaseCNNLayer"]


@dataclasses.dataclass
class GeneratedWeights:
    weight_blocks: list[list[torch.Tensor]]
    head_weight_blocks: dict[str, list[torch.Tensor]]
    shared_features: Optional[torch.Tensor] = None


def build_model(
    name: str,
    model_config: common_ht.LayerwiseModelConfig,
    dataset: common_ht.DatasetSamples,
) -> "LayerwiseModel":
    model_fn = models[name]
    return model_fn(model_config=model_config, dataset=dataset)


def get_remove_probability(max_probability: float) -> torch.Tensor:
    """Returns a random probability between 0 and `max_probability`.

    e.g.
       >>> torch.rand(size=()) 
           tensor(0.6942)
    """
    return torch.rand(size=()) * max_probability


def remove_some_samples(
    labels: torch.Tensor,
    model_config: "LayerwiseModelConfig",
    mask: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Returns a random label mask removing some labeled and unlabeled samples."""
    if (
        model_config.max_prob_remove_unlabeled <= 0.0
        and model_config.max_prob_remove_labeled <= 0.0
    ):
        return mask

    # Unlabeled samples
    if model_config.max_prob_remove_unlabeled > 0.0:
        # Dropping samples with a random probability between 0 and
        # `max_prob_remove_unlabeled`.
        prob = get_remove_probability(model_config.max_prob_remove_unlabeled)
        # Removing unlabeled samples with probability `prob`
        # A label with a value of `num_labels` indicates no label
        new_mask = (labels == model_config.num_labels).float()
        masked_uniform = new_mask * torch.rand_like(new_mask)
        mask_unlabeled = (masked_uniform > 1-prob).float()
    else:
        mask_unlabeled = torch.zeros_like(labels, dtype=torch.float32)

    # Labeled samples
    if model_config.max_prob_remove_labeled > 0.0:
        # Dropping samples with a random probability between 0 and
        # `max_prob_remove_labeled`.
        prob = get_remove_probability(model_config.max_prob_remove_labeled)
        # Removing labeled samples with probability `prob`
        new_mask =  (labels != model_config.num_labels).float()
        masked_uniform = new_mask * torch.rand_like(new_mask)
        mask_labeled = (masked_uniform > 1-prob).float()
    else:
        mask_labeled = torch.zeros_like(labels, dtype=torch.float32)

    # Combine masks
    # Boolean "or" equivalent for 3 masks (1 indicates a missing value).
    if mask is None:
        return torch.clamp(mask_labeled + mask_unlabeled, 0.0, 1.0)
    else:
        return torch.clamp(mask_labeled + mask_unlabeled + mask, 0.0, 1.0)


# ------------------------------------------------------------
#   Layer weight generators
# ------------------------------------------------------------


class _Generator(nn.Module):
    """Generic generator."""

    def __init__(self, name: str, model_config: "LayerwiseModelConfig"):
        super().__init__()

        self.name = name
        self.model_config = model_config
        self.num_weight_blocks: Optional[int] = None
        self.weight_block_size: Optional[int] = None
        self.feature_extractor: Optional[nn.Module] = None
        self.feature_extractor_class: Optional[Callable[..., nn.Module]] = None
        self.transformer_io: Optional[Union[
            util.JointTransformerIO,
            util.SeparateTransformerIO,
        ]] = None
        self.transformer: Optional[Union[
            transformer.EncoderDecoderModel,
            transformer.EncoderModel,
            transformer.SeparateEncoderDecoderModel,
        ]] = None

    def _setup(self):
        raise NotImplementedError

    def set_weight_params(self, num_weight_blocks: int, weight_block_size: int):
        self.num_weight_blocks = num_weight_blocks
        self.weight_block_size = weight_block_size

    def set_feature_extractor_class(
        self,
        feature_extractor_class: Callable[..., nn.Module],
    ) -> None:
        self.feature_extractor_class = feature_extractor_class

    def _make_feature_extractor(self):
        if self.feature_extractor_class is not None:
            self.feature_extractor = self.feature_extractor_class(
                name="feature_extractor"
            )

    def _features(
        self,
        input_tensor: torch.Tensor,
        shared_features: Optional[torch.Tensor] = None,
        enable_fe_dropout: bool = False
    ) -> torch.Tensor:
        """Returns full feature vector (per-layer and shared if specified)."""
        if self.feature_extractor is not None:
            # feature_extractor -> Activation Feature Extractor
            # shared_features   -> Image Feature Extractor
            features = self.feature_extractor(input_tensor)
            if enable_fe_dropout and self.model_config.fe_dropout > 0.0:
                # NOTE: always True
                features = F.dropout(
                    features,
                    p=self.model_config.fe_dropout,
                    training=True,
                )
        else:
            features = None

        if shared_features is not None:
            if features is not None:
                return torch.concat([features, shared_features], dim=-1)
            return shared_features

        if features is None:
            raise RuntimeError(
                "Layerwise model should have at least one of "
                "per-layer and shared feature extractors."
            )
        return features


class JointGenerator(_Generator):
    """Model that feeds the Encoder/Decoder concatenated samples and weights."""

    def _pad_features(self, features: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Pads features to fit the embedding size.

        [batch_size, feature_dim]
            ↓
        [batch_size, feature_dim + pad_size]
        """
        assert isinstance(self.transformer_io, util.JointTransformerIO)

        feature_size = int(features.shape[1])
        embedding_dim = self.transformer_io.embedding_dim
        input_embedding_size = feature_size + embedding_dim

        if self.weight_block_size is None:
            raise RuntimeError("weight_block_size must be set before calling _pad_features()")

        embedding_size = max(
            embedding_dim + self.weight_block_size,
            input_embedding_size,
        )
        if embedding_size == input_embedding_size:
            return features, embedding_size
        else:
            pad_size = embedding_size - input_embedding_size
            """
            Paddings specification for `tf.pad`:
               [
                   [0, 0],       # Dimension 0 (batch dimension):
                                 #   - pad_before = 0
                                 #   - pad_after  = 0
                                 #   → no padding is applied to the batch dimension

                   [0, pad_size] # Dimension 1 (feature dimension):
                                 #   - pad_before = 0
                                 #   - pad_after  = pad_size
                                 #   → append `pad_size` zeros to the right (end) of the feature vector
               ]

            Example:
               features = [[1, 2, 3],
                           [4, 5, 6]]
                                ↓
               paddings = [[1, 1],  # pad 1 sample before and after the batch
                           [0, 2]]  # pad 2 zeros at the end of the feature dimension
                                ↓
               tf.pad(features, paddings, mode="CONSTANT")
                                ↓
               [[0, 0, 0, 0, 0],   # padded batch (before)
                [1, 2, 3, 0, 0],   # original sample 0
                [4, 5, 6, 0, 0],   # original sample 1
                [0, 0, 0, 0, 0]]   # padded batch (after)
            """
            padded_features = F.pad(features, (0, pad_size), mode='constant', value=0)
            return padded_features, embedding_size

    def get_transformer_params(self, embedding_size: int):
        """Returns Transformer parameters."""
        num_heads = self.model_config.heads

        def get_size(frac):
            if frac <= 3.0:
                dim = int(embedding_size * frac)
                if dim % num_heads > 0:
                    # Making sure that the Transformer input dimension is divisible by
                    # the number of heads.
                    dim = math.ceil(float(dim) / num_heads) * num_heads
                return dim
            else:
                return int(frac)

        attn_act_fn = common_ht.get_transformer_activation(self.model_config)

        return transformer.TransformerParams(
            query_key_dim=get_size(self.model_config.query_key_dim_frac), # D_qk
            internal_dim=get_size(self.model_config.internal_dim_frac), # Used by `PWFeedForward`
            value_dim=get_size(self.model_config.value_dim_frac), # D_v
            num_layers=self.model_config.num_layers,
            mha_output_dim=embedding_size,
            heads=num_heads,
            dropout_rate=self.model_config.dropout_rate,
            attention_activation_fn=attn_act_fn,
            activation_fn=util.nonlinearity(self.model_config.transformer_nonlinearity),
        )

    def _setup(self):
        self._make_feature_extractor()

        assert self.num_weight_blocks
        assert self.weight_block_size
        self.transformer_io = util.JointTransformerIO(
            num_labels=self.model_config.num_labels,
            num_weights=self.num_weight_blocks,
            embedding_dim=self.model_config.embedding_dim,
            weight_block_size=self.weight_block_size,
        )

    def generate_weights(
        self,
        input_tensor: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        shared_features: Optional[torch.Tensor] = None,
        enable_fe_dropout: bool = False,
    ) -> list[torch.Tensor]:
        """Generates weights from the inputs."""
        if self.transformer_io is None:
            self._setup()

        assert isinstance(self.transformer_io, util.JointTransformerIO)

        # [B, C, H, W] -> [B, feature_layers*feature_dim]
        # [8, 16, 15, 15] -> [8, 64]
        features = self._features(
            input_tensor, shared_features, enable_fe_dropout=enable_fe_dropout
        )
        # [B, feature_layers*feature_dim] -> [batch_size, feature_size+weight_embedding_dim+pad_size]
        # [8, 64] -> [8, 256]
        features, transformer_embedding_size = self._pad_features(features)

        if self.transformer is None:
            if self.model_config.use_decoder:
                model_class = transformer.EncoderDecoderModel
            else:
                model_class = transformer.EncoderModel
            self.transformer = model_class(
                self.get_transformer_params(transformer_embedding_size),
                skip_last_nonlinearity=self.model_config.skip_last_nonlinearity,
                name="transformer",
            )
        if mask is not None:
            mask = self.transformer_io.extend_label_mask(mask)

        # [batch_size, feature_dim+pad_size] -> [num_weights + B, D_w + I_l]
        # [8, 256] -> [17, 264]
        transformer_input = self.transformer_io.encode_samples(features, labels)
        # Still [num_weights + B, D_w + I_l]
        transformer_output = self.transformer(transformer_input, mask=mask)

        return self.transformer_io.decode_weights(transformer_output)


class SeparateGenerator(_Generator):
    """Model that feeds samples to Encoder and weights to Decoder."""

    def get_encoder_params(self, embedding_size: int):
        """Returns Transformer parameters."""
        def get_size(frac):
            if frac <= 3.0:
                return int(embedding_size * frac)
            else:
                return int(frac)

        attn_act_fn = common_ht.get_transformer_activation(self.model_config)

        return transformer.TransformerParams(
            query_key_dim=get_size(self.model_config.query_key_dim_frac),
            internal_dim=get_size(self.model_config.internal_dim_frac),
            value_dim=get_size(self.model_config.value_dim_frac),
            num_layers=self.model_config.num_layers,
            mha_output_dim=embedding_size,
            heads=self.model_config.heads,
            dropout_rate=self.model_config.dropout_rate,
            attention_activation_fn=attn_act_fn,
            activation_fn=util.nonlinearity(self.model_config.transformer_nonlinearity),
        )

    def get_decoder_params(self, embedding_size: int):
        """Returns Transformer parameters."""
        def get_size(frac):
            if frac <= 3.0:
                return int(embedding_size * frac)
            else:
                return int(frac)

        attn_act_fn = common_ht.get_transformer_activation(self.model_config)

        return transformer.TransformerParams(
            query_key_dim=get_size(self.model_config.query_key_dim_frac),
            internal_dim=get_size(self.model_config.internal_dim_frac),
            value_dim=get_size(self.model_config.value_dim_frac),
            num_layers=self.model_config.num_layers,
            mha_output_dim=embedding_size,
            heads=self.model_config.heads,
            dropout_rate=self.model_config.dropout_rate,
            attention_activation_fn=attn_act_fn,
            activation_fn=util.nonlinearity(self.model_config.transformer_nonlinearity),
        )

    def _setup(self):
        self._make_feature_extractor()

        assert self.num_weight_blocks
        assert self.weight_block_size
        self.transformer_io = util.SeparateTransformerIO(
            num_labels=self.model_config.num_labels,
            num_weights=self.num_weight_blocks,
            embedding_dim=self.model_config.embedding_dim,
            weight_block_size=self.weight_block_size,
        )

    def generate_weights(
        self,
        input_tensor: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        shared_features: Optional[torch.Tensor] = None,
        enable_fe_dropout: bool = False,
    ) -> list[torch.Tensor]:
        """Generates weights from the inputs."""
        del mask
        if self.transformer_io is None:
            self._setup()
        if self.model_config.max_prob_remove_unlabeled > 0:
            raise ValueError(
                "Removing unlabeled samples is not currently supported "
                'in the "separate" weight generator.'
            )

        assert self.weight_block_size is not None, \
        "weight_block_size must be set before calling _pad_features()"
        assert isinstance(self.transformer_io, util.SeparateTransformerIO)

        features = self._features(input_tensor, shared_features, enable_fe_dropout)
        weight_dim = self.transformer_io.weight_embedding_dim + self.weight_block_size
        sample_dim = self.transformer_io.embedding_dim
        sample_dim += int(features.shape[1])

        if self.transformer is None:
            self.transformer = transformer.SeparateEncoderDecoderModel(
                encoder_params=self.get_encoder_params(sample_dim),
                decoder_params=self.get_decoder_params(weight_dim),
                skip_last_nonlinearity=self.model_config.skip_last_nonlinearity,
                name="transformer",
            )
        decoded = self.transformer(
            self.transformer_io.encode_samples(features, labels),
            self.transformer_io.encode_weights(),
        )

        return self.transformer_io.decode_weights(decoded)


# ------------------------------------------------------------
#   Layer implementations: convolutional and logits layers
# ------------------------------------------------------------


class BaseCNNLayer(nn.Module):
    """Base CNN layer used in our models."""

    def __init__(
        self,
        name: str,
        model_config: LayerwiseModelConfig,
        head_builder: Optional[HeadBuilder] = None,
        var_reg_weight: Optional[float] = None,
    ):
        super().__init__()

        self.name = name
        self.model_config = model_config
        self.num_labels = model_config.num_labels

        self.add_trainable_weights = model_config.add_trainable_weights
        self.l2_reg_weight = model_config.l2_reg_weight
        if var_reg_weight is None:
            var_reg_weight = model_config.var_reg_weight
        self.var_reg_weight = var_reg_weight

        self.feature_extractor = None
        self.head: Optional["BaseCNNLayer"] = None

        if head_builder is not None and self.model_config.train_heads:
            self.head = head_builder(
                name="head_" + self.name,
                model_config=self.model_config,
            )

        if model_config.generator == "joint":
            self.generator = JointGenerator(
                name=self.name + "_generator", model_config=self.model_config
            )
        elif model_config.generator == "separate":
            self.generator = SeparateGenerator(
                name=self.name + "_generator", model_config=self.model_config
            )

        # To save the additional regularization terms generated by `_select_weight` (to be collected in the training loop)
        self._extra_losses: list[torch.Tensor] = []
        # Support `add_trainable_weights`, and these parameters must be registered in `__init__` (required by DDP)
        # Here, `nn.ParameterDict` is used, with the keys being 'kernel', 'bias', 'gamma', and 'beta' (if present)
        # Note: We must know the shapes of these parameters and create them in `__init__`; Or provide an initialization method to be called when the model is created
        self.var_weights = nn.ParameterDict() if self.add_trainable_weights else {}
        self.initialized = False
        # Initialized by `_setup` func
        self.output_features: int = 0

    def _setup(self, inputs: torch.Tensor):
        """Input-dependent layer setup."""
        raise NotImplementedError

    def forward(
        self,
        inputs: torch.Tensor,
        selected_weights: dict[str, Optional[torch.Tensor]] = {},
        training: bool = True,
        evaluation: bool = False,
        separate_bn_variables: bool = False,
    ) -> torch.Tensor:
        raise NotImplementedError

    def _var_getter(self, weights: Optional[list[torch.Tensor]], name: str) -> Optional[torch.Tensor]:
        """Call it each time a variable is created.
        name = "conv/kernel"
                │
                ▼
        cnn_var_getter(name)
                │
                ├── Tensor ──┐
                │            ├─ (+ trainable residual)
                │            └─ (+ regularization)
                │
                └── None ───▶ fallback_fn(name)
                                │
                                └─ nn.Parameter
        """
        raise NotImplementedError

    def create(
        self,
        input_tensor: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        shared_features: Optional[torch.Tensor] = None,
        enable_fe_dropout: bool = False,
        enable_weight_generation: bool = False,
    ) -> Optional[list[torch.Tensor]]:
        """Creates a layer using a feature extractor and a Transformer."""
        if not self.initialized:
            self._setup(input_tensor)
            self.enable_weight_generation = enable_weight_generation
            self.initialized = True

        if not self.enable_weight_generation:
            return None

        # Generate CNN weights (not `nn.Parameter` parameters but `torch.Tensor` tensors)
        return self.generator.generate_weights(
            input_tensor=input_tensor,
            labels=labels,
            mask=mask,
            shared_features=shared_features,
            enable_fe_dropout=enable_fe_dropout,
        )

    def _select_weight(
        self,
        weight_blocks: Optional[list[torch.Tensor]],
        name: str,
        evaluation: bool = False,
        separate_bn_variables: bool = False,
    ):
        short_name = name.split("/")[-1] if "/" in name else name

        # 1) First, try to take it from the `weight_blocks` generated by Transformer
        built_weights = None
        if weight_blocks is not None:
            # `weight_blocks` -> `list[torch.Tensor]` (each tensor corresponds to a generated vector)
            try:
                built_weights = self._var_getter(weight_blocks, name)
            except NotImplementedError:
                built_weights = None

        # 2) If does not generate, registered in the back to the layer parameters (such as the self.weight/self.bias/self.bn)
        if built_weights is None:
            if hasattr(self, short_name):
                return getattr(self, short_name)
            return None

        var_name = short_name
        if evaluation and separate_bn_variables:
            var_name += "_eval"
        if self.add_trainable_weights:
            if var_name in self.var_weights:
                var_param = self.var_weights[var_name]  # nn.Parameter
                if self.var_reg_weight > 0.0:
                    built_weight_mag = built_weights.pow(2).sum()
                    var_weight_mag = var_param.pow(2).sum()
                    reg = self.var_reg_weight * built_weight_mag / (1e-8 + var_weight_mag)
                    self._extra_losses.append(reg)
                built_weights += var_param

        # 3) Kernel L2 reg
        if self.l2_reg_weight and self.l2_reg_weight > 0.0 and short_name.endswith("kernel"):
            self._extra_losses.append(self.l2_reg_weight * built_weights.pow(2).sum())

        return built_weights

    def apply(
        self,
        input_tensor: torch.Tensor,
        weight_blocks: Optional[list[torch.Tensor]],
        *args,
        evaluation: bool = False, # Evaluation of new task
        separate_bn_variables: bool = False,
        **kwargs,
    ):
        """Applies created layer to the input tensor.
        The weights of the convolutional/FC layer are not created directly.
        Instead, they are either generated from `weight_blocks` or real parameters are created,
        the generated weights are superimposed,
        and various regular expressions are added to these weights.

        Attributes:
           Position parameters → *args → Keyword-only parameters → **kwargs

        Child.apply(x)
        → self(x)
        → nn.Module.__call__(x)
        → Child.forward(x)
        """
        assert self.initialized, "Layer must be initialized with `self.create()` before apply"

        # Make sure it does not accumulate across forwards
        self._extra_losses.clear()
        # 'kernel', 'bias', 'gamma', 'beta'
        selected_weights: dict[str, Optional[torch.Tensor]] = {}
        for n in ("kernel", "bias", "gamma", "beta"):
            selected_weights[n] = self._select_weight(
                weight_blocks,
                name=n,
                evaluation=evaluation,
                separate_bn_variables=separate_bn_variables
            )

        # Equal to call `self.__call__(...) -> self.forward(...)`
        return self(
            input_tensor,
            selected_weights,
            evaluation=evaluation,
            separate_bn_variables=separate_bn_variables,
            *args,
            **kwargs,
        )


class ConvLayer(BaseCNNLayer):
    """Conv Layer of the CNN."""

    def __init__(
        self,
        name,
        model_config: "LayerwiseModelConfig",
        output_dim=None,
        kernel_size: Optional[int] = None,
        num_features=None,
        feature_layers=0,
        padding: str ="same",
        generate_bias: Optional[bool] = None,
        generate_bn: Optional[bool] = None,
        stride=None,
        act_fn=None,
        act_after_bn=False,
        maxpool_size=None,
        head_builder: Optional[HeadBuilder] = None,
    ):
        super().__init__(
            name=name,
            model_config=model_config,
            head_builder=head_builder,
        )

        if generate_bn is None:
            generate_bn = model_config.generate_bn
        if generate_bias is None:
            generate_bias = model_config.generate_bias
        if act_fn is None:
            act_fn = common_ht.get_cnn_activation(model_config)
        if kernel_size is None:
            kernel_size = model_config.kernel_size
        if num_features is None:
            num_features = model_config.num_features
        if output_dim is None:
            output_dim = model_config.default_num_channels
        if stride is None:
            stride = model_config.stride

        self.generate_bn = generate_bn
        self.generate_bias = generate_bias
        self.act_fn = act_fn

        assert kernel_size is not None
        self.kernel_size = kernel_size

        self.num_features = num_features

        self.feature_layers = feature_layers
        if self.feature_layers < 1:
            self.feature_layers = self.model_config.feature_layers

        # PyTorch 1.10+ supports "same" / "valid"
        assert padding in ("same", "valid")
        self.padding = padding
        self.kernel = None
        self.bias = None

        self.input_dim = 1
        self.output_dim = output_dim
        self.stride = stride
        self.maxpool_size = maxpool_size
        self.act_after_bn = act_after_bn

    def _compute_weight_sizes(self, weight_alloc: common_ht.WeightAllocation):
        """Computes the number of weight blocks, their size, axis to stack, etc."""
        assert self.input_dim > 0
        if weight_alloc == common_ht.WeightAllocation.SPATIAL:
            if self.generate_bn:
                raise ValueError(
                    "BN weight generation is not currently supported for "
                    "the spatial weight allocation."
                )
            if self.generate_bias:
                raise ValueError(
                    "Bias generation is not currently supported for "
                    "the spatial weight allocation."
                )

            self.num_blocks = int(self.kernel_size**2)
            self.block_size = self.input_dim * self.output_dim
            self.conv_weight_size = self.block_size

            self.stack_axis = 0 # BCHW
        elif weight_alloc == common_ht.WeightAllocation.OUTPUT_CHANNEL:
            self.num_blocks = self.output_dim
            self.block_size = self.input_dim * int(self.kernel_size**2)
            self.conv_weight_size = self.block_size

            if self.generate_bias:
                self.block_size += 1 # + 1 bias
            if self.generate_bn:
                self.block_size += 2 # + gamma + beta

            self.stack_axis = 1 # BCHW
        else:
            raise ValueError("Unknown WeightAllocation value.")

    def _setup(self, inputs: torch.Tensor) -> None:
        """Input-specific setup."""
        # inputs -> [BCHW]
        self.input_dim = int(inputs.shape[1])
        if self.num_features is None:
            self.num_features = max(self.output_dim, self.input_dim)
        self._compute_weight_sizes(self.model_config.weight_allocation)
        feature_extractor_class = functools.partial(
            feature_extractors.SimpleConvFeatureExtractor,
            in_channels=inputs.shape[1],
            input_size=int(inputs.shape[2]),
            feature_layers=self.feature_layers,
            feature_dim=self.num_features,
            kernel_size=self.kernel_size,
        )
        self.generator.set_weight_params(
            num_weight_blocks=self.num_blocks,
            weight_block_size=self.block_size,
        )
        self.generator.set_feature_extractor_class(feature_extractor_class)

        in_ch = inputs.shape[1]
        out_ch = self.output_dim
        self.default_kernel = nn.Parameter(torch.empty(out_ch, in_ch, self.kernel_size, self.kernel_size))
        # fan_in: Maintain the variance stability of the activation value in the forward propagation
        # fan_out: Maintain the variance stability of the gradient in the backpropagation
        nn.init.kaiming_normal_(self.default_kernel, mode="fan_in", nonlinearity="relu")
        self.default_bias = nn.Parameter(torch.zeros(out_ch))
        if self.add_trainable_weights:
            self.var_weights["kernel"] = nn.Parameter(torch.zeros_like(self.default_kernel))
            self.var_weights["bias"] = nn.Parameter(torch.zeros_like(self.default_bias))

        self.bn = nn.BatchNorm2d(out_ch)
        if self.model_config.separate_bn_vars:
            self.bn_eval = nn.BatchNorm2d(out_ch)

        if self.generate_bn:
            self.bn = nn.BatchNorm2d(out_ch, affine=False)
            if self.model_config.separate_bn_vars:
                self.bn_eval = nn.BatchNorm2d(out_ch, affine=False)

            if self.add_trainable_weights:
                self.var_weights["gamma"] = nn.Parameter(torch.ones(out_ch))  # The initial `gamma` of BN is usually 1
                self.var_weights["beta"]  = nn.Parameter(torch.zeros(out_ch)) # The initial `beta` of BN is usually 0

                if self.model_config.separate_bn_vars:
                    self.var_weights["gamma_eval"] = nn.Parameter(torch.ones(out_ch))
                    self.var_weights["beta_eval"]  = nn.Parameter(torch.zeros(out_ch))

        if self.maxpool_size is not None:
            self.maxpool = nn.MaxPool2d(
                kernel_size=self.maxpool_size,
                stride=self.maxpool_size,
                padding=0, # padding="valid"
            )
        else:
            self.maxpool = None

    def forward(
        self,
        inputs: torch.Tensor,
        selected_weights: dict[str, Optional[torch.Tensor]] = {},
        training: bool = True,
        evaluation: bool = False,
        separate_bn_variables: bool = False,
    ) -> torch.Tensor:
        # Layers should be created in `__call__` to properly build weights from
        # Transformer outputs for both training and evaluation.
        # Use the weights in `selected_weights` (priority: `selected_weights` -> `self.kernel`)
        kernel = None
        bias = None
        if selected_weights is not None:
            kernel = selected_weights.get("kernel", None)
            bias = selected_weights.get("bias", None)

        if kernel is None:
            if hasattr(self, "default_kernel"):
                kernel = self.default_kernel
            else:
                kernel = nn.Parameter(torch.empty(self.output_dim, inputs.shape[1], self.kernel_size, self.kernel_size))
                nn.init.kaiming_normal_(kernel, mode="fan_in", nonlinearity="relu")
        if bias is None:
            if hasattr(self, "default_bias"):
                bias = self.default_bias
            else:
                bias = nn.Parameter(torch.zeros(inputs.shape[1]))
        self.kernel = kernel
        self.bias = bias

        # conv
        if self.padding == "same":
            inputs, _ = util.same_pad_2d(
                inputs,
                kernel_size=self.kernel_size,
                stride=cast(tuple[int, int], self.stride),
            )
        x = F.conv2d(
            inputs,
            weight=kernel,
            bias=bias,
            stride=self.stride,
        )
        if not self.act_after_bn:
            x = self.act_fn(x)
        if self.maxpool_size is not None and self.maxpool is not None:
            x = self.maxpool(x)

        """
        # Batch normalization should always in the training mode.
        #
        # ------------------------------------------------------------------
        # In a standard CNN:
        #   - During training, training is True  -> use batch mean/variance
        #   - During testing,  training is False -> use global mean/variance (moving averages) from training
        #   Assumes train/test data come from the same distribution.
        #
        # In this transformer + few-shot setting:
        #   - Each forward pass is a new task (e.g., "cat vs dog", then "plane vs ship")
        #   - CNN weights are generated per task, not fixed
        #   - There is no stable global feature distribution
        #
        # Therefore, BatchNorm must always use batch statistics (training => True)
        # to normalize features based on the current task's data.
        """
        # This is aiming to make pytest happy, we already initialized it in `_setup` func
        # So this DDP safe
        gamma = None
        beta = None
        bn_layer = None
        if evaluation and separate_bn_variables:
            if hasattr(self, "bn_eval"):
                bn_layer = self.bn_eval
            else:
                bn_layer = nn.BatchNorm2d(self.output_dim, affine=not self.generate_bn)
            if self.add_trainable_weights:
                gamma = selected_weights.get("gamma_eval", None)
                beta = selected_weights.get("beta_eval", None)
        else:
            if hasattr(self, "bn"):
                bn_layer = self.bn
            else:
                bn_layer = nn.BatchNorm2d(self.output_dim, affine=not self.generate_bn)
            if self.add_trainable_weights:
                gamma = selected_weights.get("gamma", None)
                beta = selected_weights.get("beta", None)

        if not self.generate_bn:
            x = bn_layer(x)
        elif self.add_trainable_weights:
            # NOTE: always True
            # Episode-wise BatchNorm
            x = F.batch_norm(
                x,
                running_mean=None, # bn_layer.running_mean
                running_var=None, # bn_layer.running_var
                weight=gamma,
                bias=beta,
                training=True,
                eps=bn_layer.eps,
                # `nn.BatchNorm2d(... momentum=None)` allows `None` in the module,
                # indicating the use of default values, which is `0.1`
                momentum=bn_layer.momentum or 0.1,
            )

        if self.act_after_bn:
            x = self.act_fn(x)
        return x

    def _var_getter(self, weights: Optional[list[torch.Tensor]], name: str) -> Optional[torch.Tensor]:
        if weights is None:
            return None

        if name.endswith("kernel"):
            if self.enable_weight_generation:
                """Take the kernel part from each w
                w = [
                    conv_kernel_flat...,   # conv_weight_size
                    bias,                  # 1 bias
                    gamma,                 # 1 BN
                    beta                   # 1 BN
                ]
                    |
                    V
                ws = [
                    [k1_0, k1_1, ..., k1_N],
                    [k2_0, k2_1, ..., k2_N],
                    ...
                ]
                """
                ws = [w[:self.conv_weight_size] for w in weights]
                kernel = torch.stack(ws, dim=self.stack_axis)
                
                # PyTorch Conv2d expects: 
                #    conv.weight.shape == (out_channels, in_channels, kernel_size, kernel_size)
                # 
                # e.g.
                #    >>> conv = nn.Conv2d(3, 16, 5)
                #    >>> print(conv.weight.shape)
                #    torch.Size([16, 3, 5, 5])
                kernel = kernel.view(
                    self.output_dim,
                    self.input_dim,
                    self.kernel_size,
                    self.kernel_size,
                )
                return kernel

        if name.endswith("bias"):
            if self.enable_weight_generation and self.generate_bias:
                return torch.stack(
                    [w[self.conv_weight_size] for w in weights],
                    dim=0,
                )

        if self.generate_bn:
            offset = self.conv_weight_size + int(self.generate_bias)
            if name.endswith("gamma"):
                tensor = torch.stack([w[offset] for w in weights], dim=0)
                return GAMMA_BIAS + tensor * GAMMA_SCALE
            elif name.endswith("beta"):
                tensor = torch.stack([w[offset + 1] for w in weights], dim=0)
                return BETA_BIAS + tensor * BETA_SCALE

        return None


class VariableProtoLayer(BaseCNNLayer):
    """Variable prototype layer."""

    def __init__(
        self,
        name: str,
        model_config: LayerwiseModelConfig,
        num_features: Optional[int] = None,
        fe_kernel_size=3,
        head_builder: Optional[HeadBuilder] = None,
    ):
        super().__init__(
            name=name,
            model_config=model_config,
            head_builder=head_builder,
        )

        if num_features is None:
            num_features = model_config.num_features
        self.num_features = num_features

        self.fe_kernel_size = fe_kernel_size
        self.input_dim = 1

        # For variable prototype
        self.input_size = model_config.flows_shape
        # [num_prototypes, num_channels, prototype_size, prototype_size]
        self.prototype_shape = model_config.prototype_shape
        self.num_prototypes = model_config.prototype_shape[0]
        self.num_classes = model_config.num_labels
        self.epsilon = 1e-4 # Used when `self.prototype_activation_func == "log"`
        self.prototype_activation_func = model_config.prototype_activation_func
        self.prototype_class_identity = model_config.prototype_class_identity
        self.prototype_ids_per_class = model_config.prototype_ids_per_class
        self.proto_layer_rf_info = model_config.proto_layer_rf_info

        # 1) Training semantics (without `detach`)
        self.cached_prototype_vectors: Optional[torch.Tensor] = None
        # 2) For push and visualization semantics (`detach`+`clone`)
        self.used_prototype_vectors: Optional[torch.Tensor] = None

        # `prototype_vectors` -> `prototype_shape`=[out_channels, in_channels, kernel_h, kernel_w]
        self.prototype_vectors = nn.Parameter(
            torch.rand(self.prototype_shape),
            requires_grad=True,
        )
        # nn.init.normal_(self.prototype_vectors, mean=0.0, std=0.1) # std=0.01
        # nn.init.kaiming_normal_(self.prototype_vectors, mode="fan_in", nonlinearity="relu")

        if self.add_trainable_weights:
            self.var_weights["kernel"] = nn.Parameter(
                torch.zeros_like(self.prototype_vectors)
            )
        self.ones = nn.Parameter(
            torch.ones(self.prototype_shape),
            requires_grad=False,
        )

        self.kernel_size = self.prototype_vectors.shape[-1]
        self.output_dim = self.prototype_vectors.shape[0]
        # Cache `conv_features` for pushing prototype
        self.conv_features: Optional[torch.Tensor] = None
        self.min_distances: Optional[torch.Tensor] = None

    def _compute_weight_sizes(self):
        """Computes the number of weight blocks, their size, axis to stack, etc."""
        assert self.input_dim > 0
        self.num_blocks = self.output_dim
        self.block_size = self.input_dim * int(self.kernel_size**2)
        self.conv_weight_size = self.block_size
        self.stack_axis = 1 # BCHW

    def _setup(self, inputs: torch.Tensor) -> None:
        """Input-specific setup."""
        # inputs -> [BCHW]
        self.input_dim = int(inputs.shape[1])
        if self.num_features is None:
            self.num_features = self.input_dim
        self._compute_weight_sizes()
        # `num_blocks` -> `num_prototypes`
        self.min_distances = torch.rand((inputs.shape[0], self.num_blocks))

        feature_extractor_class = functools.partial(
            feature_extractors.SimpleConvFeatureExtractor,
            in_channels=inputs.shape[1],
            feature_layers=1,
            input_size=int(inputs.shape[2]),
            feature_dim=self.num_features,
            nonlinear_feature=self.model_config.nonlinear_feature,
            kernel_size=self.fe_kernel_size,
        )

        self.generator.set_weight_params(
            num_weight_blocks=self.num_blocks, # Equals to `num_prototypes`
            weight_block_size=self.block_size,
        )
        self.generator.set_feature_extractor_class(feature_extractor_class)

    def _l2_convolution(
        self,
        x: torch.Tensor,
        p_vectors: torch.Tensor,
    ) -> torch.Tensor:
        # p_vectors -> [num_prototypes, num_channels, prototype_size, prototype_size]=[out_channels, in_channels, kernel_h, kernel_w]
        xp = F.conv2d(input=x, weight=p_vectors)
        x = F.conv2d(input=x**2, weight=self.ones)
        p = torch.sum(p_vectors**2, dim=(1, 2, 3))
        p = p.view(-1, 1, 1)
        return F.relu(x - 2*xp + p)

    def _prototype_distances(
        self,
        conv_features: torch.Tensor,
        p_vectors: torch.Tensor,
    ) -> torch.Tensor:
        return self._l2_convolution(conv_features, p_vectors)

    def _distance_2_similarity(self, distances: torch.Tensor) -> torch.Tensor:
        if self.prototype_activation_func == "log":
            return torch.log((distances+1) / (distances+self.epsilon))
        elif self.prototype_activation_func == "linear":
            return -distances
        else:
            assert callable(self.prototype_activation_func)
            return self.prototype_activation_func(distances)

    def forward(
        self,
        inputs: torch.Tensor,
        selected_weights: dict[str, Optional[torch.Tensor]] = {},
        training: bool = True,
        evaluation: bool = False,
        separate_bn_variables: bool = False,
    ) -> torch.Tensor:
        """
        x: (batch_size, channel_size, Hf, Wf)
            ↓ 
            ↓ L2 conv (`_prototype_distances()`, prototype_shape: num_prototypes x channel_size x Hp x Wp)
            ↓ L2 conv will sum all channels
            ↓ 
        d: (batch_size, num_prototypes, Hf - Hp + 1, Wf - Wp + 1)
            ↓ 
            ↓ global min pooling
            ↓ 
        m: (batch_size, num_prototypes, 1, 1)
            ↓ 
            ↓ reshape
            ↓ 
        distances:  (batch_size, num_prototypes)
            ↓ 
            ↓ linear
            ↓ 
        logits: (batch_size, num_classes)
        """
        if selected_weights["kernel"] is not None:
            p_vectors = selected_weights["kernel"]
        else:
            p_vectors = self.prototype_vectors
        self.cached_prototype_vectors = p_vectors.detach().clone()
        self.used_prototype_vectors = p_vectors

        # (batch_size, channel_size, Hf, Wf)
        #                  ↓
        # (batch_size, num_prototypes, Hf-Hp+1, Wf-Wp+1)
        self.conv_features = inputs
        distances = self._prototype_distances(inputs, p_vectors)
        # Each proto only retains the minimum distance (best match)
        # ... -> (batch_size, num_prototypes, 1, 1)
        min_distances = -F.max_pool2d(
            -distances,
            kernel_size=(distances.size()[2], distances.size()[3]),
        )
        # (batch_size, num_prototypes, 1, 1) -> (batch_size, num_prototypes)
        min_distances = min_distances.view(-1, self.num_prototypes)
        prototype_activations = self._distance_2_similarity(min_distances)
        self.min_distances = min_distances
        return prototype_activations

    @torch.no_grad()
    def push_forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch_size, channel_size, Hf, Wf)
        assert self.conv_features is not None
        conv_output = self.conv_features
        # distances -> (batch_size, num_prototypes, Hf-Hp+1, Wf-Wp+1)
        assert self.cached_prototype_vectors is not None
        distances = self._l2_convolution(conv_output, self.cached_prototype_vectors)
        return conv_output, distances

    def _var_getter(self, weights: Optional[list[torch.Tensor]], name: str) -> Optional[torch.Tensor]:
        if weights is None:
            return None

        assert self.generator.weight_block_size is not None, \
        "`weight_block_size` must be set before calling `_pad_features`"

        # kernel: [num_prototypes, num_channels, prototype_size, prototype_size]
        if name.endswith("kernel"):
            if self.enable_weight_generation:
                """
                ws = [
                    [k1_0, k1_1, ..., k1_N],
                    [k2_0, k2_1, ..., k2_N],
                    ...
                ]
                """
                ws = [w[:self.conv_weight_size] for w in weights]
                kernel = torch.stack(ws, dim=self.stack_axis)
                kernel = kernel.view(
                    self.output_dim,
                    self.input_dim,
                    self.kernel_size,
                    self.kernel_size,
                )
                return kernel

        return None


class LogitsLayer(BaseCNNLayer):
    """Logits layer of the CNN.

       [batch, C, H, W]
            ↓
       [batch, C]
            ↓
       [batch, num_labels]
    """

    def __init__(
        self,
        name: str,
        model_config: LayerwiseModelConfig,
        num_features: Optional[int] = None,
        generate_bias: Optional[bool] = True,
        fe_kernel_size=3,
        head_builder: Optional[HeadBuilder] = None,
    ):
        super().__init__(
            name=name,
            model_config=model_config,
            head_builder=head_builder,
            # We generally do not want to regularize the last logits layer.
            var_reg_weight=0.0,
        )

        if num_features is None:
            num_features = model_config.num_features
        self.num_features = num_features
        """
        The core purpose of setting `bias=False` is to ensure that
        the classification results are entirely determined by the similarity of the prototype,
        avoid "unfounded predictions", and enhance the interpretability of the model.
        """
        self.generate_bias = generate_bias
        if self.model_config.use_variable_proto:
            self.generate_bias = False

        self.fe_kernel_size = fe_kernel_size
        self.input_dim = 1

    def _setup(self, inputs: torch.Tensor) -> None:
        """Input-specific setup."""
        # inputs -> [BCHW]
        self.input_dim = int(inputs.shape[1])

        if self.num_features is None:
            self.num_features = self.input_dim
        feature_extractor_class = functools.partial(
            feature_extractors.SimpleConvFeatureExtractor,
            in_channels=inputs.shape[1],
            feature_layers=1,
            input_size=int(inputs.shape[2]),
            feature_dim=self.num_features,
            nonlinear_feature=self.model_config.nonlinear_feature,
            kernel_size=self.fe_kernel_size,
        )

        self.linear_weight_size = self.input_dim
        self.block_size = self.linear_weight_size
        if self.generate_bias:
            self.block_size = self.linear_weight_size + 1 # + 1 bias

        self.default_fc = nn.Linear(
            self.linear_weight_size,
            self.model_config.num_labels,
        )
        self.generator.set_weight_params(
            num_weight_blocks=self.model_config.num_labels,
            weight_block_size=self.block_size,
        )
        self.generator.set_feature_extractor_class(feature_extractor_class)

        self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(
        self,
        inputs: torch.Tensor,
        selected_weights: dict[str, Optional[torch.Tensor]] = {},
        training: bool = True,
        evaluation: bool = False,
        separate_bn_variables: bool = False,
    ) -> torch.Tensor:
        # Global Average Pooling
        # [batch, channels, height, width] → [batch, channels, 1, 1]
        inputs = self.gap(inputs)
        # [batch, channels, 1, 1] → [batch, channels]
        inputs = inputs.flatten(start_dim=1)
        dropout_tensor = F.dropout(
            inputs,
            p=self.model_config.cnn_dropout_rate,
            training=training,
        )

        # [batch, channels] → [batch, num_labels]
        W = selected_weights["kernel"]
        B = selected_weights["bias"]

        if W is not None and B is not None:
            # input:  [B, in_features]
            # weight: [out_features, in_features]
            return F.linear(dropout_tensor, weight=W.t(), bias=B)
        return self.default_fc(dropout_tensor)

    def _var_getter(self, weights: Optional[list[torch.Tensor]], name: str) -> Optional[torch.Tensor]:
        if weights is None:
            return None

        assert self.generator.weight_block_size is not None, \
        "weight_block_size must be set before calling _pad_features"

        # kernel: [num_labels, in_features]
        if name.endswith("kernel"):
            ws = [w[:self.linear_weight_size] for w in weights]
            return torch.stack(ws, dim=-1)
        
        # bias: [num_labels]
        if name.endswith("bias") and self.generate_bias:
            ws = [w[self.linear_weight_size] for w in weights]
            return torch.stack(ws, dim=-1)

        return None


class FlattenLogitsLayer(LogitsLayer):
    """Logits layer of the CNN that flattens its input (instead of averaging).

       [batch, C, H, W]
            ↓
       [batch, C*H*W]
            ↓
       [batch, num_labels]
    """

    def _setup(self, inputs: torch.Tensor) -> None:
        """Input-specific setup."""
        self.input_dim = int(inputs.shape[1])
        height, width = int(inputs.shape[-2]), int(inputs.shape[-1])

        if self.num_features is None:
            self.num_features = self.input_dim
        if self.model_config.logits_feature_extractor in ["", "default", "mix"]:
            feature_extractor_class = functools.partial(
                feature_extractors.SimpleConvFeatureExtractor,
                in_channels=inputs.shape[1],
                feature_layers=1,
                feature_dim=self.num_features,
                input_size=int(inputs.shape[2]),
                nonlinear_feature=self.model_config.nonlinear_feature,
                kernel_size=self.fe_kernel_size,
            )
        elif self.model_config.logits_feature_extractor == "passthrough":
            feature_extractor_class = feature_extractors.PassthroughFeatureExtractor
        else:
            raise AssertionError("Unexpected `logits_feature_extractor` value.")
        if self.model_config.logits_feature_extractor == "mix":
            feature_extractor_class = functools.partial(
                feature_extractors.PassthroughFeatureExtractor,
                name="feature_extractor",
                wrap_class=feature_extractor_class,
            )

        self.linear_weight_size = self.input_dim*width*height
        self.block_size = self.linear_weight_size
        if self.generate_bias:
            self.block_size = self.linear_weight_size + 1 # + 1 bias

        self.default_fc = nn.Linear(
            self.linear_weight_size,
            self.model_config.num_labels,
        )
        self.generator.set_weight_params(
            num_weight_blocks=self.model_config.num_labels,
            weight_block_size=self.block_size,
        )
        self.generator.set_feature_extractor_class(feature_extractor_class)

    def forward(
        self,
        inputs: torch.Tensor,
        selected_weights: dict[str, Optional[torch.Tensor]] = {},
        training: bool = True,
        evaluation: bool = False,
        separate_bn_variables: bool = False,
    ) -> torch.Tensor:
        # [batch, C, H, W] → [batch, C*H*W]
        inputs = inputs.flatten(start_dim=1)
        dropout_tensor = F.dropout(
            inputs,
            p=self.model_config.cnn_dropout_rate,
            training=training,
        )

        # [batch, C*H*W] → [batch, num_labels]
        W = selected_weights["kernel"]
        B = selected_weights["bias"]

        if W is not None and B is not None:
            # input:  [B, in_features]
            # weight: [out_features, in_features]
            return F.linear(dropout_tensor, weight=W.t(), bias=B)
        return self.default_fc(dropout_tensor)


class ProtoLogitsLayer(LogitsLayer):
    """Logits layer of the variable prototype.

       [batch, num_prototypes]
            ↓
       [batch, num_labels]
    """

    def _setup(self, inputs: torch.Tensor) -> None:
        # inputs -> (batch_size, num_prototypes)
        self.input_dim = int(inputs.shape[-1])
        if self.num_features is None:
            self.num_features = self.input_dim

        feature_extractor_class = functools.partial(
            feature_extractors.PassthroughFeatureExtractor,
            name="passthrough_fe",
        )

        self.linear_weight_size = self.input_dim
        self.block_size = self.linear_weight_size

        self.generator.set_weight_params(
            num_weight_blocks=self.model_config.num_labels,
            weight_block_size=self.block_size,
        )
        self.generator.set_feature_extractor_class(feature_extractor_class)

        self.last_linear_weight: Optional[torch.Tensor] = None

    def forward(
        self,
        inputs: torch.Tensor,
        selected_weights: dict[str, Optional[torch.Tensor]] = {},
        training: bool = True,
        evaluation: bool = False,
        separate_bn_variables: bool = False,
    ) -> torch.Tensor:
        # inputs -> [batch_size, num_prototypes]
        # W -> [num_prototypes, num_classes]
        W = selected_weights["kernel"]
        assert W is not None

        if self.model_config.use_prior_proto:
            proto_prior = self._set_incorrect_connection()
            W = W + proto_prior.t().to(W.device)

        self.last_linear_weight = W
        # weight -> [num_classes, num_prototypes]
        return F.linear(inputs, weight=W.t(), bias=None)

    def _set_incorrect_connection(self, incorrect_strength: float = -0.5) -> torch.Tensor:
        """Construct prototype–class connection prior.

           1) Prototype makes a positive contribution (1.) to its own category
           2) A negative contribution (incorrect_strength) to other categories

        Returns:
           [num_classes, num_prototypes]
        """
        # [num_prototypes, num_classes] -> [num_classes, num_prototypes]
        positive_one_weights_locations = torch.t(
            self.model_config.prototype_class_identity
        ).float()
        negative_one_weights_locations = 1. - positive_one_weights_locations
        correct_class_connection = 1.
        incorrect_class_connection = incorrect_strength

        return (
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations
        )

    def _var_getter(self, weights: Optional[list[torch.Tensor]], name: str) -> Optional[torch.Tensor]:
        if weights is None:
            return None

        assert self.generator.weight_block_size is not None, \
        "`weight_block_size` must be set before calling _pad_features"

        # kernel: [num_classes, num_prototypes]
        if name.endswith("kernel"):
            ws = [w[:self.linear_weight_size] for w in weights]
            return torch.stack(ws, dim=-1)

        return None


# ------------------------------------------------------------
#   Layerwise model
# ------------------------------------------------------------


class LayerwiseModel(nn.Module):
    """Model specification including layer builders.

    Usage:
       1) `list(model.layers[i].parameters())`
       2) Stochastic structure:

          Input tensor
                ↓
          Activation & Shared Image FE `dropout` (always on)
                ↓
          Transformers
                ↓
          CNN conv + `BN` (always on -> per task batch statistics, NO stable global feature distribution)
                ↓
          Logits `dropout` (meta-train: off, evaluate new task: on)
    """

    def __init__(
        self,
        layers: "Sequence[Union[ConvLayer, BaseCNNLayer]]",
        model_config: LayerwiseModelConfig,
        dataset: common_ht.DatasetSamples,
    ):
        super().__init__()

        self.layers = nn.ModuleList(layers)
        self.shared_feature_extractor = feature_extractors.get_shared_feature_extractor(
            model_config
        )
        self.separate_bn_variables = model_config.separate_bn_vars

        #   = 0 : freeze all CNN layers (no backpropagation through CNN)
        #   > 0 : train the first N CNN layers
        #   < 0 : train the last |N| CNN layers
        self.number_of_trained_cnn_layers = model_config.number_of_trained_cnn_layers
        self.shared_fe_dropout = model_config.shared_fe_dropout
        self.fe_dropout = model_config.fe_dropout
        self.model_config = model_config
        self.dataset = dataset

        if (
            model_config.shared_features_dim is not None and
            dataset.real_class_min is not None and
            dataset.real_class_max is not None
        ):
            match = re.match(r'(\d+)', model_config.shared_feature_extractor)
            if match:
                num_feature_layers = int(match.group(1))
            else:
                raise ValueError(
                    f"No leading number found in shared_feature_extractor: "
                    f"'{model_config.shared_feature_extractor}'"
                )
    
            self.shared_head = feature_extractors.SharedHead(
                shared_features_dim=model_config.shared_features_dim*num_feature_layers
                    if model_config.use_darkflow else model_config.shared_features_dim,
                real_class_min=dataset.real_class_min,
                real_class_max=dataset.real_class_max,
                label_smoothing=model_config.label_smoothing,
            )
        else:
            self.shared_head = None

        self.shared_head_outputs: dict[str, Optional[torch.Tensor]] = {}
        self.layer_outputs: dict[
            str,
            tuple[Optional[torch.Tensor], Optional[torch.Tensor]],
        ] = {}

        # For variable prototype
        self.layer_kernel_sizes: list[tuple[int, int]] = []
        self.layer_strides: list[tuple[int, int]] = []
        self.layer_paddings: list[tuple[str, str]] = []

        self.training_stage: common_vp.ProtoTrainingStage = common_vp.ProtoTrainingStage.JOINT
        self.generate_weights_per_layers: list[bool] = []

    def setup(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mask_random_samples: bool = False,
        enable_fe_dropout: bool = False,
        only_shared_feature: bool = False,
    ) -> None:
        self.joint()
        self._train(
            inputs,
            labels,
            mask,
            mask_random_samples,
            enable_fe_dropout,
            only_shared_feature,
        )

        # For variable prototype
        for layer in self.layers:
            if not isinstance(layer, ConvLayer):
                continue
            self.layer_kernel_sizes.append((layer.kernel_size, layer.kernel_size))
            self.layer_strides.append((layer.stride, layer.stride))
            self.layer_paddings.append((layer.padding, layer.padding))

    def _train(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mask_random_samples: bool = False,
        enable_fe_dropout: bool = False,
        only_shared_feature: bool = False,
    ) -> GeneratedWeights:
        """Builds an entire CNN model using train inputs (Support Set).

           Loss → CNN logits → W_cnn → Transformer(θ_HT)

        1) The CNN does not save the gradient of `W_cnn` (it is not a Parameter), but PyTorch will compute gradients for it as a Tensor.
        2) The gradient of `W_cnn` will be backpropagated to the parameters `θ_HT` of the Transformer.
        3) The parameters `θ_HT` of the Transformer are the actual Parameters being optimized.

        The CNN itself does not update parameters (since its weights are generated by the Transformer).

           ```python
           # Assume `W_cnn` is the output of the Transformer
           W_cnn = transformer(support_set)  # Tensor, `requires_grad=True`

           logits = cnn(query_set, weight=W_cnn)
           loss = criterion(logits, labels)

           loss.backward()  # Automatically propagates the gradients back to `W_cnn`, then to Transformer parameters
           optimizer.step()  # Updates the Transformer
        ```

        W_cnn: is not an nn.Parameter, but a `Tensor`.
        Transformer: contains all the actual `nn.Parameters`.
        """
        all_weight_blocks = []
        all_head_blocks = {}

        # inputs -> [BCHW]
        # shared_features -> [B, feature_layers*feature_dim]
        shared_features = None
        if self.shared_feature_extractor is not None:
            shared_features = self.shared_feature_extractor(inputs)
            if enable_fe_dropout and self.shared_fe_dropout > 0.0:
                # NOTE: always True
                shared_features = F.dropout(
                    shared_features,
                    p=self.shared_fe_dropout,
                    training=True,
                )

        if only_shared_feature:
            return GeneratedWeights(
                weight_blocks=[],
                head_weight_blocks={},
                shared_features=shared_features,
            )

        if mask_random_samples:
            mask = remove_some_samples(labels, self.model_config, mask)

        assert len(self.generate_weights_per_layers) > 0
        for layer, do_gen_weights in zip(self.layers, self.generate_weights_per_layers):
            # Just to make Pylance 100% happy!!!
            layer = cast(BaseCNNLayer, layer)

            # cnn_builder
            weight_blocks = layer.create(
                input_tensor=inputs,
                labels=labels,
                mask=mask,
                shared_features=shared_features,
                enable_fe_dropout=enable_fe_dropout,
                enable_weight_generation=do_gen_weights,
            )
            all_weight_blocks.append(weight_blocks)

            if self.model_config.use_variable_proto and hasattr(layer, "generator"):
                for _, p in layer.generator.named_parameters():
                    p.requires_grad = do_gen_weights

            # cnn
            # [BCHW] -> [Bchw]
            inputs = layer.apply(
                inputs,
                weight_blocks=weight_blocks,
                separate_bn_variables=self.separate_bn_variables,
            )

            # cnn_builder_heads
            if layer.head is not None:
                head_blocks = layer.head.create(
                    input_tensor=inputs,
                    labels=labels,
                    mask=mask,
                    shared_features=shared_features,
                    enable_fe_dropout=enable_fe_dropout,
                )
                all_head_blocks[layer.name] = head_blocks
            
            layer.output_features = inputs.shape[1]

        self.shared_head_outputs = {}
        if (
            self.shared_head is not None and
            self.dataset.transformer_real_classes is not None and
            shared_features is not None
        ):
            shared_head_loss, shared_head_acc = self.shared_head(
                shared_features,
                self.dataset.transformer_real_classes,
            )
            self.shared_head_outputs["loss"] = shared_head_loss
            self.shared_head_outputs["accuracy"] = shared_head_acc
        else:
            self.shared_head_outputs["loss"] = None
            self.shared_head_outputs["accuracy"] = None

        return GeneratedWeights(
            weight_blocks=all_weight_blocks,
            head_weight_blocks=all_head_blocks,
            shared_features=shared_features,
        )

    def _evaluate(
        self,
        inputs: torch.Tensor,
        weight_blocks: Optional["GeneratedWeights"] = None,
        training: bool = True,
    ) -> torch.Tensor:
        """Passes input tensors (Query Set) through a built CNN model."""
        if weight_blocks is None:
            raise ValueError("weight_blocks must be provided")

        self.layer_outputs = {}
        for layer, layer_blocks in zip(self.layers, weight_blocks.weight_blocks):
            # Just to make Pylance 100% happy!!!
            layer = cast(BaseCNNLayer, layer)

            # cnn
            inputs = layer.apply(
                inputs,
                weight_blocks=layer_blocks,
                training=training,
                evaluation=True,
                separate_bn_variables=self.separate_bn_variables,
            )

            # cnn_builder_heads
            head = None
            if layer.head is not None:
                head = layer.head.apply(
                    inputs,
                    weight_blocks=weight_blocks.head_weight_blocks[layer.name],
                    training=training,
                    evaluation=True,
                    separate_bn_variables=self.separate_bn_variables,
                )
            self.layer_outputs[layer.name] = (inputs, head)

        return inputs

    def forward(
        self,
        transformer_samples: torch.Tensor,
        transformer_labels: torch.Tensor,
        cnn_samples: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mask_random_samples: bool = False,
        enable_fe_dropout: bool = False,
        only_shared_feature: bool = False,
        deterministic_inference: bool = True, # How to evaluate generated CNN
    ) -> Union[Optional[torch.Tensor], Optional["GeneratedWeights"]]:
        """
        deterministic_inference -> Whether to run CNN in deterministic reasoning mode (with randomness such as dropout, bn turned off)
           1) True  -> Observe the optimization process consistent with the training behavior,
              making sure that the forward calculation method used when calculating accuracy/loss
              should be the same as the one you will actually use to calculate gradients and
              update parameters later.
           2) False -> Given the generated weights of the CNN, how is the reasoning ability
              of this CNN?
        """
        weight_blocks: Optional["GeneratedWeights"] = self._train(
            transformer_samples,
            transformer_labels,
            mask,
            mask_random_samples=mask_random_samples
                if mask_random_samples
                else only_shared_feature,
            enable_fe_dropout=enable_fe_dropout
                if enable_fe_dropout
                else only_shared_feature,
            only_shared_feature=only_shared_feature,
        )
        if not only_shared_feature:
            return self._evaluate(cnn_samples, weight_blocks, training=deterministic_inference)
        else:
            return weight_blocks

    def regularization_loss(self) -> torch.Tensor:
        reg_losses = []

        for layer in self.layers:
            # Just to make Pylance 100% happy!!!
            layer = cast(BaseCNNLayer, layer)

            if hasattr(layer, "_extra_losses") and layer._extra_losses:
                reg_losses.extend(layer._extra_losses)
            if hasattr(layer, "head") and layer.head is not None:
                if layer.head._extra_losses:
                    reg_losses.extend(layer.head._extra_losses)

        if not reg_losses:
            return torch.zeros((), device=next(self.parameters()).device)

        return torch.stack(reg_losses).sum()

    def conv_info(self) -> tuple[
        list[tuple[int, int]],
        list[tuple[int, int]],
        list[tuple[str, str]]
    ]:
        return self.layer_kernel_sizes, self.layer_strides, self.layer_paddings

    def _set_generate_weights_per_layers(self):
        if self.model_config.use_variable_proto:
            if self.training_stage == common_vp.ProtoTrainingStage.WARM:
                # Only proto + logits layers
                generate_weights_per_layers = [
                    isinstance(layer, (VariableProtoLayer, ProtoLogitsLayer))
                    for layer in self.layers
                ]
            elif self.training_stage == common_vp.ProtoTrainingStage.JOINT:
                # All layers
                num_trained_layers = abs(self.number_of_trained_cnn_layers)
                # Last two layers is always generated.
                num_generated_layers = len(self.layers) - num_trained_layers - 2
                if num_generated_layers < 0:
                    raise ValueError(
                        "num_trained_layers should be smaller that the total "
                        "number of conv layers."
                    )

                is_first_trained = self.number_of_trained_cnn_layers >= 0
                if is_first_trained:
                    generate_weights_per_layers = (
                        [False]*num_trained_layers + [True]*num_generated_layers + [True]+[True]
                    )
                else:
                    generate_weights_per_layers = (
                        [True]*num_generated_layers + [False]*num_trained_layers + [True]+[True]
                    )
            elif self.training_stage == common_vp.ProtoTrainingStage.LAST_ONLY:
                # Only last logits layer
                generate_weights_per_layers = [
                    isinstance(layer, ProtoLogitsLayer)
                    for layer in self.layers
                ]
            else:
                raise ValueError(f"Unknown training stage: {self.training_stage}")
        else:
            num_trained_layers = abs(self.number_of_trained_cnn_layers)
            # Last layer is always a LogitsLayer and we always generate it.
            num_generated_layers = len(self.layers) - num_trained_layers - 1
            if num_generated_layers < 0:
                raise ValueError(
                    "num_trained_layers should be smaller that the total "
                    "number of conv layers."
                )

            is_first_trained = self.number_of_trained_cnn_layers >= 0
            if is_first_trained:
                generate_weights_per_layers = (
                    [False]*num_trained_layers + [True]*num_generated_layers + [True]
                )
            else:
                generate_weights_per_layers = (
                    [True]*num_generated_layers + [False]*num_trained_layers + [True]
                )

        self.generate_weights_per_layers = generate_weights_per_layers

    def warm(self):
        self.training_stage = common_vp.ProtoTrainingStage.WARM
        self._set_generate_weights_per_layers()

    def joint(self):
        self.training_stage = common_vp.ProtoTrainingStage.JOINT
        self._set_generate_weights_per_layers()

    def last_only(self):
        self.training_stage = common_vp.ProtoTrainingStage.LAST_ONLY
        self._set_generate_weights_per_layers()
