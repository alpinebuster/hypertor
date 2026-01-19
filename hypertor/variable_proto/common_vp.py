import enum
import dataclasses

import torch


class ProtoTrainingStage(enum.Enum):
    WARM = "warm"
    JOINT = "joint"
    LAST_ONLY = "last"


@dataclasses.dataclass
class ProtoTrainConfig:
    """Variable Prototype Training configuration."""

    # Training
    warm_epochs: int
    # Prototype pushing
    push_freq: int
    push_start: int
    push_last_steps: int
    push_warmup_steps: int

    lr_last_layer: float
    lr_proto_layer: float
    lr_conv_layer: float

    joint_lr_step_size: int
    joint_lr_gamma: float

    # Prototype parameters
    prototypes: int # number of prototypes per class
    prototype_size: list[int]  # [height, width]
    prototype_activation_func: str

    # Loss coefficients
    coef_cross_entropy: float
    coef_cluster_cost: float
    coef_separation_cost: float
    coef_reg_protos: float
    coef_reg_last: float

    def __post_init__(self):
        if self.prototype_size is None:
            self.prototype_size = [1, 1]

        if len(self.prototype_size) != 2:
            raise ValueError("`prototype_size` must be a tuple of 2 ints")

        try:
            prototype_size_h, prototype_size_w = self.prototype_size
            self.prototype_size = [
                int(prototype_size_h),
                int(prototype_size_w),
            ]
        except (TypeError, ValueError):
            # `self.prototype_size!r` -> `repr(self.prototype_size)`
            raise TypeError(
                f"`prototype_size` elements must be int-convertible, "
                f"got {self.prototype_size!r}"
            )



@dataclasses.dataclass
class VariableProtoState:
    step_updated: int = 0
    prototype_shape: tuple[int, int, int, int] = (0, 0, 0, 0)
    prototype_class_identity: torch.Tensor = torch.zeros(0, 0)
    prototype_ids_per_class: dict[int, list[int]] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        try:
            self.step_updated = int(self.step_updated)
        except (TypeError, ValueError):
            raise TypeError(
                f"`step_updated` must be int-convertible, got {self.step_updated!r}"
            )

        if len(self.prototype_shape) != 4:
            raise ValueError("`prototype_shape` must be a tuple of 4 ints")

        try:
            num_prototypes, channels, prototype_size_h, prototype_size_w = self.prototype_shape
            self.prototype_shape = (
                int(num_prototypes),
                int(channels),
                int(prototype_size_h),
                int(prototype_size_w),
            )
        except (TypeError, ValueError):
            raise TypeError(
                f"prototype_shape elements must be int-convertible, "
                f"got {self.prototype_shape!r}"
            )
