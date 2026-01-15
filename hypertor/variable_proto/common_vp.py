import dataclasses

import torch


@dataclasses.dataclass
class TrainConfig:
    """Variable Prototype Training configuration."""

    # Training
    warm_epochs: int
    # Prototype pushing
    push_freq: int
    push_start: int

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


@dataclasses.dataclass
class VariableProtoState:
    step_updated: int = 0
    prototype_shape: tuple[int, int, int, int] = (0, 0, 0, 0)
    prototype_class_identity: torch.Tensor = torch.zeros(0, 0)
    prototype_ids_per_class: dict[int, list[int]] = dataclasses.field(default_factory=dict)
