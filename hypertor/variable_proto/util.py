import torch
import numpy as np
import numpy.typing as npt


def list_of_distances(
    X: torch.Tensor,
    Y: torch.Tensor,
) -> torch.Tensor:
    """Calculate a list of distances"""
    return torch.sum(
        # X       → X_u → [N, D, 1]
        # Y.t()   → Y_u → [D, N] → [1, D, N]
        # X_u-Y_u → [N, D, N]
        # torch.sum(..., dim=1) → [N, N]
        (torch.unsqueeze(X, dim=2) - torch.unsqueeze(Y.t(), dim=0)) ** 2,
        dim=1,
    )


def find_high_activation_crop(
    activation_map: npt.NDArray[np.floating],
    percentile: int = 95,
) -> tuple[int, int, int, int]:
    """Locate high activation region"""
    threshold = np.percentile(activation_map, percentile)
    mask: np.ndarray = np.ones(activation_map.shape)
    mask[activation_map < threshold] = 0

    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0

    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > 0.5:
            lower_y = i
            break
    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > 0.5:
            upper_y = i
            break
    for j in range(mask.shape[1]):
        if np.amax(mask[:, j]) > 0.5:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:, j]) > 0.5:
            upper_x = j
            break

    return lower_y, upper_y + 1, lower_x, upper_x + 1
