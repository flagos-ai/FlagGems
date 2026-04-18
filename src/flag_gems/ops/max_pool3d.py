import logging
from typing import List, Optional, Union

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_IntOrList = Union[int, List[int]]


def _to_list(x: _IntOrList, n: int) -> List[int]:
    if isinstance(x, int):
        return [x] * n
    return list(x)


def max_pool3d(
    input: torch.Tensor,
    kernel_size: _IntOrList,
    stride: Optional[_IntOrList] = None,
    padding: _IntOrList = 0,
    dilation: _IntOrList = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> torch.Tensor:
    """3-D max pooling.

    Args:
        input: 5-D tensor (N, C, D, H, W).
        kernel_size: Size of the pooling window.
        stride: Stride of the pooling window. Defaults to kernel_size.
        padding: Zero-padding added to both sides.
        dilation: Spacing between kernel elements.
        ceil_mode: Use ceil instead of floor to compute output size.
        return_indices: If True, return (output, indices).

    Returns:
        Pooled tensor, or (pooled tensor, indices) if return_indices=True.
    """
    logger.debug("GEMS MAX_POOL3D")
    if stride is None:
        stride = kernel_size
    return F.max_pool3d(
        input,
        kernel_size=_to_list(kernel_size, 3),
        stride=_to_list(stride, 3),
        padding=_to_list(padding, 3),
        dilation=_to_list(dilation, 3),
        ceil_mode=ceil_mode,
        return_indices=return_indices,
    )
