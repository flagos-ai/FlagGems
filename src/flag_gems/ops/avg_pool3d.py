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


def avg_pool3d(
    input: torch.Tensor,
    kernel_size: _IntOrList,
    stride: Optional[_IntOrList] = None,
    padding: _IntOrList = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: Optional[int] = None,
) -> torch.Tensor:
    """3-D average pooling.

    Args:
        input: 5-D tensor (N, C, D, H, W).
        kernel_size: Size of the pooling window.
        stride: Stride of the pooling window. Defaults to kernel_size.
        padding: Zero-padding added to both sides.
        ceil_mode: Use ceil instead of floor to compute output size.
        count_include_pad: Include zero-padding in averaging.
        divisor_override: If specified, use as divisor instead of pool size.

    Returns:
        Pooled tensor.
    """
    logger.debug("GEMS AVG_POOL3D")
    if stride is None:
        stride = kernel_size
    return F.avg_pool3d(
        input,
        kernel_size=_to_list(kernel_size, 3),
        stride=_to_list(stride, 3),
        padding=_to_list(padding, 3),
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )
