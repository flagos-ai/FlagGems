import logging

from flag_gems.ops.max_pool3d_with_indices import (
    max_pool3d_with_indices,
)

logger = logging.getLogger(__name__)


def max_pool3d(
    input,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
):
    """
    FlagGems max_pool3d operator.
    Returns only pooled output.
    """

    logger.debug(
        "GEMS MAX_POOL3D"
    )

    output, _ = (
        max_pool3d_with_indices(
            input,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
        )
    )

    return output
