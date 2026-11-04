import logging

import torch

from .adaptive_avg_pool2d import adaptive_avg_pool2d

logger = logging.getLogger(__name__)


def adaptive_avg_pool1d(input: torch.Tensor, output_size):
    """Adaptive average pooling operation over 1D input (Kunlunxin).

    Implemented by reshaping to 2D and calling the vendor
    ``adaptive_avg_pool2d``, which handles the XPU Triton load-mask issue.
    """
    logger.debug("GEMS_KUNLUNXIN ADAPTIVE_AVG_POOL1D")

    assert input.ndim == 3, f"adaptive_avg_pool1d expects 3D input, got {input.ndim}D"

    input_4d = input.unsqueeze(2)  # (N, C, L) -> (N, C, 1, L)

    if isinstance(output_size, int):
        output_size_2d = (1, output_size)
    else:
        output_size_2d = (1, output_size[0])

    output_4d = adaptive_avg_pool2d(input_4d, output_size_2d)

    return output_4d.squeeze(2)