import logging

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=2048,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(
    is_tensor=[True, True, True, False, False, False],
    promotion_methods=[(0, 1, 2, "DEFAULT")],
    config=config_,
)
@triton.jit
def rrelu_with_noise_backward_func(
    grad_output, input, noise, lower, upper, training
):
    grad = grad_output.to(tl.float32)
    if training:
        result = grad * noise.to(tl.float32)
    else:
        slope = (lower + upper) * 0.5
        result = grad * tl.where(input.to(tl.float32) > 0, 1.0, slope)
    return result.to(grad_output.dtype)


def rrelu_with_noise_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    noise: torch.Tensor,
    lower: float,
    upper: float,
    training: bool,
    self_is_result: bool = False,
):
    logger.debug("GEMS_KUNLUNXIN RRELU_WITH_NOISE_BACKWARD")
    return rrelu_with_noise_backward_func(
        grad_output, input, noise, lower, upper, training
    )
