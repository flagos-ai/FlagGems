import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim
from flag_gems.utils.codegen_config_utils import CodeGenConfig

logger = logging.getLogger(f'flag_gems.runtime._ascend.ops.{__name__.split(".")[-1]}')


try:
    import torch_npu  # noqa: F401

    atan2 = tl_extra_shim.atan2
except ImportError:  # noqa: E722
    atan2 = tl_extra_shim.atan2

config_ = CodeGenConfig(
    256,
    (40, 1, 1),
    32,
    False,
    prefer_1d_tile=int(triton.__version__[0]) < 3,
)


@pointwise_dynamic(
    is_tensor=[True, True], promotion_methods=[(0, "DEFAULT")], config=config_
)
@triton.jit
def angle_func(real, imag):
    real_last, imag_last = (
        (real.to(tl.float32), imag.to(tl.float32))
        if real.dtype == tl.float16
        else (real, imag)
    )
    result = atan2(imag_last, real_last)
    return result


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def angle_float_and_int(real):
    zero = 0.0
    pi = math.pi
    real_positive = real >= zero
    result = tl.where(real_positive, zero, pi)
    return result


def angle(input_tensor: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_ASCEND ANGLE")
    if input_tensor.dtype == torch.complex32 or input_tensor.dtype == torch.complex64:
        real = input_tensor.real
        imag = input_tensor.imag
        return angle_func(real, imag)
    else:
        real = input_tensor
        return angle_float_and_int(real)
