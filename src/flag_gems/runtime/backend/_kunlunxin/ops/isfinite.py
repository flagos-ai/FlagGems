import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import tl_extra_shim

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))
_finitef = tl_extra_shim.finitef


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "ALWAYS_BOOL")])
@triton.jit
def isfinite_func_f32(x):
    # Bitwise check: finite if exponent bits are not all 1s
    # float32 exponent mask: 0x7F800000
    bits = x.to(tl.uint32, bitcast=True)
    exp_mask = tl.full(bits.shape, 0x7F800000, dtype=tl.uint32)
    return (bits & exp_mask) != exp_mask


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "ALWAYS_BOOL")])
@triton.jit
def isfinite_func_f16(x):
    # Bitwise check for float16: exponent mask 0x7C00
    bits = x.to(tl.uint16, bitcast=True)
    exp_mask = tl.full(bits.shape, 0x7C00, dtype=tl.uint16)
    return (bits & exp_mask) != exp_mask


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "ALWAYS_BOOL")])
@triton.jit
def isfinite_func(x):
    return _finitef(x.to(tl.float32))


def isfinite(
    A: torch.Tensor,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN ISFINITE")
    if A.is_floating_point():
        if A.dtype == torch.float32:
            return isfinite_func_f32(A)
        elif A.dtype == torch.float16:
            return isfinite_func_f16(A)
        else:
            # bfloat16, float64, etc. - use original approach
            return isfinite_func(A)
    else:
        return torch.full(A.shape, True, dtype=torch.bool, device=A.device)
