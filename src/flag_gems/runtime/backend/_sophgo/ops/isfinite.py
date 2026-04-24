import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "ALWAYS_BOOL")])
@triton.jit
def isfinite_func(x):
    # Use bitwise operations to check if value is finite
    # A value is finite if it's not inf and not nan
    if x.dtype.is_fp64():
        # For fp64: check if exponent bits (bits 62-52) are not all 1s
        int_bits = x.to(tl.int64, bitcast=True)
        exp_mask = 0x7FF0000000000000  # Exponent mask for fp64
        return (int_bits & exp_mask) != exp_mask
    elif x.dtype.is_fp32():
        # For fp32: check if exponent bits (bits 30-23) are not all 1s
        int_bits = x.to(tl.int32, bitcast=True)
        exp_mask = 0x7F800000  # Exponent mask for fp32
        return (int_bits & exp_mask) != exp_mask
    elif x.dtype.is_fp16():
        # For fp16: check if exponent bits (bits 14-10) are not all 1s
        int_bits = x.to(tl.int16, bitcast=True)
        exp_mask = 0x7C00  # Exponent mask for fp16
        return (int_bits & exp_mask) != exp_mask
    elif x.dtype.is_bf16():
        # For bf16: check if exponent bits (bits 14-7) are not all 1s
        int_bits = x.to(tl.int16, bitcast=True)
        exp_mask = 0x7F80  # Exponent mask for bf16
        return (int_bits & exp_mask) != exp_mask
    else:
        # For other types, convert to fp32 and check
        x_fp32 = x.to(tl.float32)
        int_bits = x_fp32.to(tl.int32, bitcast=True)
        exp_mask = 0x7F800000
        return (int_bits & exp_mask) != exp_mask


def isfinite(
    A: torch.Tensor,
) -> torch.Tensor:
    logging.debug("GEMS ISFINITE")
    if A.is_floating_point():
        return isfinite_func(A)
    else:
        return torch.full(A.shape, True, dtype=torch.bool, device=A.device)
