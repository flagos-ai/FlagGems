import logging

import torch
import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "ALWAYS_BOOL")])
@triton.jit
def isfinite_func(x):
    if x.dtype.is_fp64():
        return (x.to(tl.int64, bitcast=True) & 0x7FFFFFFFFFFFFFFF) < 0x7FF0000000000000
    elif x.dtype.is_fp32():
        return (x.to(tl.int32, bitcast=True) & 0x7FFFFFFF) < 0x7F800000
    elif x.dtype.is_fp16():
        return (x.to(tl.int16, bitcast=True) & 0x7FFF) < 0x7C00
    elif x.dtype.is_bf16():
        return (x.to(tl.int16, bitcast=True) & 0x7FFF) < 0x7F80


def isfinite(
    A: torch.Tensor,
) -> torch.Tensor:
    logger.debug("GEMS_CAMBRICON ISFINITE")
    if A.is_floating_point():
        legal_dtype = [torch.float32, torch.float16, torch.bfloat16]
        assert (
            A.dtype in legal_dtype
        ), f"isfinite input float dtype should in {str(legal_dtype)}, get {str(A.dtype)}"
        return isfinite_func(A)
    else:
        return torch.full(A.shape, True, dtype=torch.bool, device=A.device)
