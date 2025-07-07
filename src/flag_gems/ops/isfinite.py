import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

try:
    _isfinited = tl_extra_shim.isfinited
    _finitef = tl_extra_shim.finitef
except Exception:
    pass
logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "ALWAYS_BOOL")])
@triton.jit
def isfinite_func(x):
    return _isfinited(x) if x.dtype.is_fp64() else _finitef(x.to(tl.float32))


def isfinite(
    A: torch.Tensor,
) -> torch.Tensor:
    logger.debug("GEMS ISFINITE")
    if A.is_floating_point():
        return isfinite_func(A)
    else:
        return torch.full(A.shape, True, dtype=torch.bool, device=A.device)
