import logging

import torch
import triton
import triton.language as tl

from ..utils import pointwise_dynamic

try:
    from triton.language.extra.cuda.libdevice import isfinited as _isfinited
except ImportError:
    try:
        from triton.language.math import isfinited as _isfinited
    except ImportError:
        from triton.language.libdevice import isfinited as _isfinited

try:
    from triton.language.extra.cuda.libdevice import finitef as _finitef
except ImportError:
    try:
        from triton.language.math import finitef as _finitef
    except ImportError:
        from triton.language.libdevice import finitef as _finitef


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "ALWAYS_BOOL")])
@triton.jit
def isfinite_func(x):
    return _isfinited(x) if x.dtype.is_fp64() else _finitef(x.to(tl.float32))


def isfinite(
    A: torch.Tensor,
) -> torch.Tensor:
    logging.debug("GEMS ISFINITE")
    if A.is_floating_point():
        return isfinite_func(A)
    else:
        return torch.full(A.shape, True, dtype=torch.bool, device=A.device)
