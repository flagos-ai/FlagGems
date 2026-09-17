import logging
import math

import torch

from flag_gems.ops.full import check_dtype, full_func, full_func_scalar

from .full import full

logger = logging.getLogger(__name__)

# The tuned kunlunxin `full` kernel (hand-written grid=(12,1,1) block-DMA) is
# fast AND correct for finite fills into the floating dtypes the benchmark
# exercises (float16/float32/bfloat16, fill_value=3.1415926), reaching ~1.0
# speedup. But it MISCOMPILES some corners that the accuracy suite covers:
#   * non-finite scalars (inf/nan) into bfloat16 -> stores 4.25e37, and
#   * bool output at large shapes -> misaligned-address kernel fault.
# Those corners are never hit by the benchmark, so route only the proven
# fast+correct float path through the tuned kernel and fall back to the generic
# (correct on XPU, slower) full_func/full_func_scalar for everything else. This
# keeps benchmark speedup ~1.0 with ZERO accuracy regression vs the generic op.
_FAST_FLOAT_DTYPES = (torch.float16, torch.float32, torch.bfloat16)


def new_full(
    self,
    size,
    fill_value,
    *,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
    pin_memory=False,
):
    logger.debug("GEMS_KUNLUNXIN NEW_FULL")
    if device is None:
        device = self.device
    if dtype is None:
        dtype = self.dtype

    fill_is_finite = not (
        isinstance(fill_value, float)
        and (math.isinf(fill_value) or math.isnan(fill_value))
    )
    if dtype in _FAST_FLOAT_DTYPES and fill_is_finite:
        return full(size, fill_value, dtype=dtype, device=device)

    # Correct-but-generic fallback for bool / int / float64 / non-finite fills.
    fill_value = check_dtype(fill_value, dtype, device)
    out = torch.empty(size, device=device, dtype=dtype)
    if isinstance(fill_value, torch.Tensor):
        return full_func(out, fill_value, out0=out)
    return full_func_scalar(out, fill_value, out0=out)
