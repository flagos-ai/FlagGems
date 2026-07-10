import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_atan = tl_extra_shim.atan
_isnan = tl_extra_shim.isnan


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def atan2_kernel(x, y):
    # x = input (y-coord), y = other (x-coord); compute atan2(x, y).
    # The mthreads libdevice atan2 intrinsic (`__nv_atan2f`) crashes the llc
    # instruction selector for half-precision inputs whenever a tile is
    # partially filled (e.g. numel not aligned to the tile size). Rebuild
    # atan2 from the single-argument atan, which lowers correctly, plus the
    # standard quadrant correction.
    xf = x.to(tl.float32)
    yf = y.to(tl.float32)
    pi = 3.141592653589793
    half_pi = 1.5707963267948966
    ratio = tl.where(yf != 0.0, xf / yf, 0.0)
    a = _atan(ratio)
    out = tl.where(
        yf > 0.0,
        a,
        tl.where(
            yf < 0.0,
            tl.where(xf >= 0.0, a + pi, a - pi),
            tl.where(xf > 0.0, half_pi, tl.where(xf < 0.0, -half_pi, 0.0)),
        ),
    )
    nan_mask = (_isnan(xf) | _isnan(yf)).to(tl.int1)
    out = tl.where(nan_mask, float("nan"), out)
    return out


def atan2(input, other):
    logger.debug("GEMS_MTHREADS ATAN2")
    return atan2_kernel(input, other)


def atan2_out(input, other, out):
    logger.debug("GEMS_MTHREADS ATAN2_OUT")
    return atan2_kernel(input, other, out0=out)
