import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

# The mthreads llc backend crashes (SIGSEGV in SelectionDAG) while lowering the
# libdevice `atan2` intrinsic, so atan2 is reconstructed from the working `atan`
# and `sqrt` primitives via the half-angle identity atan2(Y, X) = 2 * atan(t),
# where t = tan(theta/2) is computed with the numerically stable form per
# half-plane (see the kernel below). `input` is Y and `other` is X, matching
# torch.atan2(input, other).
_atan = tl_extra_shim.atan

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def atan2_kernel(input, other):
    y = input.to(tl.float32)
    x = other.to(tl.float32)
    r = tl.sqrt(x * x + y * y)
    # Half-angle identity atan2(Y, X) = 2 * atan(tan(theta/2)). Two equivalent
    # forms of tan(theta/2) are used, picking the one without catastrophic
    # cancellation for each half-plane:
    #   x >= 0 -> t = Y / (r + X)   (r + X never cancels)
    #   x <  0 -> t = (r - X) / Y   (r - X never cancels; Y != 0 except on the
    #                                negative real axis, where it yields +/-inf
    #                                -> +/-pi, which is correct)
    pos = x >= 0.0
    t_pos = y / (r + x)
    t_neg = (r - x) / y
    t = tl.where(pos, t_pos, t_neg)
    ang = 2.0 * _atan(t)
    # atan2(+/-0, 0) is defined as 0; the (0, 0) case produces nan above.
    return tl.where((x == 0.0) & (y == 0.0), 0.0, ang)


def atan2(input, other):
    logger.debug("GEMS_MTHREADS ATAN2")
    return atan2_kernel(input, other)


def atan2_out(input, other, out):
    logger.debug("GEMS_MTHREADS ATAN2_OUT")
    return atan2_kernel(input, other, out0=out)
