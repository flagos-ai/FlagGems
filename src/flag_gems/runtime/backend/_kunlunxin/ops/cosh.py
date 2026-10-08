# Kunlunxin (XPU) override of cosh / cosh_ / cosh_out.
#
# Generic src/flag_gems/ops/cosh.py already registers ("cosh"/"cosh_"/"cosh_out"
# are all in _FULL_CONFIG), so torch.cosh* under use_gems dispatches to the
# generic pointwise_dynamic kernel 0.5*(exp(x)+exp(-x)) computed in fp32. That
# kernel is correct on the test matrix but has NO tuned CodeGenConfig on XPU:
#   - no kunlunAutoGrid / prefer_1d_tile / unroll_num -> per-shape recompiled,
#     discrete access; measured 0.015x on large shapes (41.9ms for
#     [4096,4096] fp32 vs torch 0.62ms; 166ms for [1024,65536]).
# Fix: mirror the sibling exp.py / atanh.py / arcsinh.py -- @pointwise_dynamic
# with the tuned config_ (vec-closed + kunlunAutoGrid + unroll8) so the kernel
# compiles once and uses contiguous block-DMA tiles.
#
# Numeric note (probing the naive formula on the device):
#   0.5*(exp(x)+exp(-x)) overflows to +inf for |x| > ~88.72 in fp32 while the
#   true cosh stays finite up to |x| ~ 89.41 (cosh(89.0) = 2.245e38). The
#   native xdnn cosh shares that early overflow (native torch.cosh(89.0) -> inf
#   on the device), so the generic kernel matches native exactly -- but both
#   differ from the CPU-fp64 oracle in the window (88.72, 89.41]. Fix at no
#   accuracy cost: for |x| > 87 compute cosh(|x|) = exp(|x|)/2 as
#   exp(|x| - 88.0) * (e^88/2) (|x|-88.0 is exact; e^88/2 = 8.25818127497e37 is
#   a fp32 literal, 3e-8 relative). The argument never exceeds ln2 above the
#   fp32 exp overflow point within the representable range, and the decay term
#   exp(-|x|) is < 2^-125 ~ 0 there, so the result equals cosh(|x|) up to fp32
#   rounding; measured max relative error ~5.2e-8 (vs 6.4e-8 for the direct
#   formula). NOTE: exp(|x| - ln2) via a subtraction of the fp32-rounded ln2
#   literal is NOT used -- measured ~1.4e-6 relative error (above the fp32
#   test rtol 1.3e-6) on triton-XPU. Special values preserved: +-inf -> +inf,
#   NaN -> NaN, +-0 -> 1.0, even function via tl.abs.
import logging

import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# Same config as the sibling exp.py (cosh is an exp-family op): vec-closed +
# kunlunAutoGrid + unroll8 (isCloseVectorization=True is the log/exp-family
# recipe; atanh/arcsinh use False but both are verified-correct on the full
# test matrix, and True is the safer choice for the exp family).
config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=config_)
@triton.jit
def cosh_func(x):
    xf = x.to(tl.float32)
    ax = tl.abs(xf)
    # |x| <= 87: direct formula, e^(+-|x|) both in range, error ~6.4e-8.
    small = 0.5 * (tl.exp(ax) + tl.exp(-ax))
    # |x| > 87: exp(ax)/2 = exp(ax - 88.0) * (e^88/2), exact-shift version that
    # avoids the fp32 exp overflow (88.72) up to the true representability
    # boundary 89.41; both branches are evaluated, so it must also be finite
    # for small ax (it is: exp(-88) * 8.26e37 ~ 0.49).
    large = tl.exp(ax - 88.0) * 8.258181274970006e37
    return tl.where(ax > 87.0, large, small)


def cosh(x, *, out=None):
    logger.debug("GEMS_KUNLUNXIN COSH")
    if out is None:
        return cosh_func(x)
    cosh_func(x, out0=out)
    return out


def cosh_(x):
    logger.debug("GEMS_KUNLUNXIN COSH_")
    cosh_func(x, out0=x)
    return x


def cosh_out(x, *, out=None):
    logger.debug("GEMS_KUNLUNXIN COSH_OUT")
    return cosh(x, out=out)
