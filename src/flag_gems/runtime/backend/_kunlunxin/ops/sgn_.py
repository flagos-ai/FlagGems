import logging

import torch
import triton
import triton.language as tl

from ..utils.codegen_config_utils import CodeGenConfig
from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# sgn_ (in-place sign) was NOT overridden by kunlunxin, so it fell to the
# generic ops/sgn_.py hand-written kernel (hard-coded BLOCK_SIZE=1024,
# grid=cdiv(n,1024), NO @libentry cache, no CodeGenConfig). Without the
# libentry launch cache and with a fixed narrow tile, every distinct shape
# re-JITs the kernel: IR ir-sgn_-dev1.log shows sgn_kernel_ compiled 3570
# times across 2590 modules (25MB / 323K lines), a recompilation explosion
# (same family as cat_out / tril_out / diag / deg2rad).
# Fix: rewrite as pointwise_dynamic (libentry-cached, bounded tile, autoGrid,
# wide contiguous block-DMA), mirroring the sibling unary maps abs/deg2rad.
# promotion DEFAULT keeps the input dtype (sgn preserves dtype, incl. ints);
# vec-OPEN (isCloseVectorization=False) for a plain memory-bound unary map.
#
# Kernel body: sign via two chained tl.where. This is deliberate — profiling on
# this XPU shows the *body* (not launch/IR) is the second bottleneck:
#   * bool -> float `.to(x.dtype)` conversion does NOT vectorize -> ~150 GB/s.
#   * the unordered NaN compare `x != x` (`setuo`) cannot be selected in the
#     vectorized (v16i1) path -> forces scalar fallback (~85 GB/s) or crashes.
#   * `tl.where(cond, x, x)` DOES vectorize (~1700 GB/s), so where is cheap;
#     the two-where sign form avoids both the bool-cast and the NaN compare
#     -> ~170 GB/s fp16 / ~430 GB/s fp32 (2x the bool-subtract form).
# NaN falls through both `>0` / `<0` to the 0 branch, i.e. NaN -> 0, which
# EXACTLY matches this torch build's `torch.sgn` (verified: torch cpu & xpu both
# return 0 for NaN, 0 for +/-0, +/-1 for +/-inf). So this form is both faster
# AND more reference-accurate than a NaN-propagating variant.
_sgn_config = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=False,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")], config=_sgn_config)
@triton.jit
def sgn_func(x):
    # sign(x): +1 for x>0, -1 for x<0, 0 for x==0 and for NaN (matches torch).
    res = tl.where(x > 0, 1, 0)
    res = tl.where(x < 0, -1, res)
    return res


def sgn_(A):
    logger.debug("GEMS_KUNLUNXIN SGN_")
    # Complex sgn = x / abs(x) has different semantics; defer to aten.
    if A.is_complex():
        return torch.ops.aten.sgn_(A)
    sgn_func(A, out0=A)
    return A
