import logging

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as ext

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# Kunlunxin/XPU performance override for aten::softplus / aten::softplus_backward.
#
# Forward: the old flat kernel (@libentry + always-masked load/store) ran the
# masked-memory slow path on every shape (恒真 mask ~2-4x penalty on XPU) and
# never reached the pointwise codegen knobs. Following the established exp+log
# pointwise recipe (cosh / acosh / sinh / log1p / log2 / mish) the forward is
# rewritten with pointwise_dynamic + an explicit bounded 1D-tile CodeGenConfig
# (kunlunAutoGrid=True + prefer_1d_tile + unroll_num=8 + buffer_size_limit=4096),
# with beta/threshold passed as non-specialized scalar args (is_tensor=[True,
# False, False]).
#
# The key cost driver measured on XPU is the per-lane i1 compare+select of
# `tl.where(z > threshold, z, log(1+exp(z)))` (~400-600 us on a 16.7M fp32
# tensor, see solution/softplus/): a pure log+exp kernel runs ~222-261us while
# the same kernel plus the where runs ~640-680us (and tl.minimum clamps are even
# worse). Because softplus(x) = log(1+exp(beta*x)) rounds to exactly beta*x in
# fp32 for beta*x > ~16.6, the threshold branch is redundant whenever
# threshold >= 17.0: any z above such a threshold is also above ~16.6, where
# 1+exp(z) rounds to exp(z) and log(1+exp(z)) == z within fp32/fp16/bf16
# tolerance. The beta == 1.0 && threshold >= 17.0 case therefore takes the
# unguarded fast path softplus_func_beta1 (which also skips the per-element
# division), using the fully-stable identity (x+|x|)/2 + log(1+exp(-|x|)) (==
# max(x,0)+log1p(exp(-|x|)), max written arithmetically since tl.maximum hits a
# slow XPU codegen path); every other combination keeps the exact guarded
# tl.where form so torch's threshold semantics are preserved for small random
# thresholds. The stable identity has NO |x|>88 exp() overflow: for
# x>threshold>=17, log(1+exp(-|x|)) < 4.1e-8 is below fp32 ulp (x>=17:
# ulp>=3.8e-6), so the result rounds to exactly x and matches torch; for x<-88,
# exp(-|x|) underflows to 0 and (x+|x|)/2==0, so the result is 0, matching torch.
#
# Backward: same historical fix as softplus_backward (plain @triton.jit instead
# of @libentry + NEED_MASK constexpr + explicit num_warps tier); kernel math
# unchanged (where(z>threshold, 1.0, sigmoid(z)) * grad).

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


@pointwise_dynamic(
    is_tensor=[True, False, False],
    promotion_methods=[(0, "DEFAULT")],
    config=config_,
)
@triton.jit
def softplus_func(x, beta, threshold):
    x32 = x.to(tl.float32)
    z = x32 * beta
    soft_z = tl.where(z > threshold, z, tl.log(1.0 + tl.exp(z)))
    return (soft_z / beta).to(x.dtype)


# beta == 1.0 fast path: skip the per-element `z = x * beta` and `soft_z / beta`
# (division costs ~40us on a 16.7M fp32 tensor). Uses the fully-stable identity
# softplus(x) = (x+|x|)/2 + log(1+exp(-|x|)) (== max(x,0)+log1p(exp(-|x|))):
# one exp + one log, no tl.where. The max(x,0) piece is written arithmetically
# as (x+|x|)/2 because tl.maximum/tl.minimum hit a slow non-vectorized path in
# the XPU codegen (~1500us vs ~190us on 16.7M elements) while tl.abs is cheap.
# Unlike the textbook form x+log(1+exp(-x)), this also survives x<-88 (there
# exp(-x) would overflow fp32 to inf and x+inf=inf): exp(-|x|) underflows to 0
# and (x+|x|)/2==0, so the result is 0, matching torch. The body source text is
# structurally distinct from softplus_func (no tl.where), so the in-process
# codegen cache keys cannot collide.
@pointwise_dynamic(
    is_tensor=[True, False, False],
    promotion_methods=[(0, "DEFAULT")],
    config=config_,
)
@triton.jit
def softplus_func_beta1(x, beta, threshold):
    x32 = x.to(tl.float32)
    a = tl.abs(x32)
    soft_z = (x32 + a) * 0.5 + tl.log(1.0 + tl.exp(-a))
    return soft_z.to(x.dtype)


# (numel_upper_bound, BLOCK_SIZE, num_warps) following log_sigmoid_forward
# (same exp + log transcendental structure), plus a tiny tier so sub-2048
# tensors use a 1024-wide unmasked tile instead of a padded 2048 masked tile.
_TIERS = (
    (2048, 1024, 4),
    (16384, 2048, 4),
    (262144, 8192, 8),
    (None, 16384, 16),
)


def _pick_tier(numel):
    for hi, block, warps in _TIERS:
        if hi is None or numel <= hi:
            return block, warps
    return 16384, 16


@triton.jit(do_not_specialize=["n_elements", "beta", "threshold"])
def softplus_backward_kernel(
    grad_ptr,
    x_ptr,
    out_ptr,
    n_elements,
    beta,
    threshold,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if NEED_MASK:
        mask = offset < n_elements
        grad = tl.load(grad_ptr + offset, mask=mask, other=0.0)
        x = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    else:
        grad = tl.load(grad_ptr + offset)
        x = tl.load(x_ptr + offset).to(tl.float32)
    z = x * beta
    derivative = tl.where(z > threshold, 1.0, tl.sigmoid(z))
    out = grad * derivative
    if NEED_MASK:
        tl.store(out_ptr + offset, out.to(out_ptr.dtype.element_ty), mask=mask)
    else:
        tl.store(out_ptr + offset, out.to(out_ptr.dtype.element_ty))


def softplus(self, beta=1.0, threshold=20.0):
    logger.debug("GEMS_KUNLUNXIN SOFTPLUS")
    # beta == 1.0 AND threshold >= 17.0: skip the guard (log(1+exp(z)) == z
    # within fp32/fp16/bf16 tolerance for every z above such a threshold) and
    # skip the per-element division. Any smaller threshold keeps the guarded
    # tl.where kernel so torch's threshold semantics are preserved exactly.
    if float(beta) == 1.0 and float(threshold) >= 17.0:
        return softplus_func_beta1(self, 1.0, float(threshold))
    return softplus_func(self, float(beta), float(threshold))


def softplus_backward(grad_output, self, beta=1.0, threshold=20.0):
    logger.debug("GEMS_KUNLUNXIN SOFTPLUS_BACKWARD")
    grad = grad_output if grad_output.is_contiguous() else grad_output.contiguous()
    x = self if self.is_contiguous() else self.contiguous()
    out = torch.empty_like(grad)
    n_elements = grad.numel()
    if n_elements == 0:
        return out
    block, warps = _pick_tier(n_elements)
    need_mask = (n_elements % block) != 0
    grid = (triton.cdiv(n_elements, block),)
    with torch_device_fn.device(grad.device):
        softplus_backward_kernel[grid](
            grad,
            x,
            out,
            n_elements,
            beta,
            threshold,
            BLOCK_SIZE=block,
            NEED_MASK=need_mask,
            num_warps=warps,
        )
    return out
