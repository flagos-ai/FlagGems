import logging

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as ext

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# Kunlunxin/XPU performance override for aten::softplus / softplus_backward.
# Forward: pointwise_dynamic + bounded CodeGenConfig; beta==1 && threshold>=17
# uses the unguarded fast path softplus_func_beta1, others keep the guarded form.
# Backward: historical softplus_backward fix (plain @triton.jit + NEED_MASK + tier).

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


# beta==1 fast path: stable identity (x+|x|)/2 + log(1+exp(-|x|)), no tl.where.
# (max(x,0) written arithmetically; tl.maximum hits a slow XPU codegen path.)
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


# (numel_upper_bound, BLOCK_SIZE, num_warps), following log_sigmoid_forward.
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
    # beta==1 && threshold>=17: guard is redundant (log(1+exp(z))==z within
    # fp32/fp16/bf16 tolerance), so skip it and the per-element division.
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
