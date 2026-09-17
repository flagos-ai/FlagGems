import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

MAX_BLOCK = 8192
MIN_BLOCK = 512
NUM_CLUSTERS = 12


def _pick_block(n_elements):
    target = triton.next_power_of_2(max(1, triton.cdiv(n_elements, NUM_CLUSTERS)))
    return max(MIN_BLOCK, min(MAX_BLOCK, target))


def _pick_backward_block(n_elements):
    if MIN_BLOCK < n_elements <= 2 * MIN_BLOCK:
        return 2 * MIN_BLOCK
    return _pick_block(n_elements)


@libentry()
@triton.jit(do_not_specialize=["n_elements", "beta", "threshold"])
def softplus_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    beta,
    threshold,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    x = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    z = x * beta
    soft_z = tl.where(z > threshold, z, tl.log(1 + tl.exp(z)))
    out = soft_z / beta
    tl.store(out_ptr + offset, out.to(out_ptr.dtype.element_ty), mask=mask)


@libentry()
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
        z = x * beta
        derivative = tl.where(z > threshold, 1.0, tl.sigmoid(z))
        tl.store(
            out_ptr + offset,
            (grad * derivative).to(out_ptr.dtype.element_ty),
            mask=mask,
        )
    else:
        grad = tl.load(grad_ptr + offset)
        x = tl.load(x_ptr + offset).to(tl.float32)
        z = x * beta
        derivative = tl.where(z > threshold, 1.0, tl.sigmoid(z))
        tl.store(out_ptr + offset, (grad * derivative).to(out_ptr.dtype.element_ty))


def softplus(self, beta=1.0, threshold=20.0):
    logger.debug("GEMS_KUNLUNXIN SOFTPLUS")
    x = self.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    if n_elements == 0:
        return out
    block_size = _pick_block(n_elements)
    grid = (triton.cdiv(n_elements, block_size),)
    with torch_device_fn.device(x.device):
        softplus_kernel[grid](
            x, out, n_elements, beta, threshold, BLOCK_SIZE=block_size
        )
    return out


def softplus_backward(grad_output, self, beta=1.0, threshold=20.0):
    logger.debug("GEMS_KUNLUNXIN SOFTPLUS_BACKWARD")
    grad = grad_output if grad_output.is_contiguous() else grad_output.contiguous()
    x = self if self.is_contiguous() else self.contiguous()
    out = torch.empty_like(grad)
    n_elements = grad.numel()
    if n_elements == 0:
        return out
    block_size = _pick_backward_block(n_elements)
    grid = (triton.cdiv(n_elements, block_size),)
    need_mask = (n_elements % block_size) != 0
    with torch_device_fn.device(grad.device):
        softplus_backward_kernel[grid](
            grad,
            x,
            out,
            n_elements,
            beta,
            threshold,
            BLOCK_SIZE=block_size,
            NEED_MASK=need_mask,
        )
    return out
