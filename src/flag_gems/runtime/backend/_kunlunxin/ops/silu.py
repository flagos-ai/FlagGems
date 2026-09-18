import logging

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.utils import libentry

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")], config=config_)
@triton.jit
def silu_forward(x):
    x_fp32 = x.to(tl.float32)
    y = tl.fdiv(x_fp32, (1.0 + tl.exp(-x_fp32)))
    return y


_SILU_BW_MAX_BLOCK = 65536


@libentry()
@triton.jit(do_not_specialize=["n_elements"])
def silu_backward_kernel_xpu(
    x_ptr,
    dy_ptr,
    out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    tid = pid * BLOCK + tl.arange(0, BLOCK)
    mask = tid < n_elements
    x = tl.load(x_ptr + tid, mask=mask).to(tl.float32)
    dy = tl.load(dy_ptr + tid, mask=mask).to(tl.float32)
    sigma = 1.0 / (1.0 + tl.exp(-x))
    dx = dy * sigma * (1.0 + x * (1.0 - sigma))
    tl.store(out_ptr + tid, dx.to(x_ptr.type.element_ty), mask=mask)


@libentry()
@triton.jit
def silu_backward_kernel_xpu_unmasked(
    x_ptr,
    dy_ptr,
    out_ptr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    tid = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + tid).to(tl.float32)
    dy = tl.load(dy_ptr + tid).to(tl.float32)
    sigma = 1.0 / (1.0 + tl.exp(-x))
    dx = dy * sigma * (1.0 + x * (1.0 - sigma))
    tl.store(out_ptr + tid, dx.to(x_ptr.type.element_ty))


def _silu_backward_pick_block(n_elements):
    if n_elements <= 4096:
        return min(triton.next_power_of_2(n_elements), _SILU_BW_MAX_BLOCK)
    if n_elements <= 131072:
        ctas = 8
    elif n_elements <= 2097152:
        ctas = 32
    else:
        ctas = 128
    block = (n_elements + ctas - 1) // ctas
    return min(triton.next_power_of_2(block), _SILU_BW_MAX_BLOCK)


def silu(self):
    logger.debug("GEMS_KUNLUNXIN SILU")
    output = silu_forward(self)
    return output


def silu_backward(grad_output, self):
    logger.debug("GEMS_KUNLUNXIN SILU_BACKWARD")
    x = self if self.is_contiguous() else self.contiguous()
    dy = grad_output if grad_output.is_contiguous() else grad_output.contiguous()
    n_elements = x.numel()
    if n_elements == 0:
        return torch.empty_like(x)
    grad_input = torch.empty_like(x)
    block = _silu_backward_pick_block(n_elements)
    if n_elements % block == 0:
        grid = (n_elements // block, 1, 1)
        silu_backward_kernel_xpu_unmasked[grid](
            x,
            dy,
            grad_input,
            BLOCK=block,
            num_warps=16,
            buffer_size_limit=4096,
        )
    else:
        grid = (triton.cdiv(n_elements, block), 1, 1)
        silu_backward_kernel_xpu[grid](
            x,
            dy,
            grad_input,
            n_elements,
            BLOCK=block,
            num_warps=16,
            buffer_size_limit=4096,
        )
    if grad_input.shape != self.shape or grad_input.stride() != self.stride():
        grad_input = grad_input.reshape(self.shape).as_strided(
            self.size(), self.stride()
        )
    return grad_input


def silu_(A):
    logger.debug("GEMS_KUNLUNXIN SILU_")
    out = silu_forward(A, out0=A)
    return out
