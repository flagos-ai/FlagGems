import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import tl_extra_shim

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)
pow = tl_extra_shim.pow
_tanh = tl_extra_shim.tanh


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def tanh_kernel(x):
    return _tanh(x.to(tl.float32))


_TANH_BW_MAX_BLOCK = 65536
_TANH_BW_BF16_FLAT_MAX_NUMEL = 1 << 16


@triton.jit
def tanh_backward_flat_kernel(y_ptr, dy_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    y = tl.load(y_ptr + offs).to(tl.float32)
    dy = tl.load(dy_ptr + offs).to(tl.float32)
    tl.store(out_ptr + offs, (dy * (1.0 - y * y)).to(out_ptr.dtype.element_ty))


@triton.jit
def tanh_backward_flat_masked_kernel(
    y_ptr,
    dy_ptr,
    out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    y = tl.load(y_ptr + offs, mask=mask).to(tl.float32)
    dy = tl.load(dy_ptr + offs, mask=mask).to(tl.float32)
    tl.store(
        out_ptr + offs,
        (dy * (1.0 - y * y)).to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def _tanh_backward_pick_block(n_elements):
    if n_elements <= 4096:
        return min(triton.next_power_of_2(n_elements), _TANH_BW_MAX_BLOCK)
    if n_elements <= 131072:
        ctas = 8
    elif n_elements <= 2097152:
        ctas = 32
    else:
        ctas = 128
    block = (n_elements + ctas - 1) // ctas
    return min(triton.next_power_of_2(block), _TANH_BW_MAX_BLOCK)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def tanh_backward_kernel(y, dy):
    y = y.to(tl.float32)
    return dy.to(tl.float32) * (1.0 - y * y)


def tanh(self):
    logger.debug("GEMS_KUNLUNXIN TANH")
    out = tanh_kernel(self)
    return out


def _tanh_backward_flat(grad_output, output):
    numel = output.numel()
    if numel == 0:
        return torch.empty_like(output)
    grad_input = torch.empty_strided(
        output.shape, output.stride(), dtype=output.dtype, device=output.device
    )
    block = _tanh_backward_pick_block(numel)
    if numel % block == 0:
        tanh_backward_flat_kernel[(numel // block,)](
            output,
            grad_output,
            grad_input,
            BLOCK=block,
            num_warps=16,
            buffer_size_limit=4096,
        )
    else:
        tanh_backward_flat_masked_kernel[(triton.cdiv(numel, block),)](
            output,
            grad_output,
            grad_input,
            numel,
            BLOCK=block,
            num_warps=16,
            buffer_size_limit=4096,
        )
    return grad_input


def tanh_backward(grad_output, output):
    logger.debug("GEMS_KUNLUNXIN TANH_BACKWARD")
    if (
        output.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and output.is_contiguous()
        and grad_output.is_contiguous()
        and output.dim() > 0
        and (
            output.dtype != torch.bfloat16
            or output.numel() <= _TANH_BW_BF16_FLAT_MAX_NUMEL
        )
    ):
        return _tanh_backward_flat(grad_output, output)
    in_grad = tanh_backward_kernel(output, grad_output)
    return in_grad


def tanh_(A):
    logger.debug("GEMS_KUNLUNXIN TANH_")
    out = tanh_kernel(A, out0=A)
    return out
