import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def leaky_relu_fallback(x, negative_slope):
    return tl.where(x > 0, x, x * negative_slope)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 16384}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 16384}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def leaky_relu_kernel_fp16(
    x_ptr, out_ptr, negative_slope, n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.where(x > 0, x, x * negative_slope)
    tl.store(out_ptr + offsets, y, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 16384}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 32768}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def leaky_relu_kernel_fp32(
    x_ptr, out_ptr, negative_slope, n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.where(x > 0, x, x * negative_slope)
    tl.store(out_ptr + offsets, y, mask=mask)


def _get_fast_kernel(inp):
    if inp.dtype in (torch.float16, torch.bfloat16):
        return leaky_relu_kernel_fp16
    if inp.dtype == torch.float32:
        return leaky_relu_kernel_fp32
    return None


def _leaky_relu_contiguous(inp, negative_slope, out):
    n_elements = inp.numel()
    if n_elements == 0:
        return out
    kernel = _get_fast_kernel(inp)
    if kernel is None:
        return leaky_relu_fallback(inp, negative_slope, out0=out)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(inp.device.index):
        kernel[grid](
            inp,
            out,
            negative_slope,
            n_elements,
        )
    return out


def _can_use_fast_path(inp):
    return (
        inp.layout == torch.strided
        and inp.is_cuda
        and not inp.is_quantized
        and not inp.is_complex()
        and inp.is_contiguous()
    )


def leaky_relu(inp, negative_slope=0.01):
    logger.debug("GEMS LEAKY_RELU")
    if _can_use_fast_path(inp):
        return _leaky_relu_contiguous(inp, negative_slope, torch.empty_like(inp))
    if not inp.is_cuda or inp.is_complex():
        return torch.ops.aten.leaky_relu.default.redispatch(
            _FALLBACK_KEYSET, inp, negative_slope
        )
    return leaky_relu_fallback(inp, negative_slope)


def leaky_relu_(inp, negative_slope=0.01):
    logger.debug("GEMS LEAKY_RELU_")
    if _can_use_fast_path(inp):
        return _leaky_relu_contiguous(inp, negative_slope, inp)
    if not inp.is_cuda or inp.is_complex():
        return torch.ops.aten.leaky_relu_.default.redispatch(
            _FALLBACK_KEYSET, inp, negative_slope
        )
    return leaky_relu_fallback(inp, negative_slope, out0=inp)


def leaky_relu_out(inp, negative_slope=0.01, *, out):
    logger.debug("GEMS LEAKY_RELU_OUT")
    if (
        not _can_use_fast_path(inp)
        or out.layout != torch.strided
        or out.device != inp.device
        or out.dtype != inp.dtype
    ):
        return torch.ops.aten.leaky_relu.out.redispatch(
            _FALLBACK_KEYSET, inp, negative_slope, out=out
        )

    if out.shape != inp.shape:
        out.resize_(inp.shape)

    if out.is_contiguous():
        return _leaky_relu_contiguous(inp, negative_slope, out)
    leaky_relu_fallback(inp, negative_slope, out0=out)
    return out
