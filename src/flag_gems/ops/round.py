import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def round_half_to_even_impl(x):
    """Round to nearest with ties to even (round half to even).
    x must be fp32."""
    r = tl.floor(x)
    d = x - r  # fractional part, in [0, 1) for positive, in (-1, 0] for negative

    # is_odd = (r % 2 == 1), i.e., r is odd
    # In Triton: r - 2 * floor(r/2) for odd r in [-2.5, 2.5] range is close to 1
    is_odd = tl.abs(r - 2.0 * tl.floor(r / 2.0)) > 0.5

    # For d > 0.5: always round up
    # For d == 0.5 and r is odd: round up (to make result even)
    # For d == 0.5 and r is even: stay at r (already even)
    # For d < 0.5: stay at r
    return tl.where((d > 0.5) | ((tl.abs(d - 0.5) < 1e-10) & is_odd), r + 1.0, r)


@triton.jit
def round_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    decimals: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    IS_FP32: tl.constexpr,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    # Apply round with "round half to even" rule
    if decimals == 0:
        out = x
        if IS_FP32:
            out = round_half_to_even_impl(x)
        elif IS_FP16:
            x_fp32 = tl.cast(x, tl.float32)
            out = tl.cast(round_half_to_even_impl(x_fp32), tl.float16)
        elif IS_BF16:
            x_fp32 = tl.cast(x, tl.float32)
            out = tl.cast(round_half_to_even_impl(x_fp32), tl.bfloat16)
    else:
        # For non-zero decimals, use scaling approach
        scale = 10.0**decimals
        if IS_FP32:
            x_scaled = x * scale
            out = round_half_to_even_impl(x_scaled) / scale
        elif IS_FP16:
            x_fp32 = tl.cast(x, tl.float32)
            x_scaled = x_fp32 * scale
            out = tl.cast(round_half_to_even_impl(x_scaled) / scale, tl.float16)
        elif IS_BF16:
            x_fp32 = tl.cast(x, tl.float32)
            x_scaled = x_fp32 * scale
            out = tl.cast(round_half_to_even_impl(x_scaled) / scale, tl.bfloat16)
        else:
            out = x

    tl.store(out_ptr + offsets, out, mask=mask)


def round_func(input, decimals=0):
    if not isinstance(input, torch.Tensor):
        raise TypeError("round expects a torch.Tensor.")

    if input.is_complex():
        raise TypeError("round is not supported for complex tensors.")

    # For integer types, return a copy (array-api convention)
    if input.dtype in [torch.int32, torch.int64, torch.int16, torch.int8]:
        return input.clone()

    if not input.is_contiguous():
        raise ValueError(
            "round Triton kernel currently supports only contiguous tensors."
        )

    n_elements = input.numel()
    if n_elements == 0:
        return input

    dtype = input.dtype
    IS_FP32 = dtype == torch.float32
    IS_FP16 = dtype == torch.float16
    IS_BF16 = dtype == torch.bfloat16

    output = torch.empty_like(input)

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    with torch_device_fn.device(input.device):
        round_kernel[grid](
            input,
            output,
            n_elements,
            decimals,
            BLOCK_SIZE=BLOCK_SIZE,
            IS_FP32=IS_FP32,
            IS_FP16=IS_FP16,
            IS_BF16=IS_BF16,
        )
    return output


def round(input, decimals=0):
    logger.debug("GEMS ROUND")
    return round_func(input, decimals=decimals)


def round_out(input, *, decimals=0, out=None):
    logger.debug("GEMS ROUND_OUT")
    if out is None:
        return round_func(input, decimals=decimals)

    if not isinstance(input, torch.Tensor):
        raise TypeError("round expects a torch.Tensor.")

    if input.is_complex():
        raise TypeError("round is not supported for complex tensors.")

    # For integer types, return a copy
    if input.dtype in [torch.int32, torch.int64, torch.int16, torch.int8]:
        out.copy_(input)
        return out

    if not input.is_contiguous():
        raise ValueError(
            "round Triton kernel currently supports only contiguous tensors."
        )

    n_elements = input.numel()
    if n_elements == 0:
        return out

    dtype = input.dtype
    IS_FP32 = dtype == torch.float32
    IS_FP16 = dtype == torch.float16
    IS_BF16 = dtype == torch.bfloat16

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    with torch_device_fn.device(input.device):
        round_kernel[grid](
            input,
            out,
            n_elements,
            decimals,
            BLOCK_SIZE=BLOCK_SIZE,
            IS_FP32=IS_FP32,
            IS_FP16=IS_FP16,
            IS_BF16=IS_BF16,
        )
    return out


def round_(input, *, decimals=0):
    logger.debug("GEMS ROUND_")
    if not isinstance(input, torch.Tensor):
        raise TypeError("round expects a torch.Tensor.")

    if input.is_complex():
        raise TypeError("round is not supported for complex tensors.")

    # For integer types, return input unchanged (array-api convention for integer round)
    if input.dtype in [torch.int32, torch.int64, torch.int16, torch.int8]:
        return input

    if not input.is_contiguous():
        raise ValueError(
            "round Triton kernel currently supports only contiguous tensors."
        )

    n_elements = input.numel()
    if n_elements == 0:
        return input

    dtype = input.dtype
    IS_FP32 = dtype == torch.float32
    IS_FP16 = dtype == torch.float16
    IS_BF16 = dtype == torch.bfloat16

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    with torch_device_fn.device(input.device):
        round_kernel[grid](
            input,
            input,
            n_elements,
            decimals,
            BLOCK_SIZE=BLOCK_SIZE,
            IS_FP32=IS_FP32,
            IS_FP16=IS_FP16,
            IS_BF16=IS_BF16,
        )
    return input
