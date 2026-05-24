import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems.ops import div as base_div


@triton.jit
def _div_tensor_scalar_kernel(
    x_ptr,
    out_ptr,
    scalar,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_prog = tl.num_programs(0)
    start = pid * BLOCK_SIZE
    step = num_prog * BLOCK_SIZE
    for off in range(start, n_elements, step):
        offsets = off + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = x / scalar
        tl.store(out_ptr + offsets, y, mask=mask)

def _select_block_size(n_elements, dtype):
    if n_elements >= (1 << 20):
        return 512 if dtype in (torch.float16, torch.bfloat16) else 256
    if n_elements >= (1 << 18):
        return 256 if dtype in (torch.float16, torch.bfloat16) else 128
    return 256 if dtype in (torch.float16, torch.bfloat16) else 128


def _maybe_contiguous(x, out):
    if x.is_contiguous():
        return x, out, False
    if out is None:
        return x.contiguous(), out, True
    if out.is_contiguous():
        return x.contiguous(), out, True
    return x, out, False


def _div_tensor_scalar_triton(x, scalar, out=None):
    n_elements = x.numel()
    if n_elements == 0:
        return x if out is None else out
    if n_elements == 1 and x.dtype is torch.bfloat16:
        val = x.item() / scalar
        if out is None:
            out = torch.empty_like(x)
        out.fill_(val)
        return out

    block_size = _select_block_size(n_elements, x.dtype)
    block_size = min(block_size, triton.next_power_of_2(max(n_elements, 1)))
    num_blocks = triton.cdiv(n_elements, block_size)
    grid = (num_blocks,)
    x_contig, out_contig, _ = _maybe_contiguous(x, out)
    if out_contig is None:
        out_contig = torch.empty_like(x_contig)
    num_warps = 1
    _div_tensor_scalar_kernel[grid](
        x_contig,
        out_contig,
        scalar,
        n_elements,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return out_contig


def _maybe_get_scalar_tensor(val):
    if isinstance(val, torch.Tensor) and val.numel() == 1:
        return val.item()
    return None


def true_divide(A, B):
    logging.debug("GEMS_ARM TRUE_DIVIDE")
    if os.environ.get("GEMS_DEBUG_DIV") == "1":
        a_shape = tuple(A.shape) if isinstance(A, torch.Tensor) else None
        b_shape = tuple(B.shape) if isinstance(B, torch.Tensor) else None
        print(f"[GEMS_DEBUG_DIV] true_divide: A={a_shape} B={b_shape}")
    if isinstance(A, torch.Tensor) and not isinstance(B, torch.Tensor):
        return _div_tensor_scalar_triton(A, B)
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        scalar = _maybe_get_scalar_tensor(B)
        if scalar is not None:
            return _div_tensor_scalar_triton(A, scalar)
    return base_div.true_divide(A, B)


def true_divide_(A, B):
    logging.debug("GEMS_ARM TRUE_DIVIDE_")
    if isinstance(A, torch.Tensor) and not isinstance(B, torch.Tensor):
        if A.is_contiguous():
            return _div_tensor_scalar_triton(A, B, out=A)
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        scalar = _maybe_get_scalar_tensor(B)
        if scalar is not None and A.is_contiguous():
            return _div_tensor_scalar_triton(A, scalar, out=A)
    return base_div.true_divide_(A, B)


def trunc_divide(A, B):
    logging.debug("GEMS_ARM TRUNC_DIVIDE")
    return base_div.trunc_divide(A, B)


def trunc_divide_(A, B):
    logging.debug("GEMS_ARM TRUNC_DIVIDE_")
    return base_div.trunc_divide_(A, B)


def floor_divide(A, B):
    logging.debug("GEMS_ARM FLOOR_DIVIDE")
    return base_div.floor_divide(A, B)


def floor_divide_(A, B):
    logging.debug("GEMS_ARM FLOOR_DIVIDE_")
    return base_div.floor_divide_(A, B)


def div_mode(A, B, rounding_mode=None):
    if rounding_mode is None:
        return true_divide(A, B)
    if rounding_mode == "trunc":
        return trunc_divide(A, B)
    if rounding_mode == "floor":
        return floor_divide(A, B)
    msg = (
        "div expected rounding_mode to be one of None, 'trunc', or 'floor' "
        f"but found {rounding_mode}."
    )
    raise ValueError(msg)


def div_mode_(A, B, rounding_mode=None):
    if rounding_mode is None:
        return true_divide_(A, B)
    if rounding_mode == "trunc":
        return trunc_divide_(A, B)
    if rounding_mode == "floor":
        return floor_divide_(A, B)
    msg = (
        "div expected rounding_mode to be one of None, 'trunc', or 'floor' "
        f"but found {rounding_mode}."
    )
    raise ValueError(msg)


def remainder(A, B):
    logging.debug("GEMS_ARM REMAINDER")
    return base_div.remainder(A, B)


def remainder_(A, B):
    logging.debug("GEMS_ARM REMAINDER_")
    return base_div.remainder_(A, B)
