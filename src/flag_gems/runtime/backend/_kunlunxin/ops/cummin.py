import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@triton.jit
def tl_cummin(input, index, axis=0):
    return tl.associative_scan(
        (input, index), axis, tle.minimum_with_index_tie_break_right
    )


@triton.jit
def tl_min_tie_break_right(input, index, axis=None, keep_dims=False):
    return tl.reduce(
        (input, index),
        axis,
        tle.minimum_with_index_tie_break_right,
        keep_dims=keep_dims,
    )


@libentry()
@triton.jit(do_not_specialize=["n_elements"])
def add_base_min_kernel(
    out,
    out_indices,
    partial_min,
    partial_min_indices,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    out_ptrs = out + offset
    out_indices_ptrs = out_indices + offset
    out_vals = tl.load(out_ptrs, mask=mask)
    out_indices = tl.load(out_indices_ptrs, mask=mask)

    if pid > 0:
        partial_min_ptrs = partial_min + pid - 1
        last_part_min_via_min = tl.load(partial_min_ptrs)
        partial_min_indices_ptrs = partial_min_indices + pid - 1
        last_part_min_index_via_min = tl.load(partial_min_indices_ptrs)

        final_vals = tl.minimum(out_vals, last_part_min_via_min)
        final_indices = tl.where(
            out_vals <= last_part_min_via_min, out_indices, last_part_min_index_via_min
        )
        tl.store(out_ptrs, final_vals.to(out_vals.dtype), mask=mask)
        tl.store(out_indices_ptrs, final_indices, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["n_elements"])
def scan_part_min_kernel(
    inp,
    out,
    in_indices,
    out_indices,
    partial_min,
    partial_min_indices,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    inp_ptrs = inp + offset
    inp_vals = tl.load(inp_ptrs, mask=mask, other=float("inf"))
    if (
        tl.constexpr(inp_vals.dtype.is_int64())
        or tl.constexpr(inp_vals.dtype.is_uint64())
    ) or tl.constexpr(inp_vals.dtype.is_fp64()):
        inp_vals = inp_vals
    elif tl.constexpr(inp_vals.dtype.is_int()):
        inp_vals = inp_vals.to(tl.int32)
    else:
        inp_vals = inp_vals.to(tl.float32)
    in_indices_ptrs = out_indices + offset
    in_indices_vals = tl.load(in_indices_ptrs, mask=mask)
    result, cummin_indices = tl_cummin(inp_vals, in_indices_vals, axis=0)

    # tl.min do not support min_indices_tie_break_right
    part_min_via_min, part_min_indices_via_min = tl_min_tie_break_right(
        inp_vals, in_indices_vals, axis=0
    )

    out_ptrs = out + offset
    tl.store(out_ptrs, result, mask=mask)

    out_indices_ptrs = out_indices + offset
    tl.store(out_indices_ptrs, cummin_indices, mask=mask)

    partial_min_ptrs = partial_min + pid
    tl.store(partial_min_ptrs, part_min_via_min)

    partial_min_indices_ptrs = partial_min_indices + pid
    tl.store(partial_min_indices_ptrs, part_min_indices_via_min)


def scan_then_fan_col(inp, out, out_indices, n_ele, dtype):
    # TODO(all): tune on target board
    BLOCK_SIZE = 1024
    if n_ele <= 1024 * 4:
        BLOCK_SIZE = triton.next_power_of_2(n_ele)
    part_num = math.ceil(n_ele / BLOCK_SIZE)
    partial_min = torch.empty(part_num, dtype=dtype, device=inp.device)
    partial_min_indices = torch.empty(part_num, dtype=torch.int64, device=inp.device)

    grid = (part_num,)
    with torch_device_fn.device(inp.device):
        scan_part_min_kernel[grid](
            inp,
            out,
            out_indices,
            out_indices,
            partial_min,
            partial_min_indices,
            n_ele,
            BLOCK_SIZE,
        )

    if part_num >= 2:
        scan_then_fan_col(
            partial_min, partial_min, partial_min_indices, part_num, dtype
        )
        with torch_device_fn.device(inp.device):
            add_base_min_kernel[grid](
                out, out_indices, partial_min, partial_min_indices, n_ele, BLOCK_SIZE
            )


@libentry()
@triton.jit(do_not_specialize=["part_num"])
def scan_part_min_abc_kernel(
    inp,
    out,
    in_indices,
    out_indices,
    partial_min,
    partial_min_indices,
    B,
    C,
    part_num,
    BLOCK_SIZE: tl.constexpr,
):
    pid_a = tle.program_id(0)
    pid_b = tle.program_id(1)
    pid_c = tle.program_id(2)

    a_idx = pid_a
    b_idx = pid_b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    c_idx = pid_c

    offset = a_idx * B * C + b_idx * C + c_idx
    base_part_offset = a_idx * part_num * C + c_idx
    part_offset = base_part_offset + pid_b * C

    mask = b_idx < B
    inp_ptrs = inp + offset
    inp_vals = tl.load(inp_ptrs, mask=mask, other=float("inf"))
    if (
        tl.constexpr(inp_vals.dtype.is_int64())
        or tl.constexpr(inp_vals.dtype.is_uint64())
    ) or tl.constexpr(inp_vals.dtype.is_fp64()):
        inp_vals = inp_vals
    elif tl.constexpr(inp_vals.dtype.is_int()):
        inp_vals = inp_vals.to(tl.int32)
    else:
        inp_vals = inp_vals.to(tl.float32)
    indices_offset = a_idx * B * C + b_idx * C + c_idx
    in_indices_ptrs = out_indices + indices_offset
    in_indices_vals = tl.load(in_indices_ptrs, mask=mask)
    result, cummin_indices = tl_cummin(inp_vals, in_indices_vals, axis=0)

    # tl.min do not support min_indices_tie_break_right
    part_min_via_min, part_min_indices_via_min = tl_min_tie_break_right(
        inp_vals, in_indices_vals, axis=0
    )

    out_ptrs = out + offset
    tl.store(out_ptrs, result, mask=mask)

    out_indices_ptrs = out_indices + offset
    tl.store(out_indices_ptrs, cummin_indices, mask=mask)

    partial_min_ptrs = partial_min + part_offset
    tl.store(partial_min_ptrs, part_min_via_min)

    partial_min_indices_ptrs = partial_min_indices + part_offset
    tl.store(partial_min_indices_ptrs, part_min_indices_via_min)


@libentry()
@triton.jit(do_not_specialize=["part_num"])
def add_base_min_abc_kernel(
    out,
    out_indices,
    partial_min,
    partial_min_indices,
    B,
    C,
    part_num,
    BLOCK_SIZE: tl.constexpr,
):
    pid_a = tle.program_id(0)
    pid_b = tle.program_id(1)
    pid_c = tle.program_id(2)

    a_idx = pid_a
    b_idx = pid_b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    c_idx = pid_c

    base_offset = a_idx * B * C + c_idx
    offset = base_offset + b_idx * C
    base_part_offset = a_idx * part_num * C + c_idx
    last_part_offset = base_part_offset + (pid_b - 1) * C

    mask = b_idx < B
    out_ptrs = out + offset
    out_vals = tl.load(out_ptrs, mask=mask)
    out_indices_ptrs = out_indices + offset
    out_indices = tl.load(out_indices_ptrs, mask=mask)

    if pid_b > 0:
        partial_min_ptrs = partial_min + last_part_offset
        last_part_min_via_min = tl.load(partial_min_ptrs)
        partial_min_index_ptrs = partial_min_indices + last_part_offset
        last_part_min_index_via_min = tl.load(partial_min_index_ptrs)

        final_vals = tl.minimum(out_vals, last_part_min_via_min)
        final_indices = tl.where(
            out_vals <= last_part_min_via_min, out_indices, last_part_min_index_via_min
        )
        tl.store(out_ptrs, final_vals.to(out_vals.dtype), mask=mask)
        tl.store(out_indices_ptrs, final_indices, mask=mask)


def scan_then_fan(inp, out, out_indices, A, B, C, dtype):
    # TODO(all): tune on target board
    BLOCK_SIZE = 1024
    if B <= 1024 * 4:
        BLOCK_SIZE = triton.next_power_of_2(B)
    part_num = math.ceil(B / BLOCK_SIZE)
    partial_min = torch.empty(A, part_num, C, dtype=dtype, device=inp.device)
    partial_min_indices = torch.empty(
        A, part_num, C, dtype=torch.int64, device=inp.device
    )

    grid = (A, part_num, C)
    with torch_device_fn.device(inp.device):
        scan_part_min_abc_kernel[grid](
            inp,
            out,
            out_indices,
            out_indices,
            partial_min,
            partial_min_indices,
            B,
            C,
            part_num,
            BLOCK_SIZE,
        )

    if part_num >= 2:
        scan_then_fan(
            partial_min, partial_min, partial_min_indices, A, part_num, C, dtype
        )
        with torch_device_fn.device(inp.device):
            add_base_min_abc_kernel[grid](
                out,
                out_indices,
                partial_min,
                partial_min_indices,
                B,
                C,
                part_num,
                BLOCK_SIZE,
            )


def cummin(inp, dim=1, *, dtype=None):
    logger.debug("GEMS cummin")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    shape = inp.shape
    dim = dim % inp.ndim
    M = 1
    N = shape[dim]
    for i in range(dim):
        M *= shape[i]
    inp = inp.contiguous()
    K = inp.numel() // M // N

    if dtype is None:
        dtype = inp.dtype
        if dtype is torch.bool:
            dtype = torch.int64
    out = torch.empty_like(inp, dtype=dtype)

    compute_dtype = out.dtype
    if inp.dtype == torch.float16 or inp.dtype == torch.bfloat16:
        compute_dtype = torch.float32

    if M == 1 and K == 1:
        out_indices = torch.arange(N, device=inp.device).reshape(shape)
        scan_then_fan_col(inp, out, out_indices, N, compute_dtype)
    else:
        index_shape = [N if i == dim else 1 for i in range(inp.ndim)]
        repeat_factors = [1 if i == dim else shape[i] for i in range(inp.ndim)]
        out_indices = (
            torch.arange(N, device=inp.device)
            .reshape(index_shape)
            .repeat(repeat_factors)
        )
        scan_then_fan(inp, out, out_indices, M, N, K, compute_dtype)
    return out, out_indices
