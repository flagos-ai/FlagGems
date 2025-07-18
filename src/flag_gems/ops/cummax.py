import logging
import math
from typing import List, Tuple, Union

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.limits import get_dtype_min

Tensor = torch.Tensor

logger = logging.getLogger(__name__)


@triton.jit
def tl_cummax(input, index, axis=0):
    return tl.associative_scan(
        (input, index), axis, tle.maximum_with_index_tie_break_right
    )


@triton.jit
def tl_max_tie_break_right(input, index, axis=None, keep_dims=False):
    return tl.reduce(
        (input, index),
        axis,
        tle.maximum_with_index_tie_break_right,
        keep_dims=keep_dims,
    )


@libentry()
@triton.jit(do_not_specialize=["n_elements"])
def add_base_max_kernel(
    out,
    out_indices,
    partial_max,
    partial_max_indices,
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
        partial_max_ptrs = partial_max + pid - 1
        last_part_max_via_max = tl.load(partial_max_ptrs)
        partial_max_indices_ptrs = partial_max_indices + pid - 1
        last_part_max_index_via_max = tl.load(partial_max_indices_ptrs)

        final_vals = tl.maximum(out_vals, last_part_max_via_max)
        final_indices = tl.where(
            out_vals >= last_part_max_via_max, out_indices, last_part_max_index_via_max
        )
        tl.store(out_ptrs, final_vals.to(out_vals.dtype), mask=mask)
        tl.store(out_indices_ptrs, final_indices, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["n_elements"])
def scan_part_max_kernel(
    inp,
    out,
    in_indices,
    out_indices,
    partial_max,
    partial_max_indices,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    NEED_PARTIAL: tl.constexpr,
    USE_OUT_INDICES: tl.constexpr,
):
    pid = tle.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    min_value = get_dtype_min(inp.type.element_ty)
    inp_ptrs = inp + offset
    inp_vals = tl.load(inp_ptrs, mask=mask, other=min_value)
    if (
        tl.constexpr(inp_vals.dtype.is_int64())
        or tl.constexpr(inp_vals.dtype.is_uint64())
    ) or tl.constexpr(inp_vals.dtype.is_fp64()):
        inp_vals = inp_vals
    elif tl.constexpr(inp_vals.dtype.is_int()):
        inp_vals = inp_vals.to(tl.int32)
    else:
        inp_vals = inp_vals.to(tl.float32)
    if tl.constexpr(USE_OUT_INDICES):
        in_indices_ptrs = out_indices + offset
        in_indices_vals = tl.load(in_indices_ptrs, mask=mask)
    else:
        in_indices_vals = offset
    result, cummax_indices = tl_cummax(inp_vals, in_indices_vals, axis=0)

    if tl.constexpr(NEED_PARTIAL):
        # tl.max do not support max_indices_tie_break_right
        part_max_via_max, part_max_indices_via_max = tl_max_tie_break_right(
            inp_vals, in_indices_vals, axis=0
        )

    out_ptrs = out + offset
    tl.store(out_ptrs, result, mask=mask)

    out_indices_ptrs = out_indices + offset
    tl.store(out_indices_ptrs, cummax_indices, mask=mask)

    if tl.constexpr(NEED_PARTIAL):
        partial_max_ptrs = partial_max + pid
        tl.store(partial_max_ptrs, part_max_via_max)

        partial_max_indices_ptrs = partial_max_indices + pid
        tl.store(partial_max_indices_ptrs, part_max_indices_via_max)


def scan_then_fan_col(inp, out, out_indices, n_ele, dtype, use_out_indices=False):
    # TODO(all): tune on target board
    BLOCK_SIZE = 1024
    if n_ele <= 1024 * 4:
        BLOCK_SIZE = triton.next_power_of_2(n_ele)
    part_num = math.ceil(n_ele / BLOCK_SIZE)
    need_partial = True if part_num >= 2 else False
    if need_partial:
        partial_max = torch.empty(part_num, dtype=dtype, device=inp.device)
        partial_max_indices = torch.empty(
            part_num, dtype=torch.int64, device=inp.device
        )
    else:
        partial_max = None
        partial_max_indices = None

    grid = (part_num,)
    with torch_device_fn.device(inp.device):
        scan_part_max_kernel[grid](
            inp,
            out,
            out_indices,
            out_indices,
            partial_max,
            partial_max_indices,
            n_ele,
            BLOCK_SIZE,
            need_partial,
            use_out_indices,
        )

    if part_num >= 2:
        scan_then_fan_col(
            partial_max,
            partial_max,
            partial_max_indices,
            part_num,
            dtype,
            use_out_indices=True,
        )
        with torch_device_fn.device(inp.device):
            add_base_max_kernel[grid](
                out, out_indices, partial_max, partial_max_indices, n_ele, BLOCK_SIZE
            )


@libentry()
@triton.jit(do_not_specialize=["part_num"])
def scan_part_max_abc_kernel(
    inp,
    out,
    in_indices,
    out_indices,
    partial_max,
    partial_max_indices,
    B,
    C,
    part_num,
    BLOCK_SIZE: tl.constexpr,
    NEED_PARTIAL: tl.constexpr,
    USE_OUT_INDICES: tl.constexpr,
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
    min_value = get_dtype_min(inp.type.element_ty)
    inp_vals = tl.load(inp_ptrs, mask=mask, other=min_value)
    if (
        tl.constexpr(inp_vals.dtype.is_int64())
        or tl.constexpr(inp_vals.dtype.is_uint64())
    ) or tl.constexpr(inp_vals.dtype.is_fp64()):
        inp_vals = inp_vals
    elif tl.constexpr(inp_vals.dtype.is_int()):
        inp_vals = inp_vals.to(tl.int32)
    else:
        inp_vals = inp_vals.to(tl.float32)
    if tl.constexpr(USE_OUT_INDICES):
        in_indices_ptrs = out_indices + offset
        in_indices_vals = tl.load(in_indices_ptrs, mask=mask)
    else:
        in_indices_vals = b_idx
    result, cummax_indices = tl_cummax(inp_vals, in_indices_vals, axis=0)

    if tl.constexpr(NEED_PARTIAL):
        # tl.max do not support max_indices_tie_break_right
        part_max_via_max, part_max_indices_via_max = tl_max_tie_break_right(
            inp_vals, in_indices_vals, axis=0
        )

    out_ptrs = out + offset
    tl.store(out_ptrs, result, mask=mask)

    out_indices_ptrs = out_indices + offset
    tl.store(out_indices_ptrs, cummax_indices, mask=mask)

    if tl.constexpr(NEED_PARTIAL):
        partial_max_ptrs = partial_max + part_offset
        tl.store(partial_max_ptrs, part_max_via_max)

        partial_max_indices_ptrs = partial_max_indices + part_offset
        tl.store(partial_max_indices_ptrs, part_max_indices_via_max)


@libentry()
@triton.jit(do_not_specialize=["part_num"])
def add_base_max_abc_kernel(
    out,
    out_indices,
    partial_max,
    partial_max_indices,
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
        partial_max_ptrs = partial_max + last_part_offset
        last_part_max_via_max = tl.load(partial_max_ptrs)
        partial_max_index_ptrs = partial_max_indices + last_part_offset
        last_part_max_index_via_max = tl.load(partial_max_index_ptrs)

        final_vals = tl.maximum(out_vals, last_part_max_via_max)
        final_indices = tl.where(
            out_vals >= last_part_max_via_max, out_indices, last_part_max_index_via_max
        )
        tl.store(out_ptrs, final_vals.to(out_vals.dtype), mask=mask)
        tl.store(out_indices_ptrs, final_indices, mask=mask)


def scan_then_fan(inp, out, out_indices, A, B, C, dtype, use_out_indices=False):
    # TODO(all): tune on target board
    BLOCK_SIZE = 1024
    if B <= 1024 * 4:
        BLOCK_SIZE = triton.next_power_of_2(B)
    part_num = math.ceil(B / BLOCK_SIZE)
    need_partial = True if part_num >= 2 else False
    if need_partial:
        partial_max = torch.empty(A, part_num, C, dtype=dtype, device=inp.device)
        partial_max_indices = torch.empty(
            A, part_num, C, dtype=torch.int64, device=inp.device
        )
    else:
        partial_max = None
        partial_max_indices = None

    grid = (A, part_num, C)
    with torch_device_fn.device(inp.device):
        scan_part_max_abc_kernel[grid](
            inp,
            out,
            out_indices,
            out_indices,
            partial_max,
            partial_max_indices,
            B,
            C,
            part_num,
            BLOCK_SIZE,
            need_partial,
            use_out_indices,
        )

    if part_num >= 2:
        scan_then_fan(
            partial_max,
            partial_max,
            partial_max_indices,
            A,
            part_num,
            C,
            dtype,
            use_out_indices=True,
        )
        with torch_device_fn.device(inp.device):
            add_base_max_abc_kernel[grid](
                out,
                out_indices,
                partial_max,
                partial_max_indices,
                B,
                C,
                part_num,
                BLOCK_SIZE,
            )


@libentry()
@triton.jit()
def scan_part_max_abc_loop_kernel(
    inp,
    out,
    out_indices,
    B,
    C,
    loop_num,
    BLOCK_SIZE: tl.constexpr,
):
    pid_a = tle.program_id(0)
    pid_c = tle.program_id(1)

    a_idx = pid_a
    c_idx = pid_c
    t_idx = tl.arange(0, BLOCK_SIZE)
    ac_offset = a_idx * B * C + c_idx

    # init, promote low precision types
    min_value = get_dtype_min(inp.type.element_ty)
    if tl.constexpr(inp.type.element_ty.is_fp16()) or tl.constexpr(
        inp.type.element_ty.is_bf16()
    ):
        compute_dtype = tl.float32
    elif tl.constexpr(inp.type.element_ty.is_int8()) or tl.constexpr(
        inp.type.element_ty.is_int16()
    ):
        compute_dtype = tl.int32
    else:
        compute_dtype = inp.type.element_ty

    prev_max_val = tl.full([], min_value, dtype=compute_dtype)
    prev_max_val_idx = tl.full([], 0, dtype=tl.int64)
    last_mask = t_idx == (BLOCK_SIZE - 1)

    for l_idx in tl.range(loop_num):
        b_idx = l_idx * BLOCK_SIZE + t_idx
        mask = b_idx < B
        offset = ac_offset + b_idx * C

        inp_vals = tl.load(inp + offset, mask=mask, other=min_value)
        # Only promote if necessary
        if tl.constexpr(compute_dtype != inp.type.element_ty):
            vals = inp_vals.to(compute_dtype)
        else:
            vals = inp_vals
        idxs = b_idx

        # cummax
        result, cummax_indices = tl_cummax(vals, idxs, axis=0)

        # broadcast
        prev_max_val_b = tl.broadcast_to(prev_max_val, (BLOCK_SIZE,))
        prev_max_val_idx_b = tl.broadcast_to(prev_max_val_idx, (BLOCK_SIZE,))

        # Handle NaN and tie-breaking logic
        if tl.constexpr(compute_dtype.is_floating()):
            # For floats: handle NaN propagation + tie-break right
            prev_is_nan = prev_max_val != prev_max_val
            result_is_nan = result != result
            prev_nan_mask = tl.broadcast_to(prev_is_nan, (BLOCK_SIZE,))

            use_result = result_is_nan | (~prev_nan_mask & (result >= prev_max_val_b))
        else:
            # For integers: simple tie-break right
            use_result = result >= prev_max_val_b

        final_vals = tl.where(use_result, result, prev_max_val_b)
        final_indices = tl.where(use_result, cummax_indices, prev_max_val_idx_b)

        # update global max val and idx
        prev_max_val = tl.sum(tl.where(last_mask, final_vals, 0), axis=0)
        prev_max_val_idx = tl.sum(tl.where(last_mask, final_indices, 0), axis=0)

        # store result
        tl.store(out + offset, final_vals.to(out.type.element_ty), mask=mask)
        tl.store(out_indices + offset, final_indices, mask=mask)


def scan_then_fan_loop(inp, out, out_indices, A, B, C, dtype):
    # TODO(all): tune on target board
    BLOCK_SIZE = 1024
    if B < 1024 * 4:
        BLOCK_SIZE = triton.next_power_of_2(B)
    loop_num = math.ceil(B / BLOCK_SIZE)

    grid = (A, C)
    with torch_device_fn.device(inp.device):
        scan_part_max_abc_loop_kernel[grid](
            inp,
            out,
            out_indices,
            B,
            C,
            loop_num,
            BLOCK_SIZE,
        )


def cummax(
    input: Tensor,
    dim: int,
    *,
    out: Union[Tensor, Tuple[Tensor, ...], List[Tensor], None] = None,
) -> torch.return_types.cummax:
    logger.debug("GEMS cummax")
    assert dim >= -input.ndim and dim < input.ndim, "Invalid dim"
    shape = input.shape
    dim = dim % input.ndim
    M = 1
    N = shape[dim]
    for i in range(dim):
        M *= shape[i]
    input = input.contiguous()
    K = input.numel() // M // N

    dtype = input.dtype
    if dtype is torch.bool:
        dtype = torch.int64
    out = torch.empty_like(input, dtype=dtype)
    out_indices = torch.empty_like(input, dtype=torch.int64)

    compute_dtype = out.dtype
    if input.dtype == torch.float16 or input.dtype == torch.bfloat16:
        compute_dtype = torch.float32

    if M == 1 and K == 1:
        scan_then_fan_col(input, out, out_indices, N, compute_dtype)
    elif M * K <= 16:
        scan_then_fan(input, out, out_indices, M, N, K, compute_dtype)
    else:
        scan_then_fan_loop(input, out, out_indices, M, N, K, compute_dtype)
    return out, out_indices
