# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
import os
from typing import List, Tuple, Union

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.limits import get_dtype_min

from ..utils.tie_break import maximum_with_index_tie_break_right

Tensor = torch.Tensor

logger = logging.getLogger(__name__)


@triton.jit
def tl_cummax(input, index, axis=0):
    return tl.associative_scan((input, index), axis, maximum_with_index_tie_break_right)


@triton.jit
def tl_max_tie_break_right(input, index, axis=None, keep_dims=False):
    return tl.reduce(
        (input, index),
        axis,
        maximum_with_index_tie_break_right,
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
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    out_ptrs = out + offset
    out_indices_ptrs = out_indices + offset
    out_vals = tl.load(out_ptrs, mask=mask)
    out_indices = tl.load(out_indices_ptrs, mask=mask)

    if pid > 0:
        partial_max_ptrs = partial_max + pid - 1
        prev_val = tl.load(partial_max_ptrs)
        partial_max_indices_ptrs = partial_max_indices + pid - 1
        prev_idx = tl.load(partial_max_indices_ptrs)

        # Use the tie-break barrier rather than tl.maximum: IEEE maximum()
        # returns the non-NaN operand, so a NaN in an earlier part would be
        # dropped instead of propagating (torch.cummax keeps it).
        final_vals, final_indices = maximum_with_index_tie_break_right(
            out_vals, out_indices, prev_val, prev_idx
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
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    min_value = get_dtype_min(inp.type.element_ty)
    inp_ptrs = inp + offset
    inp_vals = tl.load(inp_ptrs, mask=mask, other=min_value)
    if (
        tl.constexpr(inp_vals.dtype.is_int64())
        | tl.constexpr(inp_vals.dtype.is_uint64())
    ) | tl.constexpr(inp_vals.dtype.is_fp64()):
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
    pid_a = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_c = tl.program_id(2)

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
        | tl.constexpr(inp_vals.dtype.is_uint64())
    ) | tl.constexpr(inp_vals.dtype.is_fp64()):
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
    pid_a = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_c = tl.program_id(2)

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
        prev_val = tl.load(partial_max_ptrs)
        partial_max_index_ptrs = partial_max_indices + last_part_offset
        prev_idx = tl.load(partial_max_index_ptrs)

        # See add_base_max_kernel: NaN must survive the cross-part merge.
        final_vals, final_indices = maximum_with_index_tie_break_right(
            out_vals, out_indices, prev_val, prev_idx
        )
        tl.store(out_ptrs, final_vals.to(out_vals.dtype), mask=mask)
        tl.store(out_indices_ptrs, final_indices, mask=mask)


def scan_then_fan(inp, out, out_indices, A, B, C, dtype, use_out_indices=False):
    # TODO(all): tune on target board
    BLOCK_SIZE = 512
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
    A,
    B,
    C,
    loop_num,
    BLOCK_SIZE: tl.constexpr,
):
    num_a = tl.num_programs(0)
    num_c = tl.num_programs(1)

    t_idx = tl.arange(0, BLOCK_SIZE)
    last_mask = t_idx == (BLOCK_SIZE - 1)
    min_value = get_dtype_min(inp.type.element_ty)

    # Grid-stride over the (A, C) planes: GCU caps grid.x at 65535 and grid.y
    # at 255, so a single launch may only cover a subset of them.
    for a_idx in tl.range(tl.program_id(0), A, num_a):
        for c_idx in tl.range(tl.program_id(1), C, num_c):
            ac_offset = a_idx * B * C + c_idx

            prev_max_val = tl.full([], min_value, dtype=tl.float32)
            prev_max_idx = tl.full([], 0, dtype=tl.int32)

            for l_idx in tl.range(loop_num):
                b_idx = l_idx * BLOCK_SIZE + t_idx
                mask = b_idx < B
                offset = ac_offset + b_idx * C

                inp_vals = tl.load(inp + offset, mask=mask, other=min_value)
                if (
                    tl.constexpr(inp_vals.dtype.is_int64())
                    | tl.constexpr(inp_vals.dtype.is_uint64())
                ) | tl.constexpr(inp_vals.dtype.is_fp64()):
                    vals = inp_vals
                elif tl.constexpr(inp_vals.dtype.is_int()):
                    vals = inp_vals.to(tl.int32)
                else:
                    vals = inp_vals.to(tl.float32)
                idxs = b_idx

                result, cummax_indices = tl_cummax(vals, idxs, axis=0)

                prev_max_val_b = tl.broadcast_to(prev_max_val, (BLOCK_SIZE,))
                prev_max_idx_b = tl.broadcast_to(prev_max_idx, (BLOCK_SIZE,))

                # tl.maximum() returns the non-NaN operand; carry NaN forward
                # explicitly so it keeps propagating like torch.cummax.
                if tl.constexpr(vals.dtype.is_floating()):
                    prev_is_nan = prev_max_val != prev_max_val
                    result_is_nan = result != result
                    prev_nan_mask = tl.broadcast_to(prev_is_nan, (BLOCK_SIZE,))
                    use_result = result_is_nan | (
                        (~prev_nan_mask) & (result >= prev_max_val_b)
                    )
                else:
                    use_result = result >= prev_max_val_b

                final_vals = tl.where(use_result, result, prev_max_val_b)
                final_idx = tl.where(use_result, cummax_indices, prev_max_idx_b)
                prev_max_val = tl.sum(tl.where(last_mask, final_vals, 0.0), axis=0)
                prev_max_idx = tl.sum(tl.where(last_mask, final_idx, 0), axis=0)

                tl.store(out + offset, final_vals.to(out.type.element_ty), mask=mask)
                tl.store(out_indices + offset, final_idx, mask=mask)


@triton.jit
def _max_i64_key(a, b):
    return tl.maximum(a, b)


@libentry()
@triton.jit()
def scan_part_max_abc_loop_key_kernel(
    inp,
    out,
    out_indices,
    A,
    B,
    C,
    loop_num,
    BLOCK: tl.constexpr,
    FLOAT_INPUT: tl.constexpr,
):
    # GCU400: tl.associative_scan with a tuple combine lowers to a ~140ns/element
    # scalar chain, ~25x slower than the same-width single-value int64 scan.
    # Encode (value, index) into one signed int64 key so the scan runs on the
    # fast single-value path:  key = (code_biased << 32) | position, where
    # code_biased orders exactly like the input values (floats via a monotone
    # bit mapping with NaN pinned above +inf; -0.0 normalized to +0.0 so ties
    # break to the later index like torch.cummax). Larger keys win, so equal
    # values resolve to the later position automatically.
    num_a = tl.num_programs(0)
    num_c = tl.num_programs(1)

    t_idx = tl.arange(0, BLOCK)
    min_value = get_dtype_min(inp.type.element_ty)

    # Grid-stride over the (A, C) planes: GCU caps grid.x at 65535 and grid.y
    # at 255, so a single launch may only cover a subset of them.
    for a_idx in tl.range(tl.program_id(0), A, num_a):
        for c_idx in tl.range(tl.program_id(1), C, num_c):
            ac_offset = a_idx.to(tl.int64) * B * C + c_idx

            carry = tl.full([], -(1 << 63), dtype=tl.int64)

            for l_idx in tl.range(loop_num):
                b_idx = l_idx * BLOCK + t_idx
                mask = b_idx < B
                offset = ac_offset + b_idx * C

                inp_vals = tl.load(inp + offset, mask=mask, other=min_value)
                if tl.constexpr(FLOAT_INPUT):
                    vals = inp_vals.to(tl.float32)
                    # torch.cummax treats -0.0 == +0.0 with later-index
                    # tie-break; give both zeros the same ordering code.
                    vals = tl.where(vals == 0, 0.0, vals)
                    bits = vals.to(tl.int32, bitcast=True).to(tl.int64)
                    code = tl.where(
                        vals != vals,
                        2147483647,
                        tl.where(bits >= 0, bits, (~bits) - 2147483648),
                    )
                else:
                    code = inp_vals.to(tl.int32).to(tl.int64)
                # Masked lanes keep the minimum code and index 0 so they lose
                # every tie against real data (protects the chunk carry).
                pos = tl.where(mask, b_idx, 0).to(tl.int64)
                key = (code << 32) | pos

                r = tl.associative_scan(key, 0, _max_i64_key)
                r = tl.maximum(r, carry)
                carry = tl.max(r, axis=0)

                rk = (r >> 32).to(tl.int32)
                rb = (r & 0xFFFFFFFF).to(tl.int32)
                if tl.constexpr(FLOAT_INPUT):
                    u_bits = tl.where(rk < 0, (~rk) ^ (-2147483648), rk)
                    fv = u_bits.to(tl.float32, bitcast=True)
                else:
                    fv = rk

                tl.store(out + offset, fv.to(out.type.element_ty), mask=mask)
                tl.store(out_indices + offset, rb, mask=mask)


def scan_then_fan_loop(inp, out, out_indices, A, B, C, dtype):
    BLOCK_SIZE = 4096
    if B <= 4096:
        BLOCK_SIZE = triton.next_power_of_2(B)
    loop_num = math.ceil(B / BLOCK_SIZE)

    # GCU caps grid.x at 65535 and grid.y at 255; the kernel strides over the
    # remaining planes, so clamping the launch is always safe.
    grid = (min(A, 65535), min(C, 255))
    with torch_device_fn.device(inp.device):
        if inp.dtype in (torch.int64, torch.uint64, torch.float64):
            # 64-bit inputs cannot be packed into the int64 key; keep the
            # original tuple-scan kernel for them.
            scan_part_max_abc_loop_kernel[grid](
                inp,
                out,
                out_indices,
                A,
                B,
                C,
                loop_num,
                BLOCK_SIZE,
                num_warps=1,
            )
        else:
            float_input = inp.dtype in (torch.float16, torch.bfloat16, torch.float32)
            scan_part_max_abc_loop_key_kernel[grid](
                inp,
                out,
                out_indices,
                A,
                B,
                C,
                loop_num,
                BLOCK_SIZE,
                float_input,
                num_warps=1,
            )


def cummax(
    input: Tensor,
    dim: int,
    *,
    out: Union[Tensor, Tuple[Tensor, ...], List[Tensor], None] = None,
) -> torch.return_types.cummax:
    logger.debug("GEMS_ENFLAME CUMMAX")
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
    out_indices = torch.empty_like(input, dtype=torch.int32)

    compute_dtype = out.dtype
    if input.dtype == torch.float16 or input.dtype == torch.bfloat16:
        compute_dtype = torch.float32

    if M == 1 and K == 1:
        scan_then_fan_col(input, out, out_indices, N, compute_dtype)
    elif M * K <= 16:
        scan_then_fan(input, out, out_indices, M, N, K, compute_dtype)
    else:
        scan_then_fan_loop(input, out, out_indices, M, N, K, compute_dtype)
    return out, out_indices.to(torch.int32).to(torch.int64)


# ---------------------------------------------------------------------------
# aten::cummax dispatch guard (torch_gcu P21 defect workaround)
#
# torch_gcu's native cummax follows the IEEE maximum() semantics and returns
# the non-NaN operand, so a NaN anywhere in the scanned prefix is silently
# dropped instead of propagating like the CPU/CUDA reference semantics
# (compiler_problems P21, flaggems_cummax_aten_nan_drop.md). The accuracy
# tests' default reference leg runs torch.cummax on the GCU tensor, so the
# broken native result poisons the comparison. Route the aten::cummax
# dispatch for this backend's key to this (NaN-propagating) implementation,
# mirroring the kthvalue / resize_as_sparse_ in-repo guards: idempotent via a
# torch-attached flag, double-name import safe, and the registrar's later
# use_gems()/enable() registration of the same op is absorbed as a warning by
# op_registrar.for_each (same kernel either way).
#
# Dormant under the Long->Int replacement regime (env off): this kernel's
# index packing needs the true int64 layout, and native cummax at least keeps
# running there (garbage indices, but no int64-load abort), so the guard only
# installs when the int64 layout is active (same _gcu_int64_indices gate the
# kthvalue guard uses).
# ---------------------------------------------------------------------------
_CUMMAX_GUARD_FLAG = "_flag_gems_gcu400_cummax_aten_guard_registered"
_CUMMAX_GUARD_LIBRARY = None


def _gcu_int64_indices():
    # torch_gcu reads an int64 tensor with a true 8-byte element stride only
    # when TORCH_GCU_ENABLE_INT64_AND_UINT64 selects the int64 layout. Accepted
    # values mirror torch_gcu's parse (see the kthvalue guard's copy): "1",
    # "true", and nonzero numerics enable int64; unset / "0" / "false" keep
    # the Long->Int replacement.
    val = os.environ.get("TORCH_GCU_ENABLE_INT64_AND_UINT64")
    if val is None:
        return False
    if val == "true":
        return True
    try:
        return int(val) != 0
    except ValueError:
        return False


def _install_native_cummax_guard():
    global _CUMMAX_GUARD_LIBRARY
    if not _gcu_int64_indices():
        return
    if getattr(torch, _CUMMAX_GUARD_FLAG, False):
        return
    library = torch.library.Library("aten", "IMPL")
    try:
        library.impl("cummax", cummax, runtime.device.dispatch_key)
    except RuntimeError as error:
        logger.warning(
            "cummax %s-key registration skipped: %s",
            runtime.device.dispatch_key,
            error,
        )
        return
    _CUMMAX_GUARD_LIBRARY = library
    setattr(torch, _CUMMAX_GUARD_FLAG, True)


_install_native_cummax_guard()
