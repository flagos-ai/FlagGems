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

import torch
import triton
import triton.language as tl
from torch._prims_common import is_boolean_dtype, is_integer_dtype

from flag_gems.runtime import device as runtime_device
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)
DEFAULT_BLOCK_SIZE = 1024
CUDA_SMALL_SCAN_LIMIT = 1024 * 4
ASCEND_SCAN_LIMIT = 1024


@tl.constexpr
def get_prod_accum_type(out_dtype: tl.dtype) -> tl.dtype:
    if out_dtype.is_bf16() or out_dtype.is_fp16():
        return tl.float32
    if out_dtype.is_int():
        return tl.int64
    return out_dtype


@triton.jit
def reduce_mul(a, b):
    return a * b


@libentry()
@triton.jit(do_not_specialize=["n_elements", "part_num"])
def scan_part_product_kernel(
    inp,
    out,
    partial_product,
    n_elements,
    part_num,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    acc_dtype: tl.constexpr = get_prod_accum_type(out.type.element_ty)
    inp_vals = tl.load(inp + offset, mask=mask, other=1).to(acc_dtype)
    result = tl.cumprod(inp_vals, axis=0)
    part_product = tl.reduce(inp_vals, axis=0, combine_fn=reduce_mul)

    tl.store(out + offset, result, mask=mask)
    tl.store(partial_product + pid, part_product)


@libentry()
@triton.jit(do_not_specialize=["n_elements", "part_num"])
def multiply_base_product_kernel(
    out,
    partial_product,
    n_elements,
    part_num,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    out_vals = tl.load(out + offset, mask=mask)

    if pid > 0:
        acc_dtype: tl.constexpr = get_prod_accum_type(out.type.element_ty)
        base_product = tl.load(partial_product + pid - 1).to(acc_dtype)
        final_vals = out_vals.to(acc_dtype) * base_product
        tl.store(out + offset, final_vals, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["part_num"])
def scan_part_product_abc_kernel(
    inp,
    out,
    partial_product,
    B,
    C,
    part_num,
    BLOCK_SIZE: tl.constexpr,
):
    pid_a = ext.program_id(0)
    pid_b = ext.program_id(1)
    pid_c = ext.program_id(2)

    a_idx = pid_a
    b_idx = pid_b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    c_idx = pid_c

    offset = a_idx * B * C + b_idx * C + c_idx
    base_part_offset = a_idx * part_num * C + c_idx
    part_offset = base_part_offset + pid_b * C
    mask = b_idx < B

    acc_dtype: tl.constexpr = get_prod_accum_type(out.type.element_ty)
    inp_vals = tl.load(inp + offset, mask=mask, other=1).to(acc_dtype)
    result = tl.cumprod(inp_vals, axis=0)
    part_product = tl.reduce(inp_vals, axis=0, combine_fn=reduce_mul)

    tl.store(out + offset, result, mask=mask)
    tl.store(partial_product + part_offset, part_product)


@libentry()
@triton.jit(do_not_specialize=["part_num"])
def multiply_base_product_abc_kernel(
    out,
    partial_product,
    B,
    C,
    part_num,
    BLOCK_SIZE: tl.constexpr,
):
    pid_a = ext.program_id(0)
    pid_b = ext.program_id(1)
    pid_c = ext.program_id(2)

    a_idx = pid_a
    b_idx = pid_b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    c_idx = pid_c

    offset = a_idx * B * C + b_idx * C + c_idx
    base_part_offset = a_idx * part_num * C + c_idx
    last_part_offset = base_part_offset + (pid_b - 1) * C
    mask = b_idx < B

    out_vals = tl.load(out + offset, mask=mask)

    if pid_b > 0:
        acc_dtype: tl.constexpr = get_prod_accum_type(out.type.element_ty)
        base_product = tl.load(partial_product + last_part_offset).to(acc_dtype)
        final_vals = out_vals.to(acc_dtype) * base_product
        tl.store(out + offset, final_vals, mask=mask)


def scan_then_fan_col(inp, out, n_ele, dtype):
    BLOCK_SIZE = _scan_block_size(n_ele)
    part_num = math.ceil(n_ele / BLOCK_SIZE)
    partial_product = torch.empty(part_num, dtype=dtype, device=inp.device)

    grid = (part_num,)
    with torch_device_fn.device(inp.device):
        scan_part_product_kernel[grid](
            inp, out, partial_product, n_ele, part_num, BLOCK_SIZE
        )

    if part_num >= 2:
        partial_prefix = torch.empty_like(partial_product)
        scan_then_fan_col(partial_product, partial_prefix, part_num, dtype)
        with torch_device_fn.device(inp.device):
            multiply_base_product_kernel[grid](
                out, partial_prefix, n_ele, part_num, BLOCK_SIZE
            )


def scan_then_fan(inp, out, A, B, C, dtype):
    BLOCK_SIZE = _scan_block_size(B)
    part_num = math.ceil(B / BLOCK_SIZE)
    partial_product = torch.empty(A, part_num, C, dtype=dtype, device=inp.device)

    grid = (A, part_num, C)
    with torch_device_fn.device(inp.device):
        scan_part_product_abc_kernel[grid](
            inp, out, partial_product, B, C, part_num, BLOCK_SIZE
        )

    if part_num >= 2:
        partial_prefix = torch.empty_like(partial_product)
        scan_then_fan(partial_product, partial_prefix, A, part_num, C, dtype)
        with torch_device_fn.device(inp.device):
            multiply_base_product_abc_kernel[grid](
                out, partial_prefix, B, C, part_num, BLOCK_SIZE
            )


def _get_output_dtype(inp, dtype):
    if dtype is not None:
        return dtype
    if is_integer_dtype(inp.dtype) or is_boolean_dtype(inp.dtype):
        return torch.int64
    return inp.dtype


def _get_compute_dtype(dtype):
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    if is_integer_dtype(dtype) or is_boolean_dtype(dtype):
        return torch.int64
    return dtype


def _should_redispatch_on_ascend(dtype):
    return runtime_device.vendor_name == "ascend" and (
        is_integer_dtype(dtype) or is_boolean_dtype(dtype)
    )


def _scan_block_size(length):
    limit = (
        ASCEND_SCAN_LIMIT
        if runtime_device.vendor_name == "ascend"
        else CUDA_SMALL_SCAN_LIMIT
    )
    if length <= limit:
        return triton.next_power_of_2(length)
    return DEFAULT_BLOCK_SIZE


def cumprod_wrapper(inp, dim, dtype=None, out=None):
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    dim = dim % inp.ndim
    out_dtype = _get_output_dtype(inp, dtype)

    inp = inp.contiguous()
    if out is None:
        out = torch.empty_like(inp, dtype=out_dtype)

    if inp.numel() == 0:
        return out

    shape = inp.shape
    M = math.prod(shape[:dim])
    N = shape[dim]
    K = inp.numel() // M // N
    compute_dtype = _get_compute_dtype(out.dtype)

    if K == 1:
        reduce_then_scan_row(inp, out, M, N, compute_dtype)
    else:
        scan_then_fan(inp, out, M, N, K, compute_dtype)

    return out


_TL_SCAN_DTYPES = {
    torch.float32: tl.float32,
    torch.float64: tl.float64,
    torch.int64: tl.int64,
    torch.int32: tl.int32,
}


@libentry()
@triton.jit(do_not_specialize=["N"])
def cumprod_row_scan_chunk_kernel(
    inp_ptr,
    out_ptr,
    N,
    ACC_DTYPE: tl.constexpr,
    BN: tl.constexpr,
    NEED_TAIL: tl.constexpr,
):
    """Per-row chunked online product scan (K == 1, N > 16384).

    One program per row; BN-wide chunks chained inside the program by a
    BN-wide carry vector (carry starts at the product identity 1). The scan
    must live inside a loop that tiles the scan axis: this is the only
    tl.cumprod pattern this XPU backend lowers correctly (a standalone scan is
    mis-lowered / rejected with an XDNN dtype error)."""
    pid = tl.program_id(0)
    row_offset = pid * N
    carry = tl.full([BN], value=1, dtype=ACC_DTYPE)
    for start in range(0, N, BN):
        n_offsets = start + tl.arange(0, BN)
        if NEED_TAIL:
            mask = n_offsets < N
            x = tl.load(inp_ptr + row_offset + n_offsets, mask=mask, other=1).to(
                ACC_DTYPE
            )
        else:
            x = tl.load(inp_ptr + row_offset + n_offsets).to(ACC_DTYPE)
        r = tl.cumprod(x, axis=0) * carry
        carry *= tl.reduce(x, axis=0, combine_fn=reduce_mul)
        if NEED_TAIL:
            tl.store(out_ptr + row_offset + n_offsets, r, mask=mask)
        else:
            tl.store(out_ptr + row_offset + n_offsets, r)


def reduce_then_scan_row(x, out, M, N, compute_dtype):
    persistent_limit = (
        ASCEND_SCAN_LIMIT if runtime_device.vendor_name == "ascend" else 16384
    )
    if N <= persistent_limit:
        TILE_SIZE = triton.next_power_of_2(N)
        num_warps = 8 if TILE_SIZE > 2048 else 4
        reduce_then_scan_root_scan_kernel_row[(M, 1, 1)](
            x, out, N, TILE_SIZE, num_warps=num_warps
        )
        return out

    # N > persistent_limit: per-row chunked online scan. The scan runs in
    # ACC_DTYPE and the result is packed back into `out` (which may be a
    # narrower dtype, e.g. in-place cumprod_ on int16) by a torch-level
    # convert + copy; XPU rejects stores whose value type is wider than the
    # pointer type (XDNN "data type not matched").
    acc_tl = _TL_SCAN_DTYPES.get(compute_dtype, tl.float32)
    BN = 32768 if compute_dtype == torch.float32 else 16384
    need_tail = 1 if N % BN else 0
    scan_out = torch.empty((M, N), dtype=compute_dtype, device=x.device)
    grid = (M,)
    with torch_device_fn.device(x.device):
        cumprod_row_scan_chunk_kernel[grid](
            x,
            scan_out,
            N,
            ACC_DTYPE=acc_tl,
            BN=BN,
            NEED_TAIL=need_tail,
            num_warps=8,
            buffer_size_limit=2048,
        )
    if scan_out.dtype == out.dtype:
        torch.ops.aten._copy_from(scan_out, out, False)
    else:
        torch.ops.aten._copy_from(scan_out.to(out.dtype), out, False)
    return out


@triton.jit
def reduce_then_scan_root_scan_kernel_row(in_ptr, out_ptr, N, TILE_SIZE: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, TILE_SIZE)
    mask = offsets < N
    acc_dtype: tl.constexpr = get_prod_accum_type(out_ptr.type.element_ty)
    x = tl.load(in_ptr + pid * N + offsets, mask=mask, other=1).to(acc_dtype)
    out = tl.cumprod(x, 0)
    tl.store(out_ptr + pid * N + offsets, out, mask=mask)


def cumprod(inp, dim, *, dtype=None):
    logger.debug("GEMS_KUNLUNXIN CUMPROD")
    out_dtype = _get_output_dtype(inp, dtype)
    if is_boolean_dtype(inp.dtype):
        if is_boolean_dtype(out_dtype):
            return torch.ops.aten.cumprod.default.redispatch(
                _FALLBACK_KEYSET, inp, dim, dtype=dtype
            )
        uint8_inp = inp.to(torch.uint8)
        if runtime_device.vendor_name == "ascend":
            return torch.ops.aten.cumprod.default.redispatch(
                _FALLBACK_KEYSET, uint8_inp, dim, dtype=dtype
            )
        return cumprod_wrapper(uint8_inp, dim, out_dtype)
    if _should_redispatch_on_ascend(out_dtype):
        return torch.ops.aten.cumprod.default.redispatch(
            _FALLBACK_KEYSET, inp, dim, dtype=dtype
        )
    return cumprod_wrapper(inp, dim, dtype)


def cumprod_(inp, dim, *, dtype=None):
    logger.debug("GEMS_KUNLUNXIN CUMPROD_")
    if dtype is not None and dtype != inp.dtype:
        raise RuntimeError(
            "Bad in-place call: input tensor dtype and output tensor dtype should match"
        )
    if is_boolean_dtype(inp.dtype):
        raise NotImplementedError(
            "In-place cumprod is not supported for boolean tensors"
        )
    if _should_redispatch_on_ascend(inp.dtype):
        return torch.ops.aten.cumprod_.default.redispatch(
            _FALLBACK_KEYSET, inp, dim, dtype=dtype
        )
    if inp.numel() == 0:
        return inp
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    if inp.size(dim % inp.ndim) == 1:
        # Inclusive prefix product over an axis of length 1 is the identity,
        # so the in-place op needs no work at all.
        return inp
    if inp.is_contiguous():
        # The aliasing is safe: every program loads its own chunk before
        # storing to it (load-before-store within a program), and the
        # multi-pass tiers (scan -> fan multiply) are separated by kernel
        # boundaries. This avoids an extra empty_like allocation plus a full
        # device-to-device copy on top of the scan. `cumprod_wrapper` calls
        # `.contiguous()` internally, which is a no-op here.
        cumprod_wrapper(inp, dim, inp.dtype, out=inp)
    else:
        # Non-contiguous self: scan the contiguous copy, then write back with
        # the native strided-copy engine (`_copy_from` is not overridden by
        # flag_gems, so this avoids recursing into the gems `copy_`).
        result = cumprod_wrapper(inp, dim, inp.dtype)
        torch.ops.aten._copy_from(result, inp, False)
    return inp
