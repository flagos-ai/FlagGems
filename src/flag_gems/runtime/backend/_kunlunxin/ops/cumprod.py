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

    if not inp.is_contiguous():
        if out is None:
            out = torch.empty_like(inp, dtype=out_dtype)
        if inp.numel() == 0:
            return out
        compute_dtype = _get_compute_dtype(out.dtype)
        return _strided_scan(inp, out, dim, compute_dtype)

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
            tl.store(
                out_ptr + row_offset + n_offsets,
                r.to(out_ptr.type.element_ty),
                mask=mask,
            )
        else:
            tl.store(
                out_ptr + row_offset + n_offsets,
                r.to(out_ptr.type.element_ty),
            )


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
    # ACC_DTYPE and is cast back to the output pointee type in-kernel (the
    # store value type then matches the pointer type, so no torch-level
    # convert + `_copy_from` pass is needed).
    acc_tl = _TL_SCAN_DTYPES.get(compute_dtype, tl.float32)
    BN = 32768 if compute_dtype == torch.float32 else 16384
    need_tail = 1 if N % BN else 0
    grid = (M,)
    with torch_device_fn.device(x.device):
        cumprod_row_scan_chunk_kernel[grid](
            x,
            out,
            N,
            ACC_DTYPE=acc_tl,
            BN=BN,
            NEED_TAIL=need_tail,
            num_warps=8,
            buffer_size_limit=2048,
        )
    return out


@triton.jit
def _strided_row_base(row, meta_ptr, ND_OTHER: tl.constexpr):
    """Storage offset of the first element of virtual row ``row``.

    ``row`` is a mixed-radix index over every dim of the tensor except the
    scan dim, in program order (``meta`` is a (2, ND_OTHER) int64 tensor:
    ``meta[0]`` = dim shapes, ``meta[1]`` = dim strides).  The scan dim is
    the innermost of the virtual rows."""
    off = 0
    tmp = row
    for i in tl.static_range(ND_OTHER):
        idx = ND_OTHER - 1 - i
        s = tl.load(meta_ptr + idx)
        digit = tmp % s
        tmp = tmp // s
        off = off + digit * tl.load(meta_ptr + ND_OTHER + idx)
    return off


@libentry()
@triton.jit(do_not_specialize=["N", "stride_n"])
def cumprod_strided_row_scan_kernel(
    inp_ptr,
    out_ptr,
    meta_ptr,
    N,
    stride_n,
    ND_OTHER: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    row_off = _strided_row_base(row, meta_ptr, ND_OTHER)
    offs = tl.arange(0, TILE_SIZE)
    mask = offs < N
    acc_dtype: tl.constexpr = get_prod_accum_type(out_ptr.type.element_ty)
    x = tl.load(inp_ptr + row_off + offs * stride_n, mask=mask, other=1).to(acc_dtype)
    r = tl.cumprod(x, 0)
    tl.store(
        out_ptr + row_off + offs * stride_n,
        r.to(out_ptr.type.element_ty),
        mask=mask,
    )


@libentry()
@triton.jit(do_not_specialize=["N", "stride_n"])
def cumprod_strided_row_scan_chunk_kernel(
    inp_ptr,
    out_ptr,
    meta_ptr,
    N,
    stride_n,
    ND_OTHER: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    BN: tl.constexpr,
    NEED_TAIL: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    row_off = _strided_row_base(row, meta_ptr, ND_OTHER)
    carry = tl.full([BN], value=1, dtype=ACC_DTYPE)
    for start in range(0, N, BN):
        n_offsets = start + tl.arange(0, BN)
        if NEED_TAIL:
            mask = n_offsets < N
            x = tl.load(
                inp_ptr + row_off + n_offsets * stride_n, mask=mask, other=1
            ).to(ACC_DTYPE)
        else:
            x = tl.load(inp_ptr + row_off + n_offsets * stride_n).to(ACC_DTYPE)
        r = tl.cumprod(x, axis=0) * carry
        carry *= tl.reduce(x, axis=0, combine_fn=reduce_mul)
        if NEED_TAIL:
            tl.store(
                out_ptr + row_off + n_offsets * stride_n,
                r.to(out_ptr.type.element_ty),
                mask=mask,
            )
        else:
            tl.store(
                out_ptr + row_off + n_offsets * stride_n,
                r.to(out_ptr.type.element_ty),
            )


def _strided_scan(x, out, dim, compute_dtype):
    """Scan a non-contiguous tensor along ``dim`` into ``out`` (in place of
    ``out``).  Every element of the scan axis is reached through the tensor's
    own strides (mixed-radix row indexing), so no contiguous copy of the
    input is made — ``inp.contiguous()`` inside ``use_gems()`` would go
    through the gems ``copy_``, which faults (KL3) on strided int8 views."""
    N = x.shape[dim]
    stride_n = x.stride(dim)
    other_dims = [i for i in range(x.ndim) if i != dim]
    nd_other = len(other_dims)
    meta = torch.tensor(
        [
            [x.shape[i] for i in other_dims],
            [x.stride(i) for i in other_dims],
        ],
        dtype=torch.int64,
        device=x.device,
    )
    R = math.prod(x.shape[i] for i in other_dims)
    persistent_limit = (
        ASCEND_SCAN_LIMIT if runtime_device.vendor_name == "ascend" else 16384
    )
    if N <= persistent_limit:
        TILE_SIZE = triton.next_power_of_2(N)
        num_warps = 8 if TILE_SIZE > 2048 else 4
        with torch_device_fn.device(x.device):
            cumprod_strided_row_scan_kernel[(R,)](
                x,
                out,
                meta,
                N,
                stride_n,
                ND_OTHER=nd_other,
                TILE_SIZE=TILE_SIZE,
                num_warps=num_warps,
            )
    else:
        acc_tl = _TL_SCAN_DTYPES.get(compute_dtype, tl.float32)
        BN = 32768 if compute_dtype == torch.float32 else 16384
        need_tail = 1 if N % BN else 0
        with torch_device_fn.device(x.device):
            cumprod_strided_row_scan_chunk_kernel[(R,)](
                x,
                out,
                meta,
                N,
                stride_n,
                ND_OTHER=nd_other,
                ACC_DTYPE=acc_tl,
                BN=BN,
                NEED_TAIL=need_tail,
                num_warps=8,
                buffer_size_limit=2048,
            )
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
    # bool 输入先转 uint8，按整型路径走 triton 内核（产品 0/1 无损）。
    if is_boolean_dtype(inp.dtype):
        uint8_inp = inp.to(torch.uint8)
        return cumprod_wrapper(uint8_inp, dim, out_dtype)
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
    if inp.numel() == 0:
        return inp
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    if inp.size(dim % inp.ndim) == 1:
        # Inclusive prefix product over an axis of length 1 is the identity,
        # so the in-place op needs no work at all.
        return inp

    cumprod_wrapper(inp, dim, inp.dtype, out=inp)
    return inp
