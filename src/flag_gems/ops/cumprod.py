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

import functools
import logging
import math

import torch
import triton
import triton.language as tl
from torch._prims_common import is_boolean_dtype, is_integer_dtype

from flag_gems.runtime import device as runtime_device
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import get_device_properties, libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)
DEFAULT_BLOCK_SIZE = 1024
CUDA_SMALL_SCAN_LIMIT = 1024 * 4
ASCEND_SCAN_LIMIT = 1024
DEFAULT_NUM_SMS = 40


@functools.lru_cache
def get_num_sms(idx: int) -> int:
    return get_device_properties(idx).multi_processor_count or DEFAULT_NUM_SMS


def _get_device_index(torch_device):
    if torch_device.index is not None:
        return torch_device.index
    return torch_device_fn.current_device()


@tl.constexpr
def get_prod_accum_type(out_dtype: tl.dtype) -> tl.dtype:
    if out_dtype.is_bf16() or out_dtype.is_fp16():
        return tl.float32
    if out_dtype.is_int64() or out_dtype.is_uint64():
        return tl.int64
    if out_dtype.is_int():
        return tl.int32
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

    tl.store(out + offset, result.to(out.type.element_ty), mask=mask)
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
        tl.store(out + offset, final_vals.to(out.type.element_ty), mask=mask)


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

    tl.store(out + offset, result.to(out.type.element_ty), mask=mask)
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
        tl.store(out + offset, final_vals.to(out.type.element_ty), mask=mask)


def scan_then_fan_col(inp, out, n_ele, dtype):
    BLOCK_SIZE = _scan_block_size(n_ele)
    part_num = math.ceil(n_ele / BLOCK_SIZE)
    partial_product = torch.empty(part_num, dtype=dtype, device=inp.device)
    scan_out = out
    if part_num >= 2 and out.dtype != dtype:
        scan_out = torch.empty_like(out, dtype=dtype)

    grid = (part_num,)
    with torch_device_fn.device(inp.device):
        scan_part_product_kernel[grid](
            inp, scan_out, partial_product, n_ele, part_num, BLOCK_SIZE
        )

    if part_num >= 2:
        partial_prefix = torch.empty_like(partial_product)
        scan_then_fan_col(partial_product, partial_prefix, part_num, dtype)
        with torch_device_fn.device(inp.device):
            multiply_base_product_kernel[grid](
                scan_out, partial_prefix, n_ele, part_num, BLOCK_SIZE
            )
        if scan_out is not out:
            out.copy_(scan_out)


def scan_then_fan(inp, out, A, B, C, dtype):
    BLOCK_SIZE = _scan_block_size(B)
    part_num = math.ceil(B / BLOCK_SIZE)
    partial_product = torch.empty(A, part_num, C, dtype=dtype, device=inp.device)
    scan_out = out
    if part_num >= 2 and out.dtype != dtype:
        scan_out = torch.empty_like(out, dtype=dtype)

    grid = (A, part_num, C)
    with torch_device_fn.device(inp.device):
        scan_part_product_abc_kernel[grid](
            inp, scan_out, partial_product, B, C, part_num, BLOCK_SIZE
        )

    if part_num >= 2:
        partial_prefix = torch.empty_like(partial_product)
        scan_then_fan(partial_product, partial_prefix, A, part_num, C, dtype)
        with torch_device_fn.device(inp.device):
            multiply_base_product_abc_kernel[grid](
                scan_out, partial_prefix, B, C, part_num, BLOCK_SIZE
            )
        if scan_out is not out:
            out.copy_(scan_out)


def _get_output_dtype(inp, dtype):
    if dtype is not None:
        return dtype
    if is_integer_dtype(inp.dtype) or is_boolean_dtype(inp.dtype):
        return torch.int64
    return inp.dtype


def _get_compute_dtype(dtype):
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    if dtype is torch.int64:
        return torch.int64
    if is_integer_dtype(dtype) or is_boolean_dtype(dtype):
        return torch.int32
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

    TILE_SIZE = min(_scan_block_size(N), triton.next_power_of_2(N))
    num_warps = 8 if TILE_SIZE > 2048 else 4
    num_tiles = triton.cdiv(N, TILE_SIZE)
    max_ctas = get_num_sms(_get_device_index(x.device)) * 4
    num_ctas = min(num_tiles, max_ctas)
    ROOT_SCAN_TILE_SIZE = triton.next_power_of_2(num_ctas)
    tiles_per_cta = triton.cdiv(num_tiles, num_ctas)

    block_products = torch.empty((M, num_ctas), dtype=compute_dtype, device=x.device)
    block_inclusive_prefix = torch.empty_like(block_products)

    reduce_then_scan_block_product_kernel_row[(M, num_ctas, 1, 1)](
        x, block_products, N, tiles_per_cta, TILE_SIZE, num_warps=num_warps
    )
    reduce_then_scan_root_scan_kernel_row[(M, 1, 1)](
        block_products,
        block_inclusive_prefix,
        num_ctas,
        ROOT_SCAN_TILE_SIZE,
        num_warps=num_warps,
    )
    reduce_then_scan_block_scan_kernel_row[(M, num_ctas, 1)](
        x,
        block_inclusive_prefix,
        out,
        N,
        num_ctas,
        tiles_per_cta,
        TILE_SIZE,
        num_warps=num_warps,
    )
    return out


@triton.jit
def reduce_then_scan_block_product_kernel_row(
    in_ptr,
    block_product_ptr,
    N,
    tiles_per_cta,
    TILE_SIZE: tl.constexpr,
):
    pid_n = tl.program_id(1).to(tl.int64)
    pid_m = tl.program_id(0).to(tl.int64)
    num_programs_n = tl.num_programs(1)
    block_offset = pid_n * (tiles_per_cta * TILE_SIZE)
    block_end = min(block_offset + tiles_per_cta * TILE_SIZE, N)

    acc_dtype: tl.constexpr = get_prod_accum_type(block_product_ptr.type.element_ty)
    acc = tl.full((TILE_SIZE,), value=1, dtype=acc_dtype)
    for start in range(block_offset, block_end, TILE_SIZE):
        offsets = start + tl.arange(0, TILE_SIZE)
        x = tl.load(in_ptr + pid_m * N + offsets, mask=offsets < N, other=1).to(
            acc_dtype
        )
        acc *= x
    block_product = tl.reduce(acc, axis=0, combine_fn=reduce_mul)
    tl.store(
        block_product_ptr + pid_m * num_programs_n + pid_n,
        block_product,
        cache_modifier=".cg",
    )


@triton.jit
def reduce_then_scan_root_scan_kernel_row(in_ptr, out_ptr, N, TILE_SIZE: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, TILE_SIZE)
    mask = offsets < N
    acc_dtype: tl.constexpr = get_prod_accum_type(out_ptr.type.element_ty)
    x = tl.load(in_ptr + pid * N + offsets, mask=mask, other=1).to(acc_dtype)
    out = tl.cumprod(x, 0)
    tl.store(out_ptr + pid * N + offsets, out.to(out_ptr.type.element_ty), mask=mask)


@triton.jit
def reduce_then_scan_block_scan_kernel_row(
    in_ptr,
    previous_product_ptr,
    out_ptr,
    N,
    num_tiles_n,
    tiles_per_cta,
    TILE_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    block_offset = pid_n * (tiles_per_cta * TILE_SIZE)
    block_end = min(block_offset + tiles_per_cta * TILE_SIZE, N)
    acc_dtype: tl.constexpr = get_prod_accum_type(out_ptr.type.element_ty)

    prefix = tl.load(
        previous_product_ptr + pid_m * num_tiles_n + pid_n - 1,
        mask=pid_n > 0,
        other=1,
    ).to(acc_dtype)
    for start in range(block_offset, block_end, TILE_SIZE):
        offsets = start + tl.arange(0, TILE_SIZE)
        mask = offsets < N
        x = tl.load(in_ptr + pid_m * N + offsets, mask=mask, other=1).to(acc_dtype)
        tile_scan = prefix * tl.cumprod(x, 0)
        prefix *= tl.reduce(x, axis=0, combine_fn=reduce_mul)
        tl.store(
            out_ptr + pid_m * N + offsets,
            tile_scan.to(out_ptr.type.element_ty),
            mask=mask,
            cache_modifier=".cg",
        )


def cumprod(inp, dim, *, dtype=None):
    logger.debug("GEMS CUMPROD")
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
    logger.debug("GEMS CUMPROD_")
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
    out = cumprod_wrapper(inp, dim, inp.dtype)
    inp.copy_(out)
    return inp


@triton.jit
def _cumprod_backward_kernel(
    grad_ptr,
    input_ptr,
    output_ptr,
    grad_input_ptr,
    N,  # length along reduction dim
    stride_d,  # stride along reduction dim (same for all tensors)
    stride_b,  # batch stride (same for all tensors)
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * stride_b

    offs = tl.arange(0, BLOCK)
    mask = offs < N

    g = tl.load(
        grad_ptr + base + offs * stride_d, mask=mask, other=0.0, cache_modifier=".ca"
    ).to(tl.float32)
    inp = tl.load(
        input_ptr + base + offs * stride_d, mask=mask, other=0.0, cache_modifier=".ca"
    ).to(tl.float32)
    out = tl.load(
        output_ptr + base + offs * stride_d, mask=mask, other=0.0, cache_modifier=".ca"
    ).to(tl.float32)

    go = g * out

    # Reverse cumulative sum: rev_cs[i] = sum_{j=i}^{N-1} go[j]
    total_go = tl.sum(go, axis=0)
    inc_cs = tl.cumsum(go, axis=0)
    rev_cs = total_go - inc_cs + go

    safe_inp = tl.where(inp == 0.0, 1.0, inp)
    grad_input = tl.where(mask, rev_cs / safe_inp, 0.0)
    grad_input = tl.where(inp == 0.0, 0.0, grad_input)

    is_zero = (inp == 0.0) & mask
    has_zero = tl.max(is_zero.to(tl.int32), axis=0)

    if has_zero != 0:
        large_val = 2147483647
        zero_indices = tl.where(is_zero, offs.to(tl.int32), large_val)
        k = tl.min(zero_indices, axis=0)

        grad_input = tl.where(offs.to(tl.int32) > k, 0.0, grad_input)

        inp_mod = tl.where(offs.to(tl.int32) == k, 1.0, inp)
        inp_mod = tl.where(mask, inp_mod, 1.0)
        cumprod_mod = tl.cumprod(inp_mod, axis=0)

        gcm = g * cumprod_mod
        zero_val = tl.sum(tl.where(offs.to(tl.int32) >= k, gcm, 0.0), axis=0)

        grad_input = tl.where(offs.to(tl.int32) == k, zero_val, grad_input)

    tl.store(
        grad_input_ptr + base + offs * stride_d,
        grad_input.to(tl.float32),
        mask=mask,
        cache_modifier=".cs",
    )


@triton.jit
def _cumprod_backward_kernel_2d_batch(
    grad_ptr,
    input_ptr,
    output_ptr,
    grad_input_ptr,
    N,
    stride_d,
    stride_b0,
    stride_b1,
    BLOCK: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    base = pid0 * stride_b0 + pid1 * stride_b1

    offs = tl.arange(0, BLOCK)
    mask = offs < N

    g = tl.load(
        grad_ptr + base + offs * stride_d, mask=mask, other=0.0, cache_modifier=".ca"
    ).to(tl.float32)
    inp = tl.load(
        input_ptr + base + offs * stride_d, mask=mask, other=0.0, cache_modifier=".ca"
    ).to(tl.float32)
    out = tl.load(
        output_ptr + base + offs * stride_d, mask=mask, other=0.0, cache_modifier=".ca"
    ).to(tl.float32)

    go = g * out

    total_go = tl.sum(go, axis=0)
    inc_cs = tl.cumsum(go, axis=0)
    rev_cs = total_go - inc_cs + go

    safe_inp = tl.where(inp == 0.0, 1.0, inp)
    grad_input = tl.where(mask, rev_cs / safe_inp, 0.0)
    grad_input = tl.where(inp == 0.0, 0.0, grad_input)

    is_zero = (inp == 0.0) & mask
    has_zero = tl.max(is_zero.to(tl.int32), axis=0)

    if has_zero != 0:
        large_val = 2147483647
        zero_indices = tl.where(is_zero, offs.to(tl.int32), large_val)
        k = tl.min(zero_indices, axis=0)

        grad_input = tl.where(offs.to(tl.int32) > k, 0.0, grad_input)

        inp_mod = tl.where(offs.to(tl.int32) == k, 1.0, inp)
        inp_mod = tl.where(mask, inp_mod, 1.0)
        cumprod_mod = tl.cumprod(inp_mod, axis=0)

        gcm = g * cumprod_mod
        zero_val = tl.sum(tl.where(offs.to(tl.int32) >= k, gcm, 0.0), axis=0)

        grad_input = tl.where(offs.to(tl.int32) == k, zero_val, grad_input)

    tl.store(
        grad_input_ptr + base + offs * stride_d,
        grad_input.to(tl.float32),
        mask=mask,
        cache_modifier=".cs",
    )


@triton.jit
def _cumprod_backward_kernel_3d_batch(
    grad_ptr,
    input_ptr,
    output_ptr,
    grad_input_ptr,
    N,
    stride_d,
    stride_b0,
    stride_b1,
    stride_b2,
    BLOCK: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    pid2 = tl.program_id(2)
    base = pid0 * stride_b0 + pid1 * stride_b1 + pid2 * stride_b2

    offs = tl.arange(0, BLOCK)
    mask = offs < N

    g = tl.load(
        grad_ptr + base + offs * stride_d, mask=mask, other=0.0, cache_modifier=".ca"
    ).to(tl.float32)
    inp = tl.load(
        input_ptr + base + offs * stride_d, mask=mask, other=0.0, cache_modifier=".ca"
    ).to(tl.float32)
    out = tl.load(
        output_ptr + base + offs * stride_d, mask=mask, other=0.0, cache_modifier=".ca"
    ).to(tl.float32)

    go = g * out

    total_go = tl.sum(go, axis=0)
    inc_cs = tl.cumsum(go, axis=0)
    rev_cs = total_go - inc_cs + go

    safe_inp = tl.where(inp == 0.0, 1.0, inp)
    grad_input = tl.where(mask, rev_cs / safe_inp, 0.0)
    grad_input = tl.where(inp == 0.0, 0.0, grad_input)

    is_zero = (inp == 0.0) & mask
    has_zero = tl.max(is_zero.to(tl.int32), axis=0)

    if has_zero != 0:
        large_val = 2147483647
        zero_indices = tl.where(is_zero, offs.to(tl.int32), large_val)
        k = tl.min(zero_indices, axis=0)

        grad_input = tl.where(offs.to(tl.int32) > k, 0.0, grad_input)

        inp_mod = tl.where(offs.to(tl.int32) == k, 1.0, inp)
        inp_mod = tl.where(mask, inp_mod, 1.0)
        cumprod_mod = tl.cumprod(inp_mod, axis=0)

        gcm = g * cumprod_mod
        zero_val = tl.sum(tl.where(offs.to(tl.int32) >= k, gcm, 0.0), axis=0)

        grad_input = tl.where(offs.to(tl.int32) == k, zero_val, grad_input)

    tl.store(
        grad_input_ptr + base + offs * stride_d,
        grad_input.to(tl.float32),
        mask=mask,
        cache_modifier=".cs",
    )


def cumprod_backward(grad, input, dim, output):
    logger.debug("GEMS CUMPROD_BACKWARD")
    ndim = input.ndim
    dim = int(dim)
    if dim < 0:
        dim += ndim

    N = input.shape[dim]

    batch_size = 1
    for i in range(ndim):
        if i != dim:
            batch_size *= input.shape[i]

    grad_input = torch.empty_like(grad)

    # aten::cumprod_backward has two regimes: when the input has no zero it
    # trusts the supplied `output`, but as soon as any zero is present it
    # discards `output` and recomputes cumprod internally in high precision
    # (the passed low-precision output would otherwise inject rounding error
    # that the reference never sees). Mirror that here: for low-precision
    # dtypes with a zero anywhere, recompute the forward cumprod in fp32 so
    # the reverse-cumsum below matches the reference. This is a whole-tensor
    # decision, matching aten's global branch rather than a per-line one.
    if output.dtype in (torch.float16, torch.bfloat16) and torch.any(input == 0):
        output = torch.cumprod(input.to(torch.float32), dim)

    grad_stride = grad.stride()
    input_stride = input.stride()
    output_stride = output.stride()
    gi_stride = grad_input.stride()

    batch_dims = [i for i in range(ndim) if i != dim]

    BLOCK = 16
    while BLOCK < N:
        BLOCK *= 2

    # Choose num_warps based on N to reduce overhead for small lines
    if BLOCK <= 32:
        num_warps = 1
    elif BLOCK <= 64:
        num_warps = 2
    elif BLOCK <= 128:
        num_warps = 4
    else:
        num_warps = 8

    # Check if all tensors have the same layout (strides)
    same_layout = (
        grad_stride == input_stride
        and grad_stride == output_stride
        and grad_stride == gi_stride
    )

    if not same_layout:
        # Fall back to contiguous approach when layouts differ
        grad_2d = grad.movedim(dim, -1).reshape(-1, N).contiguous()
        input_2d = input.movedim(dim, -1).reshape(-1, N).contiguous()
        output_2d = output.movedim(dim, -1).reshape(-1, N).contiguous()
        gi_2d = torch.empty((batch_size, N), dtype=grad.dtype, device=grad.device)

        grid = (batch_size,)
        _cumprod_backward_kernel[grid](
            grad_2d,
            input_2d,
            output_2d,
            gi_2d,
            N,
            1,
            N,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        grad_input = (
            gi_2d.reshape([input.shape[i] for i in range(ndim) if i != dim] + [N])
            .movedim(-1, dim)
            .contiguous()
        )
        return grad_input

    # All tensors share the same layout - use stride-based kernels, computing
    # batch offsets arithmetically in-kernel to avoid host-side copies.
    stride_d = grad_stride[dim]

    if len(batch_dims) == 0:
        grid = (1,)
        _cumprod_backward_kernel[grid](
            grad,
            input,
            output,
            grad_input,
            N,
            stride_d,
            0,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
    elif len(batch_dims) == 1:
        stride_b = grad_stride[batch_dims[0]]
        grid = (batch_size,)
        _cumprod_backward_kernel[grid](
            grad,
            input,
            output,
            grad_input,
            N,
            stride_d,
            stride_b,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
    elif len(batch_dims) == 2:
        stride_b0 = grad_stride[batch_dims[0]]
        stride_b1 = grad_stride[batch_dims[1]]
        s0 = input.shape[batch_dims[0]]
        s1 = input.shape[batch_dims[1]]
        grid = (s0, s1)
        _cumprod_backward_kernel_2d_batch[grid](
            grad,
            input,
            output,
            grad_input,
            N,
            stride_d,
            stride_b0,
            stride_b1,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
    else:
        stride_b0 = grad_stride[batch_dims[0]]
        stride_b1 = grad_stride[batch_dims[1]]
        stride_b2 = grad_stride[batch_dims[2]]
        s0 = input.shape[batch_dims[0]]
        s1 = input.shape[batch_dims[1]]
        s2 = input.shape[batch_dims[2]]
        grid = (s0, s1, s2)
        _cumprod_backward_kernel_3d_batch[grid](
            grad,
            input,
            output,
            grad_input,
            N,
            stride_d,
            stride_b0,
            stride_b1,
            stride_b2,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )

    return grad_input
