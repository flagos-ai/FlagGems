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

# torch_gcu's native aten::cumprod allocates int64 for integer/bool inputs according to ATen promotion rules
# output, while topsatenCumprod requires input and output dtypes to match, directly triggering a C++ CHECK process
# abort (P21 archive: compiler_problems/P21/sources/flaggems_cumprod_int_promotion_
# abort.md); for floating-point tensors with requires_grad, the native implementation's out= variant also raises a
# non-differentiable error.
# This key is already occupied by torch_gcu's C++ registration, so the Python side cannot take over on PrivateUse1,
# so register a dtype-dispatch shim on the higher-priority AutogradPrivateUse1: integer/bool and
# requires_grad inputs are computed on CPU and moved back (abort -> correct values, a strict improvement); all others
# redispatch to the native PrivateUse1 path (floating-point eager behavior and performance are bit-for-bit unchanged;
# benchmark
# native baseline leg is unaffected). The test suite's reference leg (direct torch.cumprod call) therefore falls back
# to CPU for comparison with the FlagGems implementation on GCU, preserving reference independence.
_ATEN_PU1_KEYSET = torch._C.DispatchKeySet(torch._C.DispatchKey.PrivateUse1)


def _cumprod_aten_shim(self, dim, *, dtype=None):
    if not self.is_floating_point() and not self.is_complex():
        return torch.cumprod(self.cpu(), dim, dtype=dtype).to(self.device)
    if self.requires_grad and torch.is_grad_enabled():
        return torch.cumprod(self.cpu(), dim, dtype=dtype).to(self.device)
    return torch.ops.aten.cumprod.default.redispatch(
        _ATEN_PU1_KEYSET, self, dim, dtype=dtype
    )


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
        num_warps = 1
        reduce_then_scan_root_scan_kernel_row[(M, 1, 1)](
            x, out, N, TILE_SIZE, num_warps=num_warps
        )
        return out

    TILE_SIZE = min(_scan_block_size(N), triton.next_power_of_2(N))
    num_warps = 1
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
    logger.debug("GEMS_ENFLAME CUMPROD")
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
    logger.debug("GEMS_ENFLAME CUMPROD_")
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
    # Contiguous float tensors share the output buffer directly with the input, saving a temporary allocation and a full
    # D2D copy;
    # non-contiguous views get a separate copy from contiguous() inside cumprod_wrapper, so a kernel writing out by
    # contiguous
    # layout would be misaligned, hence the original path is kept; integer tensors need an int64 accumulation buffer for
    # the same reason.
    if inp.is_floating_point() and inp.is_contiguous():
        return cumprod_wrapper(inp, dim, inp.dtype, inp)
    out = cumprod_wrapper(inp, dim, inp.dtype)
    inp.copy_(out)
    return inp


@triton.jit
def _cumprod_backward_scan_kernel(
    grad_ptr,
    input_ptr,
    output_ptr,
    grad_input_ptr,
    zero_flag_ptr,
    N,
    n_tiles,
    stride_d,
    stride_b0,
    stride_b1,
    stride_b2,
    g1g2,
    g2,
    BLOCK: tl.constexpr,
):
    # One program per batch line. The reduction axis is scanned in BLOCK-sized
    # tiles from the end backwards while `running` carries the suffix sum of
    # g * output, so every element is read exactly once regardless of N. The
    # scan tile is capped so the kernel fits the GCU DSM budget for any N.
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    pid2 = tl.program_id(2)
    base = pid0 * stride_b0 + pid1 * stride_b1 + pid2 * stride_b2

    offs = tl.arange(0, BLOCK)
    running = 0.0
    has_zero = 0.0
    for tr in range(0, n_tiles):
        t = n_tiles - 1 - tr
        offs_t = t * BLOCK + offs
        m = offs_t < N
        pos = base + offs_t * stride_d
        g = tl.load(grad_ptr + pos, mask=m, other=0.0).to(tl.float32)
        out = tl.load(output_ptr + pos, mask=m, other=0.0).to(tl.float32)
        go = g * out
        tile_total = tl.sum(go, axis=0)
        inc = tl.cumsum(go, axis=0)
        rev = running + tile_total - inc + go
        inp = tl.load(input_ptr + pos, mask=m, other=0.0).to(tl.float32)
        safe = tl.where(inp == 0.0, 1.0, inp)
        grad = rev / safe
        grad = tl.where(inp == 0.0, 0.0, grad)
        tl.store(
            grad_input_ptr + pos,
            grad.to(grad_input_ptr.type.element_ty),
            mask=m,
        )
        running += tile_total
        hz = tl.max(tl.where((inp == 0.0) & m, 1.0, 0.0), axis=0)
        has_zero = tl.maximum(has_zero, hz)
    zidx = pid0 * g1g2 + pid1 * g2 + pid2
    tl.store(zero_flag_ptr + zidx, has_zero)


@triton.jit
def _cumprod_backward_zero_flag_reduce_kernel(
    zero_flag_ptr,
    tensor_flag_ptr,
    batch_total,
    BLOCK: tl.constexpr,
):
    # Collapse the per-line zero flags into one tensor-level flag: the aten
    # reference decides the high-precision recompute globally (any zero in the
    # tensor affects every line, not just the lines holding the zero).
    any_zero = 0.0
    offs = tl.arange(0, BLOCK)
    n = tl.cdiv(batch_total, BLOCK)
    for start in range(0, n * BLOCK, BLOCK):
        offs_t = start + offs
        v = tl.load(zero_flag_ptr + offs_t, mask=offs_t < batch_total, other=0.0)
        any_zero = tl.maximum(any_zero, tl.max(v, axis=0))
    tl.store(tensor_flag_ptr, any_zero)


@triton.jit
def _cumprod_backward_zero_fixup_kernel(
    grad_ptr,
    input_ptr,
    grad_input_ptr,
    tensor_flag_ptr,
    N,
    n_tiles,
    stride_d,
    stride_b0,
    stride_b1,
    stride_b2,
    BLOCK: tl.constexpr,
):
    # aten::cumprod_backward discards the supplied output for the whole tensor
    # once any zero is present and recomputes the forward cumprod in high
    # precision for every line. This kernel repeats that recompute in fp32,
    # gated by the tensor-level flag; clean tensors skip the loops and exit
    # right after the flag load. Lines without their own zero simply see
    # k == INT_MAX, which turns loop 2 into a full-line recompute.
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    pid2 = tl.program_id(2)
    has_zero = tl.load(tensor_flag_ptr)
    n2 = tl.where(has_zero > 0.0, n_tiles, 0)
    base = pid0 * stride_b0 + pid1 * stride_b1 + pid2 * stride_b2

    offs = tl.arange(0, BLOCK)
    total = 0.0
    prefix_prod = 1.0
    k = 2147483647
    for t in range(0, n2):
        offs_t = t * BLOCK + offs
        m = offs_t < N
        pos = base + offs_t * stride_d
        inp = tl.load(input_ptr + pos, mask=m, other=1.0).to(tl.float32)
        g = tl.load(grad_ptr + pos, mask=m, other=0.0).to(tl.float32)
        c = prefix_prod * tl.cumprod(inp, axis=0)
        total += tl.sum(g * c, axis=0)
        prefix_prod = tl.sum(tl.where(offs == BLOCK - 1, c, 0.0), axis=0)
        zi = tl.where((inp == 0.0) & m, offs_t, 2147483647)
        k = tl.minimum(k, tl.min(zi, axis=0))

    prefix_prod = 1.0
    run_cs = 0.0
    pump = 1.0
    g_at_k = 0.0
    zv_acc = 0.0
    after_k_prod = 1.0
    for t in range(0, n2):
        start = t * BLOCK
        offs_t = start + offs
        m = offs_t < N
        oi = offs_t
        pos = base + offs_t * stride_d
        inp = tl.load(input_ptr + pos, mask=m, other=1.0).to(tl.float32)
        g = tl.load(grad_ptr + pos, mask=m, other=0.0).to(tl.float32)
        c = prefix_prod * tl.cumprod(inp, axis=0)
        go32 = g * c
        inc = tl.cumsum(go32, axis=0)
        tile_total = tl.sum(go32, axis=0)
        rev = total - run_cs - inc + go32
        tile_has_k = (k >= start) & (k < start + BLOCK)
        pump_new = tl.sum(tl.where(oi == k - 1, c, 0.0), axis=0)
        pump = tl.where(tile_has_k, tl.where(k == start, prefix_prod, pump_new), pump)
        g_at_k = tl.where(tile_has_k, tl.sum(tl.where(oi == k, g, 0.0), axis=0), g_at_k)
        safe = tl.where(inp == 0.0, 1.0, inp)
        grad = rev / safe
        sm = m & (oi < k)
        tl.store(
            grad_input_ptr + pos,
            grad.to(grad_input_ptr.type.element_ty),
            mask=sm,
        )
        mod = tl.where(oi > k, inp, 1.0)
        csq = after_k_prod * tl.cumprod(mod, axis=0)
        zv_acc += tl.sum(tl.where(oi > k, g * csq, 0.0), axis=0)
        last_mod = tl.sum(tl.where(offs == BLOCK - 1, csq, 0.0), axis=0)
        after_k_prod = tl.where(k < start + BLOCK, last_mod, after_k_prod)
        run_cs += tile_total
        prefix_prod = tl.sum(tl.where(offs == BLOCK - 1, c, 0.0), axis=0)

    if k < N:
        zero_val = pump * (g_at_k + zv_acc)
        tl.store(
            grad_input_ptr + base + k * stride_d,
            zero_val.to(grad_input_ptr.type.element_ty),
        )


_CUMPROD_BWD_MAX_SINGLE_TILE = 4096
_CUMPROD_BWD_SCAN_TILE = 2048
_CUMPROD_BWD_FIXUP_TILE = 512
_CUMPROD_BWD_FLAG_REDUCE_TILE = 1024


def _launch_cumprod_backward_line_kernels(
    grad,
    input,
    output,
    grad_input,
    grid,
    N,
    stride_d,
    stride_b0,
    stride_b1,
    stride_b2,
    g1g2,
    g2,
):
    if N <= _CUMPROD_BWD_MAX_SINGLE_TILE:
        scan_block = max(16, triton.next_power_of_2(N))
    else:
        scan_block = _CUMPROD_BWD_SCAN_TILE
    scan_tiles = triton.cdiv(N, scan_block)
    fixup_tiles = triton.cdiv(N, _CUMPROD_BWD_FIXUP_TILE)
    batch_total = grid[0] * grid[1] * grid[2]
    # Per-line zero flags, collapsed to one tensor-level flag by the tiny
    # reduce kernel; float so the hot scan kernel keeps its zero detection in
    # the float reduction domain.
    zero_flag = torch.empty(batch_total, dtype=torch.float32, device=input.device)
    tensor_flag = torch.empty(1, dtype=torch.float32, device=input.device)
    with torch_device_fn.device(input.device):
        _cumprod_backward_scan_kernel[grid](
            grad,
            input,
            output,
            grad_input,
            zero_flag,
            N,
            scan_tiles,
            stride_d,
            stride_b0,
            stride_b1,
            stride_b2,
            g1g2,
            g2,
            BLOCK=scan_block,
            num_warps=1,
        )
        _cumprod_backward_zero_flag_reduce_kernel[(1,)](
            zero_flag,
            tensor_flag,
            batch_total,
            BLOCK=_CUMPROD_BWD_FLAG_REDUCE_TILE,
            num_warps=1,
        )
        _cumprod_backward_zero_fixup_kernel[grid](
            grad,
            input,
            grad_input,
            tensor_flag,
            N,
            fixup_tiles,
            stride_d,
            stride_b0,
            stride_b1,
            stride_b2,
            BLOCK=_CUMPROD_BWD_FIXUP_TILE,
            num_warps=1,
        )


def cumprod_backward(grad, input, dim, output):
    logger.debug("GEMS_ENFLAME CUMPROD_BACKWARD")
    ndim = input.ndim
    dim = int(dim)
    if dim < 0:
        dim += ndim

    N = input.shape[dim]
    grad_input = torch.empty_like(grad)

    if input.numel() == 0:
        return grad_input

    grad_stride = grad.stride()
    input_stride = input.stride()
    output_stride = output.stride()
    gi_stride = grad_input.stride()

    batch_dims = [i for i in range(ndim) if i != dim]

    # The stride-based kernels fold up to three batch dims into scalar stride
    # slots; more than three, or any layout mismatch between the four tensors,
    # falls back to materializing contiguous 2D lines.
    same_layout = (
        grad_stride == input_stride
        and grad_stride == output_stride
        and grad_stride == gi_stride
    )
    if not same_layout or len(batch_dims) > 3:
        grad_2d = grad.movedim(dim, -1).reshape(-1, N).contiguous()
        input_2d = input.movedim(dim, -1).reshape(-1, N).contiguous()
        output_2d = output.movedim(dim, -1).reshape(-1, N).contiguous()
        batch_size = grad_2d.shape[0]
        gi_2d = torch.empty((batch_size, N), dtype=grad.dtype, device=grad.device)

        _launch_cumprod_backward_line_kernels(
            grad_2d,
            input_2d,
            output_2d,
            gi_2d,
            (batch_size, 1, 1),
            N,
            1,
            N,
            0,
            0,
            1,
            1,
        )
        grad_input = (
            gi_2d.reshape([input.shape[i] for i in range(ndim) if i != dim] + [N])
            .movedim(-1, dim)
            .contiguous()
        )
        return grad_input

    stride_d = grad_stride[dim]
    if len(batch_dims) == 0:
        grid = (1, 1, 1)
        stride_b0, stride_b1, stride_b2 = 0, 0, 0
        g1g2, g2 = 1, 1
    elif len(batch_dims) == 1:
        grid = (input.shape[batch_dims[0]], 1, 1)
        stride_b0 = grad_stride[batch_dims[0]]
        stride_b1, stride_b2 = 0, 0
        g1g2, g2 = 1, 1
    elif len(batch_dims) == 2:
        s0, s1 = input.shape[batch_dims[0]], input.shape[batch_dims[1]]
        grid = (s0, s1, 1)
        stride_b0 = grad_stride[batch_dims[0]]
        stride_b1 = grad_stride[batch_dims[1]]
        stride_b2 = 0
        g1g2, g2 = s1, 1
    else:
        s0 = input.shape[batch_dims[0]]
        s1 = input.shape[batch_dims[1]]
        s2 = input.shape[batch_dims[2]]
        grid = (s0, s1, s2)
        stride_b0 = grad_stride[batch_dims[0]]
        stride_b1 = grad_stride[batch_dims[1]]
        stride_b2 = grad_stride[batch_dims[2]]
        g1g2, g2 = s1 * s2, s2

    _launch_cumprod_backward_line_kernels(
        grad,
        input,
        output,
        grad_input,
        grid,
        N,
        stride_d,
        stride_b0,
        stride_b1,
        stride_b2,
        g1g2,
        g2,
    )
    return grad_input
