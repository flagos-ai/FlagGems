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
from flag_gems.utils import triton_lang_extension as tle

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
    pid = tle.program_id(0)
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
    pid = tle.program_id(0)
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
    pid_a = tle.program_id(0)
    pid_b = tle.program_id(1)
    pid_c = tle.program_id(2)

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
    tl.store(out_ptr + pid * N + offsets, out, mask=mask)


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
            out_ptr + pid_m * N + offsets, tile_scan, mask=mask, cache_modifier=".cg"
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
        return torch.ops.aten.cumprod_.default.redispatch(
            _FALLBACK_KEYSET, inp, dim, dtype=dtype
        )
    if _should_redispatch_on_ascend(inp.dtype):
        return torch.ops.aten.cumprod_.default.redispatch(
            _FALLBACK_KEYSET, inp, dim, dtype=dtype
        )
    out = cumprod_wrapper(inp, dim, inp.dtype)
    inp.copy_(out)
    return inp
