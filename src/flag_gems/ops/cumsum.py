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

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import get_device_properties, libentry
from flag_gems.utils import triton_lang_extension as ext

device = device.name
logger = logging.getLogger(__name__)


@functools.lru_cache
def get_num_sms(idx: int) -> int:
    return get_device_properties(idx).multi_processor_count


@tl.constexpr
def get_scan_accum_type(inp_dtype: tl.dtype) -> tl.dtype:
    if inp_dtype.is_bf16() or inp_dtype.is_fp16():
        return tl.float32
    if inp_dtype.is_int():  # signed or not(including bool)
        return tl.int64
    else:
        return inp_dtype


@libentry()
@triton.jit(do_not_specialize=["n_elements", "part_num"])
def scan_part_sum_kernel(
    inp,
    out,
    partial_sum,
    n_elements,
    part_num,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    inp_ptrs = inp + offset
    inp_vals = tl.load(inp_ptrs, mask=mask)
    if (
        tl.constexpr(inp_vals.dtype.is_int64())
        or tl.constexpr(inp_vals.dtype.is_uint64())
    ) or tl.constexpr(inp_vals.dtype.is_fp64()):
        inp_vals = inp_vals
    elif tl.constexpr(inp_vals.dtype.is_int()):
        inp_vals = inp_vals.to(tl.int32)
    else:
        inp_vals = inp_vals.to(tl.float32)
    result = tl.cumsum(inp_vals, axis=0)

    part_sum_via_sum = tl.sum(inp_vals)

    out_ptrs = out + offset
    tl.store(out_ptrs, result, mask=mask)

    partial_sum_ptrs = partial_sum + pid
    tl.store(partial_sum_ptrs, part_sum_via_sum)


@libentry()
@triton.jit(do_not_specialize=["n_elements", "part_num"])
def add_base_sum_kernel(
    out,
    partial_sum,
    n_elements,
    part_num,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    out_ptrs = out + offset
    out_vals = tl.load(out_ptrs, mask=mask)

    if pid > 0:
        partial_sum_ptrs = partial_sum + pid - 1
        last_part_sum_via_sum = tl.load(partial_sum_ptrs)

        final_vals = out_vals + last_part_sum_via_sum
        tl.store(out_ptrs, final_vals.to(out_vals.dtype), mask=mask)


@libentry()
@triton.jit(do_not_specialize=["part_num"])
def scan_part_sum_abc_kernel(
    inp,
    out,
    partial_sum,
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
    inp_ptrs = inp + offset
    inp_vals = tl.load(inp_ptrs, mask=mask)
    if (
        tl.constexpr(inp_vals.dtype.is_int64())
        or tl.constexpr(inp_vals.dtype.is_uint64())
    ) or tl.constexpr(inp_vals.dtype.is_fp64()):
        inp_vals = inp_vals
    elif tl.constexpr(inp_vals.dtype.is_int()):
        inp_vals = inp_vals.to(tl.int32)
    else:
        inp_vals = inp_vals.to(tl.float32)
    result = tl.cumsum(inp_vals, axis=0)

    part_sum_via_sum = tl.sum(inp_vals)

    out_ptrs = out + offset
    tl.store(out_ptrs, result, mask=mask)

    partial_sum_ptrs = partial_sum + part_offset
    tl.store(partial_sum_ptrs, part_sum_via_sum)


@libentry()
@triton.jit(do_not_specialize=["part_num"])
def add_base_sum_abc_kernel(
    out,
    partial_sum,
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

    base_offset = a_idx * B * C + c_idx
    offset = base_offset + b_idx * C
    base_part_offset = a_idx * part_num * C + c_idx
    last_part_offset = base_part_offset + (pid_b - 1) * C

    mask = b_idx < B
    out_ptrs = out + offset
    out_vals = tl.load(out_ptrs, mask=mask)

    if pid_b > 0:
        partial_sum_ptrs = partial_sum + last_part_offset
        last_part_sum_via_sum = tl.load(partial_sum_ptrs)

        final_vals = out_vals + last_part_sum_via_sum
        tl.store(out_ptrs, final_vals.to(out_vals.dtype), mask=mask)


def scan_then_fan_col(inp, out, n_ele, dtype):
    # TODO(all): tune on target board
    BLOCK_SIZE = 1024
    if n_ele <= 1024 * 4:
        BLOCK_SIZE = triton.next_power_of_2(n_ele)
    part_num = math.ceil(n_ele / BLOCK_SIZE)
    partial_sum = torch.empty(part_num, dtype=dtype, device=inp.device)

    grid = (part_num,)
    with torch_device_fn.device(inp.device):
        scan_part_sum_kernel[grid](inp, out, partial_sum, n_ele, part_num, BLOCK_SIZE)

    if part_num >= 2:
        scan_then_fan_col(partial_sum, partial_sum, part_num, dtype)
        with torch_device_fn.device(inp.device):
            add_base_sum_kernel[grid](out, partial_sum, n_ele, part_num, BLOCK_SIZE)


def scan_then_fan(inp, out, A, B, C, dtype):
    # TODO(all): tune on target board
    BLOCK_SIZE = 1024
    if B <= 1024 * 4:
        BLOCK_SIZE = triton.next_power_of_2(B)
    part_num = math.ceil(B / BLOCK_SIZE)
    partial_sum = torch.empty(A, part_num, C, dtype=dtype, device=inp.device)

    grid = (A, part_num, C)
    with torch_device_fn.device(inp.device):
        scan_part_sum_abc_kernel[grid](
            inp, out, partial_sum, B, C, part_num, BLOCK_SIZE
        )

    if part_num >= 2:
        scan_then_fan(partial_sum, partial_sum, A, part_num, C, dtype)
        with torch_device_fn.device(inp.device):
            add_base_sum_abc_kernel[grid](out, partial_sum, B, C, part_num, BLOCK_SIZE)


def cumsum_wrapper(inp, dim=1, dtype=None, out=None):
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    dim = dim % inp.ndim
    inp = inp.contiguous()
    if dtype is None:
        dtype = inp.dtype
        if is_integer_dtype(dtype) or is_boolean_dtype(dtype):
            dtype = torch.int64
    if out is None:
        out = torch.empty_like(inp, dtype=dtype)

    if inp.numel() == 0:
        return out

    shape = inp.shape
    M = 1
    N = shape[dim]
    for i in range(dim):
        M *= shape[i]
    K = inp.numel() // M // N
    compute_dtype = out.dtype
    if inp.dtype == torch.float16 or inp.dtype == torch.bfloat16:
        compute_dtype = torch.float32

    if K == 1:  # row scan
        reduce_then_scan_row(inp, out, M, N, compute_dtype)
    else:  # col scan
        scan_then_fan(inp, out, M, N, K, compute_dtype)

    return out


def reduce_then_scan_row(x, out, M, N, compute_dtype):
    if N <= 16384:  # persistent
        TILE_SIZE = triton.next_power_of_2(N)
        num_warps = 8 if TILE_SIZE > 2048 else 4
        reduce_then_scan_root_scan_kernel_row[(M, 1, 1)](
            x, out, N, TILE_SIZE, num_warps=num_warps
        )
        return out

    TILE_SIZE = min(4096, triton.next_power_of_2(N))
    num_warps = 8 if TILE_SIZE > 2048 else 4
    num_tiles = triton.cdiv(N, TILE_SIZE)
    max_ctas = get_num_sms(x.device.index) * 4
    num_ctas = min(num_tiles, max_ctas)
    ROOT_SCAN_TILE_SIZE = triton.next_power_of_2(num_ctas)
    tiles_per_cta = triton.cdiv(num_tiles, num_ctas)
    block_sums = torch.empty(
        (
            M,
            num_ctas,
        ),
        dtype=compute_dtype,
        device=x.device,
    )
    block_inclusive_prefix = torch.empty(
        (
            M,
            num_ctas,
        ),
        dtype=compute_dtype,
        device=x.device,
    )

    # 3-kernel implementation
    reduce_then_scan_block_sum_kernel_row[(M, num_ctas, 1, 1)](
        x, block_sums, N, tiles_per_cta, TILE_SIZE, num_warps=num_warps
    )
    reduce_then_scan_root_scan_kernel_row[(M, 1, 1)](
        block_sums,
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
def reduce_then_scan_block_sum_kernel_row(
    in_ptr,
    block_sum_ptr,
    N,
    tiles_per_cta,
    TILE_SIZE: tl.constexpr,
):
    """The same kernel as the block sum in parallel reduce"""
    pid_n = tl.program_id(1).to(tl.int64)
    pid_m = tl.program_id(0).to(tl.int64)
    num_programs_n = tl.num_programs(1)
    block_offset = pid_n * (tiles_per_cta * TILE_SIZE)
    block_end = min(block_offset + tiles_per_cta * TILE_SIZE, N)

    acc_dtype: tl.constexpr = get_scan_accum_type(in_ptr.type.element_ty)
    acc = tl.zeros((TILE_SIZE,), dtype=acc_dtype)
    for start in range(block_offset, block_end, TILE_SIZE):
        offsets = start + tl.arange(0, TILE_SIZE)
        x = tl.load(in_ptr + pid_m * N + offsets, mask=offsets < N).to(acc_dtype)
        acc += x
    block_sum = tl.sum(acc, 0)
    tl.store(
        block_sum_ptr + pid_m * num_programs_n + pid_n, block_sum, cache_modifier=".cg"
    )


@triton.jit
def reduce_then_scan_root_scan_kernel_row(in_ptr, out_ptr, N, TILE_SIZE: tl.constexpr):
    """Almost The same kernel as the persistent scan kernel"""
    pid = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, TILE_SIZE)
    mask = offsets < N
    acc_dtype: tl.constexpr = get_scan_accum_type(in_ptr.type.element_ty)
    x = tl.load(in_ptr + pid * N + offsets, mask=mask, other=0).to(acc_dtype)
    out = tl.cumsum(x, 0)
    tl.store(out_ptr + pid * N + offsets, out, mask=mask)


@triton.jit
def reduce_then_scan_block_scan_kernel_row(
    in_ptr,
    previous_sum_ptr,
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
    acc_dtype: tl.constexpr = get_scan_accum_type(in_ptr.type.element_ty)

    prefix = tl.load(
        previous_sum_ptr + pid_m * num_tiles_n + pid_n - 1, mask=pid_n > 0, other=0
    ).to(acc_dtype)
    for start in range(block_offset, block_end, TILE_SIZE):
        offsets = start + tl.arange(0, TILE_SIZE)
        mask = offsets < N
        x = tl.load(in_ptr + pid_m * N + offsets, mask=mask).to(acc_dtype)
        tile_scan = prefix + tl.cumsum(x, 0)
        prefix += tl.sum(x, 0)
        tl.store(
            out_ptr + pid_m * N + offsets, tile_scan, mask=mask, cache_modifier=".cg"
        )


def cumsum(inp, dim=1, *, dtype=None):
    logger.debug("GEMS CUMSUM")
    # ``aten::cumsum``'s gradient is generated by autograd (no
    # ``aten::cumsum_backward`` op exists), so an explicit ``flag_gems.cumsum``
    # call under grad mode has to go through our own ``autograd.Function`` to
    # reach the Triton reverse-scan kernel. Under ``use_gems()`` the ATen
    # ``CumsumBackward0`` node stays above the device kernel and keeps using
    # the native formula, exactly like ``logcumsumexp`` and ``baddbmm``.
    if torch.is_grad_enabled() and isinstance(inp, torch.Tensor) and inp.requires_grad:
        return CumsumFunction.apply(inp, dim, dtype)
    return cumsum_wrapper(inp, dim, dtype)


def cumsum_out(inp, dim=1, *, dtype=None, out):
    logger.debug("GEMS CUMSUM_OUT")
    return cumsum_wrapper(inp, dim, dtype, out)


@libentry()
@triton.jit(do_not_specialize=["K"])
def normed_cumsum_kernel(inp, out, K, BLOCK: tl.constexpr):
    row_start = ext.program_id(0) * K
    row_off = tl.arange(0, BLOCK)
    x = tl.load(inp + row_start + row_off, mask=row_off < K, other=0)
    if x.dtype.is_fp16():
        x = x.to(tl.float32)
    y_sum = tl.sum(x, 0)
    y = tl.cumsum(x, 0)
    y = y / y_sum
    tl.store(out + row_start + row_off, y, mask=row_off < K)


@libentry()
@triton.jit(
    do_not_specialize=[
        "r",
        "t",
        "R",
        "K",
        "r_stride",
        "out_r_stride",
    ]
)
def block_cumsum_kernel(
    inp,
    out,
    sums,
    r,
    t,
    R,
    K,
    r_stride,
    k_stride,
    out_r_stride,
    out_k_stride,
    OUTPUT_SUMS: tl.constexpr,
    NORMALIZE: tl.constexpr,
    HAS_OUT_LAYOUT: tl.constexpr,
    TILE: tl.constexpr,
):
    # One CTA processes a (r, t*tile) chunk
    # rows = [ grid.y, grid.y + r )
    # cols = [ grid.x * t * tile, (grid.x + 1) * t * tile )
    gridx = ext.program_id(0).to(tl.int64)
    gridy = ext.program_id(1).to(tl.int64)
    n_chunks = ext.num_programs(0)

    for row in range(gridy * r, min((gridy + 1) * r, R)):
        curr_cumsum = tl.zeros((1,), tl.float32)
        row_offset = row * r_stride
        cols = gridx * t * TILE + tl.arange(0, TILE)
        for ti in range(0, t):
            cols_offset = cols * k_stride
            x = tl.load(inp + row_offset + cols_offset, mask=cols < K, other=0)
            if x.dtype.is_fp16() | x.dtype.is_bf16():
                x = x.to(tl.float32)
            tile_sum = tl.sum(x, 0)[None]
            tile_cumsum = tl.cumsum(x, 0) + curr_cumsum
            curr_cumsum += tile_sum
            if HAS_OUT_LAYOUT:
                cols_offset = cols * out_k_stride
                row_offset = row * out_r_stride
            tl.store(out + row_offset + cols_offset, tile_cumsum, mask=cols < K)
            if OUTPUT_SUMS:
                tl.store(sums + row * n_chunks + gridx[None], curr_cumsum)
            cols += TILE
        if NORMALIZE:
            cols = gridx * t * TILE + tl.arange(0, TILE)
            for _ in range(0, t):
                cols_offset = cols * k_stride
                if HAS_OUT_LAYOUT:
                    cols_offset = cols * out_k_stride
                    row_offset = row * out_r_stride
                x = tl.load(out + row_offset + cols_offset, mask=cols < K, other=0)
                if x.dtype.is_fp16() | x.dtype.is_bf16():
                    x = x.to(tl.float32)
                x = x / curr_cumsum
                tl.store(out + row_offset + cols_offset, x, mask=cols < K)
                cols += TILE


@libentry()
@triton.jit(
    do_not_specialize=[
        "r",
        "t",
        "R",
        "K",
        "r_stride",
        "out_r_stride",
    ]
)
def block_update_kernel(
    inp,
    base,
    rscale_ptr,
    out,
    r,
    t,
    R,
    K,
    r_stride,
    k_stride,
    out_r_stride,
    out_k_stride,
    rscale_stride,
    HAS_OUT_LAYOUT: tl.constexpr,
    TILE: tl.constexpr,
):
    # One CTA processes a (r, t*tile) chunk
    # rows = [ grid.y, grid.y + r )
    # cols = [ grid.x * t * tile, (grid.x + 1) * t * tile )
    gridx = ext.program_id(0).to(tl.int64)
    gridy = ext.program_id(1).to(tl.int64)
    n_gridx = ext.num_programs(1)

    base += gridy * n_gridx + gridx
    rscale_ptr += gridy * rscale_stride

    for row in range(gridy, min(gridy + r, R)):
        d = tl.load(base)
        rscale = tl.load(rscale_ptr)
        base += gridx
        rscale_ptr += rscale_stride
        row_offset = row * r_stride
        cols = gridx * t * TILE + tl.arange(0, TILE)
        for _ in range(0, t):
            cols_offset = cols * k_stride
            x = tl.load(inp + row_offset + cols_offset, mask=cols < K, other=0)
            x += d
            x /= rscale
            if HAS_OUT_LAYOUT:
                cols_offset = cols * out_k_stride
                row_offset = row * out_r_stride
            tl.store(out + row_offset + cols_offset, x, mask=cols < K)
            cols += TILE


GRID_Y_LIMIT = 65535


def normed_cumsum(inp, dim=-1):
    logger.debug("GEMS NORMED_CUMSUM")
    assert inp.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    dim = dim % inp.ndim
    N = inp.numel()
    K = inp.size(dim)
    # inp = inp.contiguous()
    # First and last dims are easier to handle, but transpose the middle dim to the last
    ranked_dims = sorted(range(inp.ndim), key=lambda i: inp.stride(i), reverse=True)
    is_mid_dim = dim not in (ranked_dims[0], ranked_dims[-1])
    if is_mid_dim:
        inp = inp.transpose(dim, -1).contiguous()
        dim = -1
    out = torch.empty_like(inp)
    with torch_device_fn.device(inp.device.index):
        # Pass one, scan a (batch, n_tiles * TILE) sized block within each cta
        num_sms = get_device_properties(device).multi_processor_count
        TILE = 2048
        # Each row is split into n_chunks of chunks where each chunk is compised of
        # n_tiles of tiles. Different chunks are assigned to different ctas.
        n_rows = N // K
        n_chunks = min(triton.cdiv(num_sms, n_rows), triton.cdiv(K, TILE))
        n_tiles = triton.cdiv(triton.cdiv(K, TILE), n_chunks)
        k_stride = inp.stride(dim)
        r_stride = inp.size(dim) if k_stride == 1 else 1
        if n_rows > GRID_Y_LIMIT:
            batch = triton.cdiv(n_rows, GRID_Y_LIMIT)
            n_batch = triton.cdiv(n_rows, batch)
        else:
            batch = 1
            n_batch = n_rows

        grid = (n_chunks, n_batch)
        if n_chunks == 1:
            block_cumsum_kernel[grid](
                inp,
                out,
                0,
                batch,
                n_tiles,
                n_rows,
                K,
                r_stride,
                k_stride,
                r_stride,
                k_stride,
                OUTPUT_SUMS=False,
                NORMALIZE=True,
                HAS_OUT_LAYOUT=False,
                TILE=TILE,
            )
            return out

        if inp.dtype != torch.float64:
            acc_dtype = torch.float32
        sums = torch.empty((n_rows, n_chunks), dtype=acc_dtype, device=device)
        cumsums = torch.empty_like(sums)
        block_cumsum_kernel[grid](
            inp,
            out,
            sums,
            batch,
            n_tiles,
            n_rows,
            K,
            r_stride,
            k_stride,
            r_stride,
            k_stride,
            OUTPUT_SUMS=True,
            NORMALIZE=False,
            HAS_OUT_LAYOUT=False,
            TILE=TILE,
        )
        # Pass two, scan partial cumsums
        block_cumsum_kernel[(1, n_batch)](
            sums,
            cumsums,
            0,
            batch,
            1,
            n_rows,
            n_chunks,
            n_chunks,
            1,
            n_chunks,
            1,
            OUTPUT_SUMS=False,
            NORMALIZE=False,
            HAS_OUT_LAYOUT=True,
            TILE=TILE,
        )
        # print(sums)
        rscale = cumsums[..., -1]
        block_update_kernel[grid](
            out,
            cumsums - sums,
            rscale,
            out,
            batch,
            n_tiles,
            n_rows,
            K,
            r_stride,
            k_stride,
            r_stride,
            k_stride,
            n_chunks,
            HAS_OUT_LAYOUT=False,
            TILE=TILE,
        )
        return out


# ---------------------------------------------------------------------------
# Backward of ``cumsum``
#
# ``aten::cumsum`` has no dedicated backward operator: PyTorch derives it in
# ``tools/autograd/derivatives.yaml`` into ``torch/csrc/autograd/FunctionsManual.cpp``
# (a flip + ``aten::cumsum`` + flip chain), so it cannot be replaced by an
# ``aten.*_backward`` dispatch entry. Instead, like ``logcumsumexp``/``baddbmm``,
# the first-order gradient is provided through ``torch.autograd.Function``.
#
# Mathematics: ``y_i = sum_{j <= i} x_j`` along ``dim``, so
#
#     dL/dx_i = sum_{j >= i} dL/dy_j
#
# i.e. a *reverse* cumulative sum of the upstream gradient along ``dim``.
# ``input`` contributes only shape/dtype/device.
#
# Implementation: one kernel launch, every element read and written once.
# ``tl.cumsum`` on the reversed index order gives the inclusive suffix sum, so
# no flip kernel (and hence no extra full-tensor copy) is needed. Accumulation
# is in float32, matching the numerical convention used by the other FlagGems
# cumulative backward kernels.
# ---------------------------------------------------------------------------


# Reduce axes longer than this no longer fit one scan tile without spilling, so
# ``cumsum_backward`` splits them into chunks with a cross-chunk offset pass.
_CUMSUM_BACKWARD_MAX_SINGLE_TILE = 1024


@triton.jit
def _cumsum_backward_kernel(
    grad_ptr,
    grad_in_ptr,
    reduction_size,
    inner_size,
    BLOCK_L: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    """Reverse cumulative sum along the reduce axis of an [outer, L, inner] view.

    Program id 0 indexes an outer slice, program id 1 a tile of ``BLOCK_I``
    contiguous inner lanes, so loads and stores stay coalesced along the
    innermost axis. Rows are walked in memory order while the arithmetic runs
    over ``rev = L - 1 - row``, which turns the inclusive prefix sum into the
    inclusive suffix sum required by the gradient.
    """
    outer = tl.program_id(0).to(tl.int64)
    inner_tile = tl.program_id(1)

    base = outer * reduction_size * inner_size + inner_tile * BLOCK_I

    rows = tl.arange(0, BLOCK_L)
    cols = tl.arange(0, BLOCK_I)
    rev = reduction_size - 1 - rows
    mask = (rows < reduction_size)[:, None] & (
        inner_tile * BLOCK_I + cols < inner_size
    )[None, :]

    ptrs = base + rev[:, None] * inner_size + cols[None, :]
    g = tl.load(grad_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
    cs = tl.cumsum(g, axis=0)
    tl.store(grad_in_ptr + ptrs, cs.to(grad_in_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _cumsum_backward_chunk_kernel(
    grad_ptr,
    grad_in_ptr,
    totals_ptr,
    reduction_size,
    inner_size,
    num_chunks,
    CHUNK_L: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    """Per-chunk reverse scan used when the reduce axis is too long for one tile.

    One program owns a single ``CHUNK_L`` slice of the reduce axis for a block
    of ``BLOCK_I`` inner lanes, so the grid spans
    ``outer * cdiv(inner_size, BLOCK_I) * num_chunks`` programs. Each program
    writes its local suffix sum and the chunk total, which
    ``_cumsum_backward_offset_kernel`` later turns into the cross-chunk offsets.
    """
    outer = tl.program_id(0).to(tl.int64)
    inner_tile = tl.program_id(1)
    chunk = tl.program_id(2)

    start = chunk * CHUNK_L
    # Walk the chunk from its highest row downwards so ``tl.cumsum`` is already
    # the suffix sum. Rows past the end of a partial top chunk read as 0.0,
    # which leaves the valid entries of the scan unchanged.
    rows = start + (CHUNK_L - 1 - tl.arange(0, CHUNK_L))
    cols = tl.arange(0, BLOCK_I)
    col_mask = inner_tile * BLOCK_I + cols < inner_size
    mask = (rows < reduction_size)[:, None] & col_mask[None, :]

    ptrs = (
        (outer * reduction_size + rows[:, None]) * inner_size
        + inner_tile * BLOCK_I
        + cols[None, :]
    )
    g = tl.load(grad_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
    cs = tl.cumsum(g, axis=0)
    tl.store(grad_in_ptr + ptrs, cs.to(grad_in_ptr.dtype.element_ty), mask=mask)

    total = tl.sum(g, axis=0)
    totals_ptrs = (
        outer * inner_size + inner_tile * BLOCK_I + cols
    ) * num_chunks + chunk
    tl.store(totals_ptr + totals_ptrs, total, mask=col_mask)


@triton.jit
def _cumsum_backward_offset_kernel(
    grad_in_ptr,
    totals_ptr,
    reduction_size,
    inner_size,
    num_chunks,
    CHUNK_L: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Add the suffix sum of all higher chunks to each element.

    Reuses the chunk grid of ``_cumsum_backward_chunk_kernel``. The chunk
    totals are small (``num_chunks`` per inner lane), so each program folds the
    ones above its own chunk in registers instead of paying another full pass.
    """
    outer = tl.program_id(0).to(tl.int64)
    inner_tile = tl.program_id(1)
    chunk = tl.program_id(2)

    start = chunk * CHUNK_L
    rows = start + (CHUNK_L - 1 - tl.arange(0, CHUNK_L))
    cols = tl.arange(0, BLOCK_I)
    col_mask = inner_tile * BLOCK_I + cols < inner_size
    mask = (rows < reduction_size)[:, None] & col_mask[None, :]

    cvec = tl.arange(0, BLOCK_C)
    chunk_mask = (cvec[None, :] > chunk) & (cvec[None, :] < num_chunks)
    totals_ptrs = (
        outer * inner_size + inner_tile * BLOCK_I + cols[:, None]
    ) * num_chunks + cvec[None, :]
    totals = tl.load(totals_ptr + totals_ptrs, mask=chunk_mask & col_mask[:, None])
    offset = tl.sum(totals, axis=1)

    ptrs = (
        (outer * reduction_size + rows[:, None]) * inner_size
        + inner_tile * BLOCK_I
        + cols[None, :]
    )
    out = tl.load(grad_in_ptr + ptrs, mask=mask, other=0.0)
    tl.store(
        grad_in_ptr + ptrs,
        (out + offset[None, :]).to(grad_in_ptr.dtype.element_ty),
        mask=mask,
    )


def _cumsum_backward_inner_block(block_l: int, inner_size: int) -> int:
    """Pick ``BLOCK_I`` (contiguous inner lanes per program).

    The launch has ``outer * cdiv(inner_size, BLOCK_I)`` programs. A long
    reduce axis already supplies plenty of work per program, so the tile stays
    narrow and keeps the inner lanes contiguous; short rows trade some of that
    coalescing for enough programs to fill the device. The thresholds come from
    sweeping ``BLOCK_I`` over the benchmark shapes.
    """
    if inner_size <= 1:
        return 1
    if block_l >= 512:
        return 4
    if block_l >= 256:
        return 16
    return 32


def cumsum_backward(grad_output, input, dim=None):
    """Reverse cumulative sum of ``grad_output`` along ``dim``.

    Args:
        grad_output: Gradient w.r.t. the ``cumsum`` output.
        input: Original input tensor (used only for shape/dtype/device).
        dim: Cumulative axis. When omitted the last dimension is used.

    Returns:
        Gradient w.r.t. the input, with the same shape and dtype as ``input``.
    """
    logger.debug("GEMS CUMSUM_BACKWARD")

    ndim = grad_output.ndim
    if ndim == 0:
        # A 0-D scan has a single element, so the suffix sum is the gradient.
        return grad_output.to(input.dtype).clone()

    if dim is None:
        dim = ndim - 1
    dim = int(dim) % ndim

    grad = grad_output.contiguous()
    grad_in = torch.empty(input.shape, dtype=input.dtype, device=input.device)

    reduction_size = input.shape[dim]
    if input.numel() == 0 or reduction_size == 0:
        return grad_in
    if reduction_size == 1:
        # The suffix sum of a length-1 axis is the element itself.
        grad_in.copy_(grad_output.reshape(input.shape))
        return grad_in

    inner_size = 1
    for i in range(dim + 1, ndim):
        inner_size *= input.shape[i]
    outer_size = 1
    for i in range(dim):
        outer_size *= input.shape[i]

    with torch_device_fn.device(grad.device):
        block_l = triton.next_power_of_2(reduction_size)
        if block_l <= _CUMSUM_BACKWARD_MAX_SINGLE_TILE:
            # One program per (outer slice, inner lane block) is enough: the
            # whole reduce axis fits a single scan tile.
            block_i = _cumsum_backward_inner_block(block_l, inner_size)
            grid = (outer_size, triton.cdiv(inner_size, block_i))
            _cumsum_backward_kernel[grid](
                grad,
                grad_in,
                reduction_size,
                inner_size,
                BLOCK_L=block_l,
                BLOCK_I=block_i,
            )
            return grad_in

        # A long reduce axis would need a huge register tile, so split it into
        # ``num_chunks`` scans of ``CHUNK_L`` rows plus a cross-chunk offset
        # pass. Both passes stay coalesced along the inner axis.
        chunk_l = _CUMSUM_BACKWARD_MAX_SINGLE_TILE
        block_i = _cumsum_backward_inner_block(chunk_l, inner_size)
        num_chunks = triton.cdiv(reduction_size, chunk_l)
        totals = torch.empty(
            (outer_size, inner_size, num_chunks),
            dtype=torch.float32,
            device=grad.device,
        )
        grid = (outer_size, triton.cdiv(inner_size, block_i), num_chunks)
        _cumsum_backward_chunk_kernel[grid](
            grad,
            grad_in,
            totals,
            reduction_size,
            inner_size,
            num_chunks,
            CHUNK_L=chunk_l,
            BLOCK_I=block_i,
        )
        block_c = triton.next_power_of_2(num_chunks)
        _cumsum_backward_offset_kernel[grid](
            grad_in,
            totals,
            reduction_size,
            inner_size,
            num_chunks,
            CHUNK_L=chunk_l,
            BLOCK_I=block_i,
            BLOCK_C=block_c,
        )
    return grad_in


class CumsumFunction(torch.autograd.Function):
    """Autograd entry that routes the first-order gradient to the Triton kernel.

    The second-order derivative (``cumsum`` of the incoming gradient) falls back
    to autograd's own differentiation of the kernel call, exactly as for the
    other FlagGems ``autograd.Function`` operators.
    """

    @staticmethod
    def forward(ctx, inp, dim=1, dtype=None):
        if inp.ndim == 0:
            if dim not in (0, -1):
                raise IndexError(
                    "Dimension out of range (expected to be in range of [-1, 0])"
                )
            # A 0-D scan is the identity; ``cumsum_wrapper`` expects at least one
            # dimension, so handle it here and let the backward return the
            # incoming gradient unchanged.
            dim = 0
            out = inp.to(dtype) if dtype is not None else inp.clone()
        else:
            if dim < -inp.ndim or dim >= inp.ndim:
                raise IndexError(
                    "Dimension out of range (expected to be in range of "
                    f"[-{inp.ndim}, {inp.ndim - 1}])"
                )
            dim = dim % inp.ndim
            out = cumsum_wrapper(inp, dim, dtype)
        if ctx.needs_input_grad[0]:
            # ``input`` only supplies shape/dtype/device to the backward, so a
            # detached reference is enough (and keeps the graph single-use).
            ctx.save_for_backward(inp.detach())
            ctx.dim = dim
        return out

    @staticmethod
    def backward(ctx, grad_output):
        if not ctx.needs_input_grad[0]:
            return None, None, None
        (inp,) = ctx.saved_tensors
        grad_in = cumsum_backward(grad_output, inp, ctx.dim)
        return grad_in, None, None
