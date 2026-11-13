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

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import get_device_properties, libentry

from ..utils.config_utils import MAX_GRID_DIM

logger = logging.getLogger(__name__)


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
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    inp_ptrs = inp + offset
    inp_vals = tl.load(inp_ptrs, mask=mask)
    if (
        tl.constexpr(inp_vals.dtype.is_int64())
        | tl.constexpr(inp_vals.dtype.is_uint64())
    ) | tl.constexpr(inp_vals.dtype.is_fp64()):
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
    pid = tl.program_id(0)
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
    A,
    B,
    C,
    grid_a,
    part_num,
    BLOCK_SIZE: tl.constexpr,
):
    pid_a = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_c = tl.program_id(2)

    a_idx = pid_a
    while a_idx < A:
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
            | tl.constexpr(inp_vals.dtype.is_uint64())
        ) | tl.constexpr(inp_vals.dtype.is_fp64()):
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

        a_idx += grid_a


@libentry()
@triton.jit(do_not_specialize=["part_num"])
def add_base_sum_abc_kernel(
    out,
    partial_sum,
    A,
    B,
    C,
    grid_a,
    part_num,
    BLOCK_SIZE: tl.constexpr,
):
    pid_a = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_c = tl.program_id(2)

    a_idx = pid_a
    while a_idx < A:
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

        a_idx += grid_a


def scan_then_fan_col(inp, out, n_ele, dtype):
    # TODO(all): tune on target board
    BLOCK_SIZE = 8192
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

    grid_a = min(A, MAX_GRID_DIM)
    grid = (grid_a, part_num, C)
    with torch_device_fn.device(inp.device):
        scan_part_sum_abc_kernel[grid](
            inp, out, partial_sum, A, B, C, grid_a, part_num, BLOCK_SIZE
        )

    if part_num >= 2:
        scan_then_fan(partial_sum, partial_sum, A, part_num, C, dtype)
        with torch_device_fn.device(inp.device):
            add_base_sum_abc_kernel[grid](
                out, partial_sum, A, B, C, grid_a, part_num, BLOCK_SIZE
            )


def cumsum_wrapper(inp, dim=1, dtype=None, out=None):
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
            dtype = torch.int32
    if out is None:
        out = torch.empty_like(inp, dtype=dtype)

    compute_dtype = out.dtype
    if inp.dtype == torch.float16 or inp.dtype == torch.bfloat16:
        compute_dtype = torch.float32

    if M == 1 and K == 1:
        scan_then_fan_col(inp, out, N, compute_dtype)
    else:
        scan_then_fan(inp, out, M, N, K, compute_dtype)
    return out


def cumsum(inp, dim=1, *, dtype=None):
    logger.debug("GEMS_ENFLAME CUMSUM")
    return cumsum_wrapper(inp, dim, dtype)


def cumsum_out(inp, dim=1, *, dtype=None, out):
    logger.debug("GEMS_ENFLAME CUMSUM_OUT")
    return cumsum_wrapper(inp, dim, dtype, out)


@libentry()
@triton.jit(do_not_specialize=["K"])
def normed_cumsum_kernel(inp, out, K, BLOCK: tl.constexpr):
    row_start = tl.program_id(0) * K
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
    gridx = tl.program_id(0)
    gridy = tl.program_id(1)
    n_chunks = tl.num_programs(0)

    # gcu300: don't support int64
    # gridx = tl.program_id(0).to(tl.int64)
    # gridy = tl.program_id(1).to(tl.int64)
    gridx = tl.program_id(0)
    gridy = tl.program_id(1)
    n_chunks = tl.num_programs(0)

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
    gridx = tl.program_id(0)
    gridy = tl.program_id(1)
    n_gridx = tl.num_programs(1)

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


@libentry()
@triton.jit(do_not_specialize=["n_chunks"])
def chunk_sums_scan_kernel(sums_ptr, cumsums_ptr, n_chunks, BLOCK: tl.constexpr):
    # Pass two of the multi-chunk normed cumsum path: each program scans the
    # per-chunk partial sums of one row into their prefix form with a single
    # vectorized cumsum over BLOCK lanes (n_chunks <= BLOCK).
    # A dedicated tiny kernel here avoids reusing block_cumsum_kernel with
    # HAS_OUT_LAYOUT=True, whose specialization crashes the GCU compiler
    # (mlir::IRMapping::lookup assertion) at triton_gcu compile time.
    row = tl.program_id(0)
    lane_off = tl.arange(0, BLOCK)
    lane_mask = lane_off < n_chunks
    offsets = row * n_chunks + lane_off
    chunk_sums = tl.load(sums_ptr + offsets, mask=lane_mask, other=0.0)
    prefix_sums = tl.cumsum(chunk_sums, 0)
    tl.store(cumsums_ptr + offsets, prefix_sums, mask=lane_mask)


@libentry()
@triton.jit(do_not_specialize=["K", "n_rows", "rows_per_program"])
def normed_cumsum_rows_kernel(
    inp, out, K, n_rows, rows_per_program, BLOCK: tl.constexpr
):
    # Row-parallel normed cumsum: each program normalizes `rows_per_program`
    # consecutive rows (each row is a full independent scan over BLOCK lanes).
    # Rows are distributed across grid.x programs so short-row tensors get a
    # single fully parallel pass; rows_per_program stays 1 whenever the row
    # count fits the hardware grid.x limit.
    row_base = tl.program_id(0) * rows_per_program
    row_off = tl.arange(0, BLOCK)
    for row_index in range(0, rows_per_program):
        row = row_base + row_index
        if row < n_rows:
            x = tl.load(inp + row * K + row_off, mask=row_off < K, other=0)
            if x.dtype.is_fp16() | x.dtype.is_bf16():
                x = x.to(tl.float32)
            row_sum = tl.sum(x, 0)
            y = tl.cumsum(x, 0)
            y = y / row_sum
            tl.store(out + row * K + row_off, y, mask=row_off < K)


GRID_Y_LIMIT = 255
GRID_X_LIMIT = 65535


def normed_cumsum(inp, dim=-1):
    logger.debug("GEMS_ENFLAME NORMED_CUMSUM")
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
        num_sms = get_device_properties(device).multi_processor_count  # noqa: F841
        # Tile size adapts to K: small rows keep a small tile so the single
        # chunk kernel does not pay for a huge masked BLOCK, while very long
        # rows (1D huge scans) use a large tile to cut the serial loop count.
        TILE = max(2048, min(8192, triton.next_power_of_2(K)))
        # Each row is split into n_chunks of chunks where each chunk is compised of
        # n_tiles of tiles. Different chunks are assigned to different ctas.
        n_rows = N // K
        # multi_processor_count on GCU400 reports 4, which would serialize a
        # huge 1D scan into 4 programs. Split long rows into enough chunks to
        # reach a target program count (1024) instead; rows-heavy tensors get
        # n_chunks = 1 and go through the row-parallel fast path.
        n_chunks = max(1, min(triton.cdiv(K, TILE), triton.cdiv(4096, n_rows)))
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
        # Row-parallel fast path: when rows are short (K <= TILE) and there are
        # many rows, assign one program per row so the whole tensor is scanned
        # in a single fully parallel kernel. The generic block path only spans
        # n_batch programs (<= GRID_Y_LIMIT) and serializes rows inside each
        # program, which is far below the device's parallel capacity. Tiny
        # tensors keep the generic path: measured faster for a single small row
        # group than a per-row kernel launch.
        if n_chunks == 1 and n_rows > GRID_Y_LIMIT and k_stride == 1:
            row_kernel_block = triton.next_power_of_2(K)
            # Cap grid.x at the hardware limit by letting each program handle
            # several consecutive rows when the tensor has too many rows.
            # num_warps is scaled with BLOCK so short rows don't waste lanes.
            rows_per_program = triton.cdiv(n_rows, GRID_X_LIMIT)
            # Very fine-grained programs (one short row each) pay scheduling
            # overhead; the floor is tuned per BLOCK tier (measured optimum).
            if row_kernel_block <= 512:
                rows_per_program_floor = 16
            elif row_kernel_block <= 1024:
                rows_per_program_floor = 128
            else:
                rows_per_program_floor = 8
            if rows_per_program < rows_per_program_floor:
                rows_per_program = rows_per_program_floor
            row_grid = triton.cdiv(n_rows, rows_per_program)
            if row_kernel_block <= 1024:
                row_num_warps = 1
            else:
                row_num_warps = 4
            normed_cumsum_rows_kernel[(row_grid,)](
                inp,
                out,
                K,
                n_rows,
                rows_per_program,
                BLOCK=row_kernel_block,
                num_warps=row_num_warps,
            )
            return out
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
        sums = torch.empty((n_rows, n_chunks), dtype=acc_dtype, device=inp.device)
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
        scan_block = triton.next_power_of_2(n_chunks)
        chunk_sums_scan_kernel[(n_rows,)](sums, cumsums, n_chunks, BLOCK=scan_block)
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
