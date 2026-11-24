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

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.limits import get_dtype_max, get_dtype_min

logger = logging.getLogger(__name__)

# GCU400 reduction-affinity parameters. Calibrated on board in two rounds:
#   - 2026-11-13 (task j20261113032234_423a8985): baseline of the strip-persistent + nomask hot loop.
#   - 2026-11-18 (Triton GCU 4.1.6 recalibration, per-size sweep probe4): flat switched to first-block initialization
#     + native-width accumulator (fp16 native accumulation 512M 560µs/1.92TB/s, fp32 acc 1207µs);
#     fp32 is optimal at B8192/g48 across all sizes (1M 11.9µs / 512M 1113µs); bf16 uses a uint16
#     monotonic-key kernel (sign-bit flip mapping; small sizes B16384 1M 11.2µs, large sizes B32768/nw8
#     512M 910µs/1.18TB/s); row reduction changed to "per-tile axis=1 reduction + narrow vector (BM,) carrying
#     accumulation", removing the triple overhead of padding / elementwise updates / a final large reduction of the
# (BM,BN) big accumulator
#     (512M dim=-1: fp16 2032→700µs, bf16 2543→700µs, fp32 2610→887µs).
#     grid uniformly 48 (24 SIP × 2 program/SP); a non-persistent grid degrades catastrophically in measurements.
_FLAT_GRID_CAP = 48
_FLAT_BLOCK_FP16 = 16384
_FLAT_BLOCK_FP32 = 8192
_FLAT_BLOCK_INT = 16384
_FLAT_BF16_KEY_CROSSOVER = 8 * 1024 * 1024
_FLAT_BLOCK_BF16_KEY_SMALL = 16384
_FLAT_BLOCK_BF16_KEY_LARGE = 32768
_FLAT_WARPS_BF16_KEY_LARGE = 8
_FLAT_SERIAL_BLOCK_CAP = 16384
_ROW_BLOCK_M = 128
_ROW_BLOCK_N_CAP = 512
_ROW_GRID_CAP = 48
_ROW_GRID_LIMIT = 65535
_STRIDED_BLOCK_R = 32


@triton.jit
def _aminmax_tile_min(acc, values, IS_FLOAT: tl.constexpr):
    if IS_FLOAT:
        return tl.minimum(acc, values, propagate_nan=tl.PropagateNan.ALL)
    return tl.minimum(acc, values)


@triton.jit
def _aminmax_tile_max(acc, values, IS_FLOAT: tl.constexpr):
    if IS_FLOAT:
        return tl.maximum(acc, values, propagate_nan=tl.PropagateNan.ALL)
    return tl.maximum(acc, values)


@libentry()
@triton.jit
def aminmax_flat_partial_kernel(
    inp,
    min_partial,
    max_partial,
    num_elements,
    num_full_tiles,
    BLOCK: tl.constexpr,
    GRID: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    IS_FLOAT: tl.constexpr,
):
    """Stage 1: ``GRID`` CTAs stride over stripes, each writing one partial extremum.

    The caller guarantees ``GRID <= num_full_tiles``, so each CTA initializes from the first block
    load and no extreme fill constant is needed; full tiles run mask-free and CTA 0 handles the tail.
    """
    pid = tl.program_id(0)
    lane_offset = tl.arange(0, BLOCK)
    acc_min = tl.load(inp + pid * BLOCK + lane_offset).to(ACC_DTYPE)
    acc_max = acc_min
    for tile_index in range(pid + GRID, num_full_tiles, GRID):
        element_offset = tile_index * BLOCK + lane_offset
        values = tl.load(inp + element_offset).to(ACC_DTYPE)
        acc_min = _aminmax_tile_min(acc_min, values, IS_FLOAT)
        acc_max = _aminmax_tile_max(acc_max, values, IS_FLOAT)
    tail_start = num_full_tiles * BLOCK
    if tail_start < num_elements:
        if pid == 0:
            tail_offset = tail_start + lane_offset
            tail_mask = tail_offset < num_elements
            tail_values = tl.load(inp + tail_offset, mask=tail_mask, other=0).to(
                ACC_DTYPE
            )
            acc_min = _aminmax_tile_min(
                acc_min, tl.where(tail_mask, tail_values, acc_min), IS_FLOAT
            )
            acc_max = _aminmax_tile_max(
                acc_max, tl.where(tail_mask, tail_values, acc_max), IS_FLOAT
            )
    tl.store(min_partial + pid, tl.min(acc_min).to(min_partial.type.element_ty))
    tl.store(max_partial + pid, tl.max(acc_max).to(max_partial.type.element_ty))


@libentry()
@triton.jit
def aminmax_flat_key_partial_kernel(
    inp,
    min_partial,
    max_partial,
    num_elements,
    num_full_tiles,
    BLOCK: tl.constexpr,
    GRID: tl.constexpr,
):
    """bf16-only: uint16 monotonic-key reduction.

    bf16's ``tl.minimum`` is promoted to fp32 by the backend (native accumulation fails to compile),
    and fp32 accumulation halves lane throughput versus 16-bit. Mapping bf16 bit patterns through
    ``u ^ ((u>>15)*0x7FFF + 0x8000)`` to unsigned 16-bit keys ordered like floats lets min/max run
    in the uint16 domain, with the inverse map done once per CTA. Note: NaN is just a large number
    in the key domain with no propagation guarantee (same-level approximation as the
    original minnum NaN drop; aten::aminmax's dim path has no NaN cases).
    """
    pid = tl.program_id(0)
    lane_offset = tl.arange(0, BLOCK)
    raw = tl.load(inp + pid * BLOCK + lane_offset).to(tl.uint16, bitcast=True)
    key = raw ^ ((raw >> 15) * 0x7FFF + 0x8000)
    key_min = key
    key_max = key
    for tile_index in range(pid + GRID, num_full_tiles, GRID):
        element_offset = tile_index * BLOCK + lane_offset
        raw = tl.load(inp + element_offset).to(tl.uint16, bitcast=True)
        key = raw ^ ((raw >> 15) * 0x7FFF + 0x8000)
        key_min = tl.minimum(key_min, key)
        key_max = tl.maximum(key_max, key)
    tail_start = num_full_tiles * BLOCK
    if tail_start < num_elements:
        if pid == 0:
            tail_offset = tail_start + lane_offset
            tail_mask = tail_offset < num_elements
            raw = tl.load(inp + tail_offset, mask=tail_mask, other=0).to(
                tl.uint16, bitcast=True
            )
            key = raw ^ ((raw >> 15) * 0x7FFF + 0x8000)
            key_min = tl.minimum(key_min, tl.where(tail_mask, key, key_min))
            key_max = tl.maximum(key_max, tl.where(tail_mask, key, key_max))
    key_min = tl.min(key_min)
    key_max = tl.max(key_max)
    mask_min = tl.where(key_min >= 0x8000, 0x8000, 0xFFFF).to(tl.uint16)
    mask_max = tl.where(key_max >= 0x8000, 0x8000, 0xFFFF).to(tl.uint16)
    bits_min = (key_min ^ mask_min).to(tl.uint16)
    bits_max = (key_max ^ mask_max).to(tl.uint16)
    tl.store(min_partial + pid, bits_min.to(min_partial.type.element_ty, bitcast=True))
    tl.store(max_partial + pid, bits_max.to(max_partial.type.element_ty, bitcast=True))


@libentry()
@triton.jit
def aminmax_flat_serial_kernel(
    inp,
    min_out,
    max_out,
    num_elements,
    num_full_tiles,
    BLOCK: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    IS_FLOAT: tl.constexpr,
):
    """Single-CTA direct version: one sweep writes the scalar output, skipping stage 2."""
    lane_offset = tl.arange(0, BLOCK)
    first_valid = lane_offset < num_elements
    clamped = tl.where(first_valid, lane_offset, 0)
    acc_min = tl.load(inp + clamped).to(ACC_DTYPE)
    acc_max = acc_min
    for tile_index in range(1, num_full_tiles, 1):
        element_offset = tile_index * BLOCK + lane_offset
        values = tl.load(inp + element_offset).to(ACC_DTYPE)
        acc_min = _aminmax_tile_min(acc_min, values, IS_FLOAT)
        acc_max = _aminmax_tile_max(acc_max, values, IS_FLOAT)
    tail_start = num_full_tiles * BLOCK
    if tail_start < num_elements:
        tail_offset = tail_start + lane_offset
        tail_mask = tail_offset < num_elements
        tail_values = tl.load(inp + tail_offset, mask=tail_mask, other=0).to(ACC_DTYPE)
        acc_min = _aminmax_tile_min(
            acc_min, tl.where(tail_mask, tail_values, acc_min), IS_FLOAT
        )
        acc_max = _aminmax_tile_max(
            acc_max, tl.where(tail_mask, tail_values, acc_max), IS_FLOAT
        )
    tl.store(min_out, tl.min(acc_min).to(min_out.type.element_ty))
    tl.store(max_out, tl.max(acc_max).to(max_out.type.element_ty))


@libentry()
@triton.jit
def aminmax_flat_final_kernel(
    min_partial,
    max_partial,
    min_out,
    max_out,
    P: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_F)
    valid = offsets < P
    dtype = min_partial.type.element_ty
    partial_min = tl.load(min_partial + offsets, mask=valid, other=get_dtype_max(dtype))
    partial_max = tl.load(max_partial + offsets, mask=valid, other=get_dtype_min(dtype))
    tl.store(min_out, tl.min(partial_min))
    tl.store(max_out, tl.max(partial_max))


@libentry()
@triton.jit
def aminmax_row_kernel(
    inp,
    min_out,
    max_out,
    M,
    N,
    NUM_GROUPS,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GRID: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    """Row reduction (reduced dim last, (M, N) contiguous after compression): each group handles BLOCK_M rows.

    2026-11-18 refactor: no longer carries a large (BLOCK_M, BLOCK_N) accumulator (fill + per-element
    update + final axis reduction, three costs plus DSM materialization); each tile immediately
    reduces along axis=1 into a narrow (BLOCK_M,) vector and merges -- 512M dim=-1 fp16 2032->700us,
    bf16 2543->700us. Dropping NaN in the tree matches the original (minnum merge + final tree).
    """
    pid = ext.program_id(0)
    dtype = inp.type.element_ty
    acc_dtype = tl.float32 if dtype is tl.bfloat16 else dtype
    dtype_max = get_dtype_max(acc_dtype)
    dtype_min = get_dtype_min(acc_dtype)
    num_full = N // BLOCK_N
    cols_all = tl.arange(0, BLOCK_N)[None, :]
    row_lane = tl.arange(0, BLOCK_M)
    for group in range(pid, NUM_GROUPS, GRID):
        rows = group * BLOCK_M + row_lane[:, None]
        base = inp + rows * N
        if EVEN_M and EVEN_N:
            first = tl.load(base + cols_all).to(acc_dtype)
            g_min = tl.min(first, axis=1)
            g_max = tl.max(first, axis=1)
            for tile_index in range(1, num_full):
                cols = tile_index * BLOCK_N + cols_all
                values = tl.load(base + cols).to(acc_dtype)
                g_min = tl.minimum(g_min, tl.min(values, axis=1))
                g_max = tl.maximum(g_max, tl.max(values, axis=1))
        else:
            row_mask = rows < M
            col_mask = cols_all < N
            tile_mask = row_mask & col_mask
            first = tl.load(base + cols_all, mask=tile_mask, other=0).to(acc_dtype)
            g_min = tl.min(tl.where(tile_mask, first, dtype_max), axis=1)
            g_max = tl.max(tl.where(tile_mask, first, dtype_min), axis=1)
            for tile_index in range(1, num_full):
                cols = tile_index * BLOCK_N + cols_all
                values = tl.load(base + cols, mask=tile_mask, other=0).to(acc_dtype)
                g_min = tl.minimum(
                    g_min, tl.min(tl.where(tile_mask, values, dtype_max), axis=1)
                )
                g_max = tl.maximum(
                    g_max, tl.max(tl.where(tile_mask, values, dtype_min), axis=1)
                )
            tail_start = num_full * BLOCK_N
            if num_full >= 1:
                if tail_start < N:
                    cols = tail_start + cols_all
                    tail_mask = row_mask & (cols < N)
                    values = tl.load(base + cols, mask=tail_mask, other=0).to(acc_dtype)
                    g_min = tl.minimum(
                        g_min, tl.min(tl.where(tail_mask, values, dtype_max), axis=1)
                    )
                    g_max = tl.maximum(
                        g_max, tl.max(tl.where(tail_mask, values, dtype_min), axis=1)
                    )
        row_off = group * BLOCK_M + row_lane
        if EVEN_M:
            tl.store(min_out + row_off, g_min.to(dtype))
            tl.store(max_out + row_off, g_max.to(dtype))
        else:
            rm = row_off < M
            tl.store(min_out + row_off, g_min.to(dtype), mask=rm)
            tl.store(max_out + row_off, g_max.to(dtype), mask=rm)


@libentry()
@triton.jit
def aminmax_strided_kernel(
    inp,
    min_out,
    max_out,
    R,
    INNER,
    N_ICHUNKS,
    BR: tl.constexpr,
    BI: tl.constexpr,
    EVEN_I: tl.constexpr,
):
    pid = ext.program_id(0)
    outer_index = pid // N_ICHUNKS
    inner_chunk = pid % N_ICHUNKS
    inner_cols = inner_chunk * BI + tl.arange(0, BI)
    base = inp + outer_index * R * INNER
    dtype = inp.type.element_ty
    acc_dtype = tl.float32 if dtype is tl.bfloat16 else dtype
    dtype_max = get_dtype_max(acc_dtype)
    dtype_min = get_dtype_min(acc_dtype)
    fill_min = tl.full((1,), dtype_min, acc_dtype)
    acc_min = tl.full((BI,), dtype_max, acc_dtype)
    acc_max = tl.full((BI,), dtype_min, acc_dtype)
    num_full = R // BR
    if EVEN_I:
        for tile_index in range(0, num_full):
            rows = tile_index * BR + tl.arange(0, BR)[:, None]
            values = tl.load(base + rows * INNER + inner_cols[None, :]).to(acc_dtype)
            acc_min = tl.minimum(acc_min, tl.min(values, axis=0).to(acc_dtype))
            acc_max = tl.maximum(acc_max, tl.max(values, axis=0).to(acc_dtype))
        start = num_full * BR
        for row in range(start, R):
            values = tl.load(base + row * INNER + inner_cols).to(acc_dtype)
            acc_min = tl.minimum(acc_min, values)
            acc_max = tl.maximum(acc_max, values)
        out_base = min_out + outer_index * INNER + inner_cols
        tl.store(out_base, acc_min.to(dtype))
        tl.store(max_out + outer_index * INNER + inner_cols, acc_max.to(dtype))
    else:
        inner_mask = inner_cols < INNER
        for tile_index in range(0, num_full):
            rows = tile_index * BR + tl.arange(0, BR)[:, None]
            values = tl.load(
                base + rows * INNER + inner_cols[None, :],
                mask=inner_mask[None, :],
                other=dtype_max,
            ).to(acc_dtype)
            acc_min = tl.minimum(acc_min, tl.min(values, axis=0).to(acc_dtype))
            acc_max = tl.maximum(
                acc_max,
                tl.where(inner_mask, tl.max(values, axis=0).to(acc_dtype), fill_min),
            )
        start = num_full * BR
        for row in range(start, R):
            values = tl.load(
                base + row * INNER + inner_cols, mask=inner_mask, other=dtype_max
            ).to(acc_dtype)
            acc_min = tl.minimum(acc_min, values)
            acc_max = tl.maximum(acc_max, tl.where(inner_mask, values, fill_min))
        tl.store(
            min_out + outer_index * INNER + inner_cols,
            acc_min.to(dtype),
            mask=inner_mask,
        )
        tl.store(
            max_out + outer_index * INNER + inner_cols,
            acc_max.to(dtype),
            mask=inner_mask,
        )


def _flat_acc_dtype(dtype):
    """Accumulator dtype: fp16 uses native width (min/max lose no precision, lane throughput doubles); bf16's
    ``tl.minimum`` is promoted to fp32 by the backend, so it takes the uint16 key kernel; fp32/integer stay native.
    """
    if dtype is torch.float16:
        return tl.float16
    if dtype is torch.int64:
        return tl.int64
    if dtype is torch.int32:
        return tl.int32
    return tl.float32


def _aminmax_flat_kernels(inp, min_out, max_out):
    """Whole-tensor reduction: table lookup by dtype/size -> small tensors take the
    single-CTA path, otherwise striped persistent + finalize.

    2026-11-18 on-board calibration (probe4, per-size sweep):
    - fp16: native accumulation b16384/g48/nw4 is best at all sizes (1M 8.7us, 512M 560us/1.92TB/s);
      the has_nan guard costs a fixed +2us and aten::aminmax has no NaN cases (the stock version also
      drops NaN via minnum), so no guard is used; propagate_nan measured zero cost and is kept.
    - fp32: b8192/g48/nw4 is best at all sizes (1M 11.9us, 16.7M 46.1us, 512M 1113us); b32768 only
      ties at 512M but costs 54% more at 1M.
    - bf16: uint16 key kernel; <=8M elements b16384/nw4 (1M 11.2us), >8M b32768/nw8
      (512M 910µs/1.18TB/s).
    - fp32acc is void (fp16 512M 1207us vs native 560us); grid = tile count (non-persistent) is
      catastrophically worse (512M 4162us), so 48 persistent CTAs is a hard constraint.
    """
    numel = inp.numel()
    dtype = inp.dtype
    is_float = inp.is_floating_point()
    use_key = dtype is torch.bfloat16
    if use_key:
        if numel > _FLAT_BF16_KEY_CROSSOVER:
            block = _FLAT_BLOCK_BF16_KEY_LARGE
            num_warps = _FLAT_WARPS_BF16_KEY_LARGE
        else:
            block = _FLAT_BLOCK_BF16_KEY_SMALL
            num_warps = 4
    elif dtype is torch.float32:
        block = _FLAT_BLOCK_FP32
        num_warps = 4
    elif dtype is torch.float16:
        block = _FLAT_BLOCK_FP16
        num_warps = 4
    else:
        block = _FLAT_BLOCK_INT
        num_warps = 4
    num_full = numel // block
    with torch_device_fn.device(inp.device):
        if num_full <= 1:
            serial_block = min(triton.next_power_of_2(numel), _FLAT_SERIAL_BLOCK_CAP)
            aminmax_flat_serial_kernel[(1,)](
                inp,
                min_out,
                max_out,
                numel,
                numel // serial_block,
                serial_block,
                _flat_acc_dtype(dtype),
                is_float,
                num_warps=4,
            )
            return
        grid = min(_FLAT_GRID_CAP, num_full)
        min_partial = torch.empty((grid,), dtype=dtype, device=inp.device)
        max_partial = torch.empty((grid,), dtype=dtype, device=inp.device)
        if use_key:
            aminmax_flat_key_partial_kernel[(grid,)](
                inp,
                min_partial,
                max_partial,
                numel,
                num_full,
                block,
                grid,
                num_warps=num_warps,
            )
        else:
            aminmax_flat_partial_kernel[(grid,)](
                inp,
                min_partial,
                max_partial,
                numel,
                num_full,
                block,
                grid,
                _flat_acc_dtype(dtype),
                is_float,
                num_warps=num_warps,
            )
        aminmax_flat_final_kernel[(1,)](
            min_partial,
            max_partial,
            min_out,
            max_out,
            grid,
            triton.next_power_of_2(grid),
            num_warps=4,
        )


def _aminmax_flat(inp, keepdim, out):
    """Whole-tensor reduction: striped persistent CTAs + one final CTA."""
    numel = inp.numel()
    if numel == 0:
        raise RuntimeError(
            "aminmax(): cannot compute aminmax over an empty dimension as the "
            "operation has no identity."
        )
    if not inp.is_contiguous():
        inp = inp.contiguous()
    dtype = inp.dtype
    if out is not None:
        min_out = out[0] if isinstance(out, tuple) else out
        max_out = out[1] if isinstance(out, tuple) else out
        if not keepdim:
            min_out = min_out.squeeze()
            max_out = max_out.squeeze()
    else:
        if not keepdim:
            min_out = torch.empty([], dtype=dtype, device=inp.device)
            max_out = torch.empty([], dtype=dtype, device=inp.device)
        else:
            shape = [1] * inp.dim()
            min_out = torch.empty(shape, dtype=dtype, device=inp.device)
            max_out = torch.empty(shape, dtype=dtype, device=inp.device)
    _aminmax_flat_kernels(inp, min_out, max_out)
    return min_out, max_out


def _aminmax_row(view, min_out, max_out, M, N):
    """Row-reduction launch: bm=128 / bn=min(np2(N),512) / grid=min(groups,48).

    2026-11-18 on-board calibration (512M dim=-1): bm128/bn512/g48 fp16 700us, bf16 700us,
    fp32 891us; [4096,4096] dim=-1 bf16 52.2->40.8us, fp32 57.7->38.4us. A 512 cap on bn is also
    best for bf16 (the old table's 256-wide accumulation cap was set for the large-accumulator
    structure; after the refactor a tile is dropped as soon as it is reduced, so width no longer constrains).
    """
    block_m = min(_ROW_BLOCK_M, triton.next_power_of_2(M))
    block_n = min(triton.next_power_of_2(N), _ROW_BLOCK_N_CAP)
    even_m = (M % block_m) == 0
    even_n = (N % block_n) == 0
    if not (even_m and even_n):
        # The masked branch holds both values and min/max where intermediates (three 2D
        # tiles in total): a wide accumulation dtype (fp32/bf16/int64) at bm128×bn512 exceeds the 896KB/SP
        # DSM budget (measured Required 958464 > 917504), so shrink to 64×256.
        wide_acc = view.element_size() >= 4 or view.dtype is torch.bfloat16
        block_m = min(block_m, 64)
        if wide_acc:
            block_n = min(block_n, 256)
        even_m = (M % block_m) == 0
        even_n = (N % block_n) == 0
    num_groups = triton.cdiv(M, block_m)
    grid = min(num_groups, _ROW_GRID_CAP)
    with torch_device_fn.device(view.device):
        aminmax_row_kernel[(grid,)](
            view,
            min_out,
            max_out,
            M,
            N,
            num_groups,
            block_m,
            block_n,
            grid,
            even_m,
            even_n,
            num_warps=4,
        )


def _aminmax_strided(inp, min_out, max_out, R, inner, outer):
    dtype = inp.dtype
    cap = 2048 if dtype is torch.bfloat16 else 1024
    block_inner = min(triton.next_power_of_2(inner), cap)
    n_ichunks = triton.cdiv(inner, block_inner)
    if outer * n_ichunks > _ROW_GRID_LIMIT:
        return False
    grid = (outer * n_ichunks,)
    with torch_device_fn.device(inp.device):
        aminmax_strided_kernel[grid](
            inp,
            min_out,
            max_out,
            R,
            inner,
            n_ichunks,
            _STRIDED_BLOCK_R,
            block_inner,
            (inner % block_inner) == 0,
            num_warps=4,
        )
    return True


def aminmax(inp, dim=None, keepdim=False, *, out=None):
    logger.debug("GEMS_ENFLAME AMINMAX")

    if dim is None:
        return _aminmax_flat(inp, keepdim, out)

    if isinstance(dim, int):
        dim = [dim]
    assert ((i >= -inp.ndim and i < inp.ndim) for i in dim), "Invalid dim"
    dtype = inp.dtype

    shape = list(inp.shape)
    dim = [d % inp.ndim for d in dim]
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = inp.numel() // N

    if M == 1:
        # All reduced elements form the whole tensor, but non-reduced dims may
        # still exist (they all have size 1). Keep the dim-path output shape.
        if out is not None:
            min_out = out[0] if isinstance(out, tuple) else out
            max_out = out[1] if isinstance(out, tuple) else out
        else:
            min_out = torch.empty(shape, dtype=dtype, device=inp.device)
            max_out = torch.empty(shape, dtype=dtype, device=inp.device)
        if not inp.is_contiguous():
            inp = inp.contiguous()
        _aminmax_flat_kernels(inp, min_out, max_out)
        if not keepdim:
            min_out = min_out.squeeze(dim=dim)
            max_out = max_out.squeeze(dim=dim)
        return min_out, max_out

    if out is not None:
        min_out = out[0] if isinstance(out, tuple) else out
        max_out = out[1] if isinstance(out, tuple) else out
    else:
        min_out = torch.empty(shape, dtype=dtype, device=inp.device)
        max_out = torch.empty(shape, dtype=dtype, device=inp.device)

    ndim = inp.ndim
    single_dim = dim[0] if len(dim) == 1 else None
    if single_dim is not None and single_dim != ndim - 1 and inp.is_contiguous():
        inner = 1
        for i in range(single_dim + 1, ndim):
            inner *= shape[i]
        if inner >= 16384 or inp.numel() <= 65536:
            applied = _aminmax_strided(
                inp, min_out, max_out, inp.shape[single_dim], inner, M // inner
            )
            if applied:
                if not keepdim:
                    min_out = min_out.squeeze(dim=dim)
                    max_out = max_out.squeeze(dim=dim)
                return min_out, max_out

    view = dim_compress(inp, dim)
    _aminmax_row(view, min_out, max_out, M, N)
    if not keepdim:
        min_out = min_out.squeeze(dim=dim)
        max_out = max_out.squeeze(dim=dim)
    return min_out, max_out
