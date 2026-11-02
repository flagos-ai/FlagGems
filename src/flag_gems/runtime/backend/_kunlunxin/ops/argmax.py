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

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.limits import get_dtype_min

torch_dtype_to_tl_dtype_and_min_value = {
    torch.int16: (tl.int16, torch.iinfo(torch.int16).min),
    torch.int32: (tl.int32, torch.iinfo(torch.int32).min),
    torch.float16: (tl.float16, torch.finfo(torch.float16).min),
    torch.float32: (tl.float32, torch.finfo(torch.float32).min),
    torch.bfloat16: (tl.float32, torch.finfo(torch.float32).min),
}
logger = logging.getLogger(__name__)

# N above this width cannot be reduced (with return_indices) in a single XPU
# tile: both the single-load kernel and the loop-accumulator kernel fail to
# compile ("out of resource: uni_sram"). Such N is handled by the two-stage
# reduction below.
MAX_TILE_N = 8192
# Chunk width / row-tile for the two-stage large-N path (measured fastest on XPU).
STAGE_BLOCK_N = 2048
STAGE_BLOCK_M = 32
# For a contiguous inner reduce (K == 1), route N at or above this width to the
# constexpr two-stage path even when it would fit a single tile: the runtime-N/K
# single-tile kernel degrades to discrete access and is far slower here. Below
# this, single-tile launch overhead wins and the discrete penalty is negligible.
TWO_STAGE_MIN_N = 256


@libentry()
@triton.jit
def argmax_kernel_1(
    inp,
    mid_value,
    mid_index,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    min_value = get_dtype_min(inp.type.element_ty)
    inp_val = tl.load(inp_ptrs, mask=mask, other=min_value)
    max_val, max_index = tl.max(inp_val, axis=0, return_indices=True)
    max_index = max_index + pid * BLOCK_SIZE
    mid_value_ptr = mid_value + pid
    max_index_ptr = mid_index + pid
    tl.store(mid_value_ptr, max_val)
    tl.store(max_index_ptr, max_index)


@libentry()
@triton.jit
def argmax_kernel_2(mid_value, mid_index, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid_value + offset
    mask = offset < mid_size
    min_value = get_dtype_min(mid_value.type.element_ty)
    mid_val = tl.load(mid_ptrs, mask=mask, other=min_value)
    index_val = tl.argmax(mid_val, axis=0)
    mid_index_ptrs = mid_index + index_val
    out_val = tl.load(mid_index_ptrs)
    tl.store(out, out_val)


@libentry()
@triton.jit
def argmax_stage1(
    inp,
    part_val,
    part_idx,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Stage 1 of the large-N path. Each program reduces one BLOCK_N-wide chunk
    # of a BLOCK_M row block and emits the chunk-local max value plus its
    # *global* argmax index. Output is [M, NUM_CHUNKS, K].
    pid_m = ext.program_id(0)
    pid_c = ext.program_id(1)
    pid_k = ext.program_id(2)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    chunk_start = pid_c * BLOCK_N
    n_offset = chunk_start + tl.arange(0, BLOCK_N)
    dtype = inp.type.element_ty
    min_value = get_dtype_min(dtype)
    offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
    mask = m_offset[:, None] < M and n_offset[None, :] < N
    vals = tl.load(inp + offset, mask=mask, other=min_value)
    lmax, largmax = tl.max(
        vals, axis=1, return_indices=True, return_indices_tie_break_left=True
    )
    gidx = chunk_start + largmax
    part_offset = m_offset * NUM_CHUNKS * K + pid_c * K + pid_k
    pmask = m_offset < M
    tl.store(part_val + part_offset, lmax, mask=pmask)
    tl.store(part_idx + part_offset, gidx, mask=pmask)


@libentry()
@triton.jit
def argmax_stage2(
    part_val,
    part_idx,
    out_index,
    M,
    K: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Stage 2: reduce the NUM_CHUNKS per-row partial maxes, then gather the
    # global argmax index of the winning chunk. tie_break_left keeps the
    # earliest chunk on ties, matching torch semantics.
    pid_m = ext.program_id(0)
    pid_k = ext.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    c_offset = tl.arange(0, BLOCK_C)
    dtype = part_val.type.element_ty
    min_value = get_dtype_min(dtype)
    offset = m_offset[:, None] * NUM_CHUNKS * K + c_offset[None, :] * K + pid_k
    mask = m_offset[:, None] < M and c_offset[None, :] < NUM_CHUNKS
    vals = tl.load(part_val + offset, mask=mask, other=min_value)
    _, best_c = tl.max(
        vals, axis=1, return_indices=True, return_indices_tie_break_left=True
    )
    gather = m_offset * NUM_CHUNKS * K + best_c * K + pid_k
    pmask = m_offset < M
    res = tl.load(part_idx + gather, mask=pmask)
    tl.store(out_index + m_offset * K + pid_k, res, mask=pmask)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("argmax"))
@triton.jit
def argmax_kernel_small_n(
    inp,
    out_index,
    M,
    N,
    K,
    tl_dtype: tl.constexpr,
    dtype_min_value: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Single-tile path (N <= MAX_TILE_N so a single load covers all of N).
    # Runtime N/K, matching the proven original form.
    pid_m = ext.program_id(0)
    pid_k = ext.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    if tl_dtype is tl.int16:
        tl_dtype = tl.int32
    n_offset = tl.arange(0, BLOCK_N)
    offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
    offset_index = m_offset * K + pid_k
    mask1 = m_offset < M
    mask = m_offset[:, None] < M and n_offset[None, :] < N
    inp_ptrs = inp + offset
    inp_vals = tl.load(inp_ptrs, mask=mask, other=dtype_min_value)
    _, result_index = tl.max(inp_vals, axis=1, return_indices=True)

    out_index_ptrs = out_index + offset_index

    tl.store(out_index_ptrs, result_index, mask=mask1)


# ---------------------------------------------------------------------------
# Fast mask-free row-reduction path (floats only), ported from argmin.py.
# Fixed bounded tiles that divide both M and N -> the whole kernel compiles
# mask-free (skips the XPU masked-memory slow path), and the reduced dim is
# brought innermost with a native `aten._copy_from` copy when needed.
#
# The (value, index) pair is packed into a single int64 word so a plain
# signed-int64 `tl.min` returns both the winning value and its tie-break-left
# index in one pass (no `return_indices`, which measured ~20x slower on XPU).
#
# argmin packs `(ordered << 30) | idx` and minimizes `ordered`.  argmax needs
# the *maximum* value with the *lowest* index on ties, so it minimizes the
# bitwise complement `key = ~ordered`: maximizing `ordered` == minimizing
# `key`, and on equal values the low 30 index bits still pick the earliest.
# NaN is mapped to the top of the order (key 0) so the first NaN wins, matching
# torch/XPU argmax NaN propagation.
# ---------------------------------------------------------------------------
_FULL_REDUCTION_BLOCK_SIZE = 8192
_FAST_MIN_N = 64
_FAST_BN_FP16 = (1024, 256, 512, 128, 64)
_FAST_BN_FP32 = (512, 256, 1024, 128, 64)
_FAST_BM_FP16 = (128, 64, 32, 256, 512, 16, 8, 4, 2)
_FAST_BM_FP32 = (64, 128, 32, 256, 512, 16, 8, 4, 2)


def _is_fast_dtype(dtype):
    return dtype in (torch.float16, torch.float32, torch.bfloat16)


def _pick_fast_tile(M, N, is_fp32):
    """Return (BLOCK_M, BLOCK_N) with M % BLOCK_M == 0 and N % BLOCK_N == 0,
    or None when no mask-free tile covers this shape."""
    if N < _FAST_MIN_N:
        return None
    bns = _FAST_BN_FP32 if is_fp32 else _FAST_BN_FP16
    bms = _FAST_BM_FP32 if is_fp32 else _FAST_BM_FP16
    bn = next((b for b in bns if N % b == 0), None)
    if bn is None:
        return None
    bm = next((m for m in bms if M % m == 0), None)
    if bm is None:
        return None
    return bm, bn


# fp16/bf16 pack a 16-bit order-preserving key and a 16-bit column index into a
# single int32 word (N <= MAX_TILE_N = 8192 < 2**16), halving the reduction width
# vs the fp32 int64 path (~1.5x faster measured on XPU).
_FAST16_BN = (512, 256, 1024, 128, 64)
_FAST16_BM = (128, 64, 32, 256, 16, 8, 4, 2)


def _pick_fast_tile16(M, N):
    """Return (BLOCK_M, BLOCK_N) mask-free for a 16-bit packed kernel."""
    if N < _FAST_MIN_N:
        return None
    bn = next((b for b in _FAST16_BN if N % b == 0), None)
    if bn is None:
        return None
    bm = next((m for m in _FAST16_BM if M % m == 0), None)
    if bm is None:
        return None
    return bm, bn


_IDX_BITS = tl.constexpr(30)
_IDX_MASK = tl.constexpr((1 << 30) - 1)
# Sentinel that never wins the int64 min: larger than any real packed word
# (real words are (key << 30) | idx with key <= 0xFFFFFFFF, so they top out at
# 0x3FFFFFFFFFFFFFFF).
_PACK_OTHER = tl.constexpr(0x7FFFFFFFFFFFFFFF)


@libentry()
@triton.jit
def argmax_kernel_2d(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IDX_BASE: tl.constexpr,
    IDX_SHIFT: tl.constexpr,
    NEED_MASK: tl.constexpr,
    EXTRACT: tl.constexpr,
):
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows.to(tl.int64) * N
    out_i = out + rows.to(tl.int64)
    row_mask = rows < M

    acc = tl.full([BLOCK_M, 1], value=_PACK_OTHER, dtype=tl.int64)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        if NEED_MASK:
            mask = row_mask and cols < N
            a = tl.load(inp + cols, mask=mask, other=float("-inf")).to(tl.float32)
        else:
            a = tl.load(inp + cols).to(tl.float32)
        u = a.to(tl.int32, bitcast=True)
        is_nan = (u & 0x7FFFFFFF) > 0x7F800000
        neg = u < 0
        ordered = tl.where(neg, ~u, u ^ -2147483648)
        # NaN wins a max (torch propagates NaN): give it the top of the order.
        ordered = tl.where(is_nan, -1, ordered)
        # Complement so the maximum value becomes the minimum key.
        key = ordered ^ -1
        idx = (IDX_SHIFT + off + tl.arange(0, BLOCK_N)[None, :]).to(tl.int64)
        idx = idx + rows.to(tl.int64) * IDX_BASE
        pack = ((key.to(tl.int64) & 0xFFFFFFFF) << _IDX_BITS) | idx
        blk = tl.min(pack, axis=1)[:, None]
        acc = tl.minimum(acc, blk)
    val = acc & _IDX_MASK if EXTRACT else acc
    if NEED_MASK:
        tl.store(out_i, val, mask=row_mask)
    else:
        tl.store(out_i, val)


@libentry()
@triton.jit
def argmax_kernel_2d_16(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_BF16: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    # dim-path packed reduction for fp16/bf16. Same idea as argmax_kernel_2d but
    # the order-preserving key is 16 bits and the (dim-local) column index fits
    # in 16 bits, so the packed word is a single int32.  The word is XORed with
    # the sign bit so a signed `tl.min` orders it like the unsigned key.
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows.to(tl.int64) * N
    out_i = out + rows.to(tl.int64)
    row_mask = rows < M

    acc = tl.full([BLOCK_M, 1], value=0x7FFFFFFF, dtype=tl.int32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        if NEED_MASK:
            mask = row_mask and cols < N
            a = tl.load(inp + cols, mask=mask, other=float("-inf"))
        else:
            a = tl.load(inp + cols)
        if IS_BF16:
            # bf16 loads as fp32 on XPU; its bit pattern is the top 16 bits.
            u32 = a.to(tl.float32).to(tl.int32, bitcast=True)
            u = (u32 >> 16) & 0xFFFF
            nan_hi = 0x7F80
        else:
            u = a.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
            nan_hi = 0x7C00
        is_nan = (u & 0x7FFF) > nan_hi
        # Ascending order-preserving key without a select: for a negative float
        # (sign bit set) `(0 - (u >> 15)) | 0x8000` is 0xFFFF, so the key is ~u;
        # for a non-negative float it is 0x8000, so the key is u | 0x8000.
        akey = (u ^ ((0 - (u >> 15)) | 0x8000)) & 0xFFFF
        # NaN wins a max (torch propagates NaN): give it the top of the order.
        akey = tl.where(is_nan, 0xFFFF, akey)
        # Complement so the maximum value becomes the minimum key.
        key = (~akey) & 0xFFFF
        idx = off + tl.arange(0, BLOCK_N)[None, :]
        pack = ((key << 16) | idx) ^ -2147483648
        blk = tl.min(pack, axis=1)[:, None]
        acc = tl.minimum(acc, blk)
    val = acc & 0xFFFF
    if NEED_MASK:
        tl.store(out_i, val, mask=row_mask)
    else:
        tl.store(out_i, val)


@libentry()
@triton.jit
def argmax_kernel_mid(
    packed,
    out,
    M,
    BLOCK_SIZE: tl.constexpr,
    EXTRACT: tl.constexpr,
):
    # One reduction step over packed (value | index) words.
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    vals = tl.load(packed + offset, mask=offset < M, other=_PACK_OTHER)
    v = tl.min(vals, axis=0)
    if EXTRACT:
        v = v & _IDX_MASK
    tl.store(out + pid, v)


# ---------------------------------------------------------------------------
# Hierarchical path for very large N (dim path, floats, K == 1).
#
# Measured on XPU: a pure-fp32 value max over a row runs at ~790 GB/s, while any
# per-element *index* vector (arange or a loaded index array) forces a slow path
# (~30-125 GB/s), and the int64/int32 packed word is ALU-bound (~60-150 GB/s).
# So for very large N we avoid touching every element with the index logic:
#   Pass 1  argmax_chunk_max : pure fp32 max of each CHUNK slab  (~790 GB/s)
#   Pass 2  argmax_win_chunk : pick the winning chunk on cmax[M, NC]  (tiny)
#   Pass 3  argmax_chunk_idx : rescan only the winning CHUNK for idx (1/NC data)
# Total cost is ~1 + 1/NC passes instead of 2 full passes.
# ---------------------------------------------------------------------------
_HIER_MIN_N = 131072
_HIER_BLOCK1 = 8192
_HIER_BLOCK3 = 4096
_HIER_MAX_NC = 16


@libentry()
@triton.jit
def argmax_chunk_max(inp, cmax, N, NC, BLOCK: tl.constexpr):
    row = ext.program_id(0)
    ch = ext.program_id(1)
    CHUNK = N // NC
    base = inp + row.to(tl.int64) * N + ch.to(tl.int64) * CHUNK
    acc = tl.full([BLOCK], float("-inf"), tl.float32)
    for off in range(0, CHUNK, BLOCK):
        acc = tl.maximum(acc, tl.load(base + off + tl.arange(0, BLOCK)).to(tl.float32))
    tl.store(cmax + row.to(tl.int64) * NC + ch, tl.max(acc, axis=0))


@libentry()
@triton.jit
def argmax_win_chunk(cmax, maxv, win, NC, BLOCK_C: tl.constexpr):
    # Tiny pass: argmax over the NC per-row chunk maxima (tie-break-left).
    row = ext.program_id(0)
    off = tl.arange(0, BLOCK_C)
    mask = off < NC
    v = tl.load(cmax + row.to(tl.int64) * NC + off, mask=mask, other=float("-inf")).to(
        tl.float32
    )
    u = v.to(tl.int32, bitcast=True)
    ordered = tl.where(u < 0, ~u, u ^ -2147483648)
    key = ordered ^ -1
    pack = ((key.to(tl.int64) & 0xFFFFFFFF) << 30) | off.to(tl.int64)
    best = tl.min(tl.where(mask, pack, 0x7FFFFFFFFFFFFFFF), axis=0)
    c = (best & ((1 << 30) - 1)).to(tl.int64)
    tl.store(win + row, c)
    tl.store(maxv + row, tl.load(cmax + row.to(tl.int64) * NC + c))


@libentry()
@triton.jit
def argmax_chunk_idx(inp, win, maxv, out, N, CHUNK, BLOCK: tl.constexpr):
    # Rescan only the winning chunk -> first global index of the row max.
    row = ext.program_id(0)
    c = tl.load(win + row)
    m = tl.load(maxv + row).to(tl.float32)
    base = inp + row.to(tl.int64) * N + c * CHUNK
    acc = tl.full([BLOCK], 1 << 30, tl.int32)
    for off in range(0, CHUNK, BLOCK):
        v = tl.load(base + off + tl.arange(0, BLOCK)).to(tl.float32)
        acc = tl.minimum(acc, tl.where(v == m, off + tl.arange(0, BLOCK), 1 << 30))
    tl.store(out + row, c * CHUNK + tl.min(acc, axis=0).to(tl.int64))


# ---------------------------------------------------------------------------
# return_indices dim path (floats).  Measured on XPU: a *wide* 2-D reduce with
# return_indices lowers to a cross-core smem + >=2 xpu_barrier() reduction and is
# very slow, but a *narrow* tile (BLOCK_N <= 512) keeps the reduce intra-core and
# is competitive with (often faster than) the packed int word.  tie_break_left
# keeps the earliest index, matching torch.
# ---------------------------------------------------------------------------
_MRI_MIN_N = 8
_MRI_BN_FP32 = (128, 256, 64, 512, 32, 16, 8)
_MRI_BM_FP32 = (64, 128, 32, 16, 8)
_MRI_BN_16 = (512, 256, 128, 64, 32, 16, 8)
_MRI_BM_16 = (64, 128, 32, 16, 8)
_MRI_BM_16_WIDE = (32, 64, 128, 16, 8)


def _pick_mri_tile(M, N, is_fp32):
    """Return (BLOCK_M, BLOCK_N) with M % BLOCK_M == 0 and N % BLOCK_N == 0."""
    if N < _MRI_MIN_N:
        return None
    if is_fp32:
        bns, bms = _MRI_BN_FP32, _MRI_BM_FP32
    else:
        bns = _MRI_BN_16
        # A single-chunk width (N <= 512) wants a larger row tile; a wide N
        # wants a smaller one (more programs, shorter per-program sweep).
        bms = _MRI_BM_16 if N <= 512 else _MRI_BM_16_WIDE
    bn = next((b for b in bns if N % b == 0), None)
    if bn is None:
        return None
    bm = next((m for m in bms if M % m == 0), None)
    if bm is None:
        return None
    return bm, bn


@libentry()
@triton.jit
def argmax_return_idx(inp, out, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    p = inp + rows.to(tl.int64) * N
    accv = tl.full([BLOCK_M, 1], float("-inf"), tl.float32)
    acci = tl.full([BLOCK_M, 1], 1 << 30, tl.int32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        v = tl.load(p + cols).to(tl.float32)
        cm, ci = tl.max(
            v, axis=1, return_indices=True, return_indices_tie_break_left=True
        )
        cm = cm[:, None]
        ci = ci[:, None] + off
        acci = tl.where(cm > accv, ci, acci)
        accv = tl.maximum(accv, cm)
    tl.store(out + rows.to(tl.int64), acci, mask=rows < M)


@libentry()
@triton.jit
def argmax_row_merge(
    ov,
    oi,
    out,
    P: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # Merge the P per-segment (value, global index) pairs of each row into the
    # row winner: maximum value, tie-break-left on the global index.  P is a
    # power of two and BLOCK_M divides M2, so the whole tile is mask-free.
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    off = rows[:, None] * P + tl.arange(0, P)[None, :]
    v = tl.load(ov + off).to(tl.float32)
    i = tl.load(oi + off)
    u = v.to(tl.int32, bitcast=True)
    ordered = tl.where(u < 0, ~u, u ^ -2147483648)
    key = ordered ^ -1
    pack = ((key.to(tl.int64) & 0xFFFFFFFF) << 30) | i
    best = tl.min(pack, axis=1)
    tl.store(out + rows.to(tl.int64), best & ((1 << 30) - 1))


def _argmax_hierarchical(src, out_flat, M2, N, device):
    """Return True if the hierarchical path handled this (M2, N) float tensor."""
    nc = max(2, min(_HIER_MAX_NC, N // _HIER_MIN_N))
    while nc > 1 and (N % nc or (N // nc) % _HIER_BLOCK1 or (N // nc) % _HIER_BLOCK3):
        nc //= 2
    chunk = N // nc
    if nc < 2 or N % nc or chunk % _HIER_BLOCK1 or chunk % _HIER_BLOCK3:
        return False
    cmax = torch.empty((M2 * nc,), dtype=torch.float32, device=device)
    maxv = torch.empty((M2,), dtype=torch.float32, device=device)
    win = torch.empty((M2,), dtype=torch.int64, device=device)
    with torch_device_fn.device(device):
        argmax_chunk_max[(M2, nc)](src, cmax, N, nc, _HIER_BLOCK1)
        argmax_win_chunk[(M2,)](cmax, maxv, win, nc, triton.next_power_of_2(nc))
        argmax_chunk_idx[(M2,)](src, win, maxv, out_flat, N, chunk, _HIER_BLOCK3)
    return True


# ---------------------------------------------------------------------------
# dim=None fast path for large flat inputs (floats).  Same idea as the dim path:
# a narrow-BLOCK_N return_indices row reduce runs at ~1 TB/s, then a tiny combine
# over the per-row maxima picks the global winner.  Row width / tiles measured on
# XPU.
# ---------------------------------------------------------------------------
_FLAT_ROW = 2048
_FLAT_BM = 64
_FLAT_BN = 256


@libentry()
@triton.jit
def argmax_mri_vi(
    inp,
    val,
    gidx,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IDX_BASE: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
):
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    p = inp + rows.to(tl.int64) * N
    accv = tl.full([BLOCK_M, 1], float("-inf"), tl.float32)
    acci = tl.full([BLOCK_M, 1], 1 << 30, tl.int32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        v = tl.load(p + cols).to(tl.float32)
        cm, ci = tl.max(
            v, axis=1, return_indices=True, return_indices_tie_break_left=True
        )
        cm = cm[:, None]
        ci = ci[:, None] + off
        acci = tl.where(cm > accv, ci, acci)
        accv = tl.maximum(accv, cm)
    g = IDX_BASE + rows.to(tl.int64) * ROW_STRIDE + acci.to(tl.int64)
    tl.store(val + rows.to(tl.int64), accv, mask=rows < M)
    tl.store(gidx + rows.to(tl.int64), g, mask=rows < M)


@libentry()
@triton.jit
def argmax_combine(val, gidx, out, M, BLOCK: tl.constexpr):
    off = tl.arange(0, BLOCK)
    mask = off < M
    v = tl.load(val + off, mask=mask, other=float("-inf")).to(tl.float32)
    u = v.to(tl.int32, bitcast=True)
    ordered = tl.where(u < 0, ~u, u ^ -2147483648)
    key = ordered ^ -1
    pack = ((key.to(tl.int64) & 0xFFFFFFFF) << 30) | off.to(tl.int64)
    best = tl.min(tl.where(mask, pack, 0x7FFFFFFFFFFFFFFF), axis=0)
    c = (best & ((1 << 30) - 1)).to(tl.int64)
    tl.store(out, tl.load(gidx + c))


def _argmax_flat_mri(inp, out, device):
    """dim=None fast path: narrow-BN return_indices rows + tiny combine."""
    total = inp.numel()
    row = _FLAT_ROW
    rows = total // row
    res = total - rows * row
    n_mid = rows + (1 if res else 0)
    val = torch.empty((n_mid,), dtype=torch.float32, device=device)
    gidx = torch.empty((n_mid,), dtype=torch.int64, device=device)
    flat = inp.reshape(-1)
    if rows:
        bm = _FLAT_BM
        while bm > 1 and rows % bm:
            bm //= 2
        bn = _FLAT_BN
        while bn > 1 and row % bn:
            bn //= 2
        with torch_device_fn.device(device):
            argmax_mri_vi[(triton.cdiv(rows, bm),)](
                flat, val, gidx, rows, row, bm, bn, 0, row
            )
    if res:
        bn = _FLAT_BN
        while bn > 1 and res % bn:
            bn //= 2
        with torch_device_fn.device(device):
            argmax_mri_vi[(1,)](
                flat[rows * row :],
                val[rows:],
                gidx[rows:],
                1,
                res,
                1,
                bn,
                rows * row,
                res,
            )
    with torch_device_fn.device(device):
        argmax_combine[(1,)](
            val, gidx, out.reshape(-1), n_mid, triton.next_power_of_2(n_mid)
        )


def _argmax_flat_fast(inp, out, device):
    """dim=None float fast path: rows-of-8192 2D packed pass + staged mid."""
    numel = inp.numel()
    block = _FULL_REDUCTION_BLOCK_SIZE
    is_fp32 = inp.dtype == torch.float32
    bm = 128 if not is_fp32 else 64
    bn = 1024 if not is_fp32 else 512
    rows = numel // block
    res = numel - rows * block
    n_mid = rows + (1 if res else 0)
    mid = torch.empty((n_mid,), dtype=torch.int64, device=device)
    flat = inp.reshape(-1)
    if rows:
        if rows % bm != 0:
            bm = next((m for m in _FAST_BM_FP32 if rows % m == 0), 2)
        need_mask = rows % bm != 0
        argmax_kernel_2d[(triton.cdiv(rows, bm),)](
            flat,
            mid,
            rows,
            block,
            bm,
            bn,
            block,
            0,
            need_mask,
            False,
            buffer_size_limit=2048,
        )
    if res:
        argmax_kernel_2d[(1,)](
            flat[rows * block :],
            mid[rows:],
            1,
            res,
            1,
            block,
            block,
            rows * block,
            True,
            False,
            buffer_size_limit=2048,
        )
    # staged mid reduce to a single packed word, then extract the index
    n = n_mid
    while n > block:
        nxt_m = triton.cdiv(n, block)
        nxt = torch.empty((nxt_m,), dtype=torch.int64, device=device)
        argmax_kernel_mid[(nxt_m,)](mid, nxt, n, block, False, buffer_size_limit=2048)
        mid, n = nxt, nxt_m
    argmax_kernel_mid[(1,)](
        mid,
        out.reshape(-1),
        n,
        triton.next_power_of_2(n),
        True,
        buffer_size_limit=2048,
    )


def _argmax_flat_legacy(inp, out, device):
    """dim=None path for int dtypes / small inputs (unchanged from HEAD)."""
    M = inp.numel()
    dtype = inp.dtype
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid_value = torch.empty((mid_size,), dtype=dtype, device=device)
    mid_index = torch.empty((mid_size,), dtype=torch.int64, device=device)
    with torch_device_fn.device(device):
        argmax_kernel_1[(mid_size, 1, 1)](
            inp,
            mid_value,
            mid_index,
            M,
            block_size,
        )
        argmax_kernel_2[(1, 1, 1)](mid_value, mid_index, out, mid_size, block_mid)


# ---------------------------------------------------------------------------
# Hand-written SIMD `.xpu` fast path (ported from the vendor library's
# `arg_max_min_mt.xpu` algorithm).
#
# Why: every plain-triton formulation of the argmax *index* logic is slow on
# XPU because a "per-element index vector" (arange or a loaded index array)
# cannot be vectorised by the backend (measured 31-125 GB/s), which caps the
# large-N dim reductions at sp ~0.03-0.12.  The vendor kernel instead keeps the
# running (value, index) pair in SIMD registers: a `float32x16_t` value
# accumulator reduced with `vvmax`, plus an `int32x16_t` index vector updated
# through the mask of `vvlt`.  The vector is collapsed to a scalar only once per
# row (`vrmax` -> equality mask -> min index = tie-break-left), so the index
# logic never touches per-element memory.
#
# Measured on card 6: ~840 GB/s of pure read (vs ~786 GB/s for the best
# plain-triton pure-value reduction).  It is grid-parallel: a single program is
# one 64-core cluster and only reaches ~230 GB/s, so a multi-program grid is
# always launched.
#
# Only the Kunlunxin backend is touched; shapes/dtypes that do not qualify fall
# back to this repo's own plain-triton kernels below (not an external fallback).
# ---------------------------------------------------------------------------
try:  # pragma: no cover - import guard
    import triton.experimental.tle as _tle

    _HAS_TLE = True
except Exception:  # noqa: BLE001
    _tle = None
    _HAS_TLE = False

# LM (per-core local memory) budget: the compiler rejects a kernel whose
# per-core stack exceeds 8000 B, so the staging buffer stays <= ~6 KB.
_SIMD_MAXC32 = 1536  # fp32 elements staged per program (6 KB)
_SIMD_MAXC16 = 3072  # fp16/bf16 elements staged per program (6 KB)
_SIMD_MIN_N = 256  # narrower rows are faster on the plain-triton path
_SIMD_COL_MAX_N = 512  # above this the middle-dim mode lacks parallelism
# bf16's col kernel de-interleaves int32 -> two fp32 vectors, so it costs about
# twice the fp16 one per element; keep it to short reduce axes where the
# transpose + row fallback cannot compete.
_SIMD_COL_BF16_MAX_N = 64

_PAYLOAD_OBJ = os.path.join(os.path.dirname(__file__), "..", "payload", "obj", "amax.o")

if _HAS_TLE:
    try:

        @_tle.raw.dialect("xpu3", object=_PAYLOAD_OBJ, arch=3)
        def _gems_amax_grp_f32(x, oi, M2, N, PID, NPROG): ...

        @_tle.raw.dialect("xpu3", object=_PAYLOAD_OBJ, arch=3)
        def _gems_amax_row_f32(x, oi, M2, N, PID, NPROG): ...

        @_tle.raw.dialect("xpu3", object=_PAYLOAD_OBJ, arch=3)
        def _gems_amax_grp_f16(x, oi, M2, N, PID, NPROG): ...

        @_tle.raw.dialect("xpu3", object=_PAYLOAD_OBJ, arch=3)
        def _gems_amax_grp_bf16(x, oi, M2, N, PID, NPROG): ...

        @_tle.raw.dialect("xpu3", object=_PAYLOAD_OBJ, arch=3)
        def _gems_amax_row_f16(x, oi, M2, N, PID, NPROG): ...

        @_tle.raw.dialect("xpu3", object=_PAYLOAD_OBJ, arch=3)
        def _gems_amax_row_bf16(x, oi, M2, N, PID, NPROG): ...

        @_tle.raw.dialect("xpu3", object=_PAYLOAD_OBJ, arch=3)
        def _gems_amax_rowp_f32(x, ov, oi, M2, N, P, PID, NPROG): ...

        @_tle.raw.dialect("xpu3", object=_PAYLOAD_OBJ, arch=3)
        def _gems_amax_rowp_f16(x, ov, oi, M2, N, P, PID, NPROG): ...

        @_tle.raw.dialect("xpu3", object=_PAYLOAD_OBJ, arch=3)
        def _gems_amax_rowp_bf16(x, ov, oi, M2, N, P, PID, NPROG): ...

        @_tle.raw.dialect("xpu3", object=_PAYLOAD_OBJ, arch=3)
        def _gems_amax_col_f32(x, oi, M, N, K, PID, NPROG): ...

        @_tle.raw.dialect("xpu3", object=_PAYLOAD_OBJ, arch=3)
        def _gems_amax_col_f16(x, oi, M, N, K, PID, NPROG): ...

        @_tle.raw.dialect("xpu3", object=_PAYLOAD_OBJ, arch=3)
        def _gems_amax_col_bf16(x, oi, M, N, K, PID, NPROG): ...

        @triton.jit
        def _k_grp_f32(X, OI, M2, N, NP):
            _tle.raw.call(_gems_amax_grp_f32, (X, OI, M2, N, tl.program_id(0), NP))

        @triton.jit
        def _k_row_f32(X, OI, M2, N, NP):
            _tle.raw.call(_gems_amax_row_f32, (X, OI, M2, N, tl.program_id(0), NP))

        @triton.jit
        def _k_grp_f16(X, OI, M2, N, NP):
            _tle.raw.call(_gems_amax_grp_f16, (X, OI, M2, N, tl.program_id(0), NP))

        @triton.jit
        def _k_grp_bf16(X, OI, M2, N, NP):
            _tle.raw.call(_gems_amax_grp_bf16, (X, OI, M2, N, tl.program_id(0), NP))

        @triton.jit
        def _k_row_f16(X, OI, M2, N, NP):
            _tle.raw.call(_gems_amax_row_f16, (X, OI, M2, N, tl.program_id(0), NP))

        @triton.jit
        def _k_row_bf16(X, OI, M2, N, NP):
            _tle.raw.call(_gems_amax_row_bf16, (X, OI, M2, N, tl.program_id(0), NP))

        @triton.jit
        def _k_rowp_f32(X, OV, OI, M2, N, P, NP):
            _tle.raw.call(
                _gems_amax_rowp_f32, (X, OV, OI, M2, N, P, tl.program_id(0), NP)
            )

        @triton.jit
        def _k_rowp_f16(X, OV, OI, M2, N, P, NP):
            _tle.raw.call(
                _gems_amax_rowp_f16, (X, OV, OI, M2, N, P, tl.program_id(0), NP)
            )

        @triton.jit
        def _k_rowp_bf16(X, OV, OI, M2, N, P, NP):
            _tle.raw.call(
                _gems_amax_rowp_bf16, (X, OV, OI, M2, N, P, tl.program_id(0), NP)
            )

        @triton.jit
        def _k_col_f32(X, OI, M, N, K, NP):
            _tle.raw.call(_gems_amax_col_f32, (X, OI, M, N, K, tl.program_id(0), NP))

        @triton.jit
        def _k_col_f16(X, OI, M, N, K, NP):
            _tle.raw.call(_gems_amax_col_f16, (X, OI, M, N, K, tl.program_id(0), NP))

        @triton.jit
        def _k_col_bf16(X, OI, M, N, K, NP):
            _tle.raw.call(_gems_amax_col_bf16, (X, OI, M, N, K, tl.program_id(0), NP))

    except Exception:  # pragma: no cover - toolchain without the tle.raw object channel
        _HAS_TLE = False


def _simd_kind(dtype):
    """(kind, maxc) for the SIMD row/group path, or (None, 0) if unsupported."""
    if not _HAS_TLE:
        return None, 0
    if dtype == torch.float32:
        return "f32", _SIMD_MAXC32
    if dtype in (torch.float16, torch.bfloat16):
        return ("f16" if dtype == torch.float16 else "bf16"), _SIMD_MAXC16
    return None, 0


def _simd_grid(nwork, total_elems):
    """Grid size: enough programs to keep the DMA engines saturated, capped."""
    g = triton.cdiv(max(nwork, 1), 64)
    g = max(32, min(512, g))
    return triton.next_power_of_2(g)


def _simd_split_grid(nwork):
    """Grid size for the row-split kernel (more workers than the plain path)."""
    g = triton.cdiv(max(nwork, 1), 64)
    g = max(64, min(512, g))
    return triton.next_power_of_2(g)


# Row-split tuning (card 6 measurements, see harness/solution/argmax/plan.md
# section 14):
#   * only a long row (>= 64 tile-chunks) leaves the plain row kernel worker
#     starved enough for the split to pay off;
#   * the useful segment count grows with the row length (aim for ~64 chunks
#     per worker) and shrinks with M2 (target ~8192 workers total);
#   * with M2 >= 4096 the plain kernel is already worker-rich, so only a very
#     long row justifies the extra merge pass.
_SIMD_SPLIT_MIN_CHUNKS = 64
_SIMD_SPLIT_MAX_WORKERS = 8192
_SIMD_SPLIT_MAX_P = 16


def _row_split_p(M2, N, maxc, vlen):
    """Segments per row for the row-split path; 1 means "do not split"."""
    if M2 < 2 or N >= (1 << 30):
        # The merge kernel packs the global index into 30 bits.
        return 1
    chunks = -(-N // maxc)
    if chunks < _SIMD_SPLIT_MIN_CHUNKS:
        return 1
    p = 1
    while p * _SIMD_SPLIT_MIN_CHUNKS < chunks and p < _SIMD_SPLIT_MAX_P:
        p *= 2
    wmax = 1
    while wmax * 2 * M2 <= _SIMD_SPLIT_MAX_WORKERS:
        wmax *= 2
    p = min(p, wmax, _SIMD_SPLIT_MAX_P)
    if M2 >= 4096 and chunks < 512:
        return 1
    while p > 1 and (N % p or (N // p) % vlen):
        p //= 2
    return p


def _argmax_simd(inp, out_index, dim, M, N, K):
    """Run the SIMD .xpu argmax along `dim`; True when it handled the case."""
    kind, maxc = _simd_kind(inp.dtype)
    if kind is None:
        return False
    M2 = M * K
    # triton specialises a scalar argument equal to 1 into a constexpr, which
    # `tle.raw.call` cannot hand to the payload -> keep M2 >= 2.
    if M2 < 2:
        return False
    # ---- middle-dim mode: reduce over a non-innermost axis in place --------
    # No transpose copy, and the vector spans columns rather than the reduce
    # axis, so a narrow reduce axis is fine here.  Parallelism is only M, so a
    # very wide reduce axis is better served by the row mode (which fans out
    # over M*K rows even though it pays for a transpose).
    if dim != inp.dim() - 1 and M >= 2 and N <= _SIMD_COL_MAX_N and inp.is_contiguous():
        if kind == "f32" and 16 <= K <= 64 and K % 16 == 0:
            src3 = inp.view(M, N, K)
            out_flat = out_index.reshape(-1)
            g = _simd_grid(M, M * N * K)
            with torch_device_fn.device(inp.device):
                _k_col_f32[(g,)](src3, out_flat, M, N, K, g)
            return True
        if kind == "f16" and 32 <= K <= 64 and K % 32 == 0:
            src3 = inp.view(M, N, K)
            out_flat = out_index.reshape(-1)
            g = _simd_grid(M, M * N * K)
            with torch_device_fn.device(inp.device):
                _k_col_f16[(g,)](src3, out_flat, M, N, K, g)
            return True
        # bf16 has no SIMD compare on this part, so the col kernel goes through
        # the int32 -> fp32 de-interleave (same trick as the bf16 row kernel),
        # which costs ~2x the fp16 col kernel per element while its
        # parallelism is only M.  It is still a big win for a short reduce axis
        # -- without it bf16 (64, 16, 64) falls back to a transpose + triton
        # path and loses ~3.5x -- but the transpose + row fallback parallelises
        # over M*K and is faster once N grows (measured on card 6: bf16
        # (64, 256, 64) is 0.0251ms via the fallback vs 0.0294ms via col), so
        # the bf16 col path is capped to short rows.
        if (
            kind == "bf16"
            and N <= _SIMD_COL_BF16_MAX_N
            and 32 <= K <= 64
            and K % 32 == 0
        ):
            src3 = inp.view(M, N, K)
            out_flat = out_index.reshape(-1)
            g = _simd_grid(M, M * N * K)
            with torch_device_fn.device(inp.device):
                _k_col_bf16[(g,)](src3, out_flat, M, N, K, g)
            return True
    # Narrow reduce axes are dominated by the per-row scalar collapse, where
    # this repo's plain-triton return_indices path is faster.
    if N < _SIMD_MIN_N:
        return False
    if (kind == "f32" and N % 16) or (kind != "f32" and N % 32):
        return False
    # Reorder so the reduced dim is physically innermost; the copy uses the
    # native strided engine (aten._copy_from), as in the packed path.
    perm = [d for d in range(inp.dim()) if d != dim] + [dim]
    view = inp.permute(perm)
    if view.is_contiguous():
        src = view
    else:
        src = torch.empty(list(view.shape), dtype=inp.dtype, device=inp.device)
        with torch_device_fn.device(inp.device):
            torch.ops.aten._copy_from(view, src, False)
    out_flat = out_index.reshape(-1)
    # Long rows: split each row into `p` contiguous segments (M2*p workers)
    # and merge the per-segment (value, global index) partials afterwards.
    vlen = 16 if kind == "f32" else 32
    p = _row_split_p(M2, N, maxc, vlen)
    ov = oi = None
    if p > 1:
        ov = torch.empty((M2 * p,), dtype=torch.float32, device=inp.device)
        oi = torch.empty((M2 * p,), dtype=torch.int64, device=inp.device)
    with torch_device_fn.device(inp.device):
        if N <= maxc:
            r = maxc // N
            groups = triton.cdiv(M2, r)
            g = _simd_grid(groups, M2 * N)
            if kind == "f32":
                _k_grp_f32[(g,)](src, out_flat, M2, N, g)
            elif kind == "f16":
                _k_grp_f16[(g,)](src, out_flat, M2, N, g)
            else:
                _k_grp_bf16[(g,)](src, out_flat, M2, N, g)
        elif p > 1:
            g = _simd_split_grid(M2 * p)
            if kind == "f32":
                _k_rowp_f32[(g,)](src, ov, oi, M2, N, p, g)
            elif kind == "f16":
                _k_rowp_f16[(g,)](src, ov, oi, M2, N, p, g)
            else:
                _k_rowp_bf16[(g,)](src, ov, oi, M2, N, p, g)
            bm = 64
            while bm > 1 and M2 % bm:
                bm //= 2
            argmax_row_merge[(M2 // bm,)](ov, oi, out_flat, p, bm)
        else:
            g = _simd_grid(M2, M2 * N)
            if kind == "f32":
                _k_row_f32[(g,)](src, out_flat, M2, N, g)
            elif kind == "f16":
                _k_row_f16[(g,)](src, out_flat, M2, N, g)
            else:
                _k_row_bf16[(g,)](src, out_flat, M2, N, g)
    return True


def argmax(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS_KUNLUNXIN ARGMAX")
    if dim is None:
        M = inp.numel()
        if dtype is None:
            dtype = inp.dtype
        if keepdim:
            shape = list(inp.shape)
            for i in range(0, inp.dim()):
                shape[i] = 1
            out = torch.empty(shape, dtype=torch.int64, device=inp.device)
        else:
            out = torch.empty([], dtype=torch.int64, device=inp.device)

        if _is_fast_dtype(dtype) and (1 << 20) <= M <= (1 << 30):
            _argmax_flat_mri(inp, out, inp.device)
        elif (
            _is_fast_dtype(dtype)
            and M > 2 * 1024 * 1024
            and M > _FULL_REDUCTION_BLOCK_SIZE
            and M <= (1 << 30)
        ):
            _argmax_flat_fast(inp, out, inp.device)
        else:
            _argmax_flat_legacy(inp, out, inp.device)
        return out
    else:
        assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
        shape = inp.shape
        dim = dim % inp.ndim
        if inp.numel() == 0:
            out_shape = list(shape)
            if keepdim:
                out_shape[dim] = 1
            else:
                del out_shape[dim]
            return torch.zeros(out_shape, dtype=torch.int64, device=inp.device)
        N = shape[dim]
        M = math.prod(shape[:dim])
        K = inp.numel() // M // N

        inp = inp.contiguous()

        shape_list = list(shape)
        shape_list[dim] = 1
        out_index = torch.empty(shape_list, dtype=torch.int64, device=inp.device)
        if not keepdim:
            out_index = torch.squeeze(out_index, dim)

        # argmax along a size-1 dim is trivially index 0.
        if N == 1:
            out_index.zero_()
            return out_index

        # ---- hand-written SIMD .xpu path (see the block above) -------------
        if _argmax_simd(inp, out_index, dim, M, N, K):
            return out_index

        tl_dtype, dtype_min_value = torch_dtype_to_tl_dtype_and_min_value[inp.dtype]

        # ---- fast packed row-reduction path (floats only) -----------------
        # Routing (measured on XPU):
        #   * N < _FAST_MIN_N          : narrow 2-D int64 reduce is unreliable
        #     (wrong results) and the legacy single-tile kernel is fine -> legacy.
        #   * _FAST_MIN_N <= N <= MAX_TILE_N : the packed kernel beats the
        #     runtime-N single-tile kernel (discrete access) and the two-stage
        #     path by 2-5x here.
        #   * N > MAX_TILE_N           : the two-stage path (constexpr N/K,
        #     very high grid parallelism) is faster than the packed kernel,
        #     which sweeps all of N inside a single low-occupancy program.
        # A row count below one tile (M*K < 128) leaves the packed grid with a
        # single program, where the legacy single-tile kernel wins.
        # ---- hierarchical path for very large N (K == 1, floats) -----------
        # The index logic is the bottleneck; only touch 1/NC of the data with it.
        if _is_fast_dtype(inp.dtype) and K == 1 and N >= _HIER_MIN_N:
            if _argmax_hierarchical(inp, out_index.reshape(-1), M, N, inp.device):
                return out_index

        # ---- return_indices path (narrow tile -> intra-core reduce) ---------
        if _is_fast_dtype(inp.dtype) and _MRI_MIN_N <= N < _HIER_MIN_N and M * K >= 128:
            is_fp32 = inp.dtype == torch.float32
            perm = [d for d in range(inp.dim()) if d != dim] + [dim]
            view = inp.permute(perm)
            if view.is_contiguous():
                src = view
            else:
                src = torch.empty(list(view.shape), dtype=inp.dtype, device=inp.device)
                with torch_device_fn.device(inp.device):
                    torch.ops.aten._copy_from(view, src, False)
            M2 = M * K
            tile = _pick_mri_tile(M2, N, is_fp32)
            if tile is not None:
                bm, bn = tile
                with torch_device_fn.device(inp.device):
                    argmax_return_idx[(M2 // bm,)](
                        src, out_index.reshape(-1), M2, N, bm, bn
                    )
                return out_index

        is_fp32 = inp.dtype == torch.float32
        # fp16 packs a 16-bit column index (its int16 bitcast is direct), so its
        # int32 path covers N up to 2**16.  fp32 needs the 30-bit int64 index
        # (MAX_TILE_N cap), and bf16 loads as fp32 on XPU (extra round-trip) so
        # its packed path stays slower than the two-stage legacy above the cap.
        packed_hi = 65536 if inp.dtype == torch.float16 else MAX_TILE_N
        if _is_fast_dtype(inp.dtype) and _FAST_MIN_N <= N <= packed_hi and M * K >= 128:
            # Reorder so the reduced dim (N) is physically innermost; the copy
            # uses the native strided engine (`aten._copy_from`) instead of the
            # gems `.contiguous()` override (the amax/amin 2026-08-16 lesson).
            perm = [d for d in range(inp.dim()) if d != dim] + [dim]
            view = inp.permute(perm)
            if view.is_contiguous():
                src = view
            else:
                src = torch.empty(list(view.shape), dtype=inp.dtype, device=inp.device)
                with torch_device_fn.device(inp.device):
                    torch.ops.aten._copy_from(view, src, False)
            M2 = M * K
            out_flat = out_index.reshape(-1)
            # fp16/bf16 use the int32 (16-bit key + 16-bit column index) kernel;
            # fp32 uses the int64 kernel.
            if is_fp32:
                bns, bms = _FAST_BN_FP32, _FAST_BM_FP32
                tile = _pick_fast_tile(M2, N, True)
            else:
                bns, bms = _FAST16_BN, _FAST16_BM
                tile = _pick_fast_tile16(M2, N)
            if tile is not None:
                block_m, block_n = tile
                with torch_device_fn.device(inp.device):
                    if is_fp32:
                        argmax_kernel_2d[(M2 // block_m,)](
                            src,
                            out_flat,
                            M2,
                            N,
                            block_m,
                            block_n,
                            0,
                            0,
                            False,
                            True,
                            buffer_size_limit=2048,
                        )
                    else:
                        argmax_kernel_2d_16[(M2 // block_m,)](
                            src,
                            out_flat,
                            M2,
                            N,
                            block_m,
                            block_n,
                            inp.dtype == torch.bfloat16,
                            False,
                            buffer_size_limit=2048,
                        )
                return out_index
            # Non-dividing shapes: same packed kernel, but masked. N is swept
            # in chunks, so N > MAX_TILE_N (single-load limit) is covered too.
            if M2 >= 1 and N >= 2:
                bn = next((b for b in bns if N % b == 0), None)
                if bn is not None:
                    bm = next((m for m in bms if M2 % m == 0), 2)
                    grid_m = triton.cdiv(M2, bm)
                    with torch_device_fn.device(inp.device):
                        if is_fp32:
                            argmax_kernel_2d[(grid_m,)](
                                src,
                                out_flat,
                                M2,
                                N,
                                bm,
                                bn,
                                0,
                                0,
                                True,
                                True,
                                buffer_size_limit=2048,
                            )
                        else:
                            argmax_kernel_2d_16[(grid_m,)](
                                src,
                                out_flat,
                                M2,
                                N,
                                bm,
                                bn,
                                inp.dtype == torch.bfloat16,
                                True,
                                buffer_size_limit=2048,
                            )
                    return out_index

        # Routing (see argmin.py for the full rationale):
        #   * N > MAX_TILE_N : reduce axis cannot fit one XPU tile, two-stage
        #     is mandatory.
        #   * K == 1 and N >= TWO_STAGE_MIN_N : contiguous inner reduce. The
        #     single-tile kernel uses *runtime* N/K -> discrete access that is
        #     catastrophically slow as N grows. The two-stage kernels take N/K
        #     as constexpr -> provable stride-1 block DMA (~13x faster).
        #   * otherwise (small N, or K > 1 with N <= MAX_TILE_N) : the proven
        #     single-tile kernel.
        use_two_stage = N > MAX_TILE_N or (K == 1 and N >= TWO_STAGE_MIN_N)
        if not use_two_stage:
            grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), K)  # noqa: E731
            with torch_device_fn.device(inp.device):
                argmax_kernel_small_n[grid](
                    inp,
                    out_index,
                    M,
                    N,
                    K,
                    tl_dtype,
                    dtype_min_value,
                )
            return out_index

        # Two-stage per-row reduction.
        block_n = STAGE_BLOCK_N
        block_m = STAGE_BLOCK_M
        num_chunks = triton.cdiv(N, block_n)
        part_val = torch.empty(
            (M * num_chunks * K,), dtype=inp.dtype, device=inp.device
        )
        part_idx = torch.empty(
            (M * num_chunks * K,), dtype=torch.int64, device=inp.device
        )
        grid1 = (triton.cdiv(M, block_m), num_chunks, K)
        with torch_device_fn.device(inp.device):
            argmax_stage1[grid1](
                inp,
                part_val,
                part_idx,
                M,
                N,
                K,
                num_chunks,
                block_m,
                block_n,
            )
            # A single chunk already holds the whole reduce axis: stage 1 wrote
            # the final global argmax into part_idx (layout matches out_index),
            # so skip stage 2 entirely.
            if num_chunks == 1:
                out_index.view(-1).copy_(part_idx)
                return out_index
            block_c = triton.next_power_of_2(num_chunks)
            grid2 = (triton.cdiv(M, block_m), K)
            argmax_stage2[grid2](
                part_val,
                part_idx,
                out_index,
                M,
                K,
                num_chunks,
                block_m,
                block_c,
            )
        return out_index
