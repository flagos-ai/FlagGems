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

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.limits import get_dtype_max

from .copy import copy_ as _vendor_copy
from .min import _HAS_RAW as _MIN_RAW_OK
from .min import _raw_min_dim as _min_raw_dim

logger = logging.getLogger(__name__)
torch_dtype_to_tl_dtype_and_max_value = {
    torch.int16: (tl.int16, torch.iinfo(torch.int16).max),
    torch.int32: (tl.int32, torch.iinfo(torch.int32).max),
    torch.float16: (tl.float16, torch.finfo(torch.float16).max),
    torch.float32: (tl.float32, torch.finfo(torch.float32).max),
    torch.bfloat16: (tl.float32, torch.finfo(torch.float32).max),
}

# N above this width cannot be reduced (with return_indices) in a single XPU
# tile: both the single-load kernel and the loop-accumulator kernel fail to
# compile ("out of resource: uni_sram" / "2D Shape[-1] <= core_num *
# buffer_size Limit"). Such N is handled by the two-stage reduction below.
MAX_TILE_N = 8192
# Chunk width / row-tile for the two-stage large-N path (measured fastest on XPU).
STAGE_BLOCK_N = 2048
STAGE_BLOCK_M = 32
# For a contiguous inner reduce (K == 1), route N at or above this width to the
# constexpr two-stage path even when it would fit a single tile: the runtime-N/K
# single-tile kernel degrades to discrete access and is far slower here. Below
# this, single-tile launch overhead wins and the discrete penalty is negligible.
TWO_STAGE_MIN_N = 256

# ---------------------------------------------------------------------------
# Fast mask-free row-reduction path (floats only), following the amax/amin
# 2026-08-16 recipe: fixed bounded tiles that divide both M and N -> the whole
# kernel compiles mask-free (skips the XPU masked-memory slow path), and the
# reduced dim is brought innermost with a native `aten._copy_from` copy when
# needed. The index is carried through the reduction as an integer (+ an extra
# tie/NaN-aware merge), avoiding a second scan.
#   * fp16/bf16 sweet spot: BLOCK_M=128, BLOCK_N=1024 (measured on XPU).
#   * fp32: BLOCK_M=64, BLOCK_N=512.
# Small BLOCK_N (< _FAST_MIN_N) is routed back to the legacy masked single-tile
# kernel: a narrow constexpr 2-D reduce with return_indices retriggers an XPU
# layout-inference failure.
# ---------------------------------------------------------------------------
_FULL_REDUCTION_BLOCK_SIZE = 8192
_FAST_MIN_N = 64
_FAST_BN_FP16 = (1024, 256, 512, 128, 64)
_FAST_BN_FP32 = (512, 256, 1024, 128, 64)
_FAST_BM_FP16 = (128, 64, 32, 256, 512, 16, 8, 4, 2)
_FAST_BM_FP32 = (64, 128, 32, 256, 512, 16, 8, 4, 2)

# The backend ships a hand-written SIMD min row reducer WITH an index output
# (`payload/obj/min.o`, the same object the registered `min_dim` op uses). It
# carries the running (value, index) pair in SIMD registers, so the per-element
# index logic that caps every plain-triton formulation at 30-160 GB/s never
# touches memory. It is the fastest index-carrying row reduce here, so it is
# tried before the plain-triton paths. Below this reduce-axis width the per-row
# scalar collapse dominates (plain-triton wins); bf16 additionally stays on the
# plain-triton path for very narrow axes.
_RAW_DIM_MIN_N = 64
# The payload only saturates the chip when it has many rows to fan out over: with
# few, very wide rows each core scans one long row and the 12-cluster grid leaves
# ~half the cores idle (measured 0.42 TB/s on (1024,65536) fp16 vs 1.25 TB/s on
# (16384,65536)). Splitting every row into S contiguous segments turns a few wide
# rows into many narrow ones and recovers full bandwidth; a tiny combine over the
# S per-segment (value, local-index) pairs then picks the first minimum, exactly
# torch's tie rule. Aim for ~_SEG_TARGET_ROWS effective rows while keeping each
# segment at least _SEG_MIN_W wide (below that the per-segment collapse dominates
# and S=1 is faster).
_SEG_MIN_W = 8192
_SEG_MAX_S = 32
_SEG_TARGET_ROWS = 8192
_SEG_COMBINE_BM = 64
# For fp32 the payload's per-row scalar collapse costs proportionally more than
# the packed int64 kernel on tiny, single-segment tensors (4-byte loads, no row
# fan-out): measured (64,64) fp32 payload 0.644x vs packed 0.761x. Require a
# minimum element count there; wide/segmented rows and 16-bit dtypes are
# unaffected (their payload result already beats the packed path).
_RAW_MIN_ELEMS_FP32 = 16384


def _pick_seg(M2, N):
    """Segments per row S (a power of two dividing N) for the payload. S == 1
    means "do not split" (few enough rows -> already parallel, or too narrow)."""
    s = 1
    while (
        s * 2 <= _SEG_MAX_S
        and N % (s * 2) == 0
        and N // (s * 2) >= _SEG_MIN_W
        and M2 * s < _SEG_TARGET_ROWS
    ):
        s *= 2
    return s


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


@libentry()
@triton.jit
def argmin_kernel_1(
    inp,
    mid_value,
    mid_index,
    M,
    BLOCK_SIZE: tl.constexpr,
    dtype_max_value: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    inp_val = tl.load(inp_ptrs, mask=mask, other=dtype_max_value)
    min_val, min_index = tl.min(inp_val, axis=0, return_indices=True)
    min_index = min_index + pid * BLOCK_SIZE
    mid_value_ptr = mid_value + pid
    min_index_ptr = mid_index + pid
    tl.store(mid_value_ptr, min_val)
    tl.store(min_index_ptr, min_index)


@libentry()
@triton.jit
def argmin_kernel_2(
    mid_value,
    mid_index,
    out,
    mid_size,
    BLOCK_MID: tl.constexpr,
    dtype_max_value: tl.constexpr,
):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid_value + offset
    mask = offset < mid_size
    mid_val = tl.load(mid_ptrs, mask=mask, other=dtype_max_value)
    index_val = tl.argmin(mid_val, axis=0)
    mid_index_ptrs = mid_index + index_val
    out_val = tl.load(mid_index_ptrs)
    tl.store(out, out_val)


@libentry()
@triton.jit
def argmin_stage1(
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
    # Stage 1 of the large-N (N > MAX_TILE_N) path. Each program reduces one
    # BLOCK_N-wide chunk of a BLOCK_M row block and emits the chunk-local min
    # value plus its *global* argmin index. Output is [M, NUM_CHUNKS, K].
    pid_m = ext.program_id(0)
    pid_c = ext.program_id(1)
    pid_k = ext.program_id(2)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    chunk_start = pid_c * BLOCK_N
    n_offset = chunk_start + tl.arange(0, BLOCK_N)
    dtype = inp.type.element_ty
    max_value = get_dtype_max(dtype)
    offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
    mask = m_offset[:, None] < M and n_offset[None, :] < N
    vals = tl.load(inp + offset, mask=mask, other=max_value)
    lmin, largmin = tl.min(vals, axis=1, return_indices=True)
    gidx = chunk_start + largmin
    part_offset = m_offset * NUM_CHUNKS * K + pid_c * K + pid_k
    pmask = m_offset < M
    tl.store(part_val + part_offset, lmin, mask=pmask)
    tl.store(part_idx + part_offset, gidx, mask=pmask)


@libentry()
@triton.jit
def argmin_stage2(
    part_val,
    part_idx,
    out_index,
    M,
    K: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Stage 2: reduce the NUM_CHUNKS per-row partial mins, then gather the
    # global argmin index of the winning chunk. The reduce keeps the earliest
    # chunk on ties (default first-occurrence), matching torch semantics.
    pid_m = ext.program_id(0)
    pid_k = ext.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    c_offset = tl.arange(0, BLOCK_C)
    dtype = part_val.type.element_ty
    max_value = get_dtype_max(dtype)
    offset = m_offset[:, None] * NUM_CHUNKS * K + c_offset[None, :] * K + pid_k
    mask = m_offset[:, None] < M and c_offset[None, :] < NUM_CHUNKS
    vals = tl.load(part_val + offset, mask=mask, other=max_value)
    _, best_c = tl.min(vals, axis=1, return_indices=True)
    gather = m_offset * NUM_CHUNKS * K + best_c * K + pid_k
    pmask = m_offset < M
    res = tl.load(part_idx + gather, mask=pmask)
    tl.store(out_index + m_offset * K + pid_k, res, mask=pmask)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("argmin"))
@triton.jit
def argmin_kernel_small_n(
    inp,
    out_index,
    M,
    N,
    K,
    tl_dtype: tl.constexpr,
    dtype_max_value: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Single-tile path (N <= MAX_TILE_N so a single load covers all of N),
    # preserved verbatim from the original kernel (runtime N/K, no tie-break
    # flag). A constexpr-N/K variant compiles ~13x faster for large contiguous
    # tiles but retriggers an XPU 2D-reduce layout-inference failure for narrow
    # tiles (small BLOCK_N) and for int16 -- so keep the proven original form.
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
    inp_vals = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
    _, result_index = tl.min(inp_vals, axis=1, return_indices=True)

    out_index_ptrs = out_index + offset_index

    tl.store(out_index_ptrs, result_index, mask=mask1)


# ---------------------------------------------------------------------------
# Fast 2-D row reduction with a packed (value, index) integer word.
#
# Each lane packs its fp32 value into a *tautological order-preserving* int
# (`ordered`: -inf -> 0x007FFFFF ... +inf -> 0xFF800000, NaN -> 0) into the
# HIGH 34 bits of an int64 and the global index into the LOW 30 bits:
#       packed = (ordered << 30) | idx
# A single signed-int64 `tl.min` then returns both the minimal value AND the
# lowest index, exactly reproducing torch's "first minimal" tie rule (ties
# compare on idx -> the earlier lane wins). NaN encodes as the smallest word,
# so the first NaN wins, matching torch/XPU min semantics.
#
# Each row is swept in BLOCK_N-wide chunks with a running int64 accumulator;
# the reduce is a plain int min (no `return_indices`, which measured ~20x
# slower on XPU than a value-only reduce). NEED_MASK=False compiles fully
# mask-free (M and N both divide by the block sizes). IDX_BASE scales the row
# id (rows-of-8192 view of the flat path -> global flat index); IDX_SHIFT adds
# a constant index offset (residue row of the flat path). EXTRACT controls
# whether the stored word is the packed value (flat mid chain) or the
# extracted index (direct dim output).
# ---------------------------------------------------------------------------
_IDX_BITS = tl.constexpr(30)
_IDX_MASK = tl.constexpr((1 << 30) - 1)
_PACK_PINF = tl.constexpr(0xFF800000 << 30)  # packed +inf (masked/acc sentinel)
_PACK_OTHER = tl.constexpr(0x4000000000000000)  # larger than any real packed word


@libentry()
@triton.jit
def argmin_kernel_2d(
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

    acc = tl.full([BLOCK_M, 1], value=_PACK_PINF, dtype=tl.int64)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        if NEED_MASK:
            mask = row_mask and cols < N
            a = tl.load(inp + cols, mask=mask, other=float("inf")).to(tl.float32)
        else:
            a = tl.load(inp + cols).to(tl.float32)
        u = a.to(tl.int32, bitcast=True)
        is_nan = (u & 0x7FFFFFFF) > 0x7F800000
        neg = u < 0
        ordered = tl.where(neg, ~u, u ^ -2147483648)
        # XPU device-native torch.argmin ignores NaN (a NaN lane never wins the
        # min); map NaN to the same ordering as +inf so finite minima win, and
        # an all-NaN row degrades to index 0 like the device native.
        ordered = tl.where(is_nan, -8388608, ordered)
        idx = (IDX_SHIFT + off + tl.arange(0, BLOCK_N)[None, :]).to(tl.int64)
        idx = idx + rows.to(tl.int64) * IDX_BASE
        pack = ((ordered.to(tl.int64) & 0xFFFFFFFF) << _IDX_BITS) | idx
        blk = tl.min(pack, axis=1)[:, None]
        acc = tl.minimum(acc, blk)
    val = acc & _IDX_MASK if EXTRACT else acc
    if NEED_MASK:
        tl.store(out_i, val, mask=row_mask)
    else:
        tl.store(out_i, val)


@libentry()
@triton.jit
def argmin_kernel_mid(
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
# fp16/bf16 int32 packed dim path.
#
# Same idea as argmin_kernel_2d but the order-preserving key is 16 bits and the
# (dim-local) column index fits in 16 bits, so the packed word is a single
# int32 (halves the reduction width vs the int64 path; ~1.5x faster on XPU).
# The word is XORed with the sign bit so a signed `tl.min` orders it like the
# unsigned key, and the value is minimized directly (no bitwise complement, as
# argmin minimizes the ascending-order key).
# ---------------------------------------------------------------------------
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


@libentry()
@triton.jit
def argmin_kernel_2d_16(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_BF16: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
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
            a = tl.load(inp + cols, mask=mask, other=float("inf"))
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
        # XPU device-native torch.argmin ignores NaN: map NaN to the top of the
        # order so a finite minimum always wins; an all-NaN row degrades to
        # index 0 (first lane).
        akey = tl.where(is_nan, 0xFFFF, akey)
        idx = off + tl.arange(0, BLOCK_N)[None, :]
        pack = ((akey << 16) | idx) ^ -2147483648
        blk = tl.min(pack, axis=1)[:, None]
        acc = tl.minimum(acc, blk)
    val = acc & 0xFFFF
    if NEED_MASK:
        tl.store(out_i, val, mask=row_mask)
    else:
        tl.store(out_i, val)


# ---------------------------------------------------------------------------
# return_indices dim path (floats).  Measured on XPU: a *wide* 2-D reduce with
# return_indices lowers to a cross-core smem + >=2 xpu_barrier() reduction and
# is very slow, but a *narrow* tile (BLOCK_N <= 512) keeps the reduce intra-core
# and is competitive with (often faster than) the packed int word.
# tie_break_left keeps the earliest index, matching torch.
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
def argmin_return_idx(inp, out, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    p = inp + rows.to(tl.int64) * N
    accv = tl.full([BLOCK_M, 1], float("inf"), tl.float32)
    acci = tl.full([BLOCK_M, 1], 1 << 30, tl.int32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        v = tl.load(p + cols).to(tl.float32)
        cm, ci = tl.min(
            v, axis=1, return_indices=True, return_indices_tie_break_left=True
        )
        cm = cm[:, None]
        ci = ci[:, None] + off
        acci = tl.where(cm < accv, ci, acci)
        accv = tl.minimum(accv, cm)
    tl.store(out + rows.to(tl.int64), acci, mask=rows < M)


# ---------------------------------------------------------------------------
# Hierarchical path for very large N (dim path, floats, K == 1).
#
# Measured on XPU: a pure-fp32 value min over a row runs at high bandwidth,
# while any per-element *index* vector forces a slow path, and the int64/int32
# packed word is ALU-bound. So for very large N we avoid touching every element
# with the index logic:
#   Pass 1  argmin_chunk_min : pure fp32 min of each CHUNK slab
#   Pass 2  argmin_win_chunk : pick the winning chunk on cmin[M, NC]  (tiny)
#   Pass 3  argmin_chunk_idx : rescan only the winning CHUNK for idx (1/NC data)
# Total cost is ~1 + 1/NC passes instead of 2 full passes.
# ---------------------------------------------------------------------------
_HIER_MIN_N = 131072
_HIER_BLOCK1 = 8192
_HIER_BLOCK3 = 4096
_HIER_MAX_NC = 16


@libentry()
@triton.jit
def argmin_chunk_min(inp, cmin, N, NC, BLOCK: tl.constexpr):
    row = ext.program_id(0)
    ch = ext.program_id(1)
    CHUNK = N // NC
    base = inp + row.to(tl.int64) * N + ch.to(tl.int64) * CHUNK
    acc = tl.full([BLOCK], float("inf"), tl.float32)
    for off in range(0, CHUNK, BLOCK):
        acc = tl.minimum(acc, tl.load(base + off + tl.arange(0, BLOCK)).to(tl.float32))
    tl.store(cmin + row.to(tl.int64) * NC + ch, tl.min(acc, axis=0))


@libentry()
@triton.jit
def argmin_win_chunk(cmin, minv, win, NC, BLOCK_C: tl.constexpr):
    # Tiny pass: argmin over the NC per-row chunk minima (tie-break-left).
    row = ext.program_id(0)
    off = tl.arange(0, BLOCK_C)
    mask = off < NC
    v = tl.load(cmin + row.to(tl.int64) * NC + off, mask=mask, other=float("inf")).to(
        tl.float32
    )
    u = v.to(tl.int32, bitcast=True)
    ordered = tl.where(u < 0, ~u, u ^ -2147483648)
    ordered = tl.where((u & 0x7FFFFFFF) > 0x7F800000, -8388608, ordered)
    pack = ((ordered.to(tl.int64) & 0xFFFFFFFF) << 30) | off.to(tl.int64)
    best = tl.min(tl.where(mask, pack, 0x7FFFFFFFFFFFFFFF), axis=0)
    c = (best & ((1 << 30) - 1)).to(tl.int64)
    tl.store(win + row, c)
    tl.store(minv + row, tl.load(cmin + row.to(tl.int64) * NC + c))


@libentry()
@triton.jit
def argmin_chunk_idx(inp, win, minv, out, N, CHUNK, BLOCK: tl.constexpr):
    # Rescan only the winning chunk -> first global index of the row min.
    row = ext.program_id(0)
    c = tl.load(win + row)
    m = tl.load(minv + row).to(tl.float32)
    base = inp + row.to(tl.int64) * N + c * CHUNK
    acc = tl.full([BLOCK], 1 << 30, tl.int32)
    for off in range(0, CHUNK, BLOCK):
        v = tl.load(base + off + tl.arange(0, BLOCK)).to(tl.float32)
        acc = tl.minimum(acc, tl.where(v == m, off + tl.arange(0, BLOCK), 1 << 30))
    tl.store(out + row, c * CHUNK + tl.min(acc, axis=0).to(tl.int64))


@libentry()
@triton.jit
def argmin_seg_combine(
    vals, loc, out, M, S, W, BLOCK_M: tl.constexpr, BS: tl.constexpr
):
    # Combine the S per-segment (value, local-index) pairs of a segmented payload
    # row reduce: pick the first segment holding the row minimum (tie-break-left
    # on the ordered value), then map its local index to the global column.
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rows < M
    # XPU masked loads still touch OOB addresses (backend limitation), so clamp
    # the tail rows to a real row for every load; only the store is masked.
    rows_c = tl.where(rmask, rows, M - 1)
    off = tl.arange(0, BS)  # BS == S (power of two), so no column mask is needed
    v = tl.load(vals + rows_c[:, None].to(tl.int64) * S + off[None, :]).to(tl.float32)
    u = v.to(tl.int32, bitcast=True)
    ordered = tl.where(u < 0, ~u, u ^ -2147483648)
    # XPU device-native argmin ignores NaN; order NaN above any finite value so a
    # finite minimum (and its earliest segment) always wins.
    ordered = tl.where((u & 0x7FFFFFFF) > 0x7F800000, -8388608, ordered)
    pack = ((ordered.to(tl.int64) & 0xFFFFFFFF) << 30) | off.to(tl.int64)
    best = tl.min(pack, axis=1)
    si = (best & ((1 << 30) - 1)).to(tl.int64)
    li = tl.load(loc + rows_c.to(tl.int64) * S + si)
    tl.store(out + rows.to(tl.int64), si * W + li, mask=rmask)


def _argmin_hierarchical(src, out_flat, M2, N, device):
    """Return True if the hierarchical path handled this (M2, N) float tensor."""
    nc = max(2, min(_HIER_MAX_NC, N // _HIER_MIN_N))
    while nc > 1 and (N % nc or (N // nc) % _HIER_BLOCK1 or (N // nc) % _HIER_BLOCK3):
        nc //= 2
    chunk = N // nc
    if nc < 2 or N % nc or chunk % _HIER_BLOCK1 or chunk % _HIER_BLOCK3:
        return False
    cmin = torch.empty((M2 * nc,), dtype=torch.float32, device=device)
    minv = torch.empty((M2,), dtype=torch.float32, device=device)
    win = torch.empty((M2,), dtype=torch.int64, device=device)
    with torch_device_fn.device(device):
        argmin_chunk_min[(M2, nc)](src, cmin, N, nc, _HIER_BLOCK1)
        argmin_win_chunk[(M2,)](cmin, minv, win, nc, triton.next_power_of_2(nc))
        argmin_chunk_idx[(M2,)](src, win, minv, out_flat, N, chunk, _HIER_BLOCK3)
    return True


# ---------------------------------------------------------------------------
# dim=None fast path for large flat inputs (floats).  A narrow-BLOCK_N
# return_indices row reduce runs at ~1 TB/s, then a tiny combine over the
# per-row minima picks the global winner.  Row width / tiles measured on XPU.
# ---------------------------------------------------------------------------
_FLAT_ROW = 2048
_FLAT_BM = 64
_FLAT_BN = 256


@libentry()
@triton.jit
def argmin_mri_vi(
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
    accv = tl.full([BLOCK_M, 1], float("inf"), tl.float32)
    acci = tl.full([BLOCK_M, 1], 1 << 30, tl.int32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        v = tl.load(p + cols).to(tl.float32)
        cm, ci = tl.min(
            v, axis=1, return_indices=True, return_indices_tie_break_left=True
        )
        cm = cm[:, None]
        ci = ci[:, None] + off
        acci = tl.where(cm < accv, ci, acci)
        accv = tl.minimum(accv, cm)
    g = IDX_BASE + rows.to(tl.int64) * ROW_STRIDE + acci.to(tl.int64)
    tl.store(val + rows.to(tl.int64), accv, mask=rows < M)
    tl.store(gidx + rows.to(tl.int64), g, mask=rows < M)


@libentry()
@triton.jit
def argmin_combine(val, gidx, out, M, BLOCK: tl.constexpr):
    off = tl.arange(0, BLOCK)
    mask = off < M
    v = tl.load(val + off, mask=mask, other=float("inf")).to(tl.float32)
    u = v.to(tl.int32, bitcast=True)
    ordered = tl.where(u < 0, ~u, u ^ -2147483648)
    ordered = tl.where((u & 0x7FFFFFFF) > 0x7F800000, -8388608, ordered)
    pack = ((ordered.to(tl.int64) & 0xFFFFFFFF) << 30) | off.to(tl.int64)
    best = tl.min(tl.where(mask, pack, 0x7FFFFFFFFFFFFFFF), axis=0)
    c = (best & ((1 << 30) - 1)).to(tl.int64)
    tl.store(out, tl.load(gidx + c))


def _argmin_flat_mri(inp, out, device):
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
            argmin_mri_vi[(triton.cdiv(rows, bm),)](
                flat, val, gidx, rows, row, bm, bn, 0, row
            )
    if res:
        bn = _FLAT_BN
        while bn > 1 and res % bn:
            bn //= 2
        with torch_device_fn.device(device):
            argmin_mri_vi[(1,)](
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
        argmin_combine[(1,)](
            val, gidx, out.reshape(-1), n_mid, triton.next_power_of_2(n_mid)
        )


def _argmin_flat_fast(inp, out, device):
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
        argmin_kernel_2d[(triton.cdiv(rows, bm),)](
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
        argmin_kernel_2d[(1,)](
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
        argmin_kernel_mid[(nxt_m,)](mid, nxt, n, block, False, buffer_size_limit=2048)
        mid, n = nxt, nxt_m
    argmin_kernel_mid[(1,)](
        mid,
        out.reshape(-1),
        n,
        triton.next_power_of_2(n),
        True,
        buffer_size_limit=2048,
    )


def _argmin_flat_legacy(inp, out, device):
    """dim=None path for int dtypes / small inputs (unchanged from HEAD)."""
    M = inp.numel()
    dtype = inp.dtype
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid_value = torch.empty((mid_size,), dtype=dtype, device=device)
    mid_index = torch.empty((mid_size,), dtype=torch.int64, device=device)
    tl_dtype, dtype_max_value = torch_dtype_to_tl_dtype_and_max_value[dtype]
    with torch_device_fn.device(device):
        argmin_kernel_1[(mid_size, 1, 1)](
            inp,
            mid_value,
            mid_index,
            M,
            block_size,
            dtype_max_value,
        )
        argmin_kernel_2[(1, 1, 1)](
            mid_value,
            mid_index,
            out,
            mid_size,
            block_mid,
            dtype_max_value,
        )


def argmin(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS_KUNLUNXIN ARGMIN")
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

        if _is_fast_dtype(dtype) and (1 << 20) <= M <= (1 << 24):
            # Mid-size flat inputs: narrow-BN return_indices rows + tiny combine.
            _argmin_flat_mri(inp, out, inp.device)
        elif (
            _is_fast_dtype(dtype)
            and M > 2 * 1024 * 1024
            and M > _FULL_REDUCTION_BLOCK_SIZE
        ):
            # Very large flat inputs: the packed rows-of-8192 pass wins here.
            _argmin_flat_fast(inp, out, inp.device)
        else:
            _argmin_flat_legacy(inp, out, inp.device)
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

        # argmin along a size-1 dim is trivially index 0. A return_indices
        # reduce over a size-1 axis fails XPU layout inference for BLOCK_M > 1,
        # so skip the kernel entirely.
        if N == 1:
            out_index.zero_()
            return out_index

        tl_dtype, dtype_max_value = torch_dtype_to_tl_dtype_and_max_value[inp.dtype]
        is_fp32 = inp.dtype == torch.float32

        # ---- vendor SIMD min row reducer (value + index) -------------------
        # The payload needs the reduced axis to be physically innermost and
        # contiguous, so reorder (transpose copy only when the reduced axis is
        # not already last). Unsupported shapes/dtypes keep the plain-triton
        # paths below.
        if (
            _MIN_RAW_OK
            and _is_fast_dtype(inp.dtype)
            and M * K >= 2
            and (N >= _RAW_DIM_MIN_N or inp.dtype != torch.bfloat16)
        ):
            rows = M * K
            seg = _pick_seg(rows, N)
            # Tiny single-segment fp32 tensors are faster on the packed int64
            # kernel (see _RAW_MIN_ELEMS_FP32); leave them to the paths below.
            if not (
                seg == 1
                and inp.dtype == torch.float32
                and rows * N < _RAW_MIN_ELEMS_FP32
            ):
                perm = [d for d in range(inp.dim()) if d != dim] + [dim]
                view = inp.permute(perm)
                if view.is_contiguous():
                    src = view
                else:
                    src = torch.empty(
                        list(view.shape), dtype=inp.dtype, device=inp.device
                    )
                    with torch_device_fn.device(inp.device):
                        _vendor_copy(src, view)
                if seg == 1:
                    raw = _min_raw_dim(src, src.dim() - 1, False)
                    if raw is not None:
                        idx = raw[1]  # (non-reduced dims) in original order
                        return idx.unsqueeze(dim) if keepdim else idx
                else:
                    width = N // seg
                    parts = src.reshape(rows * seg, width)
                    raw = _min_raw_dim(parts, 1, False)
                    if raw is not None:
                        vals, loc = raw
                        out_flat = torch.empty(
                            (rows,), dtype=torch.int64, device=inp.device
                        )
                        bm = _SEG_COMBINE_BM
                        bs = triton.next_power_of_2(seg)
                        with torch_device_fn.device(inp.device):
                            argmin_seg_combine[(triton.cdiv(rows, bm),)](
                                vals, loc, out_flat, rows, seg, width, bm, bs
                            )
                        out_shaped = out_flat.reshape(list(src.shape[:-1]))
                        return out_shaped.unsqueeze(dim) if keepdim else out_shaped

        # ---- hierarchical path for very large N (K == 1, floats) -----------
        # The index logic is the bottleneck; only touch 1/NC of the data with it.
        if _is_fast_dtype(inp.dtype) and K == 1 and N >= _HIER_MIN_N:
            if _argmin_hierarchical(inp, out_index.reshape(-1), M, N, inp.device):
                return out_index

        # ---- return_indices path (narrow tile -> intra-core reduce) ---------
        if _is_fast_dtype(inp.dtype) and _MRI_MIN_N <= N < _HIER_MIN_N and M * K >= 128:
            perm = [d for d in range(inp.dim()) if d != dim] + [dim]
            view = inp.permute(perm)
            if view.is_contiguous():
                src = view
            else:
                src = torch.empty(list(view.shape), dtype=inp.dtype, device=inp.device)
                with torch_device_fn.device(inp.device):
                    _vendor_copy(src, view)
            M2 = M * K
            tile = _pick_mri_tile(M2, N, is_fp32)
            if tile is not None:
                bm, bn = tile
                with torch_device_fn.device(inp.device):
                    argmin_return_idx[(M2 // bm,)](
                        src, out_index.reshape(-1), M2, N, bm, bn
                    )
                return out_index

        # ---- fast packed row-reduction path (floats only) -----------------
        # fp16 packs a 16-bit column index (its int16 bitcast is direct), so its
        # int32 path covers N up to 2**16.  fp32 needs the 30-bit int64 index
        # (MAX_TILE_N cap), and bf16 loads as fp32 on XPU (extra round-trip) so
        # its packed path stays at the MAX_TILE_N cap.
        packed_hi = 65536 if inp.dtype == torch.float16 else MAX_TILE_N
        if _is_fast_dtype(inp.dtype) and _FAST_MIN_N <= N <= packed_hi:
            # Reorder so the reduced dim (N) is physically innermost; the copy
            # uses the vendor pure-triton `copy_` override (zero aten).
            perm = [d for d in range(inp.dim()) if d != dim] + [dim]
            view = inp.permute(perm)
            if view.is_contiguous():
                src = view
            else:
                src = torch.empty(list(view.shape), dtype=inp.dtype, device=inp.device)
                with torch_device_fn.device(inp.device):
                    _vendor_copy(src, view)
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
                        argmin_kernel_2d[(M2 // block_m,)](
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
                        argmin_kernel_2d_16[(M2 // block_m,)](
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
            if M2 >= 2 and N >= 2:
                bn = next((b for b in bns if N % b == 0), None)
                if bn is not None:
                    bm = next((m for m in bms if M2 % m == 0), 2)
                    grid_m = triton.cdiv(M2, bm)
                    with torch_device_fn.device(inp.device):
                        if is_fp32:
                            argmin_kernel_2d[(grid_m,)](
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
                            argmin_kernel_2d_16[(grid_m,)](
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

        if not (N > MAX_TILE_N or (K == 1 and N >= TWO_STAGE_MIN_N)):
            grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), K)  # noqa: E731
            with torch_device_fn.device(inp.device):
                argmin_kernel_small_n[grid](
                    inp,
                    out_index,
                    M,
                    N,
                    K,
                    tl_dtype,
                    dtype_max_value,
                )
            return out_index

        # Two-stage per-row reduction (legacy path; see HEAD rationale).
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
            argmin_stage1[grid1](
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
            # the final global argmin into part_idx (layout matches out_index),
            # so skip stage 2 entirely.
            if num_chunks == 1:
                out_index.view(-1).copy_(part_idx)
                return out_index
            block_c = triton.next_power_of_2(num_chunks)
            grid2 = (triton.cdiv(M, block_m), K)
            argmin_stage2[grid2](
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
