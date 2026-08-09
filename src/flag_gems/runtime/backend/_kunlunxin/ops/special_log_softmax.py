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
#
# Kunlunxin (XPU) override of `special_log_softmax`.
#
# `special_log_softmax(self, dim, dtype=None)` is functionally identical to
# `log_softmax` over `dim` (with an optional output dtype cast).
#
# XPU dispatch (measured 2026-08-15 on XPU 5, official 9-shape matrix):
#   * the HEAD single-tile kernels silently mis-compute for N > 8192 (tl.sum
#     over >8192 lanes drops lanes on this backend): (1024,65536) maxdiff
#     ~2.1 and (1024,1048576) ~1.4 vs fp64 reference; the official benchmark
#     does not validate output so that was invisible.
#   * per-row 1D reduces are launch-bound for huge M: (64,512,512) had 32768
#     programs -> 19.4ms.
#   * 2D multirow [TILE_M, N] single-pass tile + order-preserving uint32-key
#     int max (bit ^ (0x80000000 | (bit >> 31)), radix-sort family) is the
#     fastest correct path for N <= 4096 (the fp wide-row tl.max is a serial
#     chain on XPU; int-key max measured ~2-4x faster; always-true-row mask-
#     free tiles keep contiguous block DMA).
#   * N > 4096: chunk split with FLAT grids (one program per (row, chunk));
#     runtime row loops measure ~20-60x slower on this backend, and any
#     runtime `row*N + c*BN` offset form collapses to discrete gathers, so the
#     chunk offsets are pid*BN (BN constexpr) - block DMA. Full 8192-wide
#     chunks are unmasked (masked column tiles miscompile); the row tail is
#     split into <= 4096-lane 1D masked pieces (masked reduces are exact up to
#     4096 lanes). A fused combine+pass kernel miscompiles on this backend ->
#     partial / combine / pass 3-kernel structure (verified chunk-vs-fp64).
#
# 2026-08-30 (XPU 3) retune, measured with isolated per-config probes:
#   * `tl.max` on a 2D [TILE_M, N] tile is now *cheaper* than the order-
#     preserving uint32-key int max the 2026-08-15 revision introduced
#     ((4096,4096) fp16 0.8733 -> 0.4966 ms, identical maxdiff).  The int-key
#     trick is still faster for the 1D chunk reduce, so it is kept there for the
#     legacy fall-back path only.
#   * `buffer_size_limit=2048` is the single largest lever on this backend for
#     these kernels: (4096,4096) fp16 fp-max tile 0.4966 -> 0.3450 ms, and the
#     large-N second pass 1.90 -> 0.57 ms.  It is also what makes `tl.sum` /
#     `tl.max` exact above 8192 lanes (BN=32768 without it: maxdiff 1.42).
#   * TILE_M retuned to "tile holds ~32768 elements for N < 1024, ~65536 for
#     N >= 1024"; within 1% of the per-N optimum on every probed N.
#   * the fp-max tile compiles and stays exact up to N = 65536 (N = 131072
#     fails to compile: uni_sram), so the wide-tile path now covers the whole
#     64 <= N <= 65536 power-of-2 range and beats the chunk split there
#     ((1024,8192) 0.657 -> 0.155 ms, (1024,65536) 4.768 -> 1.498 ms).
#   * N > 65536: 2D-tiled partial ([64, 512] tile per 32768-element contiguous
#     block) + 2D-tiled combine + flat second pass.  (1024,1048576) fp16
#     76.2 -> 26.7 ms.  A single kernel doing "2D axis=1 reduce then fold the
#     L partials with a 1D reduce" does NOT compile on this backend
#     (uni_sram / PassManager::run failed) - do not retry it.
#   * N == 1 is a degenerate reduction: out == x - x bit-exactly (including
#     +-inf -> NaN, matching eager ATen), so it is a flat pointwise pass
#     (M=1024: 0.0275 -> 0.0054 ms).
#   * The new paths never use a row mask: M is split into power-of-2 row
#     segments and each segment gets its own launch on an offset view.  The
#     masked 2D store in the legacy kernel below writes
#     `(TILE_M - M % TILE_M) * N` elements past the output (canary-verified,
#     pre-existing), and `tl.arange(0, N)` with non-power-of-2 N mis-lowers on
#     this backend (N=65/127/333/997/1500/3000 all wrong vs float64, plus tail
#     writes).  Both defects are pre-existing in the legacy path, which is only
#     reached now for non-power-of-2 N or N < 64; they are reported separately
#     and NOT touched here.
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

from .log_softmax import log_softmax as _log_softmax_kunlunxin

logger = logging.getLogger(__name__)

_MULTIROW_MAX_N = 4096  # 2D [TILE_M, N] single-pass tile family
_CHUNK_BN = 8192  # big-N chunk width (tl.sum/tl.max lane-safety bound)
_TAIL_PIECE = 4096  # masked 1D tail pieces kept <= 4096 lanes (exact)
# TILE_M buckets per N for the single-pass tile (probed on XPU 5).
# Non-power-of-2 N < 64 needs TILE_M >= 64 to compile correctly; handled in
# the dispatch below.
_N_TILE_M = [(16, 64), (64, 32), (256, 32), (1024, 16), (4096, 8)]
_NUM_WARPS = 8

# --- 2026-08-30 fast paths (see the module docstring for the measurements) ---
_BSL = 2048  # triton-xpu buffer_size_limit
_TILE_MIN_N = 64  # below this the legacy int-key tile is not slower
# The widest tile the fp-max kernel still compiles for is 128 KiB of input per
# program: N=65536 works for 2-byte dtypes, fails (uni_sram) for fp32, and
# N=131072 fails for every dtype. Anything wider goes to the chunk pipeline.
_TILE_MAX_BYTES = 128 * 1024
_TILE_ELEMS_SMALL = 32768  # target tile volume for N < 1024
_TILE_ELEMS_LARGE = 65536  # target tile volume for N >= 1024
_BIG_BN = 512  # inner reduce width of the large-N partial
_BIG_L = 64  # rows of the large-N partial tile (_BIG_L * _BIG_BN per program)
_BIG_R = 64  # rows per program in the large-N combine
_N1_BLOCK = 1024  # flat block for the N == 1 degenerate path
_VEC_STORE_ELEMS = 64  # any vector store touches 64 contiguous elements
_MAX_SEGMENTS = 12  # give up on the segment split past this many launches


def _is_pow2(n):
    return n > 0 and (n & (n - 1)) == 0


def _row_segments(count, unit_bytes, min_seg):
    """Split `count` rows into power-of-2 segments, lowest bit first.

    Every segment start is a multiple of the smallest segment, so each launch
    gets an aligned base pointer and a grid that divides the segment exactly -
    i.e. no row mask, no ``other=`` and no masked store anywhere.  Returns None
    when a safe split is not possible; callers then fall back to the legacy
    path instead of guessing.
    """
    if count <= 0:
        return None
    low = count & -count
    if low < min_seg or (low * unit_bytes) % 32 != 0:
        return None
    if bin(count).count("1") > _MAX_SEGMENTS:
        return None
    segments = []
    base = 0
    rem = count
    while rem:
        seg = rem & -rem
        segments.append((base, seg))
        base += seg
        rem -= seg
    return segments


@triton.jit
def _sls_tile_fp_kernel(o_ptr, i_ptr, N: tl.constexpr, TILE_M: tl.constexpr):
    """Unmasked [TILE_M, N] single-pass tile using the plain fp row max."""
    pid = tl.program_id(0)
    off = (pid * TILE_M + tl.arange(0, TILE_M))[:, None] * N + tl.arange(0, N)[None, :]
    x = tl.load(i_ptr + off).to(tl.float32)
    m = tl.max(x, 1)
    d = x - m[:, None]
    z = tl.sum(tl.exp(d), 1)
    tl.store(o_ptr + off, d - tl.log(z)[:, None])


@triton.jit
def _sls_n1_kernel(o_ptr, i_ptr, BLOCK: tl.constexpr):
    """N == 1: log_softmax degenerates to x - x (finite -> 0, +-inf -> NaN)."""
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(i_ptr + off)
    tl.store(o_ptr + off, x - x)


@triton.jit
def _sls_big_partial_kernel(pm_ptr, pz_ptr, i_ptr, L: tl.constexpr, BN: tl.constexpr):
    """One contiguous L*BN block per program, reduced as a [L, BN] tile.

    Partial slot index is pid*L + l because the flat program id already equals
    row * (blocks per row) + block, so no divmod and no runtime scalar is needed.
    """
    pid = tl.program_id(0)
    lo = tl.arange(0, L)
    off = pid * (L * BN) + lo[:, None] * BN + tl.arange(0, BN)[None, :]
    x = tl.load(i_ptr + off).to(tl.float32)
    m = tl.max(x, 1)
    z = tl.sum(tl.exp(x - m[:, None]), 1)
    po = pid * L + lo
    tl.store(pm_ptr + po, m)
    tl.store(pz_ptr + po, z)


@triton.jit
def _sls_big_combine_kernel(
    mk_ptr, lz_ptr, pm_ptr, pz_ptr, C: tl.constexpr, R: tl.constexpr
):
    """Fold C partials of R rows at a time -> (row max, row logsumexp)."""
    ro = tl.program_id(0) * R + tl.arange(0, R)
    off = ro[:, None] * C + tl.arange(0, C)[None, :]
    mc = tl.load(pm_ptr + off)
    zc = tl.load(pz_ptr + off)
    mk = tl.max(mc, 1)
    z = tl.sum(zc * tl.exp(mc - mk[:, None]), 1)
    tl.store(mk_ptr + ro, mk)
    tl.store(lz_ptr + ro, tl.log(z))


@triton.jit
def _sls_big_pass_kernel(
    o_ptr, i_ptr, mk_ptr, lz_ptr, CQ: tl.constexpr, BN: tl.constexpr
):
    """Flat unmasked second pass: out = x - row_max - row_logsumexp."""
    pid = tl.program_id(0)
    row = pid // CQ
    mk = tl.load(mk_ptr + row)
    lz = tl.load(lz_ptr + row)
    off = pid * BN + tl.arange(0, BN)
    x = tl.load(i_ptr + off).to(tl.float32)
    tl.store(o_ptr + off, x - mk - lz)


@triton.jit
def _key_u32(bits):
    return bits ^ (0x80000000 | (bits >> 31))


@triton.jit
def _decode_key(m_key):
    return (m_key ^ (0x80000000 | ((m_key >> 31) ^ 1))).to(tl.float32, bitcast=True)


@triton.jit
def _sls_singlepass_kernel(
    o_ptr, i_ptr, M, N: tl.constexpr, TILE_M: tl.constexpr, NEED_MASK: tl.constexpr
):
    """Single-pass [TILE_M, N] tile: int-key max, exp, sum, store."""
    pid = tl.program_id(0)
    mo = pid * TILE_M + tl.arange(0, TILE_M)
    no = tl.arange(0, N)
    off = mo[:, None] * N + no[None, :]
    m_mask = mo[:, None] < M
    if NEED_MASK:
        x = tl.load(i_ptr + off, mask=m_mask, other=-float("inf")).to(tl.float32)
    else:
        x = tl.load(i_ptr + off).to(tl.float32)
    bits = x.to(tl.uint32, bitcast=True)
    m_key = tl.max(_key_u32(bits), 1)
    m = _decode_key(m_key)
    e = tl.exp(x - m[:, None])
    z = tl.sum(e, 1)
    out = x - m[:, None] - tl.log(z)[:, None]
    if NEED_MASK:
        tl.store(o_ptr + off, out, mask=m_mask)
    else:
        tl.store(o_ptr + off, out)


@triton.jit
def _sls_chunk_kernel(
    pm_ptr,
    pz_ptr,
    i_ptr,
    C_FULL,
    C,
    BN: tl.constexpr,
):
    """Flat (row*C_FULL + c) grid; offsets = pid*BN (BN constexpr -> the
    [M*C_FULL, BN] read is contiguous, block DMA on XPU).
    Partial (m_c, z_c) stored at row*C + c (C includes tail slots)."""
    pid = tl.program_id(0)
    row = pid // C_FULL
    c = pid % C_FULL
    no = tl.arange(0, BN)
    off = pid * BN + no
    x = tl.load(i_ptr + off).to(tl.float32)
    bits = x.to(tl.uint32, bitcast=True)
    m_key = tl.max(_key_u32(bits), 0)
    m = _decode_key(m_key)
    z = tl.sum(tl.exp(x - m), 0)
    tl.store(pm_ptr + row * C + c, m)
    tl.store(pz_ptr + row * C + c, z)


@triton.jit
def _sls_chunk_kernel_strided(
    pm_ptr,
    pz_ptr,
    i_ptr,
    N,
    C_FULL,
    C,
    BN: tl.constexpr,
):
    """Flat (row*C_FULL + c) grid with per-row base offsets (needed when
    N % BN != 0: the flat pid*BN form drifts by the row tail)."""
    pid = tl.program_id(0)
    row = pid // C_FULL
    c = pid % C_FULL
    no = tl.arange(0, BN)
    off = row * N + c * BN + no
    x = tl.load(i_ptr + off).to(tl.float32)
    bits = x.to(tl.uint32, bitcast=True)
    m_key = tl.max(_key_u32(bits), 0)
    m = _decode_key(m_key)
    z = tl.sum(tl.exp(x - m), 0)
    tl.store(pm_ptr + row * C + c, m)
    tl.store(pz_ptr + row * C + c, z)


@triton.jit
def _sls_chunk_pass_kernel_strided(
    o_ptr,
    i_ptr,
    mk_ptr,
    lz_ptr,
    N,
    C_FULL,
    C,
    BN: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // C_FULL
    c = pid % C_FULL
    mk = tl.load(mk_ptr + row)
    lz = tl.load(lz_ptr + row)
    no = tl.arange(0, BN)
    off = row * N + c * BN + no
    x = tl.load(i_ptr + off).to(tl.float32)
    tl.store(o_ptr + off, x - mk - lz)


@triton.jit
def _sls_tail_partial_kernel(
    pm_ptr,
    pz_ptr,
    i_ptr,
    N,
    C_STRIDE,
    TAIL_BASE,
    TL: tl.constexpr,
):
    """1D masked tail piece (width <= 4096 keeps masked reduces exact).
    pm_ptr/pz_ptr point at the piece slots; scalar stores at row*C_STRIDE."""
    pid = tl.program_id(0)
    no = TAIL_BASE + tl.arange(0, TL)
    off = pid * N + no
    mask = no < N
    x = tl.load(i_ptr + off, mask=mask, other=-float("inf")).to(tl.float32)
    bits = x.to(tl.uint32, bitcast=True)
    m = _decode_key(tl.max(_key_u32(bits), 0))
    safe_m = tl.where(m == -float("inf"), 0.0, m)
    z = tl.sum(tl.exp(x - safe_m), 0)
    po = pid * C_STRIDE
    tl.store(pm_ptr + po, m)
    tl.store(pz_ptr + po, z)


@triton.jit
def _sls_chunk_combine_kernel(mk_ptr, lz_ptr, pm_ptr, pz_ptr, C, C2: tl.constexpr):
    """Combine row partials -> (row max, logsumexp); row stride = C."""
    pid = tl.program_id(0)
    co = tl.arange(0, C2)
    cmask = co < C
    po = pid * C + co
    mc = tl.load(pm_ptr + po, mask=cmask, other=-float("inf"))
    zc = tl.load(pz_ptr + po, mask=cmask, other=0.0)
    mk = tl.max(mc, 0)
    z = tl.sum(zc * tl.exp(mc - mk), 0)
    lz = tl.log(z)
    tl.store(mk_ptr + pid, mk)
    tl.store(lz_ptr + pid, lz)


@triton.jit
def _sls_chunk_pass_kernel(
    o_ptr,
    i_ptr,
    mk_ptr,
    lz_ptr,
    C_FULL,
    C,
    BN: tl.constexpr,
):
    """Flat grid (M*C_FULL): second read (offset = pid*BN) -> out."""
    pid = tl.program_id(0)
    row = pid // C_FULL
    mk = tl.load(mk_ptr + row)
    lz = tl.load(lz_ptr + row)
    no = tl.arange(0, BN)
    off = pid * BN + no
    x = tl.load(i_ptr + off).to(tl.float32)
    tl.store(o_ptr + off, x - mk - lz)


@triton.jit
def _sls_tail_pass_kernel(
    o_ptr,
    i_ptr,
    mk_ptr,
    lz_ptr,
    N,
    TAIL_BASE,
    TL: tl.constexpr,
):
    """Masked tail write (piece width <= 4096): out = x - row - logsumexp."""
    pid = tl.program_id(0)
    mk = tl.load(mk_ptr + pid)
    lz = tl.load(lz_ptr + pid)
    no = TAIL_BASE + tl.arange(0, TL)
    off = pid * N + no
    mask = no < N
    x = tl.load(i_ptr + off, mask=mask, other=0.0).to(tl.float32)
    tl.store(o_ptr + off, x - mk - lz, mask=mask)


def _fast_forward(out, inp, M, N):
    """Try the 2026-08-30 fast paths. Returns True when `out` has been filled.

    Every path here is fully unmasked; anything that cannot be covered without a
    mask returns False so the legacy kernels below stay in charge.
    """
    itemsize = inp.element_size()

    if N == 1:
        segments = _row_segments(M, itemsize, _VEC_STORE_ELEMS)
        if segments is None:
            return False
        if len(segments) == 1:
            # single power-of-2 segment: launch straight on the tensors, the
            # flatten + slice would cost more than this kernel takes
            _sls_n1_kernel[(M // min(_N1_BLOCK, M),)](
                out,
                inp,
                BLOCK=min(_N1_BLOCK, M),
                num_warps=_NUM_WARPS,
                buffer_size_limit=_BSL,
            )
            return True
        inp_flat = inp.view(-1)
        out_flat = out.view(-1)
        for base, seg in segments:
            block = min(_N1_BLOCK, seg)
            _sls_n1_kernel[(seg // block,)](
                out_flat[base:],
                inp_flat[base:],
                BLOCK=block,
                num_warps=_NUM_WARPS,
                buffer_size_limit=_BSL,
            )
        return True

    if not _is_pow2(N):
        # tl.arange(0, N) mis-lowers for non-power-of-2 N on this backend.
        return False

    if _TILE_MIN_N <= N <= _TILE_MAX_BYTES // itemsize:
        target = _TILE_ELEMS_SMALL if N < 1024 else _TILE_ELEMS_LARGE
        tile_m_max = max(1, target // N)
        segments = _row_segments(M, N * itemsize, 1)
        if segments is None:
            return False
        if len(segments) == 1:
            tile_m = min(tile_m_max, M)
            _sls_tile_fp_kernel[(M // tile_m,)](
                out,
                inp,
                N=N,
                TILE_M=tile_m,
                num_warps=_NUM_WARPS,
                buffer_size_limit=_BSL,
            )
            return True
        inp_flat = inp.view(-1)
        out_flat = out.view(-1)
        for base, seg in segments:
            tile_m = min(tile_m_max, seg)
            _sls_tile_fp_kernel[(seg // tile_m,)](
                out_flat[base * N :],
                inp_flat[base * N :],
                N=N,
                TILE_M=tile_m,
                num_warps=_NUM_WARPS,
                buffer_size_limit=_BSL,
            )
        return True

    if N > _TILE_MAX_BYTES // itemsize:
        blk = _BIG_L * _BIG_BN
        if N % blk != 0:
            return False
        blocks_per_row = N // blk
        n_partials = N // _BIG_BN
        rows_per_prog = min(_BIG_R, M & -M)
        # scratch is over-allocated: a vector store always touches 64 elements
        pad = _VEC_STORE_ELEMS
        pm = torch.empty((M * n_partials,), dtype=torch.float32, device=inp.device)
        pz = torch.empty((M * n_partials,), dtype=torch.float32, device=inp.device)
        mk = torch.empty((M + pad,), dtype=torch.float32, device=inp.device)
        lz = torch.empty((M + pad,), dtype=torch.float32, device=inp.device)
        _sls_big_partial_kernel[(M * blocks_per_row,)](
            pm,
            pz,
            inp,
            L=_BIG_L,
            BN=_BIG_BN,
            num_warps=_NUM_WARPS,
            buffer_size_limit=_BSL,
        )
        _sls_big_combine_kernel[(M // rows_per_prog,)](
            mk,
            lz,
            pm,
            pz,
            C=n_partials,
            R=rows_per_prog,
            num_warps=_NUM_WARPS,
            buffer_size_limit=_BSL,
        )
        _sls_big_pass_kernel[(M * blocks_per_row,)](
            out,
            inp,
            mk,
            lz,
            CQ=blocks_per_row,
            BN=blk,
            num_warps=_NUM_WARPS,
            buffer_size_limit=_BSL,
        )
        return True

    return False


def special_log_softmax(self, dim, dtype=None):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_LOG_SOFTMAX")

    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"
    dim = dim % self.ndim

    inp = self.contiguous()
    if dtype is not None and dtype != inp.dtype:
        inp = inp.to(dtype)

    M = 1
    for i in range(dim):
        M *= inp.shape[i]
    N = inp.shape[dim]
    K = inp.numel() // M // N

    # dim not last -> reduce dim is strided; delegate to the tuned log_softmax.
    if K != 1:
        return _log_softmax_kunlunxin(inp, dim)

    out = torch.empty_like(inp)
    if N == 0 or M == 0:
        return out

    with torch_device_fn.device(inp.device):
        if _fast_forward(out, inp, M, N):
            return out

    with torch_device_fn.device(inp.device):
        if N <= _MULTIROW_MAX_N:
            TILE_M = 4
            for n_hi, tm in _N_TILE_M:
                if N <= n_hi:
                    TILE_M = tm
                    break
            if (N & (N - 1)) and N < 64:
                TILE_M = 64  # tiny odd tiles miscompile below 64 rows
            need_mask = M % TILE_M != 0
            grid = (triton.cdiv(M, TILE_M),)
            _sls_singlepass_kernel[grid](
                out,
                inp,
                M,
                N=N,
                TILE_M=TILE_M,
                NEED_MASK=need_mask,
                num_warps=_NUM_WARPS,
            )
        else:
            C_FULL = N // _CHUNK_BN
            taillen = N - C_FULL * _CHUNK_BN
            have_tail = taillen != 0
            n = 0
            t0 = t1 = 0
            tl0 = tl1 = 0
            if have_tail:
                t0 = min(taillen, _TAIL_PIECE)
                t1 = taillen - t0
                tl0 = triton.next_power_of_2(t0)
                tl1 = triton.next_power_of_2(t1) if t1 else 0
                n = 2 if t1 else 1
            C = C_FULL + n
            C2 = triton.next_power_of_2(C)
            pm = torch.empty((M * C,), dtype=torch.float32, device=inp.device)
            pz = torch.empty((M * C,), dtype=torch.float32, device=inp.device)
            mk = torch.empty((M,), dtype=torch.float32, device=inp.device)
            lz = torch.empty((M,), dtype=torch.float32, device=inp.device)
            if have_tail:
                _sls_tail_partial_kernel[(M,)](
                    pm[C_FULL::C],
                    pz[C_FULL::C],
                    inp,
                    N,
                    C,
                    C_FULL * _CHUNK_BN,
                    tl0,
                    num_warps=_NUM_WARPS,
                )
                if t1:
                    _sls_tail_partial_kernel[(M,)](
                        pm[C_FULL + 1 :: C],
                        pz[C_FULL + 1 :: C],
                        inp,
                        N,
                        C,
                        C_FULL * _CHUNK_BN + t0,
                        tl1,
                        num_warps=_NUM_WARPS,
                    )
            if C_FULL:
                if have_tail:
                    # row base offsets are required when N % BN != 0
                    _sls_chunk_kernel_strided[(M * C_FULL,)](
                        pm,
                        pz,
                        inp,
                        N,
                        C_FULL,
                        C,
                        BN=_CHUNK_BN,
                        num_warps=_NUM_WARPS,
                    )
                else:
                    _sls_chunk_kernel[(M * C_FULL,)](
                        pm,
                        pz,
                        inp,
                        C_FULL,
                        C,
                        BN=_CHUNK_BN,
                        num_warps=_NUM_WARPS,
                    )
            _sls_chunk_combine_kernel[(M,)](
                mk,
                lz,
                pm,
                pz,
                C,
                C2=C2,
                num_warps=_NUM_WARPS,
            )
            if C_FULL:
                if have_tail:
                    _sls_chunk_pass_kernel_strided[(M * C_FULL,)](
                        out,
                        inp,
                        mk,
                        lz,
                        N,
                        C_FULL,
                        C,
                        BN=_CHUNK_BN,
                        num_warps=_NUM_WARPS,
                    )
                else:
                    _sls_chunk_pass_kernel[(M * C_FULL,)](
                        out,
                        inp,
                        mk,
                        lz,
                        C_FULL,
                        C,
                        BN=_CHUNK_BN,
                        num_warps=_NUM_WARPS,
                    )
            if have_tail:
                _sls_tail_pass_kernel[(M,)](
                    out,
                    inp,
                    mk,
                    lz,
                    N,
                    C_FULL * _CHUNK_BN,
                    tl0,
                    num_warps=_NUM_WARPS,
                )
                if t1:
                    _sls_tail_pass_kernel[(M,)](
                        out,
                        inp,
                        mk,
                        lz,
                        N,
                        C_FULL * _CHUNK_BN + t0,
                        tl1,
                        num_warps=_NUM_WARPS,
                    )

    return out
