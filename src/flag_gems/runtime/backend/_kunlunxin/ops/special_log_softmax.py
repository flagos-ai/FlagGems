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
    """Order-preserving float32 -> uint32 sort key (radix-sort family).

    The mask must be 0xFFFFFFFF for negatives and 0x80000000 for non-negatives.
    `bits >> 31` on a *uint32* yields 0/1 here (this backend lowers it as a
    logical shift), which is why the pre-2026-09-01 form
    `bits ^ (0x80000000 | (bits >> 31))` inverted the ordering of negatives.
    Bitcasting to int32 first makes the same `>> 31` an *arithmetic* shift
    (measured: key(-0.5) = 0x40FFFFFF = 0xBF000000 ^ 0xFFFFFFFF), at identical
    op count and identical measured latency.
    """
    m = (bits.to(tl.int32, bitcast=True) >> 31).to(tl.uint32, bitcast=True)
    return bits ^ (0x80000000 | m)


@triton.jit
def _decode_key(m_key):
    """Exact inverse of `_key_u32` (probe p8: bit-exact round-trip on +-0,
    subnormals, +-FLT_MAX and +-inf)."""
    s = m_key.to(tl.int32, bitcast=True) >> 31
    inv = (s ^ -1).to(tl.uint32, bitcast=True) | 0x80000000
    return (m_key ^ inv).to(tl.float32, bitcast=True)


@triton.jit
def _sls_singlepass_kernel(
    o_ptr,
    i_ptr,
    M,
    N: tl.constexpr,
    NB: tl.constexpr,
    TILE_M: tl.constexpr,
    PAD_M: tl.constexpr,
    PAD_N: tl.constexpr,
):
    """Single-pass [TILE_M, NB] tile: int-key max, exp, sum, store.

    `NB` is a power of 2 (>= 64 whenever it differs from `N`): `tl.arange(0, N)`
    with a non-power-of-2 `N` is silently widened by this backend to
    `round_up(N, 64)` lanes carrying the *continued* iota (probe_a_arange:
    N=333 -> 384 lanes valued 0..383), and for N >= 64 those extra lanes are
    neither masked out of `tl.max` / `tl.sum` nor dropped by the store, so the
    HEAD form `no = tl.arange(0, N)` read the next row into every row's
    reduction and wrote past the tensor on the last row.

    Neither dimension is ever masked here:

    * the row dimension never goes out of range because the *base* of the last
      tile is pulled back to `M - TILE_M` (`PAD_M`), so `mo` stays a contiguous
      iota.  This needs `TILE_M <= M`, which the host guarantees.  Measured on
      (1023, 4096) fp32: 6.94 ms vs HEAD's masked form 3.84 ms - but note both
      are far off the 0.227 ms that the same tile shape reaches at
      `M % TILE_M == 0` (M=1016/1024), i.e. any row-tail handling, mask or
      clamp, already costs ~17x here.  Pulling the base back in-kernel measured
      identical to clamping every row index (0.556 vs 0.555 of HEAD), so the
      cost is the row-tail regime itself, not the clamp form.  A host-side split
      into "full tiles" + "one tile based at row M - TILE_M" would keep the
      0.227 ms path, but needs a 32-byte-aligned row pitch - not attempted.
    * out-of-range column indices are clamped onto column N-1 (`PAD_N`), so
      every load and every store address stays inside the tensor (a masked
      store writes anyway on this backend, and `other=` on a masked load is
      ignored).  This is the expensive part: (1024, 997) fp32 1.94 ms vs HEAD's
      0.070 ms - HEAD was silently wrong there, so that is not a baseline.
    * both forms are idempotent: rows reduce independently along axis 1, so a
      row a shifted tile redoes yields the same m / z / out, and a duplicated
      column just re-stores column N-1's own value.  Every redundant store
      writes bit-identical bytes to an address that is written anyway.
    * the lanes that only exist because `NB > N` are removed from the two
      cross-lane reductions with an explicit `tl.where`.  The `-inf` fill is
      semantics-bearing, so it must not be delegated to `other=`.
    """
    pid = tl.program_id(0)
    if PAD_M:
        mo = tl.minimum(pid * TILE_M, M - TILE_M) + tl.arange(0, TILE_M)
    else:
        mo = pid * TILE_M + tl.arange(0, TILE_M)
    nr = tl.arange(0, NB)
    if PAD_N:
        off = mo[:, None] * N + tl.minimum(nr, N - 1)[None, :]
    else:
        off = mo[:, None] * N + nr[None, :]
    x = tl.load(i_ptr + off).to(tl.float32)
    if PAD_N:
        # the lanes that only exist because NB > N must not reach the reductions;
        # the -inf fill is semantics-bearing so `other=` (ignored here) cannot do it
        live = nr[None, :] < N
        m = tl.max(tl.where(live, x, -float("inf")), 1)
        d = x - m[:, None]
        z = tl.sum(tl.where(live, tl.exp(d), 0.0), 1)
    else:
        m_key = tl.max(_key_u32(x.to(tl.uint32, bitcast=True)), 1)
        m = _decode_key(m_key)
        d = x - m[:, None]
        z = tl.sum(tl.exp(d), 1)
    tl.store(o_ptr + off, d - tl.log(z)[:, None])


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
    # `other=` is ignored on this backend, so the out-of-range lanes hold whatever
    # was in memory: they must be removed from both reductions explicitly.
    bits = tl.where(mask, x, -float("inf")).to(tl.uint32, bitcast=True)
    m = _decode_key(tl.max(_key_u32(bits), 0))
    safe_m = tl.where(m == -float("inf"), 0.0, m)
    z = tl.sum(tl.where(mask, tl.exp(x - safe_m), 0.0), 0)
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
            while TILE_M > M:
                # the kernel's row-tail handling pulls the last tile's base back
                # to M - TILE_M, so TILE_M must not exceed M (halving keeps it a
                # power of 2, which `tl.arange` needs)
                TILE_M //= 2
            NB = triton.next_power_of_2(N)
            if NB != N and NB < 64:
                # a sub-64-lane block is widened to 64 lanes by the backend
                # regardless (probe_a_arange), so ask for the 64 lanes
                # explicitly - then every lane is clamped and gated below.
                NB = 64
            grid = (triton.cdiv(M, TILE_M),)
            _sls_singlepass_kernel[grid](
                out,
                inp,
                M,
                N=N,
                NB=NB,
                TILE_M=TILE_M,
                PAD_M=M % TILE_M != 0,
                PAD_N=NB != N,
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
