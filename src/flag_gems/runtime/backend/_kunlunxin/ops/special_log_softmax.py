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
    pm_ptr, pz_ptr, i_ptr, C_FULL, C, BN: tl.constexpr,
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
    pm_ptr, pz_ptr, i_ptr, N, C_FULL, C, BN: tl.constexpr,
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
    o_ptr, i_ptr, mk_ptr, lz_ptr, N, C_FULL, C, BN: tl.constexpr,
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
    pm_ptr, pz_ptr, i_ptr, N, C_STRIDE, TAIL_BASE, TL: tl.constexpr,
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
def _sls_chunk_combine_kernel(
    mk_ptr, lz_ptr, pm_ptr, pz_ptr, C, C2: tl.constexpr
):
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
    o_ptr, i_ptr, mk_ptr, lz_ptr, C_FULL, C, BN: tl.constexpr,
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
    o_ptr, i_ptr, mk_ptr, lz_ptr, N, TAIL_BASE, TL: tl.constexpr,
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
                out, inp, M, N=N, TILE_M=TILE_M, NEED_MASK=need_mask,
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
                    pm[C_FULL::C], pz[C_FULL::C], inp, N, C,
                    C_FULL * _CHUNK_BN, tl0, num_warps=_NUM_WARPS,
                )
                if t1:
                    _sls_tail_partial_kernel[(M,)](
                        pm[C_FULL + 1::C], pz[C_FULL + 1::C], inp, N, C,
                        C_FULL * _CHUNK_BN + t0, tl1, num_warps=_NUM_WARPS,
                    )
            if C_FULL:
                if have_tail:
                    # row base offsets are required when N % BN != 0
                    _sls_chunk_kernel_strided[(M * C_FULL,)](
                        pm, pz, inp, N, C_FULL, C, BN=_CHUNK_BN,
                        num_warps=_NUM_WARPS,
                    )
                else:
                    _sls_chunk_kernel[(M * C_FULL,)](
                        pm, pz, inp, C_FULL, C, BN=_CHUNK_BN,
                        num_warps=_NUM_WARPS,
                    )
            _sls_chunk_combine_kernel[(M,)](
                mk, lz, pm, pz, C, C2=C2, num_warps=_NUM_WARPS,
            )
            if C_FULL:
                if have_tail:
                    _sls_chunk_pass_kernel_strided[(M * C_FULL,)](
                        out, inp, mk, lz, N, C_FULL, C, BN=_CHUNK_BN,
                        num_warps=_NUM_WARPS,
                    )
                else:
                    _sls_chunk_pass_kernel[(M * C_FULL,)](
                        out, inp, mk, lz, C_FULL, C, BN=_CHUNK_BN,
                        num_warps=_NUM_WARPS,
                    )
            if have_tail:
                _sls_tail_pass_kernel[(M,)](
                    out, inp, mk, lz, N, C_FULL * _CHUNK_BN, tl0,
                    num_warps=_NUM_WARPS,
                )
                if t1:
                    _sls_tail_pass_kernel[(M,)](
                        out, inp, mk, lz, N, C_FULL * _CHUNK_BN + t0, tl1,
                        num_warps=_NUM_WARPS,
                    )

    return out