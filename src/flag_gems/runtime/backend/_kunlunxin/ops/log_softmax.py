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

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@triton.jit
def prev_multiple_of(a, b):
    # the largest x<a that x%b ==0
    return tl.cdiv(a, b) * b - b


# N above which the single-load 2D multirow tile no longer fits sram; fall back
# to the per-row online kernel.
MULTIROW_MAX_N = 8192


def _prev_pow2(x):
    x = max(1, int(x))
    return 1 << (x.bit_length() - 1)


def _multirow_tile_m(N):
    # Pack several rows per program so the [TILE_M, N] tile is one contiguous
    # block DMA. TILE_M is capped at 16: with masked tail rows a larger TILE_M
    # (e.g. 32) hits an XPU codegen bug that corrupts valid rows.
    return min(16, _prev_pow2(max(1, MULTIROW_MAX_N // N)))


# ------------------------  forward -------------------------------
# XPU dispatch (probed on XPU 5, 2026-08-19): the old masked 2D multirow
# [TILE_M, N] tile and the per-row online inner kernel are 20-100x slower
# than the special_log_softmax-style paths (measured): (4096,4096) 14.7ms,
# (10000,256) 3.0ms, (10000,65536) 1548ms. Replaced with:
#   * N <= 4096: single-pass [TILE_M, N] tile with order-preserving uint32-key
#     integer max (the fp wide-row tl.max is a serial chain on XPU; int-key
#     max ~2-4x faster), TILE_M bucketed by N, always-true row mask compiled
#     away (NEED_MASK); a masked tail launch covers M % TILE_M != 0 rows.
#   * N > 4096: chunk split partial/combine/pass with FLAT grids and
#     pid*BN (BN constexpr) offsets -> contiguous block DMA (runtime
#     row*N + c*BN offsets collapse to discrete gathers on XPU, 20-60x slower).
#     Full 8192-wide chunks are unmasked (masked column tiles miscompile);
#     a non-multiple tail is split into <= 4096-lane 1D masked pieces
#     (masked reduces are exact up to 4096 lanes). A fused combine+pass
#     kernel miscompiles on this backend -> 3-kernel structure.
# The per-row online inner kernel below is kept for the K>1 (interior-dim)
# path only (grid=(M*K,)); the old masked 2D multirow kernel is removed.
FWD_MULTIROW_MAX_N = 4096  # single-pass 2D tile family (<= 8K elems/prog)
FWD_CHUNK_BN = 8192  # big-N chunk width (tl.sum/tl.max lane-safety bound)
FWD_TAIL_PIECE = 4096  # masked 1D tail pieces kept <= 4096 lanes (exact)
# TILE_M buckets per N (probed XPU 5). Non-power-of-2 N < 64 needs TILE_M>=64
# to compile correctly; handled in the dispatch.
FWD_N_TILE_M = [(16, 64), (64, 32), (256, 16), (1024, 16), (4096, 8)]


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_inner"))
@triton.jit
def log_softmax_kernel_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    OUTPUT_K: tl.constexpr,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = ext.program_id(0)
    outer = pid_m // OUTPUT_K
    inner = pid_m % OUTPUT_K
    input_base = pid_m * N
    output_base = outer * N * OUTPUT_K + inner
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        input_offset = input_base + n_offsets
        output_offset = output_base + n_offsets * OUTPUT_K
        mask = n_offsets < N
        inp = tl.load(input_ptr + input_offset, mask=mask, other=-float("inf")).to(
            tl.float32
        )
        m = tl.max(inp, 0)
        e = tl.exp(inp - m)
        z = tl.sum(e, 0)
        log_z = tl.log(z)
        out = inp - m - log_z
        tl.store(output_ptr + output_offset, out, mask=mask)
    else:
        m = tl.full([TILE_N], value=float("-inf"), dtype=tl.float32)
        z = tl.full([TILE_N], value=0.0, dtype=tl.float32)
        input_ptr += input_base
        output_ptr += output_base

        previous_multiple = prev_multiple_of(N, TILE_N)
        for start_n in range(0, previous_multiple, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            inp = tl.load(input_ptr + n_offsets).to(tl.float32)
            m_new = tl.maximum(m, inp)
            all_neg_inf = m_new == float("-inf")
            z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
            m = m_new
        # specialize the last (partial) iteration
        for start_n in range(previous_multiple, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            inp = tl.load(input_ptr + n_offsets, mask=mask, other=-float("inf")).to(
                tl.float32
            )
            m_new = tl.maximum(m, inp)
            all_neg_inf = m_new == float("-inf")
            z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
            m = m_new

        m_reduced = tl.max(m, 0)
        z = tl.sum(z * tl.exp(m - m_reduced), 0)
        m = m_reduced
        log_z = tl.log(z)

        previous_multiple = prev_multiple_of(N, TILE_N)
        # specialize the first store iteration
        for start_n in range(0, TILE_N, TILE_N):
            n_offsets = (previous_multiple - start_n) + tl.arange(0, TILE_N)
            mask = n_offsets < N
            inp = tl.load(
                input_ptr + n_offsets,
                mask=mask,
                other=-float("inf"),
                eviction_policy="evict_first",
            ).to(tl.float32)
            o = inp - m - log_z
            tl.store(output_ptr + n_offsets * OUTPUT_K, o, mask=mask)
        for start_n in range(TILE_N, N, TILE_N):
            n_offsets = (previous_multiple - start_n) + tl.arange(0, TILE_N)
            inp = tl.load(input_ptr + n_offsets, eviction_policy="evict_first").to(
                tl.float32
            )
            o = inp - m - log_z
            tl.store(output_ptr + n_offsets * OUTPUT_K, o)
FWD_MULTIROW_MAX_N = 4096  # single-pass 2D tile family (<= 8K elems/prog)
FWD_CHUNK_BN = 8192  # big-N chunk width (tl.sum/tl.max lane-safety bound)
FWD_TAIL_PIECE = 4096  # masked 1D tail pieces kept <= 4096 lanes (exact)
# TILE_M buckets per N (probed XPU 5). Non-power-of-2 N < 64 needs TILE_M>=64
# to compile correctly; handled in the dispatch.
FWD_N_TILE_M = [(16, 64), (64, 32), (256, 16), (1024, 16), (4096, 8)]


@triton.jit
def _k_fwd_key_u32(bits):
    return bits ^ (0x80000000 | (bits >> 31))


@triton.jit
def _k_fwd_decode_key(m_key):
    return (m_key ^ (0x80000000 | ((m_key >> 31) ^ 1))).to(
        tl.float32, bitcast=True
    )


@libentry()
@triton.jit
def log_softmax_kernel_singlepass(
    output_ptr,
    input_ptr,
    M,
    N: tl.constexpr,
    TILE_M: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid_m = ext.program_id(0)
    m_offsets = pid_m * TILE_M + tl.arange(0, TILE_M)
    n_offsets = tl.arange(0, N)
    offsets = m_offsets[:, None] * N + n_offsets[None, :]
    if NEED_MASK:
        mask = m_offsets[:, None] < M
        inp = tl.load(input_ptr + offsets, mask=mask, other=-float("inf")).to(
            tl.float32
        )
    else:
        inp = tl.load(input_ptr + offsets).to(tl.float32)
    bits = inp.to(tl.uint32, bitcast=True)
    m_key = tl.max(_k_fwd_key_u32(bits), 1)
    m = _k_fwd_decode_key(m_key)
    e = tl.exp(inp - m[:, None])
    z = tl.sum(e, 1)
    out = inp - m[:, None] - tl.log(z)[:, None]
    if NEED_MASK:
        tl.store(output_ptr + offsets, out, mask=mask)
    else:
        tl.store(output_ptr + offsets, out)


@libentry()
@triton.jit
def log_softmax_kernel_singlepass_tail(
    output_ptr,
    input_ptr,
    M,
    ROW_START,
    N: tl.constexpr,
    TILE_M: tl.constexpr,
):
    """Masked tail rows of the singlepass tile (launched on full-tile grid
    remainder). Rows are masked but the tile loads the full width unmasked."""
    pid_m = ext.program_id(0)
    m_offsets = ROW_START + pid_m * TILE_M + tl.arange(0, TILE_M)
    n_offsets = tl.arange(0, N)
    offsets = m_offsets[:, None] * N + n_offsets[None, :]
    mask = m_offsets[:, None] < M
    inp = tl.load(input_ptr + offsets, mask=mask, other=-float("inf")).to(
        tl.float32
    )
    bits = inp.to(tl.uint32, bitcast=True)
    m_key = tl.max(_k_fwd_key_u32(bits), 1)
    m = _k_fwd_decode_key(m_key)
    e = tl.exp(inp - m[:, None])
    z = tl.sum(e, 1)
    out = inp - m[:, None] - tl.log(z)[:, None]
    tl.store(output_ptr + offsets, out, mask=mask)


@libentry()
@triton.jit
def log_softmax_kernel_chunk(
    partial_m_ptr,
    partial_z_ptr,
    input_ptr,
    C_FULL,
    C,
    BLOCK_N: tl.constexpr,
):
    """Flat (row*C_FULL + c) grid; offsets = pid*BN (BN constexpr -> the
    [M*C_FULL, BN] read is contiguous, block DMA on XPU). Partial (m_c, z_c)
    stored at row*C + c."""
    pid = ext.program_id(0)
    row = pid // C_FULL
    c = pid % C_FULL
    n_offsets = tl.arange(0, BLOCK_N)
    off = pid * BLOCK_N + n_offsets
    x = tl.load(input_ptr + off).to(tl.float32)
    m_key = tl.max(_k_fwd_key_u32(x.to(tl.uint32, bitcast=True)), 0)
    m = _k_fwd_decode_key(m_key)
    z = tl.sum(tl.exp(x - m), 0)
    tl.store(partial_m_ptr + row * C + c, m)
    tl.store(partial_z_ptr + row * C + c, z)


@libentry()
@triton.jit
def log_softmax_chunk_combine(
    m_ptr,
    log_z_ptr,
    partial_m_ptr,
    partial_z_ptr,
    C,
    C2: tl.constexpr,
):
    """Combine row partials -> (row max, log-sum-exp); row stride = C."""
    pid = ext.program_id(0)
    c_offsets = tl.arange(0, C2)
    cmask = c_offsets < C
    po = pid * C + c_offsets
    mc = tl.load(partial_m_ptr + po, mask=cmask, other=-float("inf"))
    zc = tl.load(partial_z_ptr + po, mask=cmask, other=0.0)
    m = tl.max(mc, 0)
    z = tl.sum(zc * tl.exp(mc - m), 0)
    log_z = tl.log(z)
    tl.store(m_ptr + pid, m)
    tl.store(log_z_ptr + pid, log_z)


@libentry()
@triton.jit
def log_softmax_chunk_pass(
    output_ptr,
    input_ptr,
    m_ptr,
    log_z_ptr,
    C_FULL,
    C,
    BLOCK_N: tl.constexpr,
):
    """Flat grid (M*C_FULL): second read (offset = pid*BN) -> out."""
    pid = ext.program_id(0)
    row = pid // C_FULL
    m = tl.load(m_ptr + row)
    log_z = tl.load(log_z_ptr + row)
    n_offsets = tl.arange(0, BLOCK_N)
    off = pid * BLOCK_N + n_offsets
    x = tl.load(input_ptr + off).to(tl.float32)
    tl.store(output_ptr + off, x - m - log_z)


@libentry()
@triton.jit
def log_softmax_chunk_strided(
    partial_m_ptr,
    partial_z_ptr,
    input_ptr,
    N,
    C_FULL,
    C,
    BLOCK_N: tl.constexpr,
):
    """Flat (row*C_FULL + c) grid with per-row base offsets (needed when
    N % BN != 0: the flat pid*BN form drifts by the row tail)."""
    pid = ext.program_id(0)
    row = pid // C_FULL
    c = pid % C_FULL
    n_offsets = tl.arange(0, BLOCK_N)
    off = row * N + c * BLOCK_N + n_offsets
    x = tl.load(input_ptr + off).to(tl.float32)
    m_key = _k_fwd_key_u32(x.to(tl.uint32, bitcast=True))
    m = _k_fwd_decode_key(tl.max(m_key, 0))
    z = tl.sum(tl.exp(x - m), 0)
    tl.store(partial_m_ptr + row * C + c, m)
    tl.store(partial_z_ptr + row * C + c, z)


@libentry()
@triton.jit
def log_softmax_chunk_pass_strided(
    output_ptr,
    input_ptr,
    m_ptr,
    log_z_ptr,
    N,
    C_FULL,
    C,
    BLOCK_N: tl.constexpr,
):
    pid = ext.program_id(0)
    row = pid // C_FULL
    m = tl.load(m_ptr + row)
    log_z = tl.load(log_z_ptr + row)
    n_offsets = tl.arange(0, BLOCK_N)
    c = pid % C_FULL
    off = row * N + c * BLOCK_N + n_offsets
    x = tl.load(input_ptr + off).to(tl.float32)
    tl.store(output_ptr + off, x - m - log_z)


@libentry()
@triton.jit
def log_softmax_tail_partial(
    partial_m_ptr,
    partial_z_ptr,
    input_ptr,
    N,
    C_STRIDE,
    TAIL_BASE,
    TL: tl.constexpr,
):
    """1D masked tail piece (width <= 4096 keeps masked reduces exact)."""
    pid = ext.program_id(0)
    n_offsets = TAIL_BASE + tl.arange(0, TL)
    off = pid * N + n_offsets
    mask = n_offsets < N
    x = tl.load(input_ptr + off, mask=mask, other=-float("inf")).to(tl.float32)
    m_key = _k_fwd_key_u32(x.to(tl.uint32, bitcast=True))
    m = _k_fwd_decode_key(tl.max(m_key, 0))
    safe_m = tl.where(m == float("-inf"), 0.0, m)
    z = tl.sum(tl.exp(x - safe_m), 0)
    po = pid * C_STRIDE
    tl.store(partial_m_ptr + po, m)
    tl.store(partial_z_ptr + po, z)


@libentry()
@triton.jit
def log_softmax_tail_pass(
    output_ptr,
    input_ptr,
    m_ptr,
    log_z_ptr,
    N,
    TAIL_BASE,
    TL: tl.constexpr,
):
    """Masked tail write (piece width <= 4096): out = x - row - logsumexp."""
    pid = ext.program_id(0)
    m = tl.load(m_ptr + pid)
    log_z = tl.load(log_z_ptr + pid)
    n_offsets = TAIL_BASE + tl.arange(0, TL)
    off = pid * N + n_offsets
    mask = n_offsets < N
    x = tl.load(input_ptr + off, mask=mask, other=0.0).to(tl.float32)
    tl.store(output_ptr + off, x - m - log_z, mask=mask)


def _fwd_singlepass(out, inp, M, N):
    tile_m = 4
    for n_hi, tm in FWD_N_TILE_M:
        if N <= n_hi:
            tile_m = tm
            break
    if (N & (N - 1)) and N < 64:
        tile_m = 64  # tiny odd-N tiles miscompile below 64 rows
    nfull, tail = divmod(M, tile_m)
    log_softmax_kernel_singlepass[(nfull, 1, 1)](
        out,
        inp,
        M,
        N,
        TILE_M=tile_m,
        NEED_MASK=False,
        buffer_size_limit=2048,
        num_warps=8,
    )
    if tail:
        grid = (1, 1, 1)
        log_softmax_kernel_singlepass_tail[grid](
            out,
            inp,
            M,
            nfull * tile_m,
            N,
            TILE_M=tile_m,
            buffer_size_limit=2048,
            num_warps=8,
        )


def _fwd_chunk_split(out, inp, M, N):
    c_full = N // FWD_CHUNK_BN
    taillen = N - c_full * FWD_CHUNK_BN
    have_tail = taillen != 0
    n = 0
    t0 = t1 = 0
    tl0 = tl1 = 0
    if have_tail:
        t0 = min(taillen, FWD_TAIL_PIECE)
        t1 = taillen - t0
        tl0 = triton.next_power_of_2(t0)
        tl1 = triton.next_power_of_2(t1) if t1 else 0
        n = 2 if t1 else 1
    C = c_full + n
    C2 = triton.next_power_of_2(C)
    pm = torch.empty((M * C,), dtype=torch.float32, device=inp.device)
    pz = torch.empty((M * C,), dtype=torch.float32, device=inp.device)
    m_out = torch.empty((M,), dtype=torch.float32, device=inp.device)
    lz = torch.empty((M,), dtype=torch.float32, device=inp.device)
    if have_tail:
        log_softmax_tail_partial[(M, 1, 1)](
            pm[c_full::C],
            pz[c_full::C],
            inp,
            N,
            C,
            c_full * FWD_CHUNK_BN,
            tl0,
            num_warps=8,
        )
        if t1:
            log_softmax_tail_partial[(M, 1, 1)](
                pm[c_full + 1::C],
                pz[c_full + 1::C],
                inp,
                N,
                C,
                c_full * FWD_CHUNK_BN + t0,
                tl1,
                num_warps=8,
            )
    if c_full:
        if have_tail:
            log_softmax_chunk_strided[(M * c_full, 1, 1)](
                pm, pz, inp, N, c_full, C, BLOCK_N=FWD_CHUNK_BN,
                buffer_size_limit=2048, num_warps=8,
            )
        else:
            log_softmax_kernel_chunk[(M * c_full, 1, 1)](
                pm, pz, inp, c_full, C, BLOCK_N=FWD_CHUNK_BN,
                buffer_size_limit=2048, num_warps=8,
            )
    log_softmax_chunk_combine[(M, 1, 1)](
        m_out, lz, pm, pz, C, C2=C2,
        buffer_size_limit=2048, num_warps=8,
    )
    if c_full:
        if have_tail:
            log_softmax_chunk_pass_strided[(M * c_full, 1, 1)](
                out, inp, m_out, lz, N, c_full, C, BLOCK_N=FWD_CHUNK_BN,
                buffer_size_limit=2048, num_warps=8,
            )
        else:
            log_softmax_chunk_pass[(M * c_full, 1, 1)](
                out, inp, m_out, lz, c_full, C, BLOCK_N=FWD_CHUNK_BN,
                buffer_size_limit=2048, num_warps=8,
            )
    if have_tail:
        log_softmax_tail_pass[(M, 1, 1)](
            out, inp, m_out, lz, N, c_full * FWD_CHUNK_BN, tl0,
            buffer_size_limit=2048, num_warps=8,
        )
        if t1:
            log_softmax_tail_pass[(M, 1, 1)](
                out, inp, m_out, lz, N, c_full * FWD_CHUNK_BN + t0, tl1,
                buffer_size_limit=2048, num_warps=8,
            )


# ------------------------  backward -------------------------------
# log_softmax backward:  scale = sum(out_grad over N); in_grad = out_grad - exp(out)*scale
#
# XPU dispatch (measured on P800): the old code sent all N<=8192 to the 2D
# [TILE_M, N] multirow tile. That tile does an axis=1 reduce that is pathological
# on XPU for medium N: as N grows TILE_M shrinks (=8192//N), the 2D reduce stops
# amortizing and gems latency explodes (N=4096 -> 14.7ms / sp 0.016, N=256 ->
# 3.0ms) while the loads/stores stay on the slow masked-memory path even when
# the mask is always true. Unmasking the always-true row mask (M % TILE_M == 0)
# flips this completely: the same [TILE_M, N] block-DMA tile becomes faster than
# the per-row 1D-reduce kernels for every N <= 4096 (e.g. N=4096 0.75->0.47ms,
# N=1024 0.16->0.03ms, N=256 3.0->0.10ms). Per-row 1D-reduce kernels are kept for
# huge rows (N > 4096, two-pass multi-tile; the unmasked full tiles run 16384
# wide, tails stay 8192 wide) and N==1 is a flat pointwise op (scale == out_grad
# element itself).
BWD_MULTIROW_MAX_N = 4096
BWD_SINGLE_TILE_MAX_N = 4096
BWD_MT_TILE_N = 8192
# per-row two-pass tile width; measurable 17-20% faster than 8192 when the row
# length is an exact multiple of it (so there is no masked tail block).
BWD_MT_TILE_N_WIDE = 16384


# single-pass per-row: N fits one TILE_N tile, out_grad cached in registers.
@libentry()
@triton.jit
def log_softmax_backward_kernel_perrow(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
    NEED_MASK: tl.constexpr = True,
):
    pid_m = ext.program_id(0)
    if pid_m < M:
        out_ptr += pid_m * N
        out_grad_ptr += pid_m * N
        in_grad_ptr += pid_m * N
        n_offsets = tl.arange(0, TILE_N)
        if NEED_MASK:
            mask = n_offsets < N
            og = tl.load(out_grad_ptr + n_offsets, mask=mask, other=0.0).to(tl.float32)
            scale = tl.sum(og, 0)
            o = tl.load(out_ptr + n_offsets, mask=mask).to(tl.float32)
            ig = og - tl.exp(o) * scale
            tl.store(in_grad_ptr + n_offsets, ig, mask=mask)
        else:
            og = tl.load(out_grad_ptr + n_offsets).to(tl.float32)
            scale = tl.sum(og, 0)
            o = tl.load(out_ptr + n_offsets).to(tl.float32)
            ig = og - tl.exp(o) * scale
            tl.store(in_grad_ptr + n_offsets, ig)


# two-pass per-row multi-tile: N>TILE_N, out_grad reloaded so the wide tile only
# ever holds one tensor at a time (avoids the reg spill that makes a single wide
# single-pass tile slow, e.g. N=8192 fp32 24.5ms single-pass vs 3.2ms two-pass).
@libentry()
@triton.jit
def log_softmax_backward_kernel_perrow_mt(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
):
    pid_m = ext.program_id(0)
    if pid_m < M:
        out_ptr += pid_m * N
        out_grad_ptr += pid_m * N
        in_grad_ptr += pid_m * N

        scale_acc = tl.zeros([TILE_N], dtype=tl.float32)
        previous_multiple = prev_multiple_of(N, TILE_N)
        for start_n in range(0, previous_multiple, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            og = tl.load(out_grad_ptr + n_offsets).to(tl.float32)
            scale_acc += og
        for start_n in range(previous_multiple, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            og = tl.load(out_grad_ptr + n_offsets, mask=mask, other=0.0).to(tl.float32)
            scale_acc += og
        scale = tl.sum(scale_acc, 0)

        for start_n in range(0, previous_multiple, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            og = tl.load(out_grad_ptr + n_offsets).to(tl.float32)
            o = tl.load(out_ptr + n_offsets).to(tl.float32)
            ig = og - tl.exp(o) * scale
            tl.store(in_grad_ptr + n_offsets, ig)
        for start_n in range(previous_multiple, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            og = tl.load(out_grad_ptr + n_offsets, mask=mask, other=0.0).to(tl.float32)
            o = tl.load(out_ptr + n_offsets, mask=mask).to(tl.float32)
            ig = og - tl.exp(o) * scale
            tl.store(in_grad_ptr + n_offsets, ig, mask=mask)


# N==1: reduction collapses to the element itself, so the whole op is a flat
# pointwise transform in_grad = og - exp(o) * og over the M flattened rows.
# BLOCK stays 256: at BLOCK=1024 the masked path of this kernel corrupts every
# 16th lane on XPU (multi-program shapes), while 256 is exact in every test.
@libentry()
@triton.jit
def log_softmax_backward_kernel_flat1(
    x_ptr,
    out_grad_ptr,
    in_grad_ptr,
    n_elem,
    BLOCK: tl.constexpr = 256,
):
    pid = ext.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    og = tl.load(out_grad_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    o = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    ig = og - tl.exp(o) * og
    tl.store(in_grad_ptr + offs, ig, mask=mask)


@libentry()
@triton.jit
def log_softmax_backward_kernel_multirow(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N: tl.constexpr,
    TILE_M: tl.constexpr,
    NEED_MASK: tl.constexpr = True,
):
    pid_m = ext.program_id(0)
    m_offsets = pid_m * TILE_M + tl.arange(0, TILE_M)
    n_offsets = tl.arange(0, N)
    offsets = m_offsets[:, None] * N + n_offsets[None, :]
    if NEED_MASK:
        mask = m_offsets[:, None] < M
        og = tl.load(out_grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        o = tl.load(out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        scale = tl.sum(og, 1)
        ig = og - tl.exp(o) * scale[:, None]
        tl.store(in_grad_ptr + offsets, ig, mask=mask)
    else:
        og = tl.load(out_grad_ptr + offsets).to(tl.float32)
        o = tl.load(out_ptr + offsets).to(tl.float32)
        scale = tl.sum(og, 1)
        ig = og - tl.exp(o) * scale[:, None]
        tl.store(in_grad_ptr + offsets, ig)


# tail rows of the multirow tile when M is not a multiple of TILE_M; launched
# with a single program after the unmasked full-tile launch.
@libentry()
@triton.jit
def log_softmax_backward_kernel_multirow_tail(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    ROW_START,
    N: tl.constexpr,
    TILE_M: tl.constexpr,
):
    pid_m = ext.program_id(0)
    m_offsets = ROW_START + pid_m * TILE_M + tl.arange(0, TILE_M)
    n_offsets = tl.arange(0, N)
    offsets = m_offsets[:, None] * N + n_offsets[None, :]
    mask = m_offsets[:, None] < M
    og = tl.load(out_grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    o = tl.load(out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scale = tl.sum(og, 1)
    ig = og - tl.exp(o) * scale[:, None]
    tl.store(in_grad_ptr + offsets, ig, mask=mask)


# large-N staged split reduction: the per-row two-pass kernel serializes the
# whole row in one program (grid=(M,)) and re-reads out_grad, which on XPU only
# reaches ~200-330 GB/s for 16384-wide rows. Replace it with the same 2D split
# pattern as any_row_stage1/2 (fully parallel over the N axis):
#   stage1: grid (M, CHUNKS) reduce each contiguous 8192-chunk of out_grad to a
#           fp32 partial[m, c];
#   stage2: grid (M,) reduce the per-row partials into scale[M];
#   stage3: grid (M, CHUNKS) flat in_grad = out_grad - exp(out) * scale[row].
# Every block reduce stays <= 8192 lanes (the XPU-safe tl.sum bound), so no
# wide 16384 register accumulator is ever materialized.
BWD_STAGED_TILE_N = 8192


@libentry()
@triton.jit
def log_softmax_backward_kernel_stage1(
    out_grad_ptr,
    partial_ptr,
    N,
    N_CHUNKS,
    BLOCK_N: tl.constexpr,
):
    pid_m = ext.program_id(0)
    pid_c = ext.program_id(1)
    offset = pid_c * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offset < N
    og = tl.load(out_grad_ptr + pid_m * N + offset, mask=mask, other=0.0).to(tl.float32)
    tl.store(partial_ptr + pid_m * N_CHUNKS + pid_c, tl.sum(og, 0))


@libentry()
@triton.jit
def log_softmax_backward_kernel_stage2(
    partial_ptr,
    scale_ptr,
    N_CHUNKS,
    BLOCK_MID: tl.constexpr,
):
    pid_m = ext.program_id(0)
    offset = tl.arange(0, BLOCK_MID)
    p = tl.load(
        partial_ptr + pid_m * N_CHUNKS + offset,
        mask=offset < N_CHUNKS,
        other=0.0,
    )
    tl.store(scale_ptr + pid_m, tl.sum(p, 0))


@libentry()
@triton.jit
def log_softmax_backward_kernel_stage3(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    scale_ptr,
    N,
    BLOCK_N: tl.constexpr,
):
    pid_m = ext.program_id(0)
    pid_c = ext.program_id(1)
    offset = pid_c * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offset < N
    scale = tl.load(scale_ptr + pid_m).to(tl.float32)
    og = tl.load(out_grad_ptr + pid_m * N + offset, mask=mask, other=0.0).to(tl.float32)
    o = tl.load(out_ptr + pid_m * N + offset, mask=mask, other=0.0).to(tl.float32)
    ig = og - tl.exp(o) * scale
    tl.store(in_grad_ptr + pid_m * N + offset, ig, mask=mask)


def _backward_launch_staged(output, grad_output, in_grad, M, N):
    n_chunks = triton.cdiv(N, BWD_STAGED_TILE_N)
    scale = torch.empty((M,), dtype=torch.float32, device=grad_output.device)
    if n_chunks == 1:
        # single 8192 chunk: stage1 writes the per-row scale directly
        log_softmax_backward_kernel_stage1[(M, 1)](
            grad_output,
            scale,
            N,
            1,
            BLOCK_N=BWD_STAGED_TILE_N,
            buffer_size_limit=2048,
            num_warps=8,
        )
    else:
        partial = torch.empty(
            (M, n_chunks), dtype=torch.float32, device=grad_output.device
        )
        log_softmax_backward_kernel_stage1[(M, n_chunks)](
            grad_output,
            partial,
            N,
            n_chunks,
            BLOCK_N=BWD_STAGED_TILE_N,
            buffer_size_limit=2048,
            num_warps=8,
        )
        log_softmax_backward_kernel_stage2[(M,)](
            partial,
            scale,
            n_chunks,
            BLOCK_MID=triton.next_power_of_2(n_chunks),
            buffer_size_limit=2048,
            num_warps=8,
        )
    log_softmax_backward_kernel_stage3[(M, n_chunks)](
        output,
        grad_output,
        in_grad,
        scale,
        N,
        BLOCK_N=BWD_STAGED_TILE_N,
        buffer_size_limit=2048,
        num_warps=8,
    )


def _forward_launch(out, inp, M, N, K=1):
    if K == 1:
        if N <= FWD_MULTIROW_MAX_N:
            _fwd_singlepass(out, inp, M, N)
        else:
            _fwd_chunk_split(out, inp, M, N)
    else:
        grid = (M * K, 1, 1)
        log_softmax_kernel_inner[grid](
            out,
            inp,
            M,
            N,
            K,
            buffer_size_limit=2048,
            isCloseVectorization=True,
            is_use_mask_zero=True,
        )


def _backward_launch(output, grad_output, in_grad, M, N):
    if N == 1:
        # in_grad[i] = og[i] - exp(o[i]) * og[i]: pure flat pointwise.
        grid = (triton.cdiv(M, 256), 1, 1)
        log_softmax_backward_kernel_flat1[grid](
            output,
            grad_output,
            in_grad,
            M,
            buffer_size_limit=2048,
            num_warps=8,
        )
    elif N <= BWD_MULTIROW_MAX_N and (N & (N - 1)) == 0:
        # small/medium N: pack TILE_M rows per program into one [TILE_M, N]
        # contiguous block-DMA tile. Requires pow2 N (tl.arange bounds).
        # non-pow2 N falls through to the per-row masked single-pass kernel.
        # TILE_M buckets ~8K elems/program fixed (min(16, 8192//N)):
        # N=256 -> 16, N=1024 -> 8, N=2048 -> 4, N=4096 -> 2. Tiles over 8K
        # elems (e.g. [4,4096]) hit XPU register-pressure OOB / illegal memory
        # access on some shapes, so the bound is kept.
        if N == 4096:
            # exception: a 4-row tile for N==4096 measures ~24% faster than
            # the 2-row default and is verified exact (fp16/fp32/bf16, M =
            # 100/4096/4098/8192/20000; no OOB, no masked-tail corruption).
            tile_m = 4
        else:
            tile_m = min(16, _prev_pow2(max(1, 8192 // N)))
        nfull, tail = divmod(M, tile_m)
        log_softmax_backward_kernel_multirow[(nfull, 1, 1)](
            output,
            grad_output,
            in_grad,
            M,
            N,
            TILE_M=tile_m,
            NEED_MASK=False,
            buffer_size_limit=2048,
            num_warps=8,
        )
        if tail:
            log_softmax_backward_kernel_multirow_tail[(1, 1, 1)](
                output,
                grad_output,
                in_grad,
                M,
                nfull * tile_m,
                N,
                TILE_M=tile_m,
                buffer_size_limit=2048,
                num_warps=8,
            )
    elif N <= BWD_SINGLE_TILE_MAX_N:
        # non-pow2 N <= 4096 (or any N in the single-tile range that the
        # multirow tile cannot express): masked single-pass per-row tile.
        grid = (M, 1, 1)
        log_softmax_backward_kernel_perrow[grid](
            output,
            grad_output,
            in_grad,
            M,
            N,
            TILE_N=triton.next_power_of_2(N),
            NEED_MASK=True,
            buffer_size_limit=2048,
            num_warps=8,
        )
    else:
        # large N (N > 4096): per-row two-pass multi-tile instead of the
        # 3-kernel staged split reduction. Measured on XPU (10000, 65536)
        # fp32: staged took 45 ms while the per-row two-pass takes 15.1 ms
        # with a 16384-wide tile and 18.6 ms with 8192. A 16384-wide tile is
        # also faster for non-exact N (e.g. (10000,40999) 14.4 vs 17.1 ms),
        # but bf16 miscompiles at 16384-wide (measured maxdiff vs torch ~11.7
        # vs 1.6e-2 at 8192), so bf16 always stays on the 8192-wide tile.
        tile_n = (
            BWD_MT_TILE_N if grad_output.dtype == torch.bfloat16 else BWD_MT_TILE_N_WIDE
        )
        grid = (M, 1, 1)
        log_softmax_backward_kernel_perrow_mt[grid](
            output,
            grad_output,
            in_grad,
            M,
            N,
            TILE_N=tile_n,
            buffer_size_limit=2048,
            num_warps=8,
        )


def log_softmax(self, dim, half_to_float=False):
    logger.debug("GEMS_KUNLUNXIN LOG_SOFTMAX")

    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"
    dim = dim % self.ndim
    M = 1
    N = self.shape[dim]
    for i in range(dim):
        M *= self.shape[i]
    inp = self.contiguous()
    if half_to_float:
        dtype = torch.float32
    else:
        dtype = self.dtype
    out = torch.empty_like(inp, dtype=dtype)
    K = inp.numel() // M // N

    with torch_device_fn.device(inp.device):
        if K > 1:
            # reduction over an interior dim: transpose to make N contiguous,
            # merge (M, K) -> M' so the fast per-row inner kernel applies.
            inp_view = inp.view(M, N, K).transpose(1, 2).contiguous()
            inp_reshaped = inp_view.view(M * K, N)
            origin_dim = out.ndim
            if origin_dim == 3:
                m, n, k = out.shape
            elif origin_dim == 2:
                m, n = out.shape
            out_reshaped = torch.empty_like(inp_reshaped, dtype=dtype)

            _forward_launch(out_reshaped, inp_reshaped, M * K, N)
            if M == 1 and origin_dim == 2:
                out = out_reshaped.view(K, N).transpose(0, 1).contiguous()
            elif M == 1 and origin_dim == 3:
                out = out_reshaped.transpose(0, 1).view(m, n, k).contiguous()
            else:
                out = out_reshaped.view(m, k, n).transpose(1, 2).contiguous()
        else:
            _forward_launch(out, inp, M, N)
    return out


def log_softmax_backward(grad_output, output, dim, input_dtype):
    logger.debug("GEMS_KUNLUNXIN LOG_SOFTMAX_BACKWARD")

    assert dim >= -output.ndim and dim < output.ndim, "Invalid dim"
    dim = dim % output.ndim
    M = 1
    N = output.shape[dim]
    for i in range(dim):
        M *= output.shape[i]

    grad_output = grad_output.contiguous()
    output = output.contiguous()
    in_grad = torch.empty_like(output, dtype=input_dtype)
    K = output.numel() // M // N

    with torch_device_fn.device(in_grad.device):
        if K > 1:
            out_grad_view = grad_output.view(M, N, K).transpose(1, 2).contiguous()
            out_view = output.view(M, N, K).transpose(1, 2).contiguous()
            out_grad_reshaped = out_grad_view.view(M * K, N)
            out_reshaped = out_view.view(M * K, N)
            in_grad_reshaped = torch.empty_like(out_reshaped, dtype=input_dtype)

            _backward_launch(
                out_reshaped, out_grad_reshaped, in_grad_reshaped, M * K, N
            )
            origin_dim = output.ndim
            if origin_dim == 3:
                m, n, k = output.shape
            elif origin_dim == 2:
                m, n = output.shape
            if M == 1 and origin_dim == 2:
                in_grad = in_grad_reshaped.view(K, N).transpose(0, 1).contiguous()
            elif M == 1 and origin_dim == 3:
                in_grad = in_grad_reshaped.transpose(0, 1).view(m, n, k).contiguous()
            else:
                in_grad = in_grad_reshaped.view(m, k, n).transpose(1, 2).contiguous()
        else:
            _backward_launch(output, grad_output, in_grad, M, N)
    return in_grad


def log_softmax_out(self, dim, half_to_float=False, *, out):
    logger.debug("GEMS_KUNLUNXIN LOG_SOFTMAX_OUT")
    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"
    dim = dim % self.ndim
    dtype = torch.float32 if half_to_float else self.dtype
    if out.dtype != dtype:
        raise RuntimeError(
            f"_log_softmax.out: expected out dtype {dtype}, got {out.dtype}"
        )
    if tuple(out.shape) != tuple(self.shape):
        out.resize_(self.shape)

    M = 1
    for i in range(dim):
        M *= self.shape[i]
    N = self.shape[dim]
    inp = self.contiguous()
    K = inp.numel() // M // N
    if K > 1:
        inp = inp.view(M, N, K).transpose(1, 2).contiguous().view(M * K, N)
    with torch_device_fn.device(inp.device):
        _forward_launch(out, inp, M, N, K)
    return out


def log_softmax_backward_out(grad_output, output, dim, input_dtype, *, out):
    logger.debug("GEMS_KUNLUNXIN LOG_SOFTMAX_BACKWARD_OUT")
    res = log_softmax_backward(grad_output, output, dim, input_dtype)
    if tuple(out.shape) != tuple(res.shape):
        out.resize_(res.shape)
    out.copy_(res)
    return out
