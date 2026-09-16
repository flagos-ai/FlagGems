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
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# Inner-dim (K==1) reduction tiers:
#  - N <= _MULTIROW_MAX_N:   one multirow tile kernel (N constexpr, block DMA,
#    order-preserving uint32-key max). The uint32 key turns the XPU fp32
#    wide-row `tl.max` serial chain (~25x slower than `tl.sum`) into a fast
#    integer reduction (~4x).
#  - N >  _MULTIROW_MAX_N:   two-kernel chunk-split (single data read, single
#    exp per element): partials (m_c, z_c) per [TILE_R, BN] chunk tile, then a
#    tiny per-row combine over C partials.
_MULTIROW_MAX_N = 4096
_CHUNK_BN = 4096

# Middle-dim (K>1) reduction tiers. The reduction cuts the strided middle axis
# of [M, N, K] (consecutive j are K elements apart), so 2D tiles are unusable:
# the XPU backend rejects `tl.<reduce>(tile, axis=0)` outright, and a
# [TILE_K, JCHUNK] axis=1 tile mis-computes. Only 1D kernels are reliable.
#  - N <= _MID_ONLINE_MAX_N:  serial per-j online kernel, TILE_K-wide
#    contiguous loads (block DMA), one program per (k-tile, m).
#  - N >  _MID_ONLINE_MAX_N:  two-kernel chunk-split over j: per-(m, k, chunk)
#    [JCHUNK] strided gathers, then a tiny per-(m, k) combine over chunks.
_MID_ONLINE_MAX_N = 1024
_MID_TILE_K = 1024
_MID_JCHUNK = 4096


@libentry()
@triton.jit
def logsumexp_kernel_multirow(
    output_ptr,
    input_ptr,
    M,
    N: tl.constexpr,
    TILE_M: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """Reduce the innermost dim N for many rows per program.

    Order-preserving uint32 key trick: float32 bits -> key = bits | 0x80000000
    for non-negative, key = ~bits (bits ^ 0xFFFFFFFF) for negative, is strictly
    increasing (radix sort family: -inf < ... < -0 < +0 < ... < +inf), so
    `tl.max(key, axis=1)` finds the per-row max on the fast integer reduction
    path. Decode with bits = key ^ 0x80000000 (key >= 0x80000000, non-negative)
    or bits = ~key = key ^ 0xFFFFFFFF (key < 0x80000000, negative).  The
    previous `bits ^ (0x80000000 | (bits >> 31))` mapped negatives to a
    *decreasing* key order (a logical >> 31 yields 1, so -inf got the largest
    key), mis-computing every all-negative row.

    N is a constexpr so ``tl.arange(0, N)`` spans exactly [0, N) and the
    ``[TILE_M, N]`` tile is one stride-1 contiguous block -> block DMA on XPU
    (a runtime N would fall back to discrete gathers). Row masking is only
    compiled in when NEED_MASK, i.e. M % TILE_M != 0.
    """
    pid = ext.program_id(0)
    m_offsets = pid * TILE_M + tl.arange(0, TILE_M)
    n_offsets = tl.arange(0, N)
    m_mask = m_offsets < M
    offsets = m_offsets[:, None] * N + n_offsets[None, :]
    if NEED_MASK:
        inp = tl.load(
            input_ptr + offsets, mask=m_mask[:, None], other=-float("inf")
        ).to(tl.float32)
    else:
        inp = tl.load(input_ptr + offsets).to(tl.float32)
    bits = inp.to(tl.uint32, bitcast=True)
    key = tl.where(bits < 0x80000000, bits | 0x80000000, bits ^ 0xFFFFFFFF)
    m_key = tl.max(key, axis=1)
    bits_m = tl.where(m_key < 0x80000000, m_key ^ 0xFFFFFFFF, m_key ^ 0x80000000)
    m = bits_m.to(tl.float32, bitcast=True)
    safe_m = tl.where(m == float("-inf"), 0.0, m)
    z = tl.sum(tl.exp(inp - safe_m[:, None]), axis=1)
    # keep native semantics for special values: NaN -> NaN, +inf -> +inf,
    # all-(-inf) rows -> -inf.
    res = tl.where(
        m == float("-inf"), m, tl.where(m == float("inf"), m, safe_m + tl.log(z))
    )
    tl.store(output_ptr + m_offsets, res, mask=m_mask)


@libentry()
@triton.jit
def logsumexp_kernel_partial(
    mrow_ptr,
    zrow_ptr,
    input_ptr,
    R,
    BN: tl.constexpr,
    TILE_R: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """Per-chunk partial (max, sum-exp) for a big innermost dim.

    Input is the flattened [rows * C, BN] view of the full 4096-chunks (each
    chunk stride-1 contiguous; BN constexpr keeps block DMA). No column
    masking -- the caller routes any tail (N % BN != 0) through the per-row
    kernel instead (masked-column reductions miscompute on this backend).
    Partial (m_c, z_c) pairs are stored compactly per chunk row; the host pads
    each row to TILE_C with -inf/0 so the combine kernel reads mask-free.
    """
    pid = ext.program_id(0)
    r_offsets = pid * TILE_R + tl.arange(0, TILE_R)
    r_mask = r_offsets < R
    n_offsets = tl.arange(0, BN)
    offsets = r_offsets[:, None] * BN + n_offsets[None, :]
    if NEED_MASK:
        a = tl.load(input_ptr + offsets, mask=r_mask[:, None], other=-float("inf")).to(
            tl.float32
        )
    else:
        a = tl.load(input_ptr + offsets).to(tl.float32)
    bits = a.to(tl.uint32, bitcast=True)
    key = tl.where(bits < 0x80000000, bits | 0x80000000, bits ^ 0xFFFFFFFF)
    m_key = tl.max(key, axis=1)
    bits_m = tl.where(m_key < 0x80000000, m_key ^ 0xFFFFFFFF, m_key ^ 0x80000000)
    m = bits_m.to(tl.float32, bitcast=True)
    safe_m = tl.where(m == float("-inf"), 0.0, m)
    z = tl.sum(tl.exp(a - safe_m[:, None]), axis=1)
    tl.store(mrow_ptr + r_offsets, m, mask=r_mask)
    tl.store(zrow_ptr + r_offsets, z, mask=r_mask)


@libentry()
@triton.jit
def logsumexp_kernel_combine(
    output_ptr,
    mrow_ptr,
    zrow_ptr,
    mtail_ptr,
    ztail_ptr,
    M,
    C_FULL: tl.constexpr,
    HAS_TAIL: tl.constexpr,
    TILE_C: tl.constexpr,
):
    """Combine the C_FULL per-chunk partials of one row plus (optionally) the
    tail partial at slot C_FULL: out = m + log(sum zc exp(mc - m))."""
    row = ext.program_id(0)
    c_offsets = tl.arange(0, TILE_C)
    mc = tl.load(mrow_ptr + row * TILE_C + c_offsets)
    zc = tl.load(zrow_ptr + row * TILE_C + c_offsets)
    if HAS_TAIL:
        m_t = tl.load(mtail_ptr + row)
        z_t = tl.load(ztail_ptr + row)
        is_tail = c_offsets == C_FULL
        mc = tl.where(is_tail, m_t, mc)
        zc = tl.where(is_tail, z_t, zc)
    m = tl.max(mc, axis=0)
    safe_m = tl.where(m == float("-inf"), 0.0, m)
    # exp(mc - safe_m) is 0 for the -inf pad chunks; zc is 0 there too.
    z = tl.sum(zc * tl.exp(mc - safe_m), axis=0)
    res = tl.where(
        m == float("-inf"), m, tl.where(m == float("inf"), m, safe_m + tl.log(z))
    )
    tl.store(output_ptr + row, res)


@libentry()
@triton.jit
def logsumexp_kernel_tail_partials(
    mrow_ptr,
    zrow_ptr,
    input_ptr,
    M,
    ROW_STRIDE,
    N,
    TILE_N: tl.constexpr,
):
    """Per-row (m, z) partials for a tail slice [M, N] strided by ROW_STRIDE.

    Tail widths are < _CHUNK_BN (<= 4096), so TILE_N is power-of-two and the
    loop body executes once; the single masked iteration is verified exact on
    this backend (unlike padded 2D-tile masked reductions). Emits compact
    per-row max m and max-shifted sum z for the combine kernel.
    """
    pid = ext.program_id(0)
    m = tl.full([TILE_N], value=float("-inf"), dtype=tl.float32)
    z = tl.full([TILE_N], value=0.0, dtype=tl.float32)
    # Same NaN guard as the online kernel: ``tl.maximum`` is maxnumf and a NaN
    # lane whose running max is still -inf keeps z = 0 (the all_neg_inf guard),
    # so the NaN is only visible through an explicit flag -- propagated to the
    # combine kernel as a NaN z partial (detected there as long as the row max
    # is finite). Gated to small TILE_N (see the online kernel).
    if TILE_N <= 64:
        nan_seen = tl.zeros([TILE_N], dtype=tl.int1)
    input_ptr += pid * ROW_STRIDE

    for start_n in range(0, N, TILE_N):
        n_offsets = start_n + tl.arange(0, TILE_N)
        mask = n_offsets < N
        a = tl.load(input_ptr + n_offsets, mask=mask, other=-float("inf")).to(
            tl.float32
        )
        if TILE_N <= 64:
            nan_seen = nan_seen | (a != a)
        m_new = tl.maximum(m, a)
        all_neg_inf = m_new == float("-inf")
        z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(a - m_new))
        m = m_new

    m_r = tl.max(m, axis=0)
    z_r = tl.sum(z * tl.exp(m - m_r), axis=0)
    # all-(-inf) tails must contribute z=0 to the combine (exp(-inf - -inf)
    # would be NaN), and all-(-inf) rows are resolved by the combine's -inf
    # guard; a NaN tail must contribute a NaN partial instead.
    if TILE_N <= 64:
        nan_r = tl.sum(nan_seen.to(tl.int32), axis=0) > 0
        tl.store(
            zrow_ptr + pid,
            tl.where(nan_r, float("nan"), tl.where(m_r == float("-inf"), 0.0, z_r)),
        )
    else:
        tl.store(zrow_ptr + pid, tl.where(m_r == float("-inf"), 0.0, z_r))
    tl.store(mrow_ptr + pid, m_r)


def _reduce_inner_small(inp, rows, N, out):
    """Single-tile multirow kernel for N <= _MULTIROW_MAX_N."""
    if N <= 64:
        TILE_M = 16
    elif N <= 256:
        TILE_M = 64
    elif N <= 1024:
        TILE_M = 32
    else:
        TILE_M = 8
    need_mask = 1 if rows % TILE_M else 0
    grid = (triton.cdiv(rows, TILE_M), 1, 1)
    logsumexp_kernel_multirow[grid](
        out,
        inp,
        rows,
        N=N,
        TILE_M=TILE_M,
        NEED_MASK=need_mask,
        num_warps=4,
        buffer_size_limit=2048,
    )


def _reduce_tail_partials(mrow, zrow, inp, rows, row_stride, tail_n):
    """Reduce a [rows, tail_n] tail-view (strided by row_stride) into compact
    (m, z) partials via the per-row online kernel."""
    TILE_N = max(1, triton.next_power_of_2(tail_n))
    grid = (rows, 1, 1)
    logsumexp_kernel_tail_partials[grid](
        mrow,
        zrow,
        inp,
        rows,
        row_stride,
        tail_n,
        TILE_N=TILE_N,
        num_warps=4,
        buffer_size_limit=2048,
    )


def _reduce_inner(inp, rows, N):
    """logsumexp over the innermost dim N of a contiguous [rows, N] tensor."""
    out = torch.empty((rows,), dtype=inp.dtype, device=inp.device)
    if N <= _MULTIROW_MAX_N and (N & (N - 1)) == 0:
        # The multirow kernel's [TILE_M, N] axis=1 tile only computes for
        # power-of-two N on this backend (N=5/6/7/9 miscompute the row max
        # and sum even with unmasked in-bounds loads), so non-pow2 N go
        # through the 1D chunk-split path below instead.
        _reduce_inner_small(inp, rows, N, out)
    else:
        # Chunk-split path: single data read, single exp per element. Full
        # 4096-chunks go through the tile kernel; any tail (N % 4096 != 0) is
        # reduced by the per-row 1D kernel (masked 2D-tile reductions
        # miscompute on this backend).
        BN = _CHUNK_BN
        C_full = N // BN
        TAIL = N - C_full * BN
        TILE_C = max(1, triton.next_power_of_2(C_full + (1 if TAIL else 0)))
        # partials compact per chunk; then per-row padded to TILE_C with
        # (-inf, 0) pad slots so the combine kernel reads mask-free.
        mrow = torch.empty((rows * C_full,), dtype=torch.float32, device=inp.device)
        zrow = torch.empty_like(mrow)
        if C_full:
            R = rows * C_full
            TILE_R = 32
            need_mask = 1 if R % TILE_R else 0
            full_view = torch.ops.aten.slice(inp, 1, 0, C_full * BN)
            # slice 的行步长 = N；partial 内核按 [R, BN] 连续布局寻址
            # （r*BN + n），直接传该视图会让 r >= 1 的行跨行错位（混入上一行
            # 的尾部列和处理段），必须物化连续视图；TAIL == 0 时 no-op。
            full_view = torch.ops.aten.contiguous(full_view)
            # reshape may copy only when the slice is non-contiguous (tail
            # cases with N % BN != 0); the aligned path is a null-op view.
            flat = torch.ops.aten.reshape(full_view, (R, BN))
            grid = (triton.cdiv(R, TILE_R), 1, 1)
            logsumexp_kernel_partial[grid](
                mrow,
                zrow,
                flat,
                R,
                BN=BN,
                TILE_R=TILE_R,
                NEED_MASK=need_mask,
                num_warps=4,
                buffer_size_limit=2048,
            )
        if C_full and TILE_C != C_full:
            mrow = mrow.view(rows, C_full)
            zrow = zrow.view(rows, C_full)
            mp = torch.full(
                (rows, TILE_C), -float("inf"), dtype=torch.float32, device=inp.device
            )
            zp = torch.zeros((rows, TILE_C), dtype=torch.float32, device=inp.device)
            mp[:, :C_full] = mrow
            zp[:, :C_full] = zrow
            mrow = mp
            zrow = zp
        elif not C_full:
            mrow = torch.full(
                (rows, TILE_C), -float("inf"), dtype=torch.float32, device=inp.device
            )
            zrow = torch.zeros((rows, TILE_C), dtype=torch.float32, device=inp.device)
        if TAIL:
            # tail slice view: [rows, TAIL] strided by N (no copy)
            tail_view = torch.ops.aten.slice(inp, 1, C_full * BN, N)
            mtail = torch.empty((rows,), dtype=torch.float32, device=inp.device)
            ztail = torch.empty_like(mtail)
            _reduce_tail_partials(mtail, ztail, tail_view, rows, N, TAIL)
        else:
            # unused sentinel pointer for the HAS_TAIL=0 build
            mtail = torch.empty((1,), dtype=torch.float32, device=inp.device)
            ztail = torch.empty_like(mtail)
        logsumexp_kernel_combine[(rows, 1, 1)](
            out,
            mrow,
            zrow,
            mtail,
            ztail,
            rows,
            C_FULL=C_full,
            HAS_TAIL=1 if TAIL else 0,
            TILE_C=TILE_C,
            num_warps=4,
            buffer_size_limit=2048,
        )
    return out


@libentry()
@triton.jit
def logsumexp_kernel_mid_online(
    input_ptr,
    output_ptr,
    N,
    K,
    TILE_K: tl.constexpr,
):
    """Serial per-j online reduction over the middle dim of a [M, N, K] input.

    One program per (k-tile, m); the [TILE_K] lane vector holds consecutive k
    of one j row (one stride-1 contiguous block -> block DMA on XPU). The
    ``tl.range(..., loop_unroll_factor=1)`` form is required: a plain
    ``range`` loop over TILE_K >= 128 lanes fails TritonXPUUnrollControl
    (out of resource: uni_sram). Masked contiguous loads are exact on this
    backend (unlike masked strided gathers), so K % TILE_K != 0 lanes use a
    k_mask with other=-inf.
    """
    pid_k = ext.program_id(0)
    pid_m = ext.program_id(1)
    k_offsets = pid_k * TILE_K + tl.arange(0, TILE_K)
    k_mask = k_offsets < K
    base = input_ptr + pid_m * N * K + k_offsets
    m = tl.full([TILE_K], value=-float("inf"), dtype=tl.float32)
    z = tl.zeros([TILE_K], dtype=tl.float32)
    # ``tl.maximum`` is maxnumf (NaN is ignored), and when a NaN lane's running
    # max is still -inf the ``all_neg`` guard keeps z=0, so the NaN would be
    # silently lost. Track it explicitly (torch logsumexp -> NaN if any input
    # element is NaN), but only for small TILE_K: the [TILE_K] i1 flag is
    # register-heavy and TILE_K >= 128 overflows uni_sram / LLVM-selects.
    if TILE_K <= 64:
        nan_seen = tl.zeros([TILE_K], dtype=tl.int1)
    for j in tl.range(0, N, loop_unroll_factor=1):
        a = tl.load(base + j * K, mask=k_mask, other=-float("inf")).to(tl.float32)
        if TILE_K <= 64:
            nan_seen = nan_seen | (a != a)
        m_new = tl.maximum(m, a)
        all_neg = m_new == -float("inf")
        z = tl.where(all_neg, z, z * tl.exp(m - m_new) + tl.exp(a - m_new))
        m = m_new
    res = tl.where(m == -float("inf"), m, tl.where(m == float("inf"), m, m + tl.log(z)))
    if TILE_K <= 64:
        res = tl.where(nan_seen, float("nan"), res)
    tl.store(output_ptr + pid_m * K + k_offsets, res, mask=k_mask)


@libentry()
@triton.jit
def logsumexp_kernel_mid_partial(
    input_ptr,
    mrow_ptr,
    zrow_ptr,
    nanrow_ptr,
    M,
    N,
    K,
    stride,
    JCHUNK: tl.constexpr,
):
    """Per-chunk (max, sum-exp, nan) partials for a big middle dim.

    One program per (chunk, m*K+k) gathers [JCHUNK] lanes of one (m, k) column
    (element (m, j0+l, k)). The strided load MUST be unmasked: masked strided
    1D gathers return garbage on this backend, so out-of-range lanes are
    clamped to N-1 (reading a duplicated in-bounds element) and then masked in
    registers with ``tl.where(j < N, ...)``. Safe-mask also keeps z_c exact for
    all-(-inf) chunks (exp(m_c - m_c) would be NaN otherwise). Partials are
    stored per chunk strided by ``stride`` = M*K for the combine kernel. A
    NaN within the chunk is recorded in nanrow (1/0): z_c alone cannot carry it
    (a chunk max of +inf also makes z_c NaN, via exp(inf - inf)), and the
    combine must distinguish the two.
    """
    pid_c = ext.program_id(0)
    pid = ext.program_id(1)  # m * K + k
    m = pid // K
    k = pid % K
    j = pid_c * JCHUNK + tl.arange(0, JCHUNK)
    jc = tl.minimum(j, N - 1)
    a = tl.load(input_ptr + m * N * K + jc * K + k).to(tl.float32)
    # mask out-of-range lanes in registers (after the clamped load, so OOB
    # lanes never read a real NaN element); a != a is then an exact NaN probe.
    a = tl.where(j < N, a, -float("inf"))
    m_c = tl.max(a, axis=0)
    safe_mc = tl.where(m_c == -float("inf"), 0.0, m_c)
    z_c = tl.sum(tl.exp(a - safe_mc), axis=0)
    tl.store(nanrow_ptr + pid_c * stride + pid, tl.sum((a != a).to(tl.int32), axis=0))
    tl.store(mrow_ptr + pid_c * stride + pid, m_c)
    tl.store(zrow_ptr + pid_c * stride + pid, z_c)


@libentry()
@triton.jit
def logsumexp_kernel_mid_combine(
    mrow_ptr,
    zrow_ptr,
    nanrow_ptr,
    output_ptr,
    nchunks,
    stride,
    TILE_C: tl.constexpr,
):
    """Combine the nchunks per-chunk partials of one (m, k): out = m + log(...).

    Strided loads use the same clamp + register-where pattern as the partial
    kernel (masked strided loads are unsafe here too); invalid slots read the
    in-bounds element nchunks-1 and are replaced with (-inf, 0, 0) in
    registers. A chunk with a NaN input records 1 in nanrow; the result is
    forced to NaN (the m == inf guard would otherwise win over the NaN).
    """
    pid = ext.program_id(0)
    c_offsets = tl.arange(0, TILE_C)
    c_mask = c_offsets < nchunks
    c = tl.minimum(c_offsets, nchunks - 1)
    mc = tl.load(mrow_ptr + c * stride + pid)
    zc = tl.load(zrow_ptr + c * stride + pid)
    nc = tl.load(nanrow_ptr + c * stride + pid)
    mc = tl.where(c_mask, mc, -float("inf"))
    zc = tl.where(c_mask, zc, 0.0)
    nc = tl.where(c_mask, nc, 0)
    m = tl.max(mc, axis=0)
    safe_m = tl.where(m == -float("inf"), 0.0, m)
    # exp(mc - safe_m) is 0 for the -inf partials; zc is 0 there too.
    z = tl.sum(zc * tl.exp(mc - safe_m), axis=0)
    res = tl.where(
        m == -float("inf"), m, tl.where(m == float("inf"), m, safe_m + tl.log(z))
    )
    res = tl.where(tl.sum(nc, axis=0) > 0, float("nan"), res)
    tl.store(output_ptr + pid, res)


def _reduce_middle_online(inp, M, N, K, out):
    """Middle-dim online reduction for N <= _MID_ONLINE_MAX_N; out is [M, K].

    fp16 is unsupported for both the [TILE_K] load and the [TILE_K] store
    inside the tl.range loop on this backend (TritonXPUUnrollControl fails at
    every TILE_K/num_warps; bf16 and fp32 are fine), so fp16 inputs/outputs go
    through fp32 temporaries.
    """
    if inp.dtype == torch.float16:
        inp32 = inp.float()
        out32 = torch.empty((M, K), dtype=torch.float32, device=inp.device)
        _launch_mid_online(inp32, M, N, K, out32)
        out.copy_(out32)
        return
    _launch_mid_online(inp, M, N, K, out)


def _launch_mid_online(inp, M, N, K, out):
    """Launch the online kernel; inp/out must not be fp16."""
    # The XPU unroll pass only accepts lane widths <= 64 or exactly 1024:
    # intermediate widths (128..512, incl. the plain no-NaN loop) overflow
    # uni_sram. Clamp to 1024 and let the k_mask drop the excess lanes.
    n2 = triton.next_power_of_2(K)
    TILE_K = n2 if n2 <= 64 else _MID_TILE_K
    grid = (triton.cdiv(K, TILE_K), M)
    logsumexp_kernel_mid_online[grid](
        inp,
        out,
        N,
        K,
        TILE_K=TILE_K,
        num_warps=4,
        buffer_size_limit=2048,
    )


def _reduce_middle_chunked(inp, M, N, K, out):
    """Two-kernel chunk-split middle-dim reduction for N > _MID_ONLINE_MAX_N.

    The partial kernel reads each input element exactly once; the tiny combine
    kernel merges the per-chunk (m_c, z_c) strided partials of each (m, k).
    """
    nchunks = triton.cdiv(N, _MID_JCHUNK)
    TILE_C = max(1, triton.next_power_of_2(nchunks))
    stride = M * K
    mrow = torch.empty((nchunks * M * K,), dtype=torch.float32, device=inp.device)
    zrow = torch.empty_like(mrow)
    nanrow = torch.empty((nchunks * M * K,), dtype=torch.int32, device=inp.device)
    logsumexp_kernel_mid_partial[(nchunks, M * K)](
        inp,
        mrow,
        zrow,
        nanrow,
        M,
        N,
        K,
        stride,
        JCHUNK=_MID_JCHUNK,
        num_warps=4,
        buffer_size_limit=2048,
    )
    logsumexp_kernel_mid_combine[(M * K,)](
        mrow,
        zrow,
        nanrow,
        out,
        nchunks,
        stride,
        TILE_C=TILE_C,
        num_warps=4,
        buffer_size_limit=2048,
    )


def _reduce_middle(inp, M, N, K):
    """logsumexp over the middle dim N of a contiguous [M, N, K] tensor (K > 1)."""
    out = torch.empty((M, K), dtype=inp.dtype, device=inp.device)
    if N <= _MID_ONLINE_MAX_N:
        _reduce_middle_online(inp, M, N, K, out)
    else:
        _reduce_middle_chunked(inp, M, N, K, out)
    return out


def logsumexp(inp, dim, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN LOGSUMEXP")

    if isinstance(dim, (list, tuple)):
        if len(dim) == 0:
            # Empty dim list means no reduction, just return the input.
            return inp.clone()
        if len(dim) != 1:
            # Multi-dim reduction: reduce sequentially over each dim, keeping
            # the reduced dims until the end (the per-dim kernels need the
            # trailing shape intact, so a keepdim chain is the exact path).
            dims = sorted(dim)
            out = inp
            for d in reversed(dims):
                out = logsumexp(out, d, True)
            if not keepdim:
                # Squeeze highest dim first so earlier squeezes never shift
                # the positions of the remaining reduced dims.
                for d in reversed(dims):
                    out = out.squeeze(dim=d)
            return out
        dim = dim[0]

    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    dim = dim % inp.ndim

    N = inp.shape[dim]
    K = 1
    for i in range(dim + 1, inp.ndim):
        K *= inp.shape[i]

    if N == 1:
        # Single-element reduction: logsumexp(x) == x -- identity view, no
        # kernel needed (this covers every K, so it must come first).
        return inp.squeeze(dim=dim) if not keepdim else inp

    # Flatten the leading dims to M and treat the input as [M, N, K].
    M = 1
    for i in range(dim):
        M *= inp.shape[i]
    inp = inp.contiguous()
    shape = list(inp.shape)
    shape[dim] = 1

    if K > 1:
        # Middle-dim reduction over the strided axis (consecutive j are K
        # apart, so 2D tiles are unusable): the 1D online / chunk-split
        # kernels above.
        with torch_device_fn.device(inp.device):
            out = _reduce_middle(inp, M, N, K).view(shape)
    else:
        # K == 1: innermost-dim reduction -> fast contiguous Triton kernels.
        # _reduce_inner indexes a [M, N] 2D view (contiguous above: no copy;
        # a plain slice of an N-D tensor would take the wrong axis).
        with torch_device_fn.device(inp.device):
            out = _reduce_inner(inp.view(M, N), M, N).view(shape)

    if not keepdim:
        out = out.squeeze(dim=dim)
    return out
