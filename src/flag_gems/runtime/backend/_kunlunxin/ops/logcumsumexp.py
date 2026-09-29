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

# K==1 (innermost-dim scan) tiers:
#  - N == 1: numeric identity (log(exp(x)) == x) -> device copy.
#  - N <= _ROW_MAX_N: one row per program, full row as a single 1D tile
#    (TILE_N = next_pow2(N); the scan is a 1D tl.cumsum, which is the only
#    scan path that lowers correctly on this XPU backend -- the 2D axis=1
#    cumsum silently mis-computes, so no multirow 2D tiles here). The tile is
#    masked whenever TILE_N != N *or* TILE_N < 64: `tl.arange` is widened to
#    round_up(TILE_N, 64) lanes, so a narrower tile would touch lanes past the
#    row on both the load and the store.
#  - N >  _ROW_MAX_N: per-row chunked online scan (BN=4096): uint32-key
#    chunk max, per-chunk cumsum with a numerically-stable running rescale.
#    A masked tail chunk (masked load with -inf + masked store) is proven ok.
# Both scan tiers derive the max-shift from a monotone uint32 key that ranks
# non-finite lanes below every finite one, so a row containing +inf/NaN still
# computes a correct finite prefix instead of overflowing to +inf/NaN. The
# chunk tier additionally pins the running sum at +inf once a +inf lane has
# been seen (the blanket max would otherwise turn the next chunk's rescale
# `inf * 0` into NaN and poison the rest of the row).
_ROW_MAX_N = 4096

# K > 1 (middle-dim): the original per-block kernel, driven sequentially from
# the host. The benchmark never hits this path (dim=-1 -> K==1); correctness
# is what matters here.
_SCAN_BLOCK = 1024


@libentry()
@triton.jit
def logcumsumexp_row_kernel(
    inp_ptr,
    out_ptr,
    N: tl.constexpr,
    TILE_N: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """One row per program; single 1D tile scan with order-preserving uint32
    key max (the fp32 wide-row `tl.max` serial chain is avoided)."""
    pid = ext.program_id(0)
    row_offset = pid * N
    n_offsets = tl.arange(0, TILE_N)
    if NEED_MASK:
        mask = n_offsets < N
        x = tl.load(
            inp_ptr + row_offset + n_offsets, mask=mask, other=-float("inf")
        ).to(tl.float32)
    else:
        x = tl.load(inp_ptr + row_offset + n_offsets).to(tl.float32)
    bits = x.to(tl.uint32, bitcast=True)
    # Monotone uint32 key (radix-sort family). `s` is the sign mask when the
    # bitcast is read as int32 and shifted *arithmetically* (0xFFFFFFFF for
    # negatives, 0 otherwise), so key = bits ^ (0x80000000 | s) reproduces the
    # standard order: -inf < negatives < -0 < +0 < positives < +inf < NaN.
    # The old `bits ^ (0x80000000 | (bits >> 31))` shifted a *uint32* (logical
    # >> 31 gives 1, i.e. mask 0x80000001) and so mapped negatives to a
    # *decreasing* key order: an all-negative row then searched from its
    # minimum and overflowed exp() to +inf for any row whose spread exceeded
    # ~88. Same defect and same fix as `_kunlunxin/ops/logsumexp.py`.
    s = (bits.to(tl.int32, bitcast=True) >> 31).to(tl.uint32, bitcast=True)
    key = bits ^ (0x80000000 | s)
    # A scan needs more than a correct max: the shift must stay *finite*
    # whenever the row holds a finite entry, otherwise a single +inf/NaN lane
    # becomes the exponent base and exp() turns the whole finite prefix into
    # +inf (or NaN). In the standard order the non-finite keys are exactly
    # {+inf, +NaN} >= 0xFF800000 at the top (and -inf/-NaN already sit at the
    # bottom, below every finite negative), so one threshold compare on the key
    # we just built blankets them: key 0x007FFFFF decodes back to -inf, so an
    # all-non-finite row still reaches the base-0 guard below and keeps
    # log(0)=-inf / exp(+inf)=+inf / NaN propagation.
    key = tl.where(key < 0xFF800000, key, 0x007FFFFF)
    m_key = tl.max(key, axis=0)
    m = tl.where(m_key < 0x80000000, m_key ^ 0xFFFFFFFF, m_key ^ 0x80000000).to(
        tl.float32, bitcast=True
    )
    # exp base: -inf or +inf rows fall back to base 0 so that e=exp(x-0)
    # keeps log(0)=-inf / exp(+inf)=+inf semantics (torch-compatible scan).
    ms = tl.where(m == -float("inf"), 0.0, tl.where(m == float("inf"), 0.0, m))
    e = tl.exp(x - ms)
    c = tl.cumsum(e, axis=0)
    res = ms + tl.log(c)
    if NEED_MASK:
        tl.store(out_ptr + row_offset + n_offsets, res, mask=mask)
    else:
        tl.store(out_ptr + row_offset + n_offsets, res)


@libentry()
@triton.jit
def logcumsumexp_chunk_kernel(
    inp_ptr,
    out_ptr,
    N,
    BN: tl.constexpr,
    NEED_TAIL: tl.constexpr,
):
    """Per-row chunked online scan for N > _ROW_MAX_N.

    Loop-carried (m_prev, s_prev) keep the scan numerically stable across
    chunks: s_prev is the max-shifted prefix sum of the row so far. The
    rescaled prefix ``carry = s_prev * exp(m_prev - ms)`` degenerates to
    ``s_prev`` whenever the chunk max does not move the running max, which
    also keeps the all-(-inf) rows free of NaN.  A non-finite prefix is pinned
    too (``s_prev == +inf`` -> ``carry = +inf``): otherwise an all-+inf chunk
    leaves ``m_prev = -inf`` with ``s_prev = +inf`` and the next finite chunk's
    ``inf * exp(-inf - ms) = inf * 0`` would turn the row into NaN instead of
    the ``+inf`` torch keeps.  1D masked load (-inf) and masked store are used
    for the tail chunk and are verified exact.
    """
    pid = ext.program_id(0)
    row_offset = pid * N
    m_prev = tl.full([BN], -float("inf"), tl.float32)
    s_prev = tl.zeros([BN], tl.float32)
    for start in range(0, N, BN):
        n_offsets = start + tl.arange(0, BN)
        if NEED_TAIL:
            mask = n_offsets < N
            x = tl.load(
                inp_ptr + row_offset + n_offsets, mask=mask, other=-float("inf")
            ).to(tl.float32)
        else:
            x = tl.load(inp_ptr + row_offset + n_offsets).to(tl.float32)
        bits = x.to(tl.uint32, bitcast=True)
        # Same monotone uint32 key + non-finite blanket as the row kernel (see
        # the long note there). This matters more here than anywhere: without
        # it a single +inf/NaN lane in one chunk sets the base for that whole
        # chunk and poisons the running (m_prev, s_prev) carry.
        s = (bits.to(tl.int32, bitcast=True) >> 31).to(tl.uint32, bitcast=True)
        key = bits ^ (0x80000000 | s)
        key = tl.where(key < 0xFF800000, key, 0x007FFFFF)
        m_key = tl.max(key, axis=0)
        m_c = tl.where(m_key < 0x80000000, m_key ^ 0xFFFFFFFF, m_key ^ 0x80000000).to(
            tl.float32, bitcast=True
        )
        m_new = tl.maximum(m_prev, m_c)
        # +inf/NaN never reach here: the key blanket above maps them to the
        # -inf key, so the running max is always a finite value or -inf.
        ms = tl.where(m_new == -float("inf"), 0.0, m_new)
        e = tl.exp(x - ms)
        cs = tl.cumsum(e, axis=0)
        # The rescale `s_prev * exp(m_prev - ms)` is only valid while s_prev is
        # finite.  As soon as a +inf lane entered the prefix, s_prev is +inf and
        # the product is inf * 0 = NaN for any chunk that moves the running max:
        # chunk0 = [+inf]*4096 leaves m_prev = -inf (its blanket max) with
        # s_prev = +inf, and the next finite chunk would then turn the whole
        # remainder of the row into NaN although torch's cumsum stays +inf from
        # the first +inf lane on.  +inf contributes +inf to the prefix sum
        # whatever the shift, so pin it.  NaN needs no special case -- NaN * 0
        # is already NaN and propagates on its own.
        carry = tl.where(
            (m_prev == m_new) | (s_prev == float("inf")),
            s_prev,
            s_prev * tl.exp(m_prev - ms),
        )
        res = ms + tl.log(cs + carry)
        if NEED_TAIL:
            tl.store(out_ptr + row_offset + n_offsets, res, mask=mask)
        else:
            tl.store(out_ptr + row_offset + n_offsets, res)
        s_prev = carry + tl.sum(e, axis=0)
        m_prev = m_new


@libentry()
@triton.jit
def logcumsumexp_block_kernel(
    inp,
    out,
    state_max,
    state_sum,
    block_start,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    FIRST_BLOCK: tl.constexpr,
):
    """K>1 (middle-dim scan) block kernel: one program per (m, k) row, host
    drives the sequential block loop. Unchanged legacy path."""
    pid = ext.program_id(0)
    m = pid // K
    k = pid % K
    n_offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = n_offsets < N
    base = m * N * K + k
    values = tl.load(inp + base + n_offsets * K, mask=mask, other=float("-inf")).to(
        tl.float32
    )
    block_max = tl.max(values, axis=0)

    if FIRST_BLOCK:
        new_max = block_max
        scaled_prefix = 0.0
    else:
        previous_max = tl.load(state_max + pid)
        previous_sum = tl.load(state_sum + pid)
        new_max = tl.maximum(previous_max, block_max)
        scaled_prefix = previous_sum * tl.exp(previous_max - new_max)

    exp_values = tl.exp(values - new_max)
    block_prefix = scaled_prefix + tl.cumsum(exp_values, axis=0)
    result = new_max + tl.log(block_prefix)
    tl.store(out + base + n_offsets * K, result, mask=mask)
    tl.store(state_max + pid, new_max)
    tl.store(state_sum + pid, scaled_prefix + tl.sum(exp_values, axis=0))


def _result_dtype(inp, dtype):
    if dtype is not None:
        return dtype
    if inp.dtype in (
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        return torch.float32
    return inp.dtype


def _scan_rows_into(inp, out, M, N):
    """K==1 fast paths: N==1 identity copy; N<=4096 single-shot row kernel;
    N>4096 chunked online scan kernel."""
    if N == 1:
        out.copy_(inp)
        return
    if N <= _ROW_MAX_N:
        TILE_N = triton.next_power_of_2(N)
        # This backend silently widens `tl.arange(0, TILE_N)` to
        # round_up(TILE_N, 64) lanes, so a TILE_N < 64 tile is 64 lanes wide.
        # The unmasked fast path would then read and write 64 - TILE_N lanes
        # past every row: neighbouring rows get clobbered (non-deterministically
        # -- whichever program stores last wins) and the final rows write beyond
        # the output allocation. Verified by sentinel-padded buffers: N=2/4/8/16/32
        # leaked 62/60/56/48/32 elements per row, N>=64 leaked 0. The boundary
        # mask `n_offsets < N` is honoured for 1D stores (unlike 2D tiles), so
        # masking whenever the tile is narrower than the 64-lane floor fixes it.
        need_mask = 1 if (TILE_N != N or TILE_N < 64) else 0
        num_warps = 8 if TILE_N > 2048 else 4
        grid = (M, 1, 1)
        logcumsumexp_row_kernel[grid](
            inp,
            out,
            N=N,
            TILE_N=TILE_N,
            NEED_MASK=need_mask,
            num_warps=num_warps,
            buffer_size_limit=2048,
        )
    else:
        BN = _ROW_MAX_N
        need_tail = 1 if N % BN else 0
        grid = (M, 1, 1)
        logcumsumexp_chunk_kernel[grid](
            inp,
            out,
            N,
            BN=BN,
            NEED_TAIL=need_tail,
            num_warps=8,
            buffer_size_limit=2048,
        )


def _logcumsumexp_impl(inp, dim, dtype=None, out_buffer=None):
    """Shared implementation for both overloads.

    K==1 writes directly into out_buffer when provided (no extra copy; the .out
    variant benefits from a full-tensor copy saved). K>1 keeps the legacy
    block kernel which needs its own destination.
    """
    assert -inp.ndim <= dim < inp.ndim, "Invalid dim"
    dim %= inp.ndim
    result_dtype = _result_dtype(inp, dtype)

    if inp.numel() == 0:
        if out_buffer is not None:
            out_buffer.resize_(inp.shape)
            return out_buffer
        return torch.empty_like(inp, dtype=result_dtype)

    shape = inp.shape
    M = 1
    for size in shape[:dim]:
        M *= size
    N = shape[dim]
    K = inp.numel() // M // N

    inp = inp.contiguous()

    if K == 1:
        if out_buffer is None:
            out = torch.empty_like(inp, dtype=result_dtype)
        else:
            out = out_buffer
        with torch_device_fn.device(inp.device):
            _scan_rows_into(inp, out, M, N)
        return out.view(shape)

    # K > 1: legacy block kernel path (middle-dim scans; not exercised by the
    # benchmark matrix, kept for correctness). Destination is always a fresh
    # tensor; the .out variant copies afterwards.
    out = torch.empty_like(inp, dtype=result_dtype)
    state_max = torch.empty((M * K,), dtype=torch.float32, device=inp.device)
    state_sum = torch.empty((M * K,), dtype=torch.float32, device=inp.device)
    block_size = triton.next_power_of_2(min(N, _SCAN_BLOCK))
    num_blocks = triton.cdiv(N, block_size)
    with torch_device_fn.device(inp.device):
        for block_idx in range(num_blocks):
            logcumsumexp_block_kernel[(M * K, 1, 1)](
                inp,
                out,
                state_max,
                state_sum,
                block_idx * block_size,
                N,
                K,
                block_size,
                FIRST_BLOCK=block_idx == 0,
                buffer_size_limit=2048,
                isCloseVectorization=True,
            )
    return out.view(shape)


def logcumsumexp(inp, dim=1, *, dtype=None):
    logger.debug("GEMS_KUNLUNXIN LOGCUMSUMEXP")
    return _logcumsumexp_impl(inp, dim, dtype, out_buffer=None)


def logcumsumexp_out(inp, dim=1, *, dtype=None, out):
    logger.debug("GEMS_KUNLUNXIN LOGCUMSUMEXP_OUT")
    result_dtype = _result_dtype(inp, dtype)
    if out.dtype != result_dtype:
        raise RuntimeError(
            f"logcumsumexp.out: expected out dtype {result_dtype}, got {out.dtype}"
        )
    if tuple(out.shape) != tuple(inp.shape):
        out.resize_(inp.shape)

    inp_c = inp.contiguous()
    out_c = out.contiguous() if not out.is_contiguous() else out
    aliasing = out_c.data_ptr() == inp_c.data_ptr()

    if (
        inp_c.dim() == out_c.dim()
        and tuple(inp_c.shape) == tuple(out_c.shape)
        and not aliasing
    ):
        # fast path: scan directly into the provided out buffer (K==1 only,
        # the block kernel path allocates its own destination).
        shape = inp_c.shape
        dim2 = dim % inp_c.ndim
        M = 1
        for size in shape[:dim2]:
            M *= size
        K = inp_c.numel() // M // shape[dim2]
        if K == 1:
            with torch_device_fn.device(inp_c.device):
                _scan_rows_into(inp_c, out_c, M, shape[dim2])
            if out_c is not out:
                out.copy_(out_c)
            return out
        # K > 1 direct path does not apply; fall through to the generic flow.
    return _generic_out_fallback(inp, dim, result_dtype, out)


def _generic_out_fallback(inp, dim, result_dtype, out):
    result = logcumsumexp(inp, dim, dtype=result_dtype)
    if tuple(out.shape) != tuple(inp.shape):
        out.resize_(inp.shape)
    out.copy_(result)
    return out
