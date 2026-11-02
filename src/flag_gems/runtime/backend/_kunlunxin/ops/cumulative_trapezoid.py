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

logger = logging.getLogger(__name__)

# Flat-tile widths for the 1-D scan path.
_TILE_LANES = {
    torch.float16: 16384,
    torch.bfloat16: 16384,
    torch.float32: 8192,
}
_MAX_TM = 8
_PACK_BLK = 16384

# ---------------------------------------------------------------------------
# Prefix sum as a matmul: a row's running integral is t @ U with
# U = triu(ones) and t the trapezoid increments, since the dot returns the
# inclusive prefix sum of its first operand.  U holds only zeros and ones and
# the dot accumulates in fp32, so the only departure from a per-element scan
# is the summation order.  The store goes straight to the (N-1)-wide result
# with stride N-1, so no staging buffer or repack pass is needed; the
# shifted load y[cols + 1] is masked.
# ---------------------------------------------------------------------------
_MM_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_MM_MIN_ROWS = 16
_MM_MIN_ELEMS = 1 << 12
_MM_MAX_ELEMS = (1 << 31) - 1
# Single-group tiles carry no running sum, so a wider row block fits.
_MM_FLAT_BUDGET = 200_000
# Group-loop path: default group width and candidate row blocks.
_MM_LOOP_G = 256
_MM_BM_CAND = (512, 256, 128, 64, 32, 16)
_MM_MIN_GRID = 8
_MM_LOOP_BUDGET = 98_304
_MM_U_CACHE = {}


def _mm_tri_u(G, dev):
    """Upper-triangular ones, cached per (width, device)."""
    key = (G, dev)
    u = _MM_U_CACHE.get(key)
    if u is None:
        u = torch.triu(torch.ones(G, G, dtype=torch.float32, device=dev))
        _MM_U_CACHE[key] = u
    return u


def _mm_plan(M, N, dtype):
    """(G, BM) for the matmul row-block scan, or None to take the scan path.

    G must divide N exactly because the group loop is unmasked along the row, and
    BM must divide M because the row index is unmasked too -- the rows past the
    last whole block are re-planned by the caller.
    """
    if dtype not in _MM_DTYPES or N < 32 or M < _MM_MIN_ROWS:
        return None
    if M * N < _MM_MIN_ELEMS or M * N > _MM_MAX_ELEMS:
        return None
    if N in (32, 64, 128, 256):
        G, budget = N, _MM_FLAT_BUDGET
    elif N % _MM_LOOP_G == 0:
        G, budget = _MM_LOOP_G, _MM_LOOP_BUDGET
    elif N % 128 == 0:
        G, budget = 128, _MM_LOOP_BUDGET
    elif N % 64 == 0:
        G, budget = 64, _MM_LOOP_BUDGET
    else:
        return None
    cand = [
        b
        for b in _MM_BM_CAND
        # The 128x128 tile is not used here.
        if b <= M and b * G + G * G <= budget and not (G == 128 and b == 128)
    ]
    if not cand:
        return None
    waste_cap = max(M // 64, 16)
    pool = [b for b in cand if M % b <= waste_cap] or cand
    return G, next((b for b in pool if M // b >= _MM_MIN_GRID), pool[0])


@triton.jit
def _cumtrapz_mm_kernel(
    y_ptr,
    out_ptr,
    N,
    NG,
    u_ptr,
    ROW: tl.constexpr,
    G: tl.constexpr,
    BM: tl.constexpr,
):
    """One program owns BM whole rows and walks their NG = N // G column groups
    in order, keeping each row's running integral in registers, so the data is
    read twice (the +1 shift) and written once.

    The dot runs in fp32 on both operands to meet the fp32 test tolerance."""
    pid = tl.program_id(0)
    rm = pid * BM + tl.arange(0, BM)
    kk = tl.arange(0, G)
    jj = tl.arange(0, G)
    u = tl.load(u_ptr + jj[:, None] * G + kk[None, :])
    base = rm[:, None] * N
    carry = tl.zeros([BM], dtype=tl.float32)
    for g in range(NG):
        cols = g * G + kk
        x = tl.load(y_ptr + base + cols[None, :]).to(tl.float32)
        xn = tl.load(
            y_ptr + base + cols[None, :] + 1,
            mask=(cols[None, :] + 1) < N,
            other=0.0,
        ).to(tl.float32)
        t = (x + xn) * 0.5
        acc = tl.dot(t, u, input_precision="ieee") + carry[:, None]
        tl.store(
            out_ptr + rm[:, None] * ROW + cols[None, :],
            acc.to(out_ptr.dtype.element_ty),
            mask=cols[None, :] < ROW,
        )
        carry = carry + tl.sum(t, axis=1)


@triton.jit
def _cumtrapz_scan_kernel(
    y_ptr,
    tmp_ptr,
    N: tl.constexpr,
    LOGN: tl.constexpr,
    B: tl.constexpr,
    TM: tl.constexpr,
):
    """Scan TM whole rows (B = TM*N lanes) with one 1-D cumsum.

    Row ends are detected with a bitmask, so N must be a power of two.  The
    scan restarts at every row boundary: for each band the prefix of all
    earlier bands is subtracted.
    """
    pid = tl.program_id(0)
    i = tl.arange(0, B)
    band = i >> LOGN
    v = tl.load(y_ptr + pid * B + i)
    vn = tl.load(y_ptr + pid * B + i + 1, mask=(i & (N - 1)) != N - 1, other=0.0)
    t = (v.to(tl.float32) + vn.to(tl.float32)) * 0.5
    t = tl.where((i & (N - 1)) == N - 1, 0.0, t)
    g = tl.cumsum(t, axis=0)
    for k in tl.static_range(1, TM):
        s = tl.sum(tl.where(band < k, t, 0.0), axis=0)
        g = g - tl.where(band == k, s, 0.0)
    tl.store(tmp_ptr + pid * B + i, g.to(tmp_ptr.dtype.element_ty))


@triton.jit
def _cumtrapz_pack_kernel(tmp_ptr, out_ptr, TOT, ROW: tl.constexpr, BLK: tl.constexpr):
    """Copy the stride-N scan into the (rows, ROW) result.

    Both the load offset and the store index are monotone, which keeps the
    access vectorised.
    """
    pid = tl.program_id(0)
    j = pid * BLK + tl.arange(0, BLK)
    m = j < TOT
    tl.store(out_ptr + j, tl.load(tmp_ptr + j + j // ROW, mask=m), mask=m)


@triton.jit
def _cumtrapz_generic_kernel(
    y_ptr,
    out_ptr,
    N,
    inner,
    stride_row_in,
    stride_row_out,
    BLOCK: tl.constexpr,
):
    """Fallback: one (outer, inner) line per program, single 1-D scan."""
    pid = tl.program_id(0)
    o = pid // inner
    i = pid % inner
    in_base = o * stride_row_in + i
    out_base = o * stride_row_out + i
    k = tl.arange(0, BLOCK)
    mask = k < (N - 1)
    a = tl.load(y_ptr + in_base + k * inner, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(y_ptr + in_base + (k + 1) * inner, mask=mask, other=0.0).to(tl.float32)
    cs = tl.cumsum((a + b) * 0.5, axis=0)
    tl.store(out_ptr + out_base + k * inner, cs, mask=mask)


def _is_pow2(x):
    return x > 0 and (x & (x - 1)) == 0


def _mm_run_into(yv, outv, M, N, dtype):
    """Scan M whole rows of yv into outv, re-planning the row tail narrower."""
    plan = _mm_plan(M, N, dtype)
    if plan is None:
        block = triton.next_power_of_2(max(N - 1, 1))
        _cumtrapz_generic_kernel[(M,)](yv, outv, N, 1, N, N - 1, BLOCK=block)
        return
    G, BM = plan
    nfull = (M // BM) * BM
    _cumtrapz_mm_kernel[(nfull // BM,)](
        yv,
        outv,
        N,
        N // G,
        _mm_tri_u(G, yv.device),
        ROW=N - 1,
        G=G,
        BM=BM,
        num_warps=4,
        buffer_size_limit=2048,
    )
    if nfull < M:
        _mm_run_into(yv[nfull:], outv[nfull:], M - nfull, N, dtype)


def cumulative_trapezoid(y, *, dx=1, dim=-1):
    logger.debug("GEMS CUMULATIVE_TRAPEZOID")
    assert y.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ), f"cumulative_trapezoid: unsupported dtype {y.dtype}"

    ndim = y.dim()
    d = dim if dim >= 0 else dim + ndim
    sizes = list(y.shape)
    N = sizes[d]
    outer = 1
    for s in sizes[:d]:
        outer *= s
    inner = 1
    for s in sizes[d + 1 :]:
        inner *= s

    out_sizes = list(sizes)
    out_sizes[d] = max(N - 1, 0)
    output = torch.empty(out_sizes, dtype=y.dtype, device=y.device)
    if N <= 1 or outer * inner == 0:
        return output if dx == 1 else output * dx

    yc = y.contiguous()

    if inner == 1 and outer >= _MM_MIN_ROWS:
        _mm_run_into(yc.view(outer, N), output.view(outer, N - 1), outer, N, y.dtype)
        if dx != 1:
            output = output * dx
        return output

    tm = 0
    if inner == 1 and _is_pow2(N) and y.dtype in _TILE_LANES:
        tm = min(max(_TILE_LANES[y.dtype] // N, 1), _MAX_TM)
        tm = min(tm, outer & (-outer))
        if tm < 1 or N * tm < 64:
            tm = 0
    if tm:
        B = N * tm
        tmp = torch.empty(outer * N, dtype=y.dtype, device=y.device)
        _cumtrapz_scan_kernel[(outer // tm,)](
            yc, tmp, N=N, LOGN=N.bit_length() - 1, B=B, TM=tm
        )
        tot = outer * (N - 1)
        blk = min(_PACK_BLK, triton.next_power_of_2(tot))
        _cumtrapz_pack_kernel[(triton.cdiv(tot, blk),)](
            tmp, output, tot, ROW=N - 1, BLK=blk
        )
        if dx != 1:
            output = output * dx
        return output

    stride_row_in = N * inner
    stride_row_out = (N - 1) * inner
    block = triton.next_power_of_2(max(N - 1, 1))
    _cumtrapz_generic_kernel[(outer * inner,)](
        yc, output, N, inner, stride_row_in, stride_row_out, BLOCK=block
    )
    if dx != 1:
        output = output * dx
    return output
