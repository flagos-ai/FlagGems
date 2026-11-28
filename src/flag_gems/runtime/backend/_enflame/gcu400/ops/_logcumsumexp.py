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
from flag_gems.utils import libentry, tl_extra_shim
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

_INT_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)

# Fixed grid: GCU400 grid dimensions have a limit of 255, and rows may be far larger than the hardware limit,
# so each program processes multiple rows in a strided manner via tl.range instead of assigning one program per row.
_GRID_DIM = 48

# Number of consecutive elements processed per Hillis-Steele round. A fixed width allows reuse of the compile cache,
# and also avoids exploding compile time from constructing oversized tiles based on row length.
_STEP_BLOCK = 1024

# ---------------------------------------------------------------------------
# Real-number fast path (GCU400 has no tt.scan, and custom associative_scan / 2D cumsum /
# scalar branches inside loops are all measured pathologies).
#
# Structure: each program claims several rows and advances serially along a row in chunks of
# width V. The in-chunk prefix comes from one [R,V]x[V,V] triangular dot, and the cross-chunk
# carry is accumulated in the exp domain (acc = sum of exp over all earlier chunks), so a chunk
# costs exactly two cross-lane ops: the dot and one sum reduce. That count is what sets the
# speed -- every cross-lane op pushes the whole tile through dsm at roughly 8 B/element
# (~100 GB/s measured), so ~0.16-0.55 us per op depending on tile size, and the kernel time
# tracks ops * elements rather than arithmetic.
#
# The whole row therefore shares a single shift: the max of its first chunk, paid once per row
# (PER_CHUNK_SHIFT keeps the exact per-chunk max for single-chunk rows, where it is free).
# Rows that the shift cannot cover -- an exponent underflowing to 0, or an accumulator large
# enough to hit the magnitude encoding -- set a per-row flag and are recomputed exactly by the
# repair kernel. That branch must stay out of the hot kernel (a branch inside the loop measured
# a 2.9-3.7x compile-time tax even when never taken).
#
# The dot's 0*inf pollution is avoided with magnitude encoding (NaN->1e35, +inf->1e30; restated
# as NaN/+inf when the prefix sum exceeds 1e34/1e29).
# ---------------------------------------------------------------------------
_SCAN_CAP = 512
# Chunk width: 128 keeps the triangular operand small enough (a 256-wide tri needs >1 MB of dsm
# and caps R at 8), and narrower chunks measured worse (V64/R32: 32.1 ms against 11.0 ms for
# V128/R32 on (1024,65536)).
_SCAN_V = 128
# Tile size: 4096 lanes (V128/R32) beat both 2048- (R16: 9.87 ms) and 8192-lane (R64: 7.15 ms)
# tiles on (1024,65536) at 5.53 ms, which is why the lane budget stops there.
_SCAN_LANES = 4096
_SCAN_R_MAX = 32
# The block count is kept at >= _SCAN_MIN_BLOCKS so short rows still spread over the machine,
# and never below _SCAN_R_MIN (a 1-row dot came back all-NaN).
_SCAN_R_MIN = 8
_SCAN_MIN_BLOCKS = 16
# dsm is what bounds R at a given V: a 256-wide tri already needs >1 MB at R=16.
_SCAN_R_CAP = {256: 8, 128: 32}


@triton.jit
def _logaddexp(a, b):
    """Real-valued logaddexp: pairwise max shift so low-magnitude prefixes never underflow."""
    lo = tl.minimum(a, b)
    hi = tl.maximum(a, b)
    diff = lo - hi
    # GCU400's vector exp degrades on deeply negative arguments: measured in this
    # context, exp(-1000) returns exp(-16) (e^-16 = 1.1253517e-7), which corrupts the
    # extreme prefix of ATen's log1p(exp(diff)) whose value should be 0 (the
    # [-1000.0, 0.0] test case is therefore off by 1.1e-7 and fails bitwise equality).
    # For diff < -16 use the reciprocal form 1/exp(-diff): the argument is then always
    # non-negative, which bypasses the degradation, and over that range log1p(e) ~ e, so
    # the ~1 ulp relative error of the reciprocal form keeps the absolute error below
    # 1e-14.
    e = tl.where(
        diff < -16.0,
        1.0 / tl.exp(tl.maximum(-diff, 0.0)),
        tl.exp(tl.maximum(diff, -16.0)),
    )
    merged = tl.log(1.0 + e) + hi
    # With two +-inf values, ATen returns the first argument unchanged, avoiding inf - inf producing NaN.
    both_inf = (lo == hi) & (tl.abs(lo) == float("inf"))
    res = tl.where(both_inf, a, merged)
    # fmin/fmax ignore NaN, so NaN must be made explicitly infectious.
    return tl.where((a != a) | (b != b), float("nan"), res)


@triton.jit
def _complex_log1p(x, y):
    """Complex log1p: use log1p of the squared modulus to keep accuracy near the origin."""
    zabs = tl.sqrt(x * x + y * y)
    theta = tl_extra_shim.atan2(y, x + 1.0)
    r = x * (2.0 + x) + y * y
    small = tl.where(r == 0.0, x, 0.5 * tl_extra_shim.log1p(r))
    big = tl.log(tl.sqrt((x + 1.0) * (x + 1.0) + y * y))
    return tl.where(zabs < 0.5, small, big), theta


@triton.jit
def _complex_exp(x, y):
    """Exponential of a finite complex number: exp(x) * (cos(y) + i sin(y))."""
    exp_abs = tl.exp(x)
    return exp_abs * tl.cos(y), exp_abs * tl.sin(y)


@triton.jit
def _complex_exp_pos_inf(y):
    """ATen branch for real part +inf: only the phase participates, avoiding NaN from inf * 0."""
    cos_v = tl.cos(y)
    sin_v = tl.sin(y)
    real = tl.where(cos_v == 0.0, 0.0, float("inf") * cos_v)
    imag = tl.where(sin_v == 0.0, 0.0, float("inf") * sin_v)
    return real, imag


@triton.jit
def _complex_logaddexp(ar, ai, br, bi):
    """Complex logaddexp: sort by real part, reproducing ATen's +-inf and NaN branches."""
    a_nan = (ar != ar) | (ai != ai)
    b_nan = (br != br) | (bi != bi)
    a_first = ar < br
    lo_r = tl.where(b_nan, br, tl.where(a_nan, ar, tl.where(a_first, ar, br)))
    lo_i = tl.where(b_nan, bi, tl.where(a_nan, ai, tl.where(a_first, ai, bi)))
    hi_r = tl.where(b_nan, br, tl.where(a_nan, ar, tl.where(a_first, br, ar)))
    hi_i = tl.where(b_nan, bi, tl.where(a_nan, ai, tl.where(a_first, bi, ai)))

    nan_out = (lo_r != lo_r) | (lo_i != lo_i)
    both_inf = (lo_r == hi_r) & (tl.abs(lo_r) == float("inf"))
    neg_inf = both_inf & (lo_r < 0.0)

    e_r, e_i = _complex_exp(lo_r - hi_r, lo_i - hi_i)
    l_r, l_i = _complex_log1p(e_r, e_i)
    gen_r = l_r + hi_r
    gen_i = l_i + hi_i

    e1_r, e1_i = _complex_exp_pos_inf(lo_i)
    e2_r, e2_i = _complex_exp_pos_inf(hi_i)
    p_r, p_i = _complex_log1p(e1_r + e2_r - 1.0, e1_i + e2_i)

    out_r = tl.where(
        nan_out, float("nan"), tl.where(neg_inf, lo_r, tl.where(both_inf, p_r, gen_r))
    )
    out_i = tl.where(
        nan_out, float("nan"), tl.where(neg_inf, lo_i, tl.where(both_inf, p_i, gen_i))
    )
    return out_r, out_i


@libentry()
@triton.jit(do_not_specialize=["N", "K", "rows", "shift"])
def _logcumsumexp_step_kernel(
    src,
    dst,
    N,
    K,
    rows,
    shift,
    BLOCK: tl.constexpr,
    IS_COMPLEX: tl.constexpr,
    REAL_INPUT: tl.constexpr,
    IS_FP64: tl.constexpr,
    SRC_IS_INPUT: tl.constexpr,
):
    """One Hillis-Steele round: dst[i] = logaddexp(src[i - shift], src[i]).

    Each round reads shifted addresses with tl.load, avoiding tt.scan /
    tt.gather. shift goes 1, 2, 4 ...; after ceil(log2(N)) launches
    dst[i] equals logaddexp of x[0..i]. The first round can read directly from the input buffer,
    saving an extra initialization kernel.
    """
    pid_r = tle.program_id(0)
    pid_b = tle.program_id(1)
    num_row_programs = tl.num_programs(0)
    idx = pid_b * BLOCK + tl.arange(0, BLOCK)
    valid = idx < N
    left_valid = valid & (idx >= shift)
    acc_ty = tl.float64 if IS_FP64 else tl.float32

    for row_id in tl.range(pid_r, rows, num_row_programs):
        base = (row_id // K) * N * K + (row_id % K)
        if IS_COMPLEX:
            cur_off = (base + idx * K) * 2
            left_off = (base + (idx - shift) * K) * 2
            if SRC_IS_INPUT and REAL_INPUT:
                # Real inputs are promoted to complex results: the imaginary part is always 0 and each logical element
                # has only 1 value.
                cur_r = tl.load(src + base + idx * K, mask=valid, other=0).to(acc_ty)
                cur_i = cur_r * 0.0
                left_r = tl.load(
                    src + base + (idx - shift) * K, mask=left_valid, other=0
                ).to(acc_ty)
                left_i = left_r * 0.0
            else:
                cur_r = tl.load(src + cur_off, mask=valid, other=0).to(acc_ty)
                cur_i = tl.load(src + cur_off + 1, mask=valid, other=0).to(acc_ty)
                left_r = tl.load(src + left_off, mask=left_valid, other=0).to(acc_ty)
                left_i = tl.load(src + left_off + 1, mask=left_valid, other=0).to(
                    acc_ty
                )
            cur_r = tl.where(valid, cur_r, float("-inf"))
            cur_i = tl.where(valid, cur_i, 0.0)
            left_r = tl.where(left_valid, left_r, float("-inf"))
            left_i = tl.where(left_valid, left_i, 0.0)
            comb_r, comb_i = _complex_logaddexp(left_r, left_i, cur_r, cur_i)
            # Lanes before shift do not combine and keep their current state, avoiding
            # the complex (-inf, 0) identity from changing the imaginary part in the both_inf branch.
            res_r = tl.where(left_valid, comb_r, cur_r)
            res_i = tl.where(left_valid, comb_i, cur_i)
            tl.store(dst + cur_off, res_r.to(dst.dtype.element_ty), mask=valid)
            tl.store(dst + cur_off + 1, res_i.to(dst.dtype.element_ty), mask=valid)
        else:
            cur = tl.load(src + base + idx * K, mask=valid, other=0).to(acc_ty)
            left = tl.load(src + base + (idx - shift) * K, mask=left_valid, other=0).to(
                acc_ty
            )
            cur = tl.where(valid, cur, float("-inf"))
            left = tl.where(left_valid, left, float("-inf"))
            combined = _logaddexp(left, cur)
            res = tl.where(left_valid, combined, cur)
            tl.store(dst + base + idx * K, res.to(dst.dtype.element_ty), mask=valid)


@libentry()
@triton.jit(do_not_specialize=["numel"])
def _logcumsumexp_identity_kernel(
    src,
    dst,
    numel,
    BLOCK: tl.constexpr,
    IS_COMPLEX: tl.constexpr,
    REAL_INPUT: tl.constexpr,
):
    """With N == 1 the prefix is the input itself; this also performs dtype conversion and zero-fills the complex
    imaginary part."""
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    if IS_COMPLEX:
        if REAL_INPUT:
            x = tl.load(src + offsets, mask=mask, other=0)
            tl.store(dst + offsets * 2, x.to(dst.dtype.element_ty), mask=mask)
            zero = x * 0.0
            tl.store(dst + offsets * 2 + 1, zero.to(dst.dtype.element_ty), mask=mask)
        else:
            xr = tl.load(src + offsets * 2, mask=mask, other=0)
            xi = tl.load(src + offsets * 2 + 1, mask=mask, other=0)
            tl.store(dst + offsets * 2, xr.to(dst.dtype.element_ty), mask=mask)
            tl.store(dst + offsets * 2 + 1, xi.to(dst.dtype.element_ty), mask=mask)
    else:
        x = tl.load(src + offsets, mask=mask, other=0)
        tl.store(dst + offsets, x.to(dst.dtype.element_ty), mask=mask)


@libentry()
@triton.jit(do_not_specialize=["N", "K", "rows", "num_blocks", "num_chunks"])
def _logcumsumexp_scan_kernel(
    src,
    dst,
    flags,
    N,
    K,
    rows,
    num_blocks,
    num_chunks,
    R: tl.constexpr,
    V: tl.constexpr,
    MASKED: tl.constexpr,
    PER_CHUNK_SHIFT: tl.constexpr,
):
    """Single pass of the real fast path: intra-chunk prefixes via a triangular dot, chunks folded

    in the exp domain. Each program claims several [R, V] row blocks with stride rb; the whole
    row shares one shift (PER_CHUNK_SHIFT=False: the max of its first chunk), so a chunk costs
    one dot plus one reduce:

    (1) e = exp(x - shift); encode magnitude (NaN->1e35, +inf->1e30) so the dot's 0*inf does not
    produce NaN; (2) s = e @ tri gives the intra-chunk prefix sum (fp32 ieee dot); (3) the running
    row prefix acc is kept as a sum of exp, so the tail is just acc + s, restored to NaN/+inf at
    the 1e34/1e29 thresholds (finite values never reach them) and logged back with the shift;
    (4) the same reduce that accumulates the chunk into acc also reports rows whose exponent hit
    zero -- any zero prefix means the whole chunk prefix was lost to the shift, so the row is
    flagged and recomputed exactly by the repair launch.

    A per-chunk max would be as expensive as the dot itself (a tile reduce moves the whole tile
    through dsm, ~8 B/element, which is what bounds this kernel), so it is paid once per row
    instead: exp(x - shift) stays in range because the shift is the row max of its first chunk,
    and any row where it is not ends up flagged. PER_CHUNK_SHIFT=True keeps the old behaviour for
    single-chunk rows, where the shift is free anyway.

    MASKED selects between the general form and the aligned form (rows % R == 0 and N % V == 0),
    where the row/column predicates disappear entirely. The masked form costs ~7% on every
    iteration and, on a large tile, is what pushes dsm past its limit, so aligned shapes take
    the maskless branch.
    """
    pid = tle.program_id(0)
    nprog = tl.num_programs(0)
    lanes = tl.arange(0, V)
    ri = tl.arange(0, V)[:, None]
    ci = tl.arange(0, V)[None, :]
    tri = tl.where(ri <= ci, 1.0, 0.0)

    for rb in tl.range(pid, num_blocks, nprog):
        r = rb * R + tl.arange(0, R)
        rv = r < rows
        base = (r // K) * N * K + (r % K)
        acc = tl.zeros([R], dtype=tl.float32)
        uf = tl.zeros([R], dtype=tl.float32)
        if PER_CHUNK_SHIFT:
            shift = tl.zeros([R], dtype=tl.float32)
        else:
            # Row anchor: the first chunk's max. It is an upper bound for the chunk it comes
            # from and, for any realistic row, close enough to the rest of the row that
            # exp(x - shift) neither overflows nor loses the sign of the prefix.
            if MASKED:
                lv0 = lanes < N
                x0 = tl.load(
                    src + base[:, None] + lanes[None, :] * K,
                    mask=rv[:, None] & lv0[None, :],
                    other=float("-inf"),
                ).to(tl.float32)
            else:
                x0 = tl.load(src + base[:, None] + lanes[None, :] * K).to(tl.float32)
            m0 = tl.max(
                tl.where((x0 != x0) | (x0 == float("inf")), float("-inf"), x0), axis=1
            )
            shift = tl.where(m0 == float("-inf"), 0.0, m0)
        for c in tl.range(0, num_chunks):
            offs = c * V + lanes
            if MASKED:
                lv = offs < N
                x = tl.load(
                    src + base[:, None] + offs[None, :] * K,
                    mask=rv[:, None] & lv[None, :],
                    other=float("-inf"),
                ).to(tl.float32)
            else:
                x = tl.load(src + base[:, None] + offs[None, :] * K).to(tl.float32)
            if PER_CHUNK_SHIFT:
                cand = tl.where((x != x) | (x == float("inf")), float("-inf"), x)
                m_c = tl.max(cand, axis=1)
                sh = tl.where(m_c == float("-inf"), 0.0, m_c)
            else:
                sh = shift
            e = tl.exp(x - sh[:, None])
            e = tl.where(x != x, 1.0e35, tl.where(x == float("inf"), 1.0e30, e))
            s = tl.dot(e, tri, input_precision="ieee")
            t = acc[:, None] + s
            res = tl.where(
                t >= 1.0e34,
                float("nan"),
                tl.where(t >= 1.0e29, float("inf"), sh[:, None] + tl.log(t)),
            )
            # Encoded sum: a zero exponent becomes 1e30, so the accumulator still reports a
            # fully underflowed prefix through the same reduce that updates acc.
            if MASKED:
                stot = tl.sum(
                    tl.where(lv[None, :], tl.where(e == 0.0, 1.0e30, e), 0.0), axis=1
                )
            else:
                stot = tl.sum(tl.where(e == 0.0, 1.0e30, e), axis=1)
            acc = acc + stot
            # The accumulator reaching 1e29 means some later output could be squashed onto the
            # +inf sentinel above (the decoded magnitude no longer fits), so repair those rows
            # too. It also catches the 1e30/1e35 sentinels themselves, i.e. non-finite inputs.
            uf = tl.maximum(uf, tl.where(acc >= 1.0e29, 1.0, 0.0))
            if MASKED:
                tl.store(
                    dst + base[:, None] + offs[None, :] * K,
                    res.to(dst.dtype.element_ty),
                    mask=rv[:, None] & lv[None, :],
                )
            else:
                tl.store(
                    dst + base[:, None] + offs[None, :] * K,
                    res.to(dst.dtype.element_ty),
                )
        if MASKED:
            tl.store(flags + r, uf, mask=rv)
        else:
            tl.store(flags + r, uf)


@libentry()
@triton.jit(do_not_specialize=["N", "K", "rows", "num_blocks"])
def _logcumsumexp_repair_kernel(
    src,
    dst,
    flags,
    N,
    K,
    rows,
    num_blocks,
    R: tl.constexpr,
):
    """Exact recompute for underflowed rows: rows with flag 0 return immediately; flagged rows

    The repair branch must live in a separate kernel -- even a never-taken branch body inside the
    hot kernel measured a 2.9-3.7x compile-time tax. Normal data (intra-chunk dynamic range
    < ~87) never triggers this kernel recompute loop.
    """
    pid = tle.program_id(0)
    nprog = tl.num_programs(0)
    for rb in tl.range(pid, num_blocks, nprog):
        r = rb * R + tl.arange(0, R)
        rv = r < rows
        f = tl.load(flags + r, mask=rv, other=0.0)
        if tl.sum(f) > 0.0:
            base = (r // K) * N * K + (r % K)
            acc = tl.full([R], float("-inf"), dtype=tl.float32)
            for i in tl.range(0, N):
                lm = i < N
                xi = tl.load(src + base + i * K, mask=rv & lm, other=float("-inf")).to(
                    tl.float32
                )
                acc = _logaddexp(acc, xi)
                tl.store(
                    dst + base + i * K,
                    acc.to(dst.dtype.element_ty),
                    mask=rv & lm,
                )


def _normalize_dim(inp, dim):
    # maybe_wrap_dim treats a 0-dimensional tensor as 1-dimensional, so scalars also accept dim=0 / -1.
    ndim = max(inp.ndim, 1)
    if dim < -ndim or dim >= ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of [{-ndim}, "
            f"{ndim - 1}], but got {dim})"
        )
    return dim % ndim


def _scan_tile(rows, N):
    """Pick the (V, R) scan tile.

    V is the chunk width: rows up to 256 wide are scanned in a single chunk, wider rows
    use the 128-wide chunk that keeps the triangular operand small enough for a wide R.
    R then eats the rest of the 4096-lane tile budget, shrunk for short row counts so the
    launch still produces at least _SCAN_MIN_BLOCKS blocks.
    """
    if N <= 256:
        scan_v = max(16, triton.next_power_of_2(N))
    else:
        scan_v = _SCAN_V
    # tl.arange needs a power of two, so round the target row count down to one.
    blocks_target = max(1, rows // _SCAN_MIN_BLOCKS)
    blocks_target = 1 << (blocks_target.bit_length() - 1)
    scan_r = min(
        _SCAN_R_MAX,
        blocks_target,
        _SCAN_LANES // scan_v,
        _SCAN_R_CAP.get(scan_v, _SCAN_R_MAX),
    )
    # A 1-row dot is degenerate and came back all-NaN, so never go below _SCAN_R_MIN;
    # the row mask covers the extra rows.
    return scan_v, max(_SCAN_R_MIN, scan_r)


def _resolve_dtype(inp, dtype):
    if dtype is None:
        dtype = inp.dtype
    if dtype in _INT_DTYPES or dtype == torch.bool:
        dtype = torch.float32
    return dtype


def _real_view(tensor):
    """Provide the kernel with a contiguous real buffer that can be addressed linearly."""
    tensor = tensor.contiguous()
    if tensor.is_complex():
        return torch.view_as_real(tensor)
    return tensor


def _logcumsumexp_wrapper(inp, dim=1, dtype=None, out=None):
    if not isinstance(dim, int):
        raise TypeError(f"dim must be an int, but got {type(dim).__name__}")
    dim = _normalize_dim(inp, dim)

    result_dtype = _resolve_dtype(inp, dtype)
    if inp.is_complex() and not result_dtype.is_complex:
        raise RuntimeError(
            "logcumsumexp: cannot compute a complex input into the real dtype "
            f"{result_dtype}"
        )

    if out is None:
        out = torch.empty(inp.shape, dtype=result_dtype, device=inp.device)
    else:
        # out= contract: same device, same dtype, resizable shape, and the caller's tensor must be returned.
        if out.device != inp.device:
            raise RuntimeError(
                "Expected all tensors to be on the same device, but got self "
                f"is on {inp.device}, different from other tensors on {out.device}"
            )
        if out.dtype != result_dtype:
            raise RuntimeError(
                f"out must have dtype {result_dtype}, but got {out.dtype}"
            )
        if tuple(out.shape) != tuple(inp.shape):
            out.resize_(inp.shape)

    if inp.numel() == 0:
        return out

    ndim = max(inp.ndim, 1)
    shape = list(inp.shape) if inp.ndim else [1]
    M = 1
    for i in range(dim):
        M *= shape[i]
    N = shape[dim]
    K = 1
    for i in range(dim + 1, ndim):
        K *= shape[i]

    is_complex = out.is_complex()
    acc_is_fp64 = result_dtype in (torch.float64, torch.complex128)
    real_input = not inp.is_complex()
    acc_dtype = torch.float64 if acc_is_fp64 else torch.float32

    # Non-contiguous outputs are computed in contiguous scratch and copied back to the caller's layout.
    if out.is_contiguous():
        target = out
        scratch = None
    else:
        target = torch.empty(inp.shape, dtype=result_dtype, device=inp.device)
        scratch = out

    src_view = _real_view(inp)
    dst_view = _real_view(target)
    rows = M * K
    num_blocks = triton.cdiv(N, _STEP_BLOCK)
    grid = (min(_GRID_DIM, rows), num_blocks)
    identity_block = 1024
    identity_grid = (triton.cdiv(inp.numel(), identity_block),)

    with torch_device_fn.device(inp.device):
        if N == 1:
            _logcumsumexp_identity_kernel[identity_grid](
                src_view,
                dst_view,
                inp.numel(),
                BLOCK=identity_block,
                IS_COMPLEX=is_complex,
                REAL_INPUT=real_input,
                num_warps=4,
            )
        elif not is_complex and not acc_is_fp64:
            # Real-number fast path: one scan + underflow repair, 2 launches in total
            # (compared with ceil(log2(N)) full-tensor launches for Hillis-Steele).
            scan_v, scan_r = _scan_tile(rows, N)
            scan_blocks = triton.cdiv(rows, scan_r)
            scan_grid = (min(scan_blocks, _SCAN_CAP),)
            flags = torch.empty(rows, dtype=torch.float32, device=inp.device)
            num_chunks = triton.cdiv(N, scan_v)
            masked = rows % scan_r != 0 or N % scan_v != 0
            _logcumsumexp_scan_kernel[scan_grid](
                src_view,
                dst_view,
                flags,
                N,
                K,
                rows,
                scan_blocks,
                num_chunks,
                R=scan_r,
                V=scan_v,
                MASKED=masked,
                PER_CHUNK_SHIFT=num_chunks == 1,
                num_warps=4,
            )
            # The repair pass stays unconditional: even a single-chunk row can
            # underflow inside the chunk (chunk-wide max loses terms below
            # exp(-87)), which large_dynamic_range cases exercise.
            _logcumsumexp_repair_kernel[scan_grid](
                src_view,
                dst_view,
                flags,
                N,
                K,
                rows,
                scan_blocks,
                R=scan_r,
                num_warps=4,
            )
        else:
            state_elems = inp.numel() * (2 if is_complex else 1)
            buf_a = torch.empty(state_elems, dtype=acc_dtype, device=inp.device)
            buf_b = torch.empty(state_elems, dtype=acc_dtype, device=inp.device)
            current = src_view
            shift = 1
            step = 0
            while shift < N:
                is_final = shift * 2 >= N
                dest = dst_view if is_final else (buf_a if step % 2 == 0 else buf_b)
                _logcumsumexp_step_kernel[grid](
                    current,
                    dest,
                    N,
                    K,
                    rows,
                    shift,
                    BLOCK=_STEP_BLOCK,
                    IS_COMPLEX=is_complex,
                    REAL_INPUT=real_input,
                    IS_FP64=acc_is_fp64,
                    SRC_IS_INPUT=step == 0,
                    num_warps=4,
                )
                current = dest
                shift *= 2
                step += 1

    if scratch is not None:
        scratch.copy_(target)
        return scratch
    return out


def _logcumsumexp(inp, dim=1, *, dtype=None):
    logger.debug("GEMS_ENFLAME _LOGCUMSUMEXP")
    return _logcumsumexp_wrapper(inp, dim, dtype)


def _logcumsumexp_out(inp, dim=1, *, dtype=None, out):
    logger.debug("GEMS_ENFLAME _LOGCUMSUMEXP_OUT")
    return _logcumsumexp_wrapper(inp, dim, dtype, out)


# Shared registration state is attached to torch: this file is loaded once each under three module paths,
# ``gcu400.ops._logcumsumexp``,
# ``_enflame.gcu400.ops._logcumsumexp``, and the full package path, so attaching the handle to torch ensures
# registration happens only once per process.
#
_ATEN_STATE_ATTR = "_flag_gems_gcu400_logcumsumexp_aten_state"


# Module-level handle keeps the Library (and its kernels) alive for the
# process; the registration itself must never break the flag_gems import.


# ---------------------------------------------------------------------------
# aten::isnan guard. torch_gcu's topsatenIsNan has no ComplexFloat entry, so
# ``torch.isnan(complex64 gcu)`` dies with TOPSATEN_STATUS_NOT_SUPPORT ->
# ``Check failed: 0`` -> SIGABRT (no Python exception). The accuracy suite
# checks this op's complex output with raw ``torch.isnan`` calls, and the
# defect is unreachable from op code, so route complex inputs through a tiny
# Triton kernel here and delegate every other dtype to the native topsaten
# kernel captured before the override (the linalg_multi_dot / index_reduce
# pattern). Non-complex behavior stays bit-identical to the native kernel.
# ---------------------------------------------------------------------------

_ISNAN_BLOCK = 1024

_ISNAN_DEVICE_KEYSET = torch._C.DispatchKeySet(
    getattr(torch._C.DispatchKey, runtime.device.dispatch_key)
)

_NATIVE_ISNAN = None
if hasattr(torch.library, "get_kernel"):
    try:
        _NATIVE_ISNAN = torch.library.get_kernel(
            torch.ops.aten.isnan.default, runtime.device.dispatch_key
        )
    except Exception:
        _NATIVE_ISNAN = None


@libentry()
@triton.jit(do_not_specialize=["numel"])
def _isnan_complex_kernel(src, dst, numel, BLOCK: tl.constexpr):
    """dst[i] = isnan(re) | isnan(im) over the (..., 2) real view of complex."""
    pid = tle.program_id(0)
    num_programs = tl.num_programs(0)
    for start in tl.range(pid * BLOCK, numel, num_programs * BLOCK):
        idx = start + tl.arange(0, BLOCK)
        mask = idx < numel
        # Explicit i64 widening: a 2**31-element complex buffer would make the
        # ``* 2`` byte offset wrap in i32 and silently poison the pointers.
        off = idx.to(tl.int64)
        re = tl.load(src + off * 2, mask=mask, other=0.0)
        im = tl.load(src + off * 2 + 1, mask=mask, other=0.0)
        res = (re != re) | (im != im)
        tl.store(dst + off, res, mask=mask)


def _isnan_aten(self):
    if self.is_complex() and self.dtype == torch.complex64:
        x = self if self.is_contiguous() else self.contiguous()
        numel = x.numel()
        out = torch.empty(x.shape, dtype=torch.bool, device=x.device)
        if numel == 0:
            return out
        grid = (min(triton.cdiv(numel, _ISNAN_BLOCK), 65535),)
        with torch_device_fn.device(x.device):
            _isnan_complex_kernel[grid](
                torch.view_as_real(x), out, numel, BLOCK=_ISNAN_BLOCK, num_warps=4
            )
        return out
    return _NATIVE_ISNAN.call_boxed(_ISNAN_DEVICE_KEYSET, self)


# Module-level handle keeps the Library (and its kernels) alive for the
# process; the registration itself must never break the flag_gems import.
