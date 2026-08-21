import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

from .vector_norm import vector_norm


# ---------------------------------------------------------------------------
# Flat L2-norm (norm p == 2 / p == None, dim == None) fast path.
#
# Design (XPU): the old many-stage Kahan tree launched numel/8 tiny programs
# per stage (~190ms for big tensors). This path instead treats the flat input
# as rows of ROWLEN=8192 lanes -- the documented XPU-safe lane count for a
# tl.sum without buffer_size_limit -- and reduces them with a mask-free 2D
# [BLOCK_M, BLOCK_N] row kernel (reduce-INSIDE [BLOCK_M, 1] accumulator, the
# amax-proven form), then keeps re-reducing until a scalar, taking sqrt at the
# very end. Non-divisible remainders are accumulated by exact unmasked tail
# kernels (no masked loads: the XPU masked-tail + tl.sum combination reads
# out-of-range lanes). All accumulation is fp32; the sqrt result is cast to
# x.dtype on store, matching torch.norm's semantics for NaN/+-inf too.
_ROW_LEN = 8192
_L2_SMALL_LIMIT = _ROW_LEN
_FP16_BLOCK_M = (128, 64, 256, 32, 16, 8, 4, 2)
_FP32_BLOCK_M = (64, 128, 32, 16, 8, 4, 2)
_FP16_BLOCK_N = (1024, 512, 256, 128, 64, 32, 16)
_FP32_BLOCK_N = (512, 1024, 256, 128, 64, 32, 16)


def _pick_row_tile(n_rows, is_fp32):
    """Pick a mask-free (BLOCK_M, BLOCK_N) pair: n_rows % BLOCK_M == 0 and
    ROW_LEN % BLOCK_N == 0, preferring the sweep-tuned sweet spots."""
    bms = _FP32_BLOCK_M if is_fp32 else _FP16_BLOCK_M
    bns = _FP32_BLOCK_N if is_fp32 else _FP16_BLOCK_N
    bm = next((m for m in bms if n_rows % m == 0), 1)
    bn = next((b for b in bns if _ROW_LEN % b == 0), 1)
    return bm, bn


@libentry()
@triton.jit
def _l2_small_kernel(X, Out, N):
    """Exact unmasked sum-of-squares over 0 < N <= ROW_LEN elements."""
    total = tl.zeros((), dtype=tl.float32)
    full = (N // 8) * 8
    for off in tl.range(0, full, 8):
        v = tl.load(X + off + tl.arange(0, 8)).to(tl.float32)
        total += tl.sum(v * v)
    for off in tl.range(0, N - full):
        v = tl.load(X + full + off).to(tl.float32)
        total += v * v
    tl.store(Out, tl.sqrt(total))


@libentry()
@triton.jit
def _l2_tail_kernel(X, Out, TAIL_BASE, TAIL_N, SQUARED: tl.constexpr):
    """Exact unmasked chunk accumulation (squares when SQUARED). All reads are
    strictly in-bounds, so no mask is needed."""
    base = TAIL_BASE.to(tl.int64)
    total = tl.zeros((), dtype=tl.float32)
    full = (TAIL_N // 8) * 8
    for off in tl.range(0, full, 8):
        v = tl.load(X + base + off + tl.arange(0, 8)).to(tl.float32)
        if SQUARED:
            v = v * v
        total += tl.sum(v)
    for off in tl.range(0, TAIL_N - full):
        v = tl.load(X + base + full + off).to(tl.float32)
        if SQUARED:
            v = v * v
        total += v
    tl.store(Out, total)


@libentry()
@triton.jit
def _l2_row_kernel(
    X,
    Partial,
    N_ROWS,
    ROW_LEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SQUARED: tl.constexpr,
):
    """Mask-free 2D row reduction: each program reduces BLOCK_M rows, each of
    ROW_LEN lanes (looped in BLOCK_N chunks), into one fp32 partial per row.
    Callers guarantee N_ROWS % BLOCK_M == 0 and ROW_LEN % BLOCK_N == 0."""
    pid = ext.program_id(0).to(tl.int64)
    row_ids = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    acc = tl.zeros([BLOCK_M, 1], dtype=tl.float32)
    for off in range(0, ROW_LEN, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        a = tl.load(X + row_ids * ROW_LEN + cols).to(tl.float32)
        if SQUARED:
            a = a * a
        acc += tl.sum(a, axis=1)[:, None]
    tl.store(Partial + row_ids, acc)


@libentry()
@triton.jit
def _l2_final_kernel(Partial, Out, N):
    """Exact unmasked sum of <= 8192 fp32 partials, then sqrt into Out."""
    total = tl.zeros((), dtype=tl.float32)
    full = (N // 8) * 8
    for off in tl.range(0, full, 8):
        total += tl.sum(tl.load(Partial + off + tl.arange(0, 8)).to(tl.float32))
    for off in tl.range(0, N - full):
        total += tl.load(Partial + full + off).to(tl.float32)
    tl.store(Out, tl.sqrt(total))


def _norm_scalar_l2(x):
    n = x.numel()
    x1 = x.view(-1)
    out = torch.empty((), dtype=x.dtype, device=x.device)
    if n <= _L2_SMALL_LIMIT:
        _l2_small_kernel[(1,)](x1, out, n)
        return out

    rows = n // _ROW_LEN
    tail = n - rows * _ROW_LEN
    mid_cnt = rows + (1 if tail else 0)
    mid = torch.empty((mid_cnt,), dtype=torch.float32, device=x.device)
    is_fp32 = x.dtype == torch.float32

    bm, bn = _pick_row_tile(rows, is_fp32)
    _l2_row_kernel[(rows // bm,)](
        x1, mid, rows, _ROW_LEN, bm, bn, SQUARED=True
    )
    if tail:
        _l2_tail_kernel[(1,)](x1, mid[rows:], rows * _ROW_LEN, tail, SQUARED=True)

    while mid_cnt > _ROW_LEN:
        rows2 = mid_cnt // _ROW_LEN
        rem2 = mid_cnt - rows2 * _ROW_LEN
        next_cnt = rows2 + (1 if rem2 else 0)
        nxt = torch.empty((next_cnt,), dtype=torch.float32, device=x.device)
        bm2, bn2 = _pick_row_tile(rows2, False)
        if rows2:
            _l2_row_kernel[(rows2 // bm2,)](
                mid, nxt, rows2, _ROW_LEN, bm2, bn2, SQUARED=False
            )
        if rem2:
            _l2_tail_kernel[(1,)](
                mid, nxt[rows2:], rows2 * _ROW_LEN, rem2, SQUARED=False
            )
        mid = nxt
        mid_cnt = next_cnt

    _l2_final_kernel[(1,)](mid, out, mid_cnt)
    return out


def norm(x, p=2, dim=None, keepdim=False):
    return vector_norm(x, ord=2 if p is None else p, dim=dim, keepdim=keepdim)


def norm_scalar(x, p=2):
    if (
        p in (None, 2)
        and x.is_contiguous()
        and x.numel() > 0
        and x.dtype in (torch.float16, torch.float32, torch.bfloat16)
    ):
        return _norm_scalar_l2(x)
    return norm(x, p=p, dim=None, keepdim=False)


def norm_scalaropt_dim(x, p, dim, keepdim=False):
    return norm(x, p=p, dim=dim, keepdim=keepdim)
