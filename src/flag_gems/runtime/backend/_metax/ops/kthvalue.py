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


_CB = 4096  # radix chunk size
_D1_MAX = 8192  # max single-block row size (32-bit dtypes)
_D1_MAX_64 = 4096  # max single-block row size (64-bit dtypes)
_SMALL_K = 8  # use iterative top-k selection when k <= _SMALL_K
_BIG = tl.constexpr(1 << 62)
# host-side plain constants (passed into kernels as constexpr args)
_HF_FLT_MAX = 3.4028234663852886e38
_HF_DBL_MAX = 1.7976931348623157e308
_HF_HALF_MAX = 65504.0
_HF_BF16_MAX = 3.3895313892515355e38
_HF_I32_MAX = 2147483647
_HF_I64_MAX = 9223372036854775807
# kernel-side constexpr constants
_FLT_MAX = tl.constexpr(_HF_FLT_MAX)
_DBL_MAX = tl.constexpr(_HF_DBL_MAX)
_HALF_MAX = tl.constexpr(_HF_HALF_MAX)
_BF16_MAX = tl.constexpr(_HF_BF16_MAX)
_I32_MAX = tl.constexpr(_HF_I32_MAX)
_I64_MAX = tl.constexpr(_HF_I64_MAX)
_POS_INF = tl.constexpr(float("inf"))
_NEG_INF = tl.constexpr(float("-inf"))
_INT_MIN32 = tl.constexpr(-2147483648)
_INT_MIN64 = tl.constexpr(-9223372036854775808)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
@triton.jit
def _decompose(e, shape_ptr, stride_ptr, DIM: tl.constexpr, RANK: tl.constexpr):
    # `e` is the output row-major index over dims != DIM (output layout).
    # Convert it to the input memory base offset of that slice.
    rem = e
    base = tl.full((), 0, tl.int64)
    for t in tl.static_range(RANK - 1, -1, -1):
        if t != DIM:
            sz = tl.load(shape_ptr + t).to(tl.int64)
            st = tl.load(stride_ptr + t).to(tl.int64)
            idx = rem % sz
            rem = rem // sz
            base += idx * st
    return base


@triton.jit
def _to_key(x, IS_FP: tl.constexpr, W: tl.constexpr):
    # Monotonic key in *unsigned* order (radix bytes are unsigned).
    if W == 32:
        b = x.to(tl.int32, bitcast=True)
        if IS_FP:
            b = tl.where(b == _INT_MIN32, 0, b)
            return tl.where(b < 0, b ^ -1, b ^ _INT_MIN32)
        else:
            return b ^ _INT_MIN32
    else:
        b = x.to(tl.int64, bitcast=True)
        if IS_FP:
            b = tl.where(b == _INT_MIN64, 0, b)
            return tl.where(b < 0, b ^ -1, b ^ _INT_MIN64)
        else:
            return b ^ _INT_MIN64


@triton.jit
def _unmap(key, IS_FP: tl.constexpr, W: tl.constexpr):
    if W == 32:
        k = key.to(tl.int32)
        if IS_FP:
            bits = tl.where(k < 0, k ^ _INT_MIN32, k ^ -1)
            return bits.to(tl.float32, bitcast=True)
        else:
            return k ^ _INT_MIN32
    else:
        if IS_FP:
            bits = tl.where(key < 0, key ^ _INT_MIN64, key ^ -1)
            return bits.to(tl.float64, bitcast=True)
        else:
            return key ^ _INT_MIN64


# ---------------------------------------------------------------------------
# Path 1a: D == 1
# ---------------------------------------------------------------------------
@triton.jit
def _kth_d1_kernel(
    inp_ptr,
    shape_ptr,
    stride_ptr,
    out_v_ptr,
    out_i_ptr,
    DIM: tl.constexpr,
    RANK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    base = _decompose(pid, shape_ptr, stride_ptr, DIM, RANK)
    x = tl.load(inp_ptr + base)
    tl.store(out_v_ptr + pid, x)
    tl.store(out_i_ptr + pid, 0)


# ---------------------------------------------------------------------------
# Path 1b: small k (k <= 8): iterative min-removal selection, no sort
# ---------------------------------------------------------------------------
@triton.jit
def _kth_smallk_kernel(
    inp_ptr,
    shape_ptr,
    stride_ptr,
    out_v_ptr,
    out_i_ptr,
    D,
    k,
    SENT: tl.constexpr,
    IS_FP: tl.constexpr,
    UPCAST16: tl.constexpr,
    LAST_TIE: tl.constexpr,
    BLOCK: tl.constexpr,
    DIM: tl.constexpr,
    RANK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    base = _decompose(pid, shape_ptr, stride_ptr, DIM, RANK)
    sd = tl.load(stride_ptr + DIM).to(tl.int64)
    offs = tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < D
    xo = tl.load(inp_ptr + base + offs * sd, mask=mask, other=0)
    if UPCAST16:
        xf = xo.to(tl.float32)
        x = tl.where(mask, xf, SENT)
    else:
        x = tl.where(mask, xo, SENT)
    if IS_FP:
        nan = x != x
        avail = mask & (~nan)
    else:
        nan = mask & False
        avail = mask
    rem = x
    m = tl.full((), 0, x.dtype)
    sel_pos = _BIG
    for i in tl.range(0, k):
        cand = tl.where(avail, rem, SENT)
        m = tl.min(cand, axis=0)
        eq = (rem == m) & avail
        sel_pos = tl.min(tl.where(eq, offs, _BIG), axis=0)
        avail = avail & (offs != sel_pos)
    v = m
    if LAST_TIE:
        eqv = (x == v) & mask
        if IS_FP:
            eqv = eqv & (~nan)
        idx = tl.max(tl.where(eqv, offs, -1), axis=0)
    else:
        idx = sel_pos
    if IS_FP:
        nn = tl.sum(tl.where(nan & mask, 1, 0), axis=0).to(tl.int64)
        k_nan = k > (D - nn)
        if LAST_TIE:
            nan_idx = tl.max(tl.where(nan & mask, offs, -1), axis=0)
            nan_v = tl.sum(tl.where(nan & (offs == nan_idx), x, 0.0), axis=0)
        else:
            r_nan = k - (D - nn)
            csum_nan = tl.cumsum(nan.to(tl.int32), axis=0).to(tl.int64)
            nan_sel = nan & (csum_nan == r_nan) & mask
            nan_idx = tl.min(tl.where(nan_sel, offs, _BIG), axis=0)
            nan_v = tl.sum(tl.where(nan_sel, x, 0.0), axis=0)
        v = tl.where(k_nan, nan_v, v)
        idx = tl.where(k_nan, nan_idx, idx)
        v = tl.where(v == 0, tl.zeros_like(v), v)  # normalize -0.0 -> +0.0
    if UPCAST16:
        if SENT == _HALF_MAX:
            v = v.to(tl.float16)
        else:
            v = v.to(tl.bfloat16)
    tl.store(out_v_ptr + pid, v)
    tl.store(out_i_ptr + pid, idx)


# ---------------------------------------------------------------------------
# Path 1c: large k (> 8): per-slice in-register sort
# ---------------------------------------------------------------------------
@triton.jit
def _kth_single_kernel(
    inp_ptr,
    shape_ptr,
    stride_ptr,
    out_v_ptr,
    out_i_ptr,
    D,
    k,
    SENT: tl.constexpr,
    IS_FP: tl.constexpr,
    UPCAST16: tl.constexpr,
    BLOCK: tl.constexpr,
    DIM: tl.constexpr,
    RANK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    base = _decompose(pid, shape_ptr, stride_ptr, DIM, RANK)
    sd = tl.load(stride_ptr + DIM).to(tl.int64)
    offs = tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < D
    xo = tl.load(inp_ptr + base + offs * sd, mask=mask, other=0)
    if UPCAST16:
        xf = xo.to(tl.float32)
        x = tl.where(mask, xf, SENT)
    else:
        x = tl.where(mask, xo, SENT)
    if IS_FP:
        nan = x != x
        pos_inf = x == _POS_INF
        neg_inf = x == _NEG_INF
        xr = tl.where(neg_inf, -SENT, x)
        xr = tl.where(mask & (~nan) & (~pos_inf), xr, SENT)
        s = tl.sort(xr, dim=0)
        kk = k - 1
        v = tl.sum(tl.where(offs == kk, s, tl.zeros_like(s)), axis=0)
        n_neg = tl.sum(tl.where(neg_inf, 1, 0), axis=0).to(tl.int64)
        n_pos = tl.sum(tl.where(pos_inf, 1, 0), axis=0).to(tl.int64)
        nn = tl.sum(tl.where(nan, 1, 0), axis=0).to(tl.int64)
        if n_neg + n_pos + nn == 0:
            nless = tl.sum(tl.where(mask & (x < v), 1, 0), axis=0).to(tl.int64)
            r = k - nless
            eq = (x == v) & mask
            csum = tl.cumsum(eq.to(tl.int32), axis=0).to(tl.int64)
            idx = tl.min(tl.where(eq & (csum == r), offs, _BIG), axis=0)
        else:
            fin = D - n_neg - n_pos - nn
            csum_neg = tl.cumsum(neg_inf.to(tl.int32), axis=0).to(tl.int64)
            neg_mask = neg_inf & (csum_neg == k) & mask
            neg_idx = tl.min(tl.where(neg_mask, offs, _BIG), axis=0)
            nless_all = tl.sum(tl.where(mask & (x < v), 1, 0), axis=0).to(tl.int64)
            r_fin = k - nless_all
            eq = (x == v) & (~nan) & (~pos_inf) & (~neg_inf) & mask
            csum = tl.cumsum(eq.to(tl.int32), axis=0).to(tl.int64)
            fin_idx = tl.min(tl.where(eq & (csum == r_fin), offs, _BIG), axis=0)
            r_pos = k - n_neg - fin
            csum_pos = tl.cumsum(pos_inf.to(tl.int32), axis=0).to(tl.int64)
            pos_mask = pos_inf & (csum_pos == r_pos) & mask
            pos_idx = tl.min(tl.where(pos_mask, offs, _BIG), axis=0)
            r_nan = k - n_neg - fin - n_pos
            csum_nan = tl.cumsum(nan.to(tl.int32), axis=0).to(tl.int64)
            nan_mask = nan & (csum_nan == r_nan) & mask
            nan_idx = tl.min(tl.where(nan_mask, offs, _BIG), axis=0)
            nan_v = tl.sum(tl.where(nan_mask, x, tl.zeros_like(x)), axis=0)
            k_neg = k <= n_neg
            k_pos = (k > n_neg + fin) & (k <= n_neg + fin + n_pos)
            k_nan = k > n_neg + fin + n_pos
            v = tl.where(
                k_neg, _NEG_INF, tl.where(k_pos, _POS_INF, tl.where(k_nan, nan_v, v))
            )
            idx = tl.where(
                k_neg,
                neg_idx,
                tl.where(k_pos, pos_idx, tl.where(k_nan, nan_idx, fin_idx)),
            )
        v = tl.where(v == 0, tl.zeros_like(v), v)  # normalize -0.0 -> +0.0
    else:
        s = tl.sort(x, dim=0)
        kk = k - 1
        v = tl.sum(tl.where(offs == kk, s, tl.zeros_like(s)), axis=0)
        nless = tl.sum(tl.where(mask & (x < v), 1, 0), axis=0).to(tl.int64)
        eq = (x == v) & mask
        csum = tl.cumsum(eq.to(tl.int32), axis=0).to(tl.int64)
        r = k - nless
        idx = tl.min(tl.where(eq & (csum == r), offs, _BIG), axis=0)
    if UPCAST16:
        if SENT == _HALF_MAX:
            v = v.to(tl.float16)
        else:
            v = v.to(tl.bfloat16)
    tl.store(out_v_ptr + pid, v)
    tl.store(out_i_ptr + pid, idx)


# ---------------------------------------------------------------------------
# Path 2: radix-select pipeline (D > D1_MAX)
# ---------------------------------------------------------------------------
@triton.jit
def _radix_count_kernel(
    inp_ptr,
    shape_ptr,
    stride_ptr,
    hist_ptr,
    lo_ptr,
    D,
    S,
    G,
    DIGIT: tl.constexpr,
    NBYTES: tl.constexpr,
    CB: tl.constexpr,
    IS_FP: tl.constexpr,
    W: tl.constexpr,
    UP16: tl.constexpr,
    DIM: tl.constexpr,
    RANK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    s = pid // G
    c = pid % G
    base = _decompose(s, shape_ptr, stride_ptr, DIM, RANK)
    sd = tl.load(stride_ptr + DIM).to(tl.int64)
    lo = tl.load(lo_ptr + s)
    offs = c * CB + tl.arange(0, CB).to(tl.int64)
    mask = offs < D
    x = tl.load(inp_ptr + base + offs * sd, mask=mask, other=0)
    if UP16:
        x = x.to(tl.float32)
    key = _to_key(x, IS_FP, W)
    if DIGIT == 0:
        rmask = mask
    else:
        rshift = 8 * (NBYTES - DIGIT)
        rmask = mask & ((key >> rshift) == (lo >> rshift))
    dshift = 8 * (NBYTES - 1 - DIGIT)
    digit = ((key >> dshift) & 255).to(tl.int32)
    h = tl.histogram(tl.where(rmask, digit, 255), num_bins=256)
    tl.atomic_add(hist_ptr + s * 256 + tl.arange(0, 256), h)


@triton.jit
def _radix_narrow_kernel(
    hist_ptr,
    lo_ptr,
    krem_ptr,
    vkey_ptr,
    S,
    k,
    DIGIT: tl.constexpr,
    NBYTES: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, 256).to(tl.int64)
    h = tl.load(hist_ptr + pid * 256 + offs)
    lo = tl.load(lo_ptr + pid)
    if DIGIT == 0:
        krem = tl.full((), k, tl.int64)
    else:
        krem = tl.load(krem_ptr + pid)
    csum = tl.cumsum(h.to(tl.int64), axis=0)
    ge = csum >= krem
    d = tl.min(tl.where(ge, offs, 256), axis=0)
    before = tl.sum(tl.where(offs < d, h.to(tl.int64), 0), axis=0)
    new_krem = krem - before
    width = 1 << (8 * (NBYTES - 1 - DIGIT))
    if DIGIT == 0:
        new_lo = d * width
    else:
        new_lo = lo + d * width
    tl.store(lo_ptr + pid, new_lo)
    tl.store(krem_ptr + pid, new_krem)
    tl.store(hist_ptr + pid * 256 + offs, tl.zeros_like(h))
    if DIGIT == NBYTES - 1:
        tl.store(vkey_ptr + pid, new_lo)


@triton.jit
def _radix_lesseq_kernel(
    inp_ptr,
    shape_ptr,
    stride_ptr,
    less_ptr,
    eq_ptr,
    vkey_ptr,
    D,
    S,
    G,
    CB: tl.constexpr,
    IS_FP: tl.constexpr,
    W: tl.constexpr,
    UP16: tl.constexpr,
    DIM: tl.constexpr,
    RANK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    s = pid // G
    c = pid % G
    base = _decompose(s, shape_ptr, stride_ptr, DIM, RANK)
    sd = tl.load(stride_ptr + DIM).to(tl.int64)
    vkey = tl.load(vkey_ptr + s)
    offs = c * CB + tl.arange(0, CB).to(tl.int64)
    mask = offs < D
    x = tl.load(inp_ptr + base + offs * sd, mask=mask, other=0)
    if UP16:
        x = x.to(tl.float32)
    key = _to_key(x, IS_FP, W)
    if W == 32:
        less = tl.sum(
            tl.where(mask & ((key ^ _INT_MIN32) < (vkey ^ _INT_MIN32)), 1, 0), axis=0
        )
    else:
        less = tl.sum(
            tl.where(mask & ((key ^ _INT_MIN64) < (vkey ^ _INT_MIN64)), 1, 0), axis=0
        )
    eq = tl.sum(tl.where(mask & (key == vkey), 1, 0), axis=0)
    tl.store(less_ptr + pid, less)
    tl.store(eq_ptr + pid, eq)


@triton.jit
def _radix_index_kernel(
    inp_ptr,
    shape_ptr,
    stride_ptr,
    vkey_ptr,
    less_ptr,
    eq_ptr,
    out_v_ptr,
    out_i_ptr,
    D,
    S,
    G,
    k,
    CB: tl.constexpr,
    IS_FP: tl.constexpr,
    W: tl.constexpr,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    DIM: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    base = _decompose(pid, shape_ptr, stride_ptr, DIM, RANK)
    sd = tl.load(stride_ptr + DIM).to(tl.int64)
    vkey = tl.load(vkey_ptr + pid)
    total_less = tl.full((), 0, tl.int64)
    for start in tl.range(0, G, BLOCK):
        offs_l = start + tl.arange(0, BLOCK).to(tl.int64)
        m = offs_l < G
        lv = tl.load(less_ptr + pid * G + offs_l, mask=m, other=0)
        total_less += tl.sum(lv.to(tl.int64), axis=0)
    r = k - total_less
    cstar = tl.full((), 0, tl.int64)
    local_r = tl.full((), 0, tl.int64)
    p = tl.full((), 0, tl.int64)
    for c in tl.range(0, G):
        e = tl.load(eq_ptr + pid * G + c).to(tl.int64)
        hit = (p < r) & (r <= p + e)
        cstar = tl.where(hit, c, cstar)
        local_r = tl.where(hit, r - p, local_r)
        p = p + e
    offs = cstar * CB + tl.arange(0, CB).to(tl.int64)
    mask = offs < D
    x = tl.load(inp_ptr + base + offs * sd, mask=mask, other=0)
    if IS_FP16 or IS_BF16:
        x = x.to(tl.float32)
    key = _to_key(x, IS_FP, W)
    eq = (key == vkey) & mask
    csum = tl.cumsum(eq.to(tl.int32), axis=0).to(tl.int64)
    cand = tl.where(eq & (csum == local_r), offs, _BIG)
    idx = tl.min(cand, axis=0)
    v = _unmap(vkey, IS_FP, W)
    if IS_FP16:
        v = v.to(tl.float16)
    elif IS_BF16:
        v = v.to(tl.bfloat16)
    tl.store(out_v_ptr + pid, v)
    tl.store(out_i_ptr + pid, idx)


# ---------------------------------------------------------------------------
# host wrapper
# ---------------------------------------------------------------------------
_scratch = {}
_meta = {}


def _scalar(v):
    if isinstance(v, torch.Tensor):
        return v.item()
    return v


def _get_meta(shape, stride, device):
    key = (shape, stride, device)
    t = _meta.get(key)
    if t is None:
        t = (
            torch.tensor(shape, dtype=torch.int64, device=device),
            torch.tensor(stride, dtype=torch.int64, device=device),
        )
        _meta[key] = t
    return t


def _get_scratch(key, factory):
    buf = _scratch.get(key)
    if buf is None:
        buf = factory()
        _scratch[key] = buf
    return buf


def kthvalue(input, k, dim=-1, keepdim=False):
    k = int(_scalar(k))
    if dim is None:
        dim = -1
    dim = int(_scalar(dim))
    keepdim = bool(_scalar(keepdim))
    inp = input
    rank = inp.dim()
    if dim < 0:
        dim += rank
    if dim < 0 or dim >= rank:
        raise IndexError(
            f"Dimension out of range (expected to be in range of [{-rank}, {rank - 1}], but got {dim})"
        )
    D = inp.shape[dim]
    if D == 0:
        raise IndexError(
            f"kthvalue(): Expected reduction dim {dim} to have non-zero size."
        )
    if k < 1 or k > D:
        raise RuntimeError(
            f"kthvalue(): selected number k out of range for dimension {dim}"
        )
    out_shape = list(inp.shape)
    if keepdim:
        out_shape[dim] = 1
    else:
        del out_shape[dim]
    values = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
    indices = torch.empty(out_shape, dtype=torch.int64, device=inp.device)
    S = inp.numel() // D
    shape_t, stride_t = _get_meta(tuple(inp.shape), tuple(inp.stride()), inp.device)
    dt = inp.dtype
    is_fp = dt.is_floating_point
    if D == 1:
        _kth_d1_kernel[(S,)](
            inp, shape_t, stride_t, values, indices, DIM=dim, RANK=rank, num_warps=1
        )
        return values, indices
    if dt in (torch.float64, torch.int64):
        d1max = _D1_MAX_64
    else:
        d1max = _D1_MAX
    if D <= d1max:
        BLOCK = 1
        while BLOCK < D:
            BLOCK <<= 1
        if dt == torch.float32:
            sent, up16 = _HF_FLT_MAX, False
        elif dt == torch.float64:
            sent, up16 = _HF_DBL_MAX, False
        elif dt == torch.float16:
            sent, up16 = _HF_HALF_MAX, True
        elif dt == torch.bfloat16:
            sent, up16 = _HF_BF16_MAX, True
        elif dt == torch.int32:
            sent, up16 = _HF_I32_MAX, False
        elif dt == torch.int64:
            sent, up16 = _HF_I64_MAX, False
        else:
            sent, up16 = int(torch.iinfo(dt).max), False
        if BLOCK <= 128:
            nw = 1
        elif BLOCK <= 512:
            nw = 2
        elif BLOCK <= 2048:
            nw = 4
        else:
            nw = 8
        if k <= _SMALL_K:
            _kth_smallk_kernel[(S,)](
                inp,
                shape_t,
                stride_t,
                values,
                indices,
                D,
                k,
                SENT=sent,
                IS_FP=is_fp,
                UPCAST16=up16,
                LAST_TIE=(D <= 4),
                BLOCK=BLOCK,
                DIM=dim,
                RANK=rank,
                num_warps=nw,
            )
        else:
            _kth_single_kernel[(S,)](
                inp,
                shape_t,
                stride_t,
                values,
                indices,
                D,
                k,
                SENT=sent,
                IS_FP=is_fp,
                UPCAST16=up16,
                BLOCK=BLOCK,
                DIM=dim,
                RANK=rank,
                num_warps=nw,
            )
        return values, indices
    # ---- radix path ----
    if dt == torch.float32:
        is_fp, w, nbytes, is_fp16, is_bf16 = True, 32, 4, False, False
    elif dt == torch.int32:
        is_fp, w, nbytes, is_fp16, is_bf16 = False, 32, 4, False, False
    elif dt == torch.float64:
        is_fp, w, nbytes, is_fp16, is_bf16 = True, 64, 8, False, False
    elif dt == torch.int64:
        is_fp, w, nbytes, is_fp16, is_bf16 = False, 64, 8, False, False
    elif dt == torch.float16:
        is_fp, w, nbytes, is_fp16, is_bf16 = True, 32, 4, True, False
    elif dt == torch.bfloat16:
        is_fp, w, nbytes, is_fp16, is_bf16 = True, 32, 4, False, True
    else:
        raise NotImplementedError("kthvalue: unsupported dtype %s" % dt)
    CB = _CB
    G = (D + CB - 1) // CB
    kt = torch.int32 if w == 32 else torch.int64
    dev = inp.device
    skey = ("radix", dev, S, G, int(kt == torch.int32))
    hist, lo, krem, vkey, less, eq = _get_scratch(
        skey,
        lambda: (
            torch.zeros(S * 256, dtype=torch.int32, device=dev),
            torch.zeros(S, dtype=kt, device=dev),
            torch.empty(S, dtype=torch.int64, device=dev),
            torch.empty(S, dtype=kt, device=dev),
            torch.empty(S * G, dtype=torch.int32, device=dev),
            torch.empty(S * G, dtype=torch.int32, device=dev),
        ),
    )
    grid_c = (S * G,)
    grid_s = (S,)
    for t in range(nbytes):
        _radix_count_kernel[grid_c](
            inp,
            shape_t,
            stride_t,
            hist,
            lo,
            D,
            S,
            G,
            DIGIT=t,
            NBYTES=nbytes,
            CB=CB,
            IS_FP=is_fp,
            W=w,
            UP16=(is_fp16 or is_bf16),
            DIM=dim,
            RANK=rank,
            num_warps=4,
        )
        _radix_narrow_kernel[grid_s](
            hist, lo, krem, vkey, S, k, DIGIT=t, NBYTES=nbytes, num_warps=1
        )
    _radix_lesseq_kernel[grid_c](
        inp,
        shape_t,
        stride_t,
        less,
        eq,
        vkey,
        D,
        S,
        G,
        CB=CB,
        IS_FP=is_fp,
        W=w,
        UP16=(is_fp16 or is_bf16),
        DIM=dim,
        RANK=rank,
        num_warps=4,
    )
    _radix_index_kernel[grid_s](
        inp,
        shape_t,
        stride_t,
        vkey,
        less,
        eq,
        values,
        indices,
        D,
        S,
        G,
        k,
        CB=CB,
        IS_FP=is_fp,
        W=w,
        IS_FP16=is_fp16,
        IS_BF16=is_bf16,
        DIM=dim,
        RANK=rank,
        num_warps=4,
        BLOCK=256,
    )
    return values, indices
