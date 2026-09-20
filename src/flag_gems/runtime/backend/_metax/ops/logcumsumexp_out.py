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


_TL_DT = {
    torch.float32: tl.float32,
    torch.float64: tl.float64,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


@triton.jit
def _lce_combine(m1, s1, m2, s2):
    m = tl.maximum(m1, m2)
    s = s1 * tl.exp(m1 - m) + s2 * tl.exp(m2 - m)
    return m, s


@triton.jit
def _lce_add(a, b):
    return a + b


# ---------------- single-kernel chunked scan (fp32 compute) ----------------
# tile (CHUNK_C, BLOCK_P); scan along axis 0 (the cumsum dim c)
# offsets = row*ROW_STRIDE + (p0 + arange(BLOCK_P))*STRIDE_P + (c0 + arange(CHUNK_C))*STRIDE_SCAN
@triton.jit
def _lce_single(
    x_ptr,
    o_ptr,
    C,
    P,
    ROW_STRIDE,
    STRIDE_P,
    STRIDE_SCAN,
    CHUNK_C: tl.constexpr,
    BLOCK_P: tl.constexpr,
    OUT_DT: tl.constexpr,
    IN_DT: tl.constexpr,
    NO_MASK: tl.constexpr,
):
    pid_p = tl.program_id(0)
    row = tl.program_id(1)
    p0 = pid_p * BLOCK_P
    pp = p0 + tl.arange(0, BLOCK_P)
    p_mask = pp < P
    p_off = row * ROW_STRIDE + pp * STRIDE_P
    idx = tl.arange(0, CHUNK_C)
    m_carry = tl.full([BLOCK_P], float("-inf"), dtype=tl.float32)
    s_carry = tl.zeros([BLOCK_P], dtype=tl.float32)
    for c0 in range(0, C, CHUNK_C):
        cc = c0 + idx
        ptrs = p_off[None, :] + cc[:, None] * STRIDE_SCAN
        if NO_MASK:
            x = tl.load(x_ptr + ptrs).to(IN_DT).to(tl.float32)
            m_elem = x
            s_elem = tl.full([CHUNK_C, BLOCK_P], 1.0, dtype=tl.float32)
        else:
            ok = (cc < C)[:, None] & p_mask[None, :]
            x = (
                tl.load(x_ptr + ptrs, mask=ok, other=float("-inf"))
                .to(IN_DT)
                .to(tl.float32)
            )
            m_elem = tl.where(ok, x, float("-inf"))
            s_elem = tl.where(ok, 1.0, 0.0)
        scan_m, scan_s = tl.associative_scan(
            (m_elem, s_elem), axis=0, combine_fn=_lce_combine
        )
        M = tl.maximum(m_carry[None, :], scan_m)
        S = s_carry[None, :] * tl.exp(m_carry[None, :] - M) + scan_s * tl.exp(
            scan_m - M
        )
        if NO_MASK:
            tl.store(o_ptr + ptrs, (M + tl.log(S)).to(OUT_DT))
        else:
            tl.store(o_ptr + ptrs, (M + tl.log(S)).to(OUT_DT), mask=ok)
        last = idx[:, None] == (CHUNK_C - 1)
        m_carry = tl.sum(tl.where(last, M, 0.0), axis=0)
        s_carry = tl.sum(tl.where(last, S, 0.0), axis=0)


# single-chunk variant with no carry chain: only valid when C <= CHUNK_C
@triton.jit
def _lce_single_nc(
    x_ptr,
    o_ptr,
    C,
    P,
    ROW_STRIDE,
    STRIDE_P,
    STRIDE_SCAN,
    CHUNK_C: tl.constexpr,
    BLOCK_P: tl.constexpr,
    OUT_DT: tl.constexpr,
    IN_DT: tl.constexpr,
    NO_MASK: tl.constexpr,
):
    pid_p = tl.program_id(0)
    row = tl.program_id(1)
    p0 = pid_p * BLOCK_P
    pp = p0 + tl.arange(0, BLOCK_P)
    p_mask = pp < P
    p_off = row * ROW_STRIDE + pp * STRIDE_P
    idx = tl.arange(0, CHUNK_C)
    ptrs = p_off[None, :] + idx[:, None] * STRIDE_SCAN
    if NO_MASK:
        x = tl.load(x_ptr + ptrs).to(IN_DT).to(tl.float32)
        m_elem = x
        s_elem = tl.full([CHUNK_C, BLOCK_P], 1.0, dtype=tl.float32)
    else:
        ok = (idx < C)[:, None] & p_mask[None, :]
        x = tl.load(x_ptr + ptrs, mask=ok, other=float("-inf")).to(IN_DT).to(tl.float32)
        m_elem = tl.where(ok, x, float("-inf"))
        s_elem = tl.where(ok, 1.0, 0.0)
    scan_m, scan_s = tl.associative_scan(
        (m_elem, s_elem), axis=0, combine_fn=_lce_combine
    )
    if NO_MASK:
        tl.store(o_ptr + ptrs, (scan_m + tl.log(scan_s)).to(OUT_DT))
    else:
        tl.store(o_ptr + ptrs, (scan_m + tl.log(scan_s)).to(OUT_DT), mask=ok)


# additive-scan carry kernel: per-chunk max reduction, exp normalization, pure-add scan
@triton.jit
def _lce_single_add(
    x_ptr,
    o_ptr,
    C,
    P,
    ROW_STRIDE,
    STRIDE_P,
    STRIDE_SCAN,
    CHUNK_C: tl.constexpr,
    BLOCK_P: tl.constexpr,
    OUT_DT: tl.constexpr,
    IN_DT: tl.constexpr,
    NO_MASK: tl.constexpr,
):
    pid_p = tl.program_id(0)
    row = tl.program_id(1)
    p0 = pid_p * BLOCK_P
    pp = p0 + tl.arange(0, BLOCK_P)
    p_mask = pp < P
    p_off = row * ROW_STRIDE + pp * STRIDE_P
    idx = tl.arange(0, CHUNK_C)
    m_carry = tl.full([BLOCK_P], float("-inf"), dtype=tl.float32)
    s_carry = tl.zeros([BLOCK_P], dtype=tl.float32)
    for c0 in range(0, C, CHUNK_C):
        cc = c0 + idx
        ptrs = p_off[None, :] + cc[:, None] * STRIDE_SCAN
        if NO_MASK:
            x = tl.load(x_ptr + ptrs).to(IN_DT).to(tl.float32)
        else:
            ok = (cc < C)[:, None] & p_mask[None, :]
            x = (
                tl.load(x_ptr + ptrs, mask=ok, other=float("-inf"))
                .to(IN_DT)
                .to(tl.float32)
            )
        m_chunk = tl.max(x, axis=0)
        e = tl.exp(x - m_chunk[None, :])
        if not NO_MASK:
            e = tl.where(ok, e, 0.0)
        s_pref = tl.associative_scan(e, axis=0, combine_fn=_lce_add)
        M = tl.maximum(m_carry[None, :], m_chunk[None, :])
        S = s_carry[None, :] * tl.exp(m_carry[None, :] - M) + s_pref * tl.exp(
            m_chunk[None, :] - M
        )
        if NO_MASK:
            tl.store(o_ptr + ptrs, (M + tl.log(S)).to(OUT_DT))
        else:
            tl.store(o_ptr + ptrs, (M + tl.log(S)).to(OUT_DT), mask=ok)
        last = idx[:, None] == (CHUNK_C - 1)
        m_carry = tl.sum(tl.where(last, M, 0.0), axis=0)
        s_carry = tl.sum(tl.where(last, S, 0.0), axis=0)


# ---------------- two-pass (fp32 compute), 2D row scans (cols == 1) ----------------
@triton.jit
def _lce_agg(
    x_ptr,
    m_ptr,
    s_ptr,
    C,
    NCHUNK,
    CHUNK_C: tl.constexpr,
    IN_DT: tl.constexpr,
    NO_MASK: tl.constexpr,
):
    pid_c = tl.program_id(0)
    row = tl.program_id(1)
    c0 = pid_c * CHUNK_C
    idx = tl.arange(0, CHUNK_C)
    cc = c0 + idx
    ptrs = row * C + cc
    if NO_MASK:
        x = tl.load(x_ptr + ptrs).to(IN_DT).to(tl.float32)
        m = tl.max(x, axis=0)
        s = tl.sum(tl.exp(x - m), axis=0)
    else:
        ok = cc < C
        x = tl.load(x_ptr + ptrs, mask=ok, other=float("-inf")).to(IN_DT).to(tl.float32)
        m = tl.max(x, axis=0)
        s = tl.sum(tl.where(ok, tl.exp(x - m), 0.0), axis=0)
    tl.store(m_ptr + row * NCHUNK + pid_c, m)
    tl.store(s_ptr + row * NCHUNK + pid_c, s)


@triton.jit
def _lce_apply(
    x_ptr,
    o_ptr,
    m_ptr,
    pm_ptr,
    ps_ptr,
    C,
    NCHUNK,
    CHUNK_C: tl.constexpr,
    OUT_DT: tl.constexpr,
    IN_DT: tl.constexpr,
    NO_MASK: tl.constexpr,
):
    # additive-scan apply: e = exp(x - m_agg), prefix-sum of e (pure add scan),
    # then merge the exclusive carry via M = max(m_carry, m_agg).
    pid_c = tl.program_id(0)
    row = tl.program_id(1)
    m_agg = tl.load(m_ptr + row * NCHUNK + pid_c)
    m_carry = tl.load(pm_ptr + row * NCHUNK + pid_c)
    s_carry = tl.load(ps_ptr + row * NCHUNK + pid_c)
    c0 = pid_c * CHUNK_C
    idx = tl.arange(0, CHUNK_C)
    cc = c0 + idx
    ptrs = row * C + cc
    if NO_MASK:
        x = tl.load(x_ptr + ptrs).to(IN_DT).to(tl.float32)
        e = tl.exp(x - m_agg)
    else:
        ok = cc < C
        x = tl.load(x_ptr + ptrs, mask=ok, other=float("-inf")).to(IN_DT).to(tl.float32)
        e = tl.where(ok, tl.exp(x - m_agg), 0.0)
    s_pref = tl.associative_scan(e, axis=0, combine_fn=_lce_add)
    M = tl.maximum(m_carry, m_agg)
    S = s_carry * tl.exp(m_carry - M) + s_pref * tl.exp(m_agg - M)
    if NO_MASK:
        tl.store(o_ptr + ptrs, (M + tl.log(S)).to(OUT_DT))
    else:
        tl.store(o_ptr + ptrs, (M + tl.log(S)).to(OUT_DT), mask=ok)


# exclusive prefix over each row's chunk aggregates (tiny kernel)
@triton.jit
def _lce_exclprefix(m_ptr, s_ptr, pm_ptr, ps_ptr, NCHUNK):
    row = tl.program_id(0)
    m_carry = tl.full([], float("-inf"), dtype=tl.float32)
    s_carry = tl.zeros([], dtype=tl.float32)
    base = row * NCHUNK
    for j in range(NCHUNK):
        mj = tl.load(m_ptr + base + j)
        sj = tl.load(s_ptr + base + j)
        tl.store(pm_ptr + base + j, m_carry)
        tl.store(ps_ptr + base + j, s_carry)
        m_new = tl.maximum(m_carry, mj)
        s_carry = s_carry * tl.exp(m_carry - m_new) + sj * tl.exp(mj - m_new)
        m_carry = m_new


# ---------------- fp64 two-pass: fp32 exp/log + fp64 accumulation ----------------
@triton.jit
def _lce_agg64f(
    x_ptr,
    m_ptr,
    s_ptr,
    C,
    NCHUNK,
    CHUNK_C: tl.constexpr,
):
    pid_c = tl.program_id(0)
    row = tl.program_id(1)
    c0 = pid_c * CHUNK_C
    idx = tl.arange(0, CHUNK_C)
    cc = c0 + idx
    ok = cc < C
    ptrs = row * C + cc
    x = tl.load(x_ptr + ptrs, mask=ok, other=float("-inf"))
    m = tl.max(x, axis=0)
    e = tl.exp((x - m).to(tl.float32)).to(tl.float64)
    s = tl.sum(tl.where(ok, e, 0.0), axis=0)
    tl.store(m_ptr + row * NCHUNK + pid_c, m)
    tl.store(s_ptr + row * NCHUNK + pid_c, s)


@triton.jit
def _lce_apply64f(
    x_ptr,
    o_ptr,
    m_ptr,
    s_ptr,
    C,
    NCHUNK,
    CHUNK_C: tl.constexpr,
    OUT_DT: tl.constexpr,
):
    pid_c = tl.program_id(0)
    row = tl.program_id(1)
    m_carry = tl.full([], float("-inf"), dtype=tl.float64)
    s_carry = tl.zeros([], dtype=tl.float64)
    base_agg = row * NCHUNK
    for j in range(pid_c):
        mj = tl.load(m_ptr + base_agg + j)
        sj = tl.load(s_ptr + base_agg + j)
        m_new = tl.maximum(m_carry, mj)
        e1 = tl.exp((m_carry - m_new).to(tl.float32)).to(tl.float64)
        e2 = tl.exp((mj - m_new).to(tl.float32)).to(tl.float64)
        s_carry = s_carry * e1 + sj * e2
        m_carry = m_new
    m = m_carry
    s = s_carry
    c0 = pid_c * CHUNK_C
    for k in tl.static_range(CHUNK_C):
        ck = c0 + k
        okk = ck < C
        xk = tl.load(x_ptr + row * C + ck, mask=okk, other=float("-inf"))
        m_new = tl.maximum(m, xk)
        e1 = tl.exp((m - m_new).to(tl.float32)).to(tl.float64)
        e2 = tl.exp((xk - m_new).to(tl.float32)).to(tl.float64)
        s = s * e1 + e2
        m = m_new
        tl.store(
            o_ptr + row * C + ck,
            (m + tl.log(s.to(tl.float32)).to(tl.float64)).to(OUT_DT),
            mask=okk,
        )


# ---------------- fp64 exact serial (fallback for non-2D / cols>1 fp64) ----------------
@triton.jit
def _lce_serial64(
    x_ptr,
    o_ptr,
    C,
    P,
    ROW_STRIDE,
    STRIDE_P,
    STRIDE_SCAN,
    BLOCK_P: tl.constexpr,
    OUT_DT: tl.constexpr,
):
    pid_p = tl.program_id(0)
    row = tl.program_id(1)
    p0 = pid_p * BLOCK_P
    pp = p0 + tl.arange(0, BLOCK_P)
    p_mask = pp < P
    base = row * ROW_STRIDE + pp * STRIDE_P
    m = tl.full([BLOCK_P], float("-inf"), dtype=tl.float64)
    s = tl.zeros([BLOCK_P], dtype=tl.float64)
    for c in range(C):
        x = tl.load(x_ptr + base + c * STRIDE_SCAN, mask=p_mask, other=float("-inf"))
        m_new = tl.maximum(m, x)
        s = s * tl.exp(m - m_new) + tl.exp(x - m_new)
        m = m_new
        tl.store(
            o_ptr + base + c * STRIDE_SCAN, (m + tl.log(s)).to(OUT_DT), mask=p_mask
        )


# ---------------- generic fallback for non-contiguous inputs ----------------
# Decomposes (row, col) into original dim indices and computes base offsets in-kernel.
@triton.jit
def _lce_generic32(
    x_ptr,
    o_ptr,
    C,
    ROWS,
    COLS,
    STRIDE_C,
    PRE0,
    PRE1,
    PRE2,
    PRE3,
    POST0,
    POST1,
    POST2,
    POST3,
    ST0,
    ST1,
    ST2,
    ST3,
    ST4,
    ST5,
    ST6,
    ST7,
    N_PRE: tl.constexpr,
    N_POST: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
    OUT_DT: tl.constexpr,
    IN_DT: tl.constexpr,
):
    pid_c = tl.program_id(0)
    row = tl.program_id(1)
    col = pid_c * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    col_mask = col < COLS
    # col decomposition (row-major over post dims)
    cc = col
    base_col = tl.zeros([BLOCK_COLS], dtype=tl.int64)
    for k in tl.static_range(4):
        if k < N_POST:
            sz = tl.where(
                k == 0, POST0, tl.where(k == 1, POST1, tl.where(k == 2, POST2, POST3))
            )
            st = tl.where(
                k == 0, ST4, tl.where(k == 1, ST5, tl.where(k == 2, ST6, ST7))
            )
            base_col += (cc % sz) * st
            cc = cc // sz
    # row decomposition
    rr = row
    base_row = 0
    for k in tl.static_range(4):
        if k < N_PRE:
            sz = tl.where(
                k == 0, PRE0, tl.where(k == 1, PRE1, tl.where(k == 2, PRE2, PRE3))
            )
            st = tl.where(
                k == 0, ST0, tl.where(k == 1, ST1, tl.where(k == 2, ST2, ST3))
            )
            base_row += (rr % sz) * st
            rr = rr // sz
    base = base_row + base_col
    m = tl.full([BLOCK_COLS], float("-inf"), dtype=tl.float32)
    s = tl.zeros([BLOCK_COLS], dtype=tl.float32)
    for c in range(C):
        x = (
            tl.load(x_ptr + base + c * STRIDE_C, mask=col_mask, other=float("-inf"))
            .to(IN_DT)
            .to(tl.float32)
        )
        m_new = tl.maximum(m, x)
        s = s * tl.exp(m - m_new) + tl.exp(x - m_new)
        m = m_new
        tl.store(o_ptr + base + c * STRIDE_C, (m + tl.log(s)).to(OUT_DT), mask=col_mask)


def logcumsumexp_out(inp, dim=1, *, dtype=None, out):
    nd = inp.dim()
    dim = dim % nd
    shp = inp.shape
    rows = 1
    for s in shp[:dim]:
        rows *= s
    C = shp[dim]
    cols = 1
    for s in shp[dim + 1 :]:
        cols *= s

    comp = dtype if dtype is not None else inp.dtype
    out_dt = _TL_DT[out.dtype]

    if comp == torch.float64:
        if cols == 1:
            CHUNK_C = 128 if C <= 4096 else (256 if C <= 65536 else 512)
            nchunk = triton.cdiv(C, CHUNK_C)
            m = torch.empty((rows, nchunk), device=inp.device, dtype=torch.float64)
            s = torch.empty((rows, nchunk), device=inp.device, dtype=torch.float64)
            grid = (nchunk, rows)
            _lce_agg64f[grid](inp, m, s, C, nchunk, CHUNK_C=CHUNK_C, num_warps=4)
            _lce_apply64f[grid](
                inp, out, m, s, C, nchunk, CHUNK_C=CHUNK_C, OUT_DT=out_dt, num_warps=4
            )
        else:
            grid = (triton.cdiv(cols, 64), rows)
            _lce_serial64[grid](
                inp,
                out,
                C,
                cols,
                C * cols,
                1,
                cols,
                BLOCK_P=64,
                OUT_DT=out_dt,
                num_warps=4,
            )
        return out

    in_dt = _TL_DT.get(inp.dtype, tl.float32)
    if not inp.is_contiguous():
        pre = shp[:dim]
        post = shp[dim + 1 :]
        strides = inp.stride()

        def _pad(seq, n):
            return tuple(seq) + (1,) * (n - len(seq))

        PRE = _pad(pre, 4)
        POST = _pad(post, 4)
        ST = _pad(strides[:dim], 4) + _pad(strides[dim + 1 :], 4)
        grid = (triton.cdiv(cols, 64), rows)
        _lce_generic32[grid](
            inp,
            out,
            C,
            rows,
            cols,
            strides[dim],
            PRE[0],
            PRE[1],
            PRE[2],
            PRE[3],
            POST[0],
            POST[1],
            POST[2],
            POST[3],
            ST[0],
            ST[1],
            ST[2],
            ST[3],
            ST[4],
            ST[5],
            ST[6],
            ST[7],
            N_PRE=len(pre),
            N_POST=len(post),
            BLOCK_COLS=64,
            OUT_DT=out_dt,
            IN_DT=in_dt,
            num_warps=4,
        )
        return out

    if cols == 1:
        if (C > 16384 or (rows < 512 and C >= 1024)) and rows * C > 262144:
            CHUNK_C = 1024 if C >= 65536 else 2048
            nchunk = triton.cdiv(C, CHUNK_C)
            no_mask = C % CHUNK_C == 0
            m = torch.empty((rows, nchunk), device=inp.device, dtype=torch.float32)
            s = torch.empty((rows, nchunk), device=inp.device, dtype=torch.float32)
            pm = torch.empty((rows, nchunk), device=inp.device, dtype=torch.float32)
            ps = torch.empty((rows, nchunk), device=inp.device, dtype=torch.float32)
            grid = (nchunk, rows)
            _lce_agg[grid](
                inp,
                m,
                s,
                C,
                nchunk,
                CHUNK_C=CHUNK_C,
                IN_DT=in_dt,
                NO_MASK=no_mask,
                num_warps=2,
            )
            _lce_exclprefix[(rows,)](m, s, pm, ps, nchunk, num_warps=2)
            _lce_apply[grid](
                inp,
                out,
                m,
                pm,
                ps,
                C,
                nchunk,
                CHUNK_C=CHUNK_C,
                OUT_DT=out_dt,
                IN_DT=in_dt,
                NO_MASK=no_mask,
                num_warps=2,
            )
        else:
            if C <= 64:
                _lce_single_nc[(triton.cdiv(rows, 16), 1)](
                    inp,
                    out,
                    C,
                    rows,
                    0,
                    C,
                    1,
                    CHUNK_C=64,
                    BLOCK_P=16,
                    OUT_DT=out_dt,
                    IN_DT=in_dt,
                    NO_MASK=(C == 64 and rows % 16 == 0),
                    num_warps=4,
                )
            elif C <= 256:
                _lce_single_nc[(triton.cdiv(rows, 8), 1)](
                    inp,
                    out,
                    C,
                    rows,
                    0,
                    C,
                    1,
                    CHUNK_C=256,
                    BLOCK_P=8,
                    OUT_DT=out_dt,
                    IN_DT=in_dt,
                    NO_MASK=(C == 256 and rows % 8 == 0),
                    num_warps=4,
                )
            elif C <= 1024:
                _lce_single_nc[(triton.cdiv(rows, 4), 1)](
                    inp,
                    out,
                    C,
                    rows,
                    0,
                    C,
                    1,
                    CHUNK_C=1024,
                    BLOCK_P=4,
                    OUT_DT=out_dt,
                    IN_DT=in_dt,
                    NO_MASK=(C == 1024 and rows % 4 == 0),
                    num_warps=4,
                )
            else:
                _lce_single_add[(triton.cdiv(rows, 4), 1)](
                    inp,
                    out,
                    C,
                    rows,
                    0,
                    C,
                    1,
                    CHUNK_C=128,
                    BLOCK_P=4,
                    OUT_DT=out_dt,
                    IN_DT=in_dt,
                    NO_MASK=(C % 128 == 0 and rows % 4 == 0),
                    num_warps=2,
                )
    else:
        grid = (triton.cdiv(cols, 64), rows)
        _lce_single[grid](
            inp,
            out,
            C,
            cols,
            C * cols,
            1,
            cols,
            CHUNK_C=64,
            BLOCK_P=64,
            OUT_DT=out_dt,
            IN_DT=in_dt,
            NO_MASK=(C % 64 == 0 and cols % 64 == 0),
            num_warps=4,
        )
    return out
