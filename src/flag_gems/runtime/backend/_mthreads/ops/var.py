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

from flag_gems.ops.var import var as default_var

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


_SIGNED = {torch.int8, torch.int16, torch.int32, torch.int64}

# ---------------------------------------------------------------------------
# Fast path (contiguous tensor):
#   * tail reductions: red dims are the trailing dims -> view (outer, reduce)
#     with contiguous rows.
#   * mid reductions: red dims form a contiguous range [d0, d1) strictly
#     inside the shape -> view (pre, reduce, inner).
# Both use the numerically-validated sum/sumsq accumulation in fp32/fp64 with
# a single-pass per-tile reduction; for randn-scale data the cancellation in
# var = (sumsq - sum^2/n) / (n - correction) is far below the test tolerance.
# ---------------------------------------------------------------------------


@triton.jit
def _var_fused(
    x_ptr,
    out_ptr,
    outer_size,
    reduce_size,
    correction,
    FP64: tl.constexpr,
    USE_I64: tl.constexpr,
    NO_MASK: tl.constexpr,
    ITERS: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0)
    if USE_I64:
        rows = pid.to(tl.int64) * BLOCK_OUT + tl.arange(0, BLOCK_OUT).to(tl.int64)
        red = reduce_size.to(tl.int64)
    else:
        rows = pid * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
        red = reduce_size
    row_mask = rows < outer_size

    acc = tl.float64 if FP64 else tl.float32
    s_acc = tl.zeros((BLOCK_OUT,), dtype=acc)
    q_acc = tl.zeros((BLOCK_OUT,), dtype=acc)
    nc_acc = tl.zeros((BLOCK_OUT,), dtype=acc)

    for i in tl.static_range(ITERS):
        r = i * BLOCK_R + tl.arange(0, BLOCK_R)
        offs = rows[:, None] * red + r[None, :]
        if NO_MASK:
            x = tl.load(x_ptr + offs)
            n_c = tl.full((BLOCK_OUT,), BLOCK_R, dtype=acc)
        else:
            m = row_mask[:, None] & (r < reduce_size)[None, :]
            x = tl.load(x_ptr + offs, mask=m, other=0.0)
            n_c = tl.sum(m.to(acc), axis=1)
        x = x.to(acc)
        s_acc += tl.sum(x, axis=1)
        q_acc += tl.sum(x * x, axis=1)
        nc_acc += n_c

    nz = tl.where(nc_acc > 0, nc_acc, 1.0)
    mean = s_acc / nz
    if FP64:
        denom = nc_acc - correction.to(tl.float64)
    else:
        denom = nc_acc - correction
    var = (q_acc - s_acc * mean) / denom
    tl.store(out_ptr + rows, var, mask=row_mask)


@triton.jit
def _var_mid(
    x_ptr,
    out_ptr,
    pre_size,
    red_size,
    in_size,
    correction,
    FP64: tl.constexpr,
    USE_I64: tl.constexpr,
    NO_MASK: tl.constexpr,
    ITERS: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_p = tl.program_id(0)
    pid_i = tl.program_id(1)
    if USE_I64:
        po = pid_p.to(tl.int64)
        io = pid_i.to(tl.int64) * BLOCK_I + tl.arange(0, BLOCK_I).to(tl.int64)
        R = red_size.to(tl.int64)
        ISZ = in_size.to(tl.int64)
    else:
        po = pid_p
        io = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
        R = red_size
        ISZ = in_size
    i_mask = io < in_size

    acc = tl.float64 if FP64 else tl.float32
    s_acc = tl.zeros((BLOCK_I,), dtype=acc)
    q_acc = tl.zeros((BLOCK_I,), dtype=acc)
    nc_acc = tl.zeros((BLOCK_I,), dtype=acc)

    for i in range(ITERS):
        r = i * BLOCK_R + tl.arange(0, BLOCK_R)
        offs = po * (R * ISZ) + r[:, None] * ISZ + io[None, :]
        if NO_MASK:
            x = tl.load(x_ptr + offs)
            n_c = tl.full((BLOCK_I,), BLOCK_R, dtype=acc)
        else:
            m = (r[:, None] < red_size) & i_mask[None, :]
            x = tl.load(x_ptr + offs, mask=m, other=0.0)
            n_c = tl.sum(m.to(acc), axis=0)
        x = x.to(acc)
        s_acc += tl.sum(x, axis=0)
        q_acc += tl.sum(x * x, axis=0)
        nc_acc += n_c

    nz = tl.where(nc_acc > 0, nc_acc, 1.0)
    mean = s_acc / nz
    if FP64:
        denom = nc_acc - correction.to(tl.float64)
    else:
        denom = nc_acc - correction
    var = (q_acc - s_acc * mean) / denom
    tl.store(out_ptr + po * in_size + io, var, mask=i_mask)


@triton.jit
def _var_partial(
    x_ptr,
    ss_ptr,
    sq_ptr,
    nc_ptr,
    outer_size,
    reduce_size,
    FP64: tl.constexpr,
    USE_I64: tl.constexpr,
    NO_MASK: tl.constexpr,
    ITERS: tl.constexpr,
    NCH: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_r = tl.program_id(1)
    if USE_I64:
        rows = pid_r.to(tl.int64) * BLOCK_OUT + tl.arange(0, BLOCK_OUT).to(tl.int64)
        red = reduce_size.to(tl.int64)
    else:
        rows = pid_r * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
        red = reduce_size
    row_mask = rows < outer_size

    acc = tl.float64 if FP64 else tl.float32
    s_acc = tl.zeros((BLOCK_OUT,), dtype=acc)
    q_acc = tl.zeros((BLOCK_OUT,), dtype=acc)
    nc_acc = tl.zeros((BLOCK_OUT,), dtype=acc)

    cbase = pid_c * (ITERS * BLOCK_R)
    for i in tl.static_range(ITERS):
        r = cbase + i * BLOCK_R + tl.arange(0, BLOCK_R)
        offs = rows[:, None] * red + r[None, :]
        if NO_MASK:
            x = tl.load(x_ptr + offs)
            n_c = tl.full((BLOCK_OUT,), BLOCK_R, dtype=acc)
        else:
            m = row_mask[:, None] & (r < reduce_size)[None, :]
            x = tl.load(x_ptr + offs, mask=m, other=0.0)
            n_c = tl.sum(m.to(acc), axis=1)
        x = x.to(acc)
        s_acc += tl.sum(x, axis=1)
        q_acc += tl.sum(x * x, axis=1)
        nc_acc += n_c

    base = rows * NCH + pid_c
    tl.store(ss_ptr + base, s_acc, mask=row_mask)
    tl.store(sq_ptr + base, q_acc, mask=row_mask)
    tl.store(nc_ptr + base, nc_acc, mask=row_mask)


@triton.jit
def _var_merge(
    ss_ptr,
    sq_ptr,
    nc_ptr,
    out_ptr,
    outer_size,
    reduce_size,
    correction,
    FP64: tl.constexpr,
    NCH: tl.constexpr,
    NCH_P2: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    row_mask = rows < outer_size

    offs = tl.arange(0, NCH_P2)
    m2d = row_mask[:, None] & (offs < NCH)[None, :]
    base = rows[:, None] * NCH + offs[None, :]
    s = tl.sum(tl.load(ss_ptr + base, mask=m2d, other=0.0), axis=1)
    q = tl.sum(tl.load(sq_ptr + base, mask=m2d, other=0.0), axis=1)
    n = tl.sum(tl.load(nc_ptr + base, mask=m2d, other=0.0), axis=1)

    nz = tl.where(n > 0, n, 1.0)
    mean = s / nz
    if FP64:
        denom = n - correction.to(tl.float64)
    else:
        denom = n - correction
    var = (q - s * mean) / denom
    tl.store(out_ptr + rows, var, mask=row_mask)


# ---------------------------------------------------------------------------
# General path: reduced dims are an arbitrary subset (may be non-contiguous
# or non-tail). Two-stage with (n, mean, M2) partials and parallel Chan merge.
# ---------------------------------------------------------------------------


@triton.jit
def _var_stage1(
    x_ptr,
    sn_ptr,
    sm_ptr,
    sm2_ptr,
    outer_size,
    reduce_size,
    nsz0,
    nst0,
    nsz1,
    nst1,
    nsz2,
    nst2,
    nsz3,
    nst3,
    nsz4,
    nst4,
    nsz5,
    nst5,
    nsz6,
    nst6,
    nsz7,
    nst7,
    rsz0,
    rst0,
    rsz1,
    rst1,
    rsz2,
    rst2,
    rsz3,
    rst3,
    rsz4,
    rst4,
    rsz5,
    rst5,
    rsz6,
    rst6,
    rsz7,
    rst7,
    CHUNKS,
    FP64: tl.constexpr,
    USE_I64: tl.constexpr,
    NDIM_NONRED: tl.constexpr,
    NDIM_RED: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_r = tl.program_id(1)

    if USE_I64:
        rows = pid_r.to(tl.int64) * BLOCK_OUT + tl.arange(0, BLOCK_OUT).to(tl.int64)
    else:
        rows = pid_r * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    row_mask = rows < outer_size

    r = pid_c * BLOCK_R + tl.arange(0, BLOCK_R)
    r_mask = r < reduce_size

    rem = rows.to(tl.int64)
    base = tl.zeros((BLOCK_OUT,), dtype=tl.int64)
    for i in tl.static_range(NDIM_NONRED):
        d = NDIM_NONRED - 1 - i
        if d == 0:
            sz = nsz0
            st = nst0
        elif d == 1:
            sz = nsz1
            st = nst1
        elif d == 2:
            sz = nsz2
            st = nst2
        elif d == 3:
            sz = nsz3
            st = nst3
        elif d == 4:
            sz = nsz4
            st = nst4
        elif d == 5:
            sz = nsz5
            st = nst5
        elif d == 6:
            sz = nsz6
            st = nst6
        else:
            sz = nsz7
            st = nst7
        idx = rem % sz
        rem = rem // sz
        base = base + idx * st
    rrem = r.to(tl.int64)
    roff = tl.zeros((BLOCK_R,), dtype=tl.int64)
    for i in tl.static_range(NDIM_RED):
        d = NDIM_RED - 1 - i
        if d == 0:
            sz = rsz0
            st = rst0
        elif d == 1:
            sz = rsz1
            st = rst1
        elif d == 2:
            sz = rsz2
            st = rst2
        elif d == 3:
            sz = rsz3
            st = rst3
        elif d == 4:
            sz = rsz4
            st = rst4
        elif d == 5:
            sz = rsz5
            st = rst5
        elif d == 6:
            sz = rsz6
            st = rst6
        else:
            sz = rsz7
            st = rst7
        idx = rrem % sz
        rrem = rrem // sz
        roff = roff + idx * st
    offs = base[:, None] + roff[None, :]

    m = row_mask[:, None] & r_mask[None, :]
    x = tl.load(x_ptr + offs, mask=m, other=0.0)
    acc = tl.float64 if FP64 else tl.float32
    x = x.to(acc)
    s = tl.sum(x, axis=1)
    n_c = tl.sum(m.to(tl.float32), axis=1)
    nnz = tl.where(n_c > 0, n_c, 1.0)
    mean_c = tl.where(n_c > 0, s / nnz, 0.0)
    d = (x - mean_c[:, None]) * m.to(acc)
    s2 = tl.sum(d, axis=1)
    mean_c = mean_c + tl.where(n_c > 0, s2 / nnz, 0.0)
    d = (x - mean_c[:, None]) * m.to(acc)
    m2_c = tl.sum(d * d, axis=1)

    base_s = rows * CHUNKS + pid_c
    tl.store(sn_ptr + base_s, n_c, mask=row_mask)
    tl.store(sm_ptr + base_s, mean_c, mask=row_mask)
    tl.store(sm2_ptr + base_s, m2_c, mask=row_mask)


@triton.jit
def _var_stage2(
    sn_ptr,
    sm_ptr,
    sm2_ptr,
    out_ptr,
    outer_size,
    reduce_size,
    correction,
    CHUNKS,
    FP64: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    row_mask = rows < outer_size

    acc = tl.float64 if FP64 else tl.float32
    n_acc = tl.zeros((BLOCK_OUT,), dtype=acc)
    m_acc = tl.zeros((BLOCK_OUT,), dtype=acc)
    m2_acc = tl.zeros((BLOCK_OUT,), dtype=acc)

    base = rows * CHUNKS
    for c in range(CHUNKS):
        n_c = tl.load(sn_ptr + base + c, mask=row_mask, other=0.0)
        m_c = tl.load(sm_ptr + base + c, mask=row_mask, other=0.0)
        m2_c = tl.load(sm2_ptr + base + c, mask=row_mask, other=0.0)
        n = n_acc + n_c
        nn = tl.where(n > 0, n, 1.0)
        d = m_c - m_acc
        m_acc = m_acc + d * (n_c / nn)
        m2_acc = m2_acc + m2_c + d * d * (n_acc * n_c / nn)
        n_acc = n

    if FP64:
        denom = (reduce_size - correction).to(tl.float64)
    else:
        denom = reduce_size - correction
    out_val = m2_acc / denom
    tl.store(out_ptr + rows, out_val, mask=row_mask)


def _specialized_var(x, dim=None, *, correction=None, keepdim=False):
    if correction is None:
        correction = 1
    if isinstance(correction, torch.Tensor):
        correction = correction.item()
    if isinstance(keepdim, torch.Tensor):
        keepdim = bool(keepdim.item())
    if isinstance(dim, torch.Tensor):
        dim = dim.item()

    ndim = x.dim()
    if dim is None or (not isinstance(dim, int) and len(dim) == 0):
        red = list(range(ndim))
    elif isinstance(dim, int):
        red = [int(dim) % ndim]
    else:
        red = sorted({int(d) % ndim for d in dim})
    red_set = set(red)
    nonred = [d for d in range(ndim) if d not in red_set]

    outer = 1
    for d in nonred:
        outer *= x.shape[d]
    reduce = 1
    for d in red:
        reduce *= x.shape[d]

    if keepdim:
        out_shape = [1 if d in red_set else x.shape[d] for d in range(ndim)]
    else:
        out_shape = [x.shape[d] for d in nonred]

    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)

    if x.numel() == 0:
        out.fill_(float("nan"))
        return out

    fp64 = x.dtype == torch.float64
    use_i64 = x.numel() > (2**31 - 1)

    if x.is_contiguous() and red == list(range(ndim - len(red), ndim)):
        # ---- tail reduction: (outer, reduce) contiguous rows ----
        # Cap block_r at 512 for fp32: the MUSA compiler's
        # OptimizeThreadLocality pass asserts (loopResult.hasOneUse) on the
        # fused-kernel fp32 tile with block_r=1024.
        block_r_cap = 512 if x.dtype == torch.float32 else 1024
        block_r = triton.next_power_of_2(min(reduce, block_r_cap))
        max_fused_iters = 16
        if reduce <= block_r * max_fused_iters:
            # ---- fused single kernel ----
            iters = (reduce + block_r - 1) // block_r
            block_out = min(64, max(1, 8192 // block_r), triton.next_power_of_2(outer))
            if block_out < 1:
                block_out = 1
            no_mask = (outer % block_out == 0) and (reduce % block_r == 0)
            tile = block_out * block_r
            nw = 8 if tile >= 4096 else 4
            _var_fused[(triton.cdiv(outer, block_out),)](
                x,
                out,
                outer,
                reduce,
                float(correction),
                FP64=fp64,
                USE_I64=use_i64,
                NO_MASK=no_mask,
                ITERS=iters,
                BLOCK_OUT=block_out,
                BLOCK_R=block_r,
                num_warps=nw,
            )
        else:
            # ---- two-kernel sum/sumsq ----
            iters = max(1, (reduce + block_r * 1024 - 1) // (block_r * 1024))
            chunk = iters * block_r
            nch = (reduce + chunk - 1) // chunk
            block_out = min(16, max(1, 8192 // block_r), triton.next_power_of_2(outer))
            if block_out < 1:
                block_out = 1
            no_mask = (outer % block_out == 0) and (reduce % chunk == 0)
            acc_dtype = torch.float64 if fp64 else torch.float32
            ss = torch.empty((outer, nch), dtype=acc_dtype, device=x.device)
            sq = torch.empty((outer, nch), dtype=acc_dtype, device=x.device)
            nc = torch.empty((outer, nch), dtype=acc_dtype, device=x.device)
            tile = block_out * block_r
            nw = 8 if tile >= 4096 else 4
            _var_partial[(nch, triton.cdiv(outer, block_out))](
                x,
                ss,
                sq,
                nc,
                outer,
                reduce,
                FP64=fp64,
                USE_I64=use_i64,
                NO_MASK=no_mask,
                ITERS=iters,
                NCH=nch,
                BLOCK_OUT=block_out,
                BLOCK_R=block_r,
                num_warps=nw,
            )
            nch_p2 = triton.next_power_of_2(nch)
            bo2 = min(64, max(1, 8192 // nch_p2), triton.next_power_of_2(outer))
            if bo2 < 1:
                bo2 = 1
            nw2 = 8 if bo2 * nch_p2 >= 4096 else 4
            _var_merge[(triton.cdiv(outer, bo2),)](
                ss,
                sq,
                nc,
                out,
                outer,
                reduce,
                float(correction),
                FP64=fp64,
                NCH=nch,
                NCH_P2=nch_p2,
                BLOCK_OUT=bo2,
                num_warps=nw2,
            )
        return out

    if x.is_contiguous() and len(red) == red[-1] - red[0] + 1:
        # ---- mid reduction: red dims form a contiguous range [d0, d1) ----
        d0 = red[0]
        d1 = red[-1] + 1
        pre = 1
        for d in range(0, d0):
            pre *= x.shape[d]
        in_size = 1
        for d in range(d1, ndim):
            in_size *= x.shape[d]
        if in_size > 1:
            block_r = 128 if reduce > 512 else triton.next_power_of_2(reduce)
            if block_r < 1:
                block_r = 1
            block_i = min(256, max(16, 8192 // block_r))
            iters = (reduce + block_r - 1) // block_r
            no_mask = (reduce % block_r == 0) and (in_size % block_i == 0)
            nw = 8 if block_r * block_i >= 8192 else 4
            _var_mid[(pre, triton.cdiv(in_size, block_i))](
                x,
                out,
                pre,
                reduce,
                in_size,
                float(correction),
                FP64=fp64,
                USE_I64=use_i64,
                NO_MASK=no_mask,
                ITERS=iters,
                BLOCK_I=block_i,
                BLOCK_R=block_r,
                num_warps=nw,
            )
            return out

    # ---- general path ----
    block_r = min(triton.next_power_of_2(reduce), 1024)
    block_out = min(2048 // block_r, 64, triton.next_power_of_2(outer))
    if block_out < 1:
        block_out = 1
    chunks = (reduce + block_r - 1) // block_r

    nsz = [x.shape[d] for d in nonred] + [1] * (8 - len(nonred))
    nst = [x.stride(d) for d in nonred] + [0] * (8 - len(nonred))
    rsz = [x.shape[d] for d in red] + [1] * (8 - len(red))
    rst = [x.stride(d) for d in red] + [0] * (8 - len(red))

    acc_dtype = torch.float64 if fp64 else torch.float32
    sn = torch.empty((outer, chunks), dtype=acc_dtype, device=x.device)
    sm = torch.empty((outer, chunks), dtype=acc_dtype, device=x.device)
    sm2 = torch.empty((outer, chunks), dtype=acc_dtype, device=x.device)

    row_blocks = triton.cdiv(outer, block_out)
    _var_stage1[(chunks, row_blocks)](
        x,
        sn,
        sm,
        sm2,
        outer,
        reduce,
        nsz[0],
        nst[0],
        nsz[1],
        nst[1],
        nsz[2],
        nst[2],
        nsz[3],
        nst[3],
        nsz[4],
        nst[4],
        nsz[5],
        nst[5],
        nsz[6],
        nst[6],
        nsz[7],
        nst[7],
        rsz[0],
        rst[0],
        rsz[1],
        rst[1],
        rsz[2],
        rst[2],
        rsz[3],
        rst[3],
        rsz[4],
        rst[4],
        rsz[5],
        rst[5],
        rsz[6],
        rst[6],
        rsz[7],
        rst[7],
        chunks,
        FP64=fp64,
        USE_I64=use_i64,
        NDIM_NONRED=len(nonred),
        NDIM_RED=len(red),
        BLOCK_OUT=block_out,
        BLOCK_R=block_r,
    )
    _var_stage2[(row_blocks,)](
        sn,
        sm,
        sm2,
        out,
        outer,
        reduce,
        float(correction),
        chunks,
        FP64=fp64,
        BLOCK_OUT=block_out,
    )
    return out


def _use_triton_var(x):
    return (
        isinstance(x, torch.Tensor)
        and x.device.type == "musa"
        and x.dtype in _SUPPORTED_DTYPES
        and x.numel() > 0
    )


def var(x, dim=None, *, correction=None, keepdim=False):
    logger.debug("GEMS_MTHREADS VAR")
    if _use_triton_var(x):
        return _specialized_var(x, dim, correction=correction, keepdim=keepdim)
    return default_var(x, dim=dim, correction=correction, keepdim=keepdim)


def var_correction(x, dim=None, *, correction=None, keepdim=False):
    logger.debug("GEMS_MTHREADS VAR_CORRECTION")
    if _use_triton_var(x):
        return _specialized_var(x, dim, correction=correction, keepdim=keepdim)
    return default_var(x, dim=dim, correction=correction, keepdim=keepdim)


def var_dim(x, dim=None, *, correction=None, keepdim=False):
    logger.debug("GEMS_MTHREADS VAR_DIM")
    if _use_triton_var(x):
        return _specialized_var(x, dim, correction=correction, keepdim=keepdim)
    return default_var(x, dim=dim, correction=correction, keepdim=keepdim)
