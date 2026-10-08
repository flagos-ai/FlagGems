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

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Fast paths (contiguous last-dim diff, no prepend/append): pure affine
# unmasked loads/stores into a padded output buffer. The padding columns hold
# out-of-row garbage that is dropped by a host-side narrow view, so no masks
# or offset clamps ever appear on the hot path (they measurably disable the
# vectorized block loads/stores on this backend).
# ---------------------------------------------------------------------------


@triton.jit
def _flat1_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    # out_flat[i] = in_flat[i+1] - in_flat[i]; padded out has width xdim
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    v0 = tl.load(x_ptr + i)
    v1 = tl.load(x_ptr + i + 1)
    tl.store(out_ptr + i, v1 - v0)


@triton.jit
def _flat2_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    # out_flat[i] = (in[i+2] - in[i+1]) - (in[i+1] - in[i]) (reference op order)
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    v0 = tl.load(x_ptr + i)
    v1 = tl.load(x_ptr + i + 1)
    v2 = tl.load(x_ptr + i + 2)
    t = v1 - v0
    d1 = v2 - v1
    tl.store(out_ptr + i, d1 - t)


@triton.jit
def _row2d1_kernel(
    x_ptr, out_ptr, xdim, out_len, ostride, ROWS: tl.constexpr, COLS: tl.constexpr
):
    pid = tl.program_id(0)
    num_blk = tl.cdiv(out_len, COLS)
    blk = pid % num_blk
    rg = pid // num_blk
    rows = rg * ROWS + tl.arange(0, ROWS)
    j = blk * COLS + tl.arange(0, COLS)
    off = rows[:, None] * xdim + j[None, :]
    v0 = tl.load(x_ptr + off)
    v1 = tl.load(x_ptr + off + 1)
    ooff = rows[:, None] * ostride + j[None, :]
    tl.store(out_ptr + ooff, v1 - v0)


@triton.jit
def _row2d2_kernel(
    x_ptr, out_ptr, xdim, out_len, ostride, ROWS: tl.constexpr, COLS: tl.constexpr
):
    pid = tl.program_id(0)
    num_blk = tl.cdiv(out_len, COLS)
    blk = pid % num_blk
    rg = pid // num_blk
    rows = rg * ROWS + tl.arange(0, ROWS)
    j = blk * COLS + tl.arange(0, COLS)
    off = rows[:, None] * xdim + j[None, :]
    v0 = tl.load(x_ptr + off)
    v1 = tl.load(x_ptr + off + 1)
    v2 = tl.load(x_ptr + off + 2)
    t = v1 - v0
    d1 = v2 - v1
    ooff = rows[:, None] * ostride + j[None, :]
    tl.store(out_ptr + ooff, d1 - t)


# ---------------------------------------------------------------------------
# General/slow path (prepend/append, inner > 1, or n > 2): masked loads with
# clamped offsets keep every address valid; masked store writes exact shape.
# ---------------------------------------------------------------------------


@triton.jit
def _load_ext(
    x_ptr,
    pre_ptr,
    app_ptr,
    xb,
    pb,
    ab,
    p,
    pdim,
    xdim,
    adim,
    inner,
    HAS_PRE: tl.constexpr,
    HAS_APP: tl.constexpr,
):
    # value of the virtual concatenated array ext = cat([prepend, x, append])
    if HAS_PRE or HAS_APP:
        mx = (p >= pdim) & (p < pdim + xdim)
        px = tl.minimum(tl.maximum(p - pdim, 0), xdim - 1)
        v = tl.load(x_ptr + xb + px * inner, mask=mx, other=0)
        if HAS_PRE:
            mp = p < pdim
            pp = tl.minimum(p, pdim - 1)
            v = v + tl.load(pre_ptr + pb + pp * inner, mask=mp, other=0)
        if HAS_APP:
            ma = p >= pdim + xdim
            pa = tl.minimum(tl.maximum(p - pdim - xdim, 0), adim - 1)
            v = v + tl.load(app_ptr + ab + pa * inner, mask=ma, other=0)
    else:
        ps = tl.minimum(p, xdim - 1)
        v = tl.load(x_ptr + xb + ps * inner)
    return v


@triton.jit
def _diff1_kernel(
    x_ptr,
    pre_ptr,
    app_ptr,
    out_ptr,
    xdim,
    pdim,
    adim,
    inner,
    out_len,
    BLOCK: tl.constexpr,
    HAS_PRE: tl.constexpr,
    HAS_APP: tl.constexpr,
):
    pid = tl.program_id(0)
    num_blk = tl.cdiv(out_len, BLOCK)
    blk = pid % num_blk
    r = pid // num_blk
    e = r // inner
    i = r % inner
    j = blk * BLOCK + tl.arange(0, BLOCK)
    smask = j < out_len
    xb = e * (xdim * inner) + i
    pb = e * (pdim * inner) + i
    ab = e * (adim * inner) + i
    v0 = _load_ext(
        x_ptr,
        pre_ptr,
        app_ptr,
        xb,
        pb,
        ab,
        j,
        pdim,
        xdim,
        adim,
        inner,
        HAS_PRE,
        HAS_APP,
    )
    v1 = _load_ext(
        x_ptr,
        pre_ptr,
        app_ptr,
        xb,
        pb,
        ab,
        j + 1,
        pdim,
        xdim,
        adim,
        inner,
        HAS_PRE,
        HAS_APP,
    )
    tl.store(out_ptr + e * (out_len * inner) + i + j * inner, v1 - v0, mask=smask)


@triton.jit
def _diff2_kernel(
    x_ptr,
    pre_ptr,
    app_ptr,
    out_ptr,
    xdim,
    pdim,
    adim,
    inner,
    out_len,
    BLOCK: tl.constexpr,
    HAS_PRE: tl.constexpr,
    HAS_APP: tl.constexpr,
):
    pid = tl.program_id(0)
    num_blk = tl.cdiv(out_len, BLOCK)
    blk = pid % num_blk
    r = pid // num_blk
    e = r // inner
    i = r % inner
    j = blk * BLOCK + tl.arange(0, BLOCK)
    smask = j < out_len
    xb = e * (xdim * inner) + i
    pb = e * (pdim * inner) + i
    ab = e * (adim * inner) + i
    v0 = _load_ext(
        x_ptr,
        pre_ptr,
        app_ptr,
        xb,
        pb,
        ab,
        j,
        pdim,
        xdim,
        adim,
        inner,
        HAS_PRE,
        HAS_APP,
    )
    v1 = _load_ext(
        x_ptr,
        pre_ptr,
        app_ptr,
        xb,
        pb,
        ab,
        j + 1,
        pdim,
        xdim,
        adim,
        inner,
        HAS_PRE,
        HAS_APP,
    )
    v2 = _load_ext(
        x_ptr,
        pre_ptr,
        app_ptr,
        xb,
        pb,
        ab,
        j + 2,
        pdim,
        xdim,
        adim,
        inner,
        HAS_PRE,
        HAS_APP,
    )
    t = v1 - v0
    d1 = v2 - v1
    tl.store(out_ptr + e * (out_len * inner) + i + j * inner, d1 - t, mask=smask)


# ---------------------------------------------------------------------------
# Host dispatch
# ---------------------------------------------------------------------------


def _pick_flat_block(total, dtype):
    pref = 16384 if dtype.itemsize <= 2 else 8192
    for b in (pref, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16):
        if total % b == 0 and total // b <= 65535:
            return b
    return None


def _run_fast(x, n_, dimn, shape, outer, xdim, out_len):
    total = outer * xdim
    dtype = x.dtype
    BLOCK = _pick_flat_block(total, dtype)
    if BLOCK is not None:
        padded = torch.empty(shape, dtype=dtype, device=x.device)  # width == xdim
        grid = (total // BLOCK,)
        if n_ == 1:
            _flat1_kernel[grid](x, padded, BLOCK=BLOCK, num_warps=2)
        else:
            _flat2_kernel[grid](x, padded, BLOCK=BLOCK, num_warps=2)
        return padded.narrow(dimn, 0, out_len)

    ROWS = (
        16
        if outer % 16 == 0
        else (
            8 if outer % 8 == 0 else 4 if outer % 4 == 0 else 2 if outer % 2 == 0 else 1
        )
    )
    COLS = 512 if out_len > 512 else triton.next_power_of_2(max(out_len, 16))
    nb = triton.cdiv(out_len, COLS)
    pad_len = nb * COLS
    pshape = list(shape)
    pshape[dimn] = pad_len
    padded = torch.empty(pshape, dtype=dtype, device=x.device)
    grid = ((outer // ROWS) * nb,)
    if n_ == 1:
        _row2d1_kernel[grid](
            x, padded, xdim, out_len, pad_len, ROWS=ROWS, COLS=COLS, num_warps=4
        )
    else:
        _row2d2_kernel[grid](
            x, padded, xdim, out_len, pad_len, ROWS=ROWS, COLS=COLS, num_warps=4
        )
    return padded.narrow(dimn, 0, out_len)


def _run_slow(
    x,
    n_,
    dimn,
    shape,
    outer,
    inner,
    xdim,
    pdim,
    adim,
    out_len,
    prepend,
    append,
    has_pre,
    has_app,
):
    oshape = list(shape)
    oshape[dimn] = out_len
    out = torch.empty(oshape, dtype=x.dtype, device=x.device)
    pre_ptr = prepend if has_pre else x
    app_ptr = append if has_app else x
    BLOCK = triton.next_power_of_2(min(max(out_len, 16), 4096))
    num_blk = triton.cdiv(out_len, BLOCK)
    grid = (outer * inner * num_blk,)

    if n_ == 1:
        _diff1_kernel[grid](
            x,
            pre_ptr,
            app_ptr,
            out,
            xdim,
            pdim,
            adim,
            inner,
            out_len,
            BLOCK=BLOCK,
            HAS_PRE=has_pre,
            HAS_APP=has_app,
        )
    elif n_ == 2:
        _diff2_kernel[grid](
            x,
            pre_ptr,
            app_ptr,
            out,
            xdim,
            pdim,
            adim,
            inner,
            out_len,
            BLOCK=BLOCK,
            HAS_PRE=has_pre,
            HAS_APP=has_app,
        )
    else:
        # General n: sequential first-difference passes over the extended array.
        total = pdim + xdim + adim
        tshape = list(shape)
        tshape[dimn] = total - 1
        buf = torch.empty(tshape, dtype=x.dtype, device=x.device)
        b1 = triton.next_power_of_2(min(max(total - 1, 16), 4096))
        g1 = (outer * inner * triton.cdiv(total - 1, b1),)
        _diff1_kernel[g1](
            x,
            pre_ptr,
            app_ptr,
            buf,
            xdim,
            pdim,
            adim,
            inner,
            total - 1,
            BLOCK=b1,
            HAS_PRE=has_pre,
            HAS_APP=has_app,
        )
        cur = buf
        clen = total - 1
        for k in range(n_ - 2):
            nclen = clen - 1
            if k == n_ - 3:
                dst = out
            else:
                tshape = list(shape)
                tshape[dimn] = nclen
                dst = torch.empty(tshape, dtype=x.dtype, device=x.device)
            bk = triton.next_power_of_2(min(max(clen, 16), 4096))
            gk = (outer * inner * triton.cdiv(clen, bk),)
            _diff1_kernel[gk](
                cur,
                x,
                x,
                dst,
                clen,
                0,
                0,
                inner,
                nclen,
                BLOCK=bk,
                HAS_PRE=False,
                HAS_APP=False,
            )
            cur = dst
            clen = nclen
    return out


def diff(x, n, dim, prepend, append):
    ndim = x.dim()
    dimn = int(dim) % ndim
    shape = list(x.shape)
    outer = 1
    for s in shape[:dimn]:
        outer *= s
    inner = 1
    for s in shape[dimn + 1 :]:
        inner *= s
    xdim = shape[dimn]
    pdim = int(prepend.shape[dimn]) if prepend is not None else 0
    adim = int(append.shape[dimn]) if append is not None else 0
    n_ = int(n)
    out_len = pdim + xdim + adim - n_

    if out_len <= 0:
        oshape = list(shape)
        oshape[dimn] = out_len
        return torch.empty(oshape, dtype=x.dtype, device=x.device)

    has_pre = prepend is not None and pdim > 0
    has_app = append is not None and adim > 0

    if (not has_pre) and (not has_app) and inner == 1 and n_ <= 2:
        return _run_fast(x, n_, dimn, shape, outer, xdim, out_len)
    return _run_slow(
        x,
        n_,
        dimn,
        shape,
        outer,
        inner,
        xdim,
        pdim,
        adim,
        out_len,
        prepend,
        append,
        has_pre,
        has_app,
    )
