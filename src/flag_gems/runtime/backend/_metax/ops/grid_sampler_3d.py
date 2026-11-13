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

_BLOCK_S = 128
_SM_CACHE = {}


@triton.jit
def _rint_even(x):
    """Round half to even (ATen uses std::nearbyint)."""
    f = tl.floor(x)
    d = x - f
    even = tl.floor(f * 0.5) * 2.0 == f
    up = tl.where(d > 0.5, 1.0, 0.0)
    up = tl.where(d == 0.5, tl.where(even, 0.0, 1.0), up)
    return f + up


@triton.jit
def _reflect(u, mn, span):
    """ATen reflect_coordinates with mn = twice_low/2, span = (twice_high-twice_low)/2."""
    v = tl.abs(u - mn)
    flips = tl.floor(v / span)
    extra = v - flips * span
    even = tl.floor(flips * 0.5) * 2.0 == flips
    r = tl.where(even, extra + mn, span - extra + mn)
    return tl.where(span > 0.0, r, 0.0)


@triton.jit
def _gs3d_body(
    in_ptr,
    grid_ptr,
    out_ptr,
    C,
    D_in,
    H_in,
    W_in,
    D_out,
    H_out,
    W_out,
    out_spatial,
    in_sn,
    in_sc,
    in_sd,
    in_sh,
    in_sw,
    g_sn,
    g_sd,
    g_sh,
    g_sw,
    g_last,
    INTERP: tl.constexpr,
    PADDING: tl.constexpr,
    ALIGN: tl.constexpr,
    FP64: tl.constexpr,
    USE_INT64: tl.constexpr,
    CH_BLOCK: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_s = tl.program_id(0)
    n = tl.program_id(1)
    c0 = tl.program_id(2) * CH_BLOCK

    s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    ms = s < out_spatial

    wi = s % W_out
    t = s // W_out
    hi = t % H_out
    di = t // H_out

    if USE_INT64:
        g_off = (
            n.to(tl.int64) * g_sn
            + di.to(tl.int64) * g_sd
            + hi.to(tl.int64) * g_sh
            + wi.to(tl.int64) * g_sw
        )
    else:
        g_off = n * g_sn + di * g_sd + hi * g_sh + wi * g_sw

    gx = tl.load(grid_ptr + g_off, mask=ms, other=0.0)
    gy = tl.load(grid_ptr + g_off + g_last, mask=ms, other=0.0)
    gz = tl.load(grid_ptr + g_off + 2 * g_last, mask=ms, other=0.0)

    if not FP64:
        gx = gx.to(tl.float32)
        gy = gy.to(tl.float32)
        gz = gz.to(tl.float32)

    Wf = W_in.to(tl.float32)
    Hf = H_in.to(tl.float32)
    Df = D_in.to(tl.float32)

    if ALIGN:
        sx = (gx + 1.0) * 0.5 * (Wf - 1.0)
        sy = (gy + 1.0) * 0.5 * (Hf - 1.0)
        sz = (gz + 1.0) * 0.5 * (Df - 1.0)
    else:
        sx = ((gx + 1.0) * Wf - 1.0) * 0.5
        sy = ((gy + 1.0) * Hf - 1.0) * 0.5
        sz = ((gz + 1.0) * Df - 1.0) * 0.5

    if PADDING == 1:
        sx = tl.minimum(tl.maximum(sx, 0.0), Wf - 1.0)
        sy = tl.minimum(tl.maximum(sy, 0.0), Hf - 1.0)
        sz = tl.minimum(tl.maximum(sz, 0.0), Df - 1.0)
    elif PADDING == 2:
        if ALIGN:
            sx = _reflect(sx, 0.0, Wf - 1.0)
            sy = _reflect(sy, 0.0, Hf - 1.0)
            sz = _reflect(sz, 0.0, Df - 1.0)
        else:
            sx = _reflect(sx, -0.5, Wf)
            sy = _reflect(sy, -0.5, Hf)
            sz = _reflect(sz, -0.5, Df)
        sx = tl.minimum(tl.maximum(sx, 0.0), Wf - 1.0)
        sy = tl.minimum(tl.maximum(sy, 0.0), Hf - 1.0)
        sz = tl.minimum(tl.maximum(sz, 0.0), Df - 1.0)

    # Keep finite values only: non-finite coordinates become far out of bounds so
    # that every bounds mask fails instead of casting NaN/inf to an integer.
    BIG = 1.0e9
    sx = tl.where(tl.abs(sx) < BIG, sx, -BIG)
    sy = tl.where(tl.abs(sy) < BIG, sy, -BIG)
    sz = tl.where(tl.abs(sz) < BIG, sz, -BIG)

    if USE_INT64:
        out_base = n.to(tl.int64) * C * out_spatial + s.to(tl.int64)
    else:
        out_base = n * C * out_spatial + s

    if INTERP == 1:
        ix = _rint_even(sx)
        iy = _rint_even(sy)
        iz = _rint_even(sz)
        valid = (
            ms
            & (ix >= 0.0)
            & (ix < Wf)
            & (iy >= 0.0)
            & (iy < Hf)
            & (iz >= 0.0)
            & (iz < Df)
        )
        ixi = ix.to(tl.int32)
        iyi = iy.to(tl.int32)
        izi = iz.to(tl.int32)
        if USE_INT64:
            off0 = (
                n.to(tl.int64) * in_sn
                + izi.to(tl.int64) * in_sd
                + iyi.to(tl.int64) * in_sh
                + ixi.to(tl.int64) * in_sw
            )
        else:
            off0 = n * in_sn + izi * in_sd + iyi * in_sh + ixi * in_sw

        for ch in range(CH_BLOCK):
            c = tl.minimum(c0 + ch, C - 1)
            cvalid = c0 + ch < C
            if USE_INT64:
                off = off0 + c.to(tl.int64) * in_sc
                ooff = out_base + c.to(tl.int64) * out_spatial
            else:
                off = off0 + c * in_sc
                ooff = out_base + c * out_spatial
            val = tl.load(in_ptr + off, mask=valid & cvalid, other=0.0)
            tl.store(out_ptr + ooff, val, mask=ms & cvalid)
    else:
        x0f = tl.floor(sx)
        y0f = tl.floor(sy)
        z0f = tl.floor(sz)
        x1f = x0f + 1.0
        y1f = y0f + 1.0
        z1f = z0f + 1.0
        wx1 = sx - x0f
        wy1 = sy - y0f
        wz1 = sz - z0f
        wx0 = 1.0 - wx1
        wy0 = 1.0 - wy1
        wz0 = 1.0 - wz1

        Wm = Wf - 1.0
        Hm = Hf - 1.0
        Dm = Df - 1.0
        mx0 = (x0f >= 0.0) & (x0f <= Wm)
        mx1 = (x1f >= 0.0) & (x1f <= Wm)
        my0 = (y0f >= 0.0) & (y0f <= Hm)
        my1 = (y1f >= 0.0) & (y1f <= Hm)
        mz0 = (z0f >= 0.0) & (z0f <= Dm)
        mz1 = (z1f >= 0.0) & (z1f <= Dm)

        m000 = ms & mz0 & my0 & mx0
        m001 = ms & mz0 & my0 & mx1
        m010 = ms & mz0 & my1 & mx0
        m011 = ms & mz0 & my1 & mx1
        m100 = ms & mz1 & my0 & mx0
        m101 = ms & mz1 & my0 & mx1
        m110 = ms & mz1 & my1 & mx0
        m111 = ms & mz1 & my1 & mx1

        w00 = wz0 * wy0
        w01 = wz0 * wy1
        w10 = wz1 * wy0
        w11 = wz1 * wy1
        w000 = w00 * wx0
        w001 = w00 * wx1
        w010 = w01 * wx0
        w011 = w01 * wx1
        w100 = w10 * wx0
        w101 = w10 * wx1
        w110 = w11 * wx0
        w111 = w11 * wx1

        x0 = x0f.to(tl.int32)
        y0 = y0f.to(tl.int32)
        z0 = z0f.to(tl.int32)
        x1 = x1f.to(tl.int32)
        y1 = y1f.to(tl.int32)
        z1 = z1f.to(tl.int32)

        if USE_INT64:
            base0 = (
                n.to(tl.int64) * in_sn
                + z0.to(tl.int64) * in_sd
                + y0.to(tl.int64) * in_sh
                + x0.to(tl.int64) * in_sw
            )
            dz = (z1.to(tl.int64) - z0.to(tl.int64)) * in_sd
            dy = (y1.to(tl.int64) - y0.to(tl.int64)) * in_sh
            dx = (x1.to(tl.int64) - x0.to(tl.int64)) * in_sw
        else:
            base0 = n * in_sn + z0 * in_sd + y0 * in_sh + x0 * in_sw
            dz = (z1 - z0) * in_sd
            dy = (y1 - y0) * in_sh
            dx = (x1 - x0) * in_sw

        for ch in range(CH_BLOCK):
            c = tl.minimum(c0 + ch, C - 1)
            cvalid = c0 + ch < C
            if USE_INT64:
                base = base0 + c.to(tl.int64) * in_sc
                ooff = out_base + c.to(tl.int64) * out_spatial
            else:
                base = base0 + c * in_sc
                ooff = out_base + c * out_spatial

            if FP64:
                acc = tl.zeros([BLOCK_S], dtype=tl.float64)
            else:
                acc = tl.zeros([BLOCK_S], dtype=tl.float32)

            v = tl.load(in_ptr + base, mask=m000 & cvalid, other=0.0)
            acc += v.to(acc.dtype) * w000
            v = tl.load(in_ptr + base + dx, mask=m001 & cvalid, other=0.0)
            acc += v.to(acc.dtype) * w001
            v = tl.load(in_ptr + base + dy, mask=m010 & cvalid, other=0.0)
            acc += v.to(acc.dtype) * w010
            v = tl.load(in_ptr + base + dy + dx, mask=m011 & cvalid, other=0.0)
            acc += v.to(acc.dtype) * w011
            v = tl.load(in_ptr + base + dz, mask=m100 & cvalid, other=0.0)
            acc += v.to(acc.dtype) * w100
            v = tl.load(in_ptr + base + dz + dx, mask=m101 & cvalid, other=0.0)
            acc += v.to(acc.dtype) * w101
            v = tl.load(in_ptr + base + dz + dy, mask=m110 & cvalid, other=0.0)
            acc += v.to(acc.dtype) * w110
            v = tl.load(in_ptr + base + dz + dy + dx, mask=m111 & cvalid, other=0.0)
            acc += v.to(acc.dtype) * w111

            tl.store(out_ptr + ooff, acc.to(out_ptr.dtype.element_ty), mask=ms & cvalid)


@triton.jit
def _gs3d_kernel(
    in_ptr,
    grid_ptr,
    out_ptr,
    C,
    D_in,
    H_in,
    W_in,
    D_out,
    H_out,
    W_out,
    out_spatial,
    in_sn,
    in_sc,
    in_sd,
    in_sh,
    in_sw,
    g_sn,
    g_sd,
    g_sh,
    g_sw,
    g_last,
    INTERP: tl.constexpr,
    PADDING: tl.constexpr,
    ALIGN: tl.constexpr,
    FP64: tl.constexpr,
    USE_INT64: tl.constexpr,
    CH_BLOCK: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    _gs3d_body(
        in_ptr,
        grid_ptr,
        out_ptr,
        C,
        D_in,
        H_in,
        W_in,
        D_out,
        H_out,
        W_out,
        out_spatial,
        in_sn,
        in_sc,
        in_sd,
        in_sh,
        in_sw,
        g_sn,
        g_sd,
        g_sh,
        g_sw,
        g_last,
        INTERP,
        PADDING,
        ALIGN,
        FP64,
        USE_INT64,
        CH_BLOCK,
        BLOCK_S,
    )


@triton.jit
def _gs3d_contig_kernel(
    in_ptr,
    grid_ptr,
    out_ptr,
    C,
    D_in,
    H_in,
    W_in,
    D_out,
    H_out,
    W_out,
    INTERP: tl.constexpr,
    PADDING: tl.constexpr,
    ALIGN: tl.constexpr,
    FP64: tl.constexpr,
    USE_INT64: tl.constexpr,
    CH_BLOCK: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    out_spatial = D_out * H_out * W_out
    if USE_INT64:
        in_sc = D_in.to(tl.int64) * H_in * W_in
        in_sn = C.to(tl.int64) * in_sc
        in_sd = H_in.to(tl.int64) * W_in
        in_sh = W_in.to(tl.int64)
        g_sn = out_spatial.to(tl.int64) * 3
        g_sd = H_out.to(tl.int64) * W_out * 3
        g_sh = W_out.to(tl.int64) * 3
    else:
        in_sc = D_in * H_in * W_in
        in_sn = C * in_sc
        in_sd = H_in * W_in
        in_sh = W_in
        g_sn = out_spatial * 3
        g_sd = H_out * W_out * 3
        g_sh = W_out * 3
    _gs3d_body(
        in_ptr,
        grid_ptr,
        out_ptr,
        C,
        D_in,
        H_in,
        W_in,
        D_out,
        H_out,
        W_out,
        out_spatial,
        in_sn,
        in_sc,
        in_sd,
        in_sh,
        1,
        g_sn,
        g_sd,
        g_sh,
        3,
        1,
        INTERP,
        PADDING,
        ALIGN,
        FP64,
        USE_INT64,
        CH_BLOCK,
        BLOCK_S,
    )


_INTERP_MAP = {"bilinear": 0, "nearest": 1, "bicubic": 2}
_PAD_MAP = {"zeros": 0, "border": 1, "reflection": 2}


def _as_mode(value, mapping, name):
    if isinstance(value, str):
        try:
            return mapping[value.lower()]
        except KeyError:
            raise RuntimeError(
                "grid_sampler_3d(): Expected %s to be one of %s, but got %r"
                % (name, ", ".join(sorted(mapping)), value)
            )
    return int(value)


def _sm_count(device):
    idx = device.index
    if idx is None:
        idx = torch.cuda.current_device()
    n = _SM_CACHE.get(idx)
    if n is None:
        try:
            n = torch.cuda.get_device_properties(idx).multi_processor_count
        except Exception:
            n = 128
        _SM_CACHE[idx] = n
    return n


def grid_sampler_3d(
    input, grid, interpolation_mode=0, padding_mode=0, align_corners=False
):
    ishape = input.shape
    gshape = grid.shape
    if len(ishape) != 5 or len(gshape) != 5:
        raise RuntimeError(
            "grid_sampler_3d(): Expected input and grid to be 5D tensors"
        )

    interp = interpolation_mode
    if interp.__class__ is not int:
        interp = _as_mode(interp, _INTERP_MAP, "interpolation_mode")
    if interp != 0 and interp != 1:
        raise RuntimeError(
            "grid_sampler_3d(): Expected interpolation_mode to be 0 (bilinear) "
            "or 1 (nearest), but got %d" % interp
        )
    pad = padding_mode
    if pad.__class__ is not int:
        pad = _as_mode(pad, _PAD_MAP, "padding_mode")
    if pad < 0 or pad > 2:
        raise RuntimeError(
            "grid_sampler_3d(): Expected padding_mode to be 0 (zeros), "
            "1 (border) or 2 (reflection), but got %d" % pad
        )
    if gshape[4] != 3:
        raise RuntimeError(
            "grid_sampler_3d(): Expected grid to have size 3 in last dimension"
        )

    N, C, D_in, H_in, W_in = ishape
    D_out = gshape[1]
    H_out = gshape[2]
    W_out = gshape[3]
    out_spatial = D_out * H_out * W_out

    out = input.new_empty((N, C, D_out, H_out, W_out))
    if out_spatial == 0 or N == 0 or C == 0:
        return out

    in_spatial = D_in * H_in * W_in
    big = out_spatial if out_spatial > in_spatial else in_spatial
    use64 = N * C * big > 2147483647 or N * out_spatial * 3 > 2147483647

    blocks = -(-out_spatial // _BLOCK_S)
    ch = 1
    if C > 1 and blocks * N >= 2 * _sm_count(input.device):
        ch = 4 if C > 4 else C

    align = False if align_corners is None else align_corners
    fp64 = input.dtype is torch.float64
    nz = -(-C // ch)

    if input.is_contiguous() and grid.is_contiguous():
        _gs3d_contig_kernel[(blocks, N, nz)](
            input,
            grid,
            out,
            C,
            D_in,
            H_in,
            W_in,
            D_out,
            H_out,
            W_out,
            interp,
            pad,
            align,
            fp64,
            use64,
            ch,
            _BLOCK_S,
            num_warps=2,
        )
    else:
        in_sn, in_sc, in_sd, in_sh, in_sw = input.stride()
        g_sn, g_sd, g_sh, g_sw, g_last = grid.stride()
        _gs3d_kernel[(blocks, N, nz)](
            input,
            grid,
            out,
            C,
            D_in,
            H_in,
            W_in,
            D_out,
            H_out,
            W_out,
            out_spatial,
            in_sn,
            in_sc,
            in_sd,
            in_sh,
            in_sw,
            g_sn,
            g_sd,
            g_sh,
            g_sw,
            g_last,
            interp,
            pad,
            align,
            fp64,
            use64,
            ch,
            _BLOCK_S,
            num_warps=2,
        )
    return out
