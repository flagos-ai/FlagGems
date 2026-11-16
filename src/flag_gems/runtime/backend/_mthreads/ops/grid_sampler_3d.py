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

from flag_gems.ops.grid_sampler_3d import grid_sampler_3d as default_grid_sampler_3d
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# Moore Threads hardware does not support fp64.
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


@triton.jit
def _reflect(v, lo, hi):
    # ATen reflect_coordinates: lo/hi are the "twice" bounds as floats.
    mn = lo * 0.5
    span = (hi - lo) * 0.5
    a = tl.abs(v - mn)
    extra = a % span
    fl = tl.floor(a / span)
    half = fl * 0.5
    even = half == tl.floor(half)
    res = tl.where(even, extra + mn, span - extra + mn)
    return tl.where(span == 0.0, 0.0, res)


@triton.jit
def _coord(g, size_f, padding_mode: tl.constexpr, align_corners: tl.constexpr):
    # exact ATen unnormalize expressions (keep order of operations)
    if align_corners:
        v = ((g + 1.0) * 0.5) * (size_f - 1.0)
    else:
        v = ((g + 1.0) * size_f - 1.0) * 0.5
    if padding_mode == 1:  # border
        v = tl.minimum(tl.maximum(v, 0.0), size_f - 1.0)
    elif padding_mode == 2:  # reflection
        if align_corners:
            v = _reflect(v, 0.0, 2.0 * (size_f - 1.0))
        else:
            v = _reflect(v, -1.0, 2.0 * size_f - 1.0)
        v = tl.minimum(tl.maximum(v, 0.0), size_f - 1.0)
    return v


@triton.jit
def _rint_even(x):
    # round-half-to-even (std::nearbyint semantics)
    f = tl.floor(x)
    d = x - f
    half = f * 0.5
    even = half == tl.floor(half)
    up = (d > 0.5) | ((d == 0.5) & (~even))
    return tl.where(up, f + 1.0, f)


# ---------------------------------------------------------------------------
# trilinear (interpolation_mode == 0)
#
# Launch: (cdiv(Ho*Wo, BW), Do*N*C).  Each program writes one (n, c, od)
# output plane's (oh, ow) tile, so every output element is exposed to the
# device at once (no serial channel loop) and the store is contiguous.
# ---------------------------------------------------------------------------


@triton.jit
def _gs3d_tri_kernel(
    inp,
    grid,
    out,
    C,
    D,
    H,
    W,
    Do,
    Ho,
    Wo,
    iN,
    iC,
    iD,
    iH,
    iW,
    gN,
    gD,
    gH,
    gW,
    gC,
    align_corners: tl.constexpr,
    padding_mode: tl.constexpr,
    BW: tl.constexpr,
    LOG_WO: tl.constexpr,
    CH: tl.constexpr,
):
    pid = tl.program_id(0)
    rest = tl.program_id(1)
    od = rest % Do
    t = rest // Do
    nch = (C + (CH - 1)) // CH
    chunk = t % nch
    n = t // nch
    # clamped so the last chunk always covers CH contiguous channels; a clamped
    # chunk may rewrite channels already written with identical values
    c0 = tl.minimum(chunk * CH, C - CH)

    Wf = W * 1.0
    Hf = H * 1.0
    Df = D * 1.0
    Wm = Wf - 1.0
    Hm = Hf - 1.0
    Dm = Df - 1.0

    idx = pid * BW + tl.arange(0, BW)
    HW = Ho * Wo
    sm = idx < HW
    if LOG_WO >= 0:
        oh = idx >> LOG_WO
        ow = idx & ((1 << LOG_WO) - 1)
    else:
        oh = idx // Wo
        ow = idx - oh * Wo

    gp = grid + n * gN + od * gD + oh * gH + ow * gW
    gx = tl.load(gp, mask=sm, other=0.0).to(tl.float32)
    gy = tl.load(gp + gC, mask=sm, other=0.0).to(tl.float32)
    gz = tl.load(gp + 2 * gC, mask=sm, other=0.0).to(tl.float32)

    X = _coord(gx, Wf, padding_mode, align_corners)
    Y = _coord(gy, Hf, padding_mode, align_corners)
    Z = _coord(gz, Df, padding_mode, align_corners)

    x0 = tl.floor(X)
    y0 = tl.floor(Y)
    z0 = tl.floor(Z)
    x1 = x0 + 1.0
    y1 = y0 + 1.0
    z1 = z0 + 1.0
    tx = X - x0
    ty = Y - y0
    tz = Z - z0

    bz0 = (z0 >= 0.0) & (z0 < Df)
    bz1 = (z1 >= 0.0) & (z1 < Df)
    by0 = (y0 >= 0.0) & (y0 < Hf)
    by1 = (y1 >= 0.0) & (y1 < Hf)
    bx0 = (x0 >= 0.0) & (x0 < Wf)
    bx1 = (x1 >= 0.0) & (x1 < Wf)

    # each tap is addressed from its own clamped index: with zeros padding the
    # base coordinate can be -1 while its +1 tap is 0, so a single stride-add
    # from the base tap would read the wrong voxel.
    dzt = tl.minimum(tl.maximum(z0, 0.0), Dm).to(tl.int32) * iD
    dzt1 = tl.minimum(tl.maximum(z1, 0.0), Dm).to(tl.int32) * iD
    dyt = tl.minimum(tl.maximum(y0, 0.0), Hm).to(tl.int32) * iH
    dyt1 = tl.minimum(tl.maximum(y1, 0.0), Hm).to(tl.int32) * iH
    dxt = tl.minimum(tl.maximum(x0, 0.0), Wm).to(tl.int32) * iW
    dxt1 = tl.minimum(tl.maximum(x1, 0.0), Wm).to(tl.int32) * iW

    zy00 = dzt + dyt
    zy01 = dzt + dyt1
    zy10 = dzt1 + dyt
    zy11 = dzt1 + dyt1

    m000 = sm & bz0 & by0 & bx0
    m001 = sm & bz0 & by0 & bx1
    m010 = sm & bz0 & by1 & bx0
    m011 = sm & bz0 & by1 & bx1
    m100 = sm & bz1 & by0 & bx0
    m101 = sm & bz1 & by0 & bx1
    m110 = sm & bz1 & by1 & bx0
    m111 = sm & bz1 & by1 & bx1

    ibase = inp + n * iN
    obase = out + n * (C * Do * HW) + od * HW + idx
    oC = Do * HW
    ot = out.dtype.element_ty
    for k in tl.static_range(CH):
        c = c0 + k
        ib = ibase + c * iC
        v000 = tl.load(ib + (zy00 + dxt), mask=m000, other=0.0).to(tl.float32)
        v001 = tl.load(ib + (zy00 + dxt1), mask=m001, other=0.0).to(tl.float32)
        v010 = tl.load(ib + (zy01 + dxt), mask=m010, other=0.0).to(tl.float32)
        v011 = tl.load(ib + (zy01 + dxt1), mask=m011, other=0.0).to(tl.float32)
        v100 = tl.load(ib + (zy10 + dxt), mask=m100, other=0.0).to(tl.float32)
        v101 = tl.load(ib + (zy10 + dxt1), mask=m101, other=0.0).to(tl.float32)
        v110 = tl.load(ib + (zy11 + dxt), mask=m110, other=0.0).to(tl.float32)
        v111 = tl.load(ib + (zy11 + dxt1), mask=m111, other=0.0).to(tl.float32)

        # separable trilinear lerp: algebraically identical to ATen's weighted
        # sum, but with the eight three-factor weights folded into seven lerps
        c00 = v000 + tx * (v001 - v000)
        c01 = v010 + tx * (v011 - v010)
        c10 = v100 + tx * (v101 - v100)
        c11 = v110 + tx * (v111 - v110)
        d0 = c00 + ty * (c01 - c00)
        d1 = c10 + ty * (c11 - c10)
        acc = d0 + tz * (d1 - d0)

        tl.store(obase + c * oC, acc.to(ot), mask=sm)


# ---------------------------------------------------------------------------
# nearest (interpolation_mode == 1)
# ---------------------------------------------------------------------------


@triton.jit
def _gs3d_near_kernel(
    inp,
    grid,
    out,
    C,
    D,
    H,
    W,
    Do,
    Ho,
    Wo,
    iN,
    iC,
    iD,
    iH,
    iW,
    gN,
    gD,
    gH,
    gW,
    gC,
    align_corners: tl.constexpr,
    padding_mode: tl.constexpr,
    BW: tl.constexpr,
    LOG_WO: tl.constexpr,
):
    pid = tl.program_id(0)
    rest = tl.program_id(1)
    od = rest % Do
    nc = rest // Do
    n = nc // C
    c = nc - n * C

    Wf = W * 1.0
    Hf = H * 1.0
    Df = D * 1.0
    Wm = Wf - 1.0
    Hm = Hf - 1.0
    Dm = Df - 1.0

    idx = pid * BW + tl.arange(0, BW)
    HW = Ho * Wo
    sm = idx < HW
    if LOG_WO >= 0:
        oh = idx >> LOG_WO
        ow = idx & ((1 << LOG_WO) - 1)
    else:
        oh = idx // Wo
        ow = idx - oh * Wo

    gp = grid + n * gN + od * gD + oh * gH + ow * gW
    gx = tl.load(gp, mask=sm, other=0.0).to(tl.float32)
    gy = tl.load(gp + gC, mask=sm, other=0.0).to(tl.float32)
    gz = tl.load(gp + 2 * gC, mask=sm, other=0.0).to(tl.float32)

    X = _coord(gx, Wf, padding_mode, align_corners)
    Y = _coord(gy, Hf, padding_mode, align_corners)
    Z = _coord(gz, Df, padding_mode, align_corners)

    xi = _rint_even(X)
    yi = _rint_even(Y)
    zi = _rint_even(Z)

    b = sm & (xi >= 0.0) & (xi < Wf) & (yi >= 0.0) & (yi < Hf) & (zi >= 0.0) & (zi < Df)

    off = (
        tl.minimum(tl.maximum(zi, 0.0), Dm).to(tl.int32) * iD
        + tl.minimum(tl.maximum(yi, 0.0), Hm).to(tl.int32) * iH
        + tl.minimum(tl.maximum(xi, 0.0), Wm).to(tl.int32) * iW
    )

    ib = inp + n * iN + c * iC
    v = tl.load(ib + off, mask=b, other=0.0).to(tl.float32)
    ob = out + n * (C * Do * HW) + c * (Do * HW) + od * HW + idx
    tl.store(ob, v.to(out.dtype.element_ty), mask=sm)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def _use_triton_kernel(x):
    """Whether ``x`` can be handled by the MUSA kernels."""
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if x.is_complex():
        return False
    return True


def grid_sampler_3d(input, grid, interpolation_mode, padding_mode, align_corners):
    logger.debug("GEMS_MTHREADS GRID_SAMPLER_3D")

    if not _use_triton_kernel(input) or not _use_triton_kernel(grid):
        return default_grid_sampler_3d(
            input, grid, interpolation_mode, padding_mode, align_corners
        )

    with torch_device_fn.device(input.device):
        return _launch_grid_sampler_3d(
            input, grid, interpolation_mode, padding_mode, align_corners
        )


def _launch_grid_sampler_3d(
    input, grid, interpolation_mode, padding_mode, align_corners
):
    if input.dim() != 5:
        raise RuntimeError("grid_sampler(): expected 5D input")
    if grid.dim() != 5:
        raise RuntimeError("grid_sampler(): expected 5D grid")
    if grid.size(-1) != 3:
        raise RuntimeError(
            "grid_sampler(): expected grid to have size 3 in last dimension"
        )
    if grid.size(0) != input.size(0):
        raise RuntimeError(
            "grid_sampler(): expected grid and input to have same batch size"
        )
    if input.dtype != grid.dtype:
        raise RuntimeError("grid_sampler(): expected input and grid to have same dtype")
    if input.device != grid.device:
        raise RuntimeError(
            "grid_sampler(): expected input and grid to be on same device"
        )
    if interpolation_mode == 2:
        raise RuntimeError("bicubic interpolation is not supported for 3D")
    if interpolation_mode != 0 and interpolation_mode != 1:
        raise RuntimeError("grid_sampler(): invalid interpolation mode")
    if padding_mode < 0 or padding_mode > 2:
        raise RuntimeError("grid_sampler(): invalid padding mode")

    N = input.size(0)
    C = input.size(1)
    D = input.size(2)
    H = input.size(3)
    W = input.size(4)
    Do = grid.size(1)
    Ho = grid.size(2)
    Wo = grid.size(3)

    out = torch.empty((N, C, Do, Ho, Wo), dtype=input.dtype, device=input.device)
    if N == 0 or C == 0 or Do * Ho * Wo == 0:
        return out

    iN, iC, iD, iH, iW = input.stride()
    gN, gD, gH, gW, gC = grid.stride()

    BW = 128
    # division-free oh/ow split when the output width is a power of two
    LOG_WO = -1
    if Wo > 0 and (Wo & (Wo - 1)) == 0:
        LOG_WO = Wo.bit_length() - 1
    tiles = triton.cdiv(Ho * Wo, BW)
    # amortize the coordinate/mask block over a few channels while keeping
    # enough CTAs to fill the device
    CH = 1
    for cand in (16, 8, 4, 2):
        if C >= cand and tiles * Do * N * ((C + cand - 1) // cand) >= 2048:
            CH = cand
            break
    launch = (tiles, Do * N * ((C + CH - 1) // CH))

    if interpolation_mode == 0:
        _gs3d_tri_kernel[launch](
            input,
            grid,
            out,
            C,
            D,
            H,
            W,
            Do,
            Ho,
            Wo,
            iN,
            iC,
            iD,
            iH,
            iW,
            gN,
            gD,
            gH,
            gW,
            gC,
            align_corners=bool(align_corners),
            padding_mode=padding_mode,
            BW=BW,
            LOG_WO=LOG_WO,
            CH=CH,
            num_warps=4,
        )
    else:
        _gs3d_near_kernel[(tiles, Do * N * C)](
            input,
            grid,
            out,
            C,
            D,
            H,
            W,
            Do,
            Ho,
            Wo,
            iN,
            iC,
            iD,
            iH,
            iW,
            gN,
            gD,
            gH,
            gW,
            gC,
            align_corners=bool(align_corners),
            padding_mode=padding_mode,
            BW=BW,
            LOG_WO=LOG_WO,
            num_warps=4,
        )
    return out
