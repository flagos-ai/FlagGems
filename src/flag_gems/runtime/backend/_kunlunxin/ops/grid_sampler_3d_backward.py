# Kunlunxin (XPU) specialization of aten::grid_sampler_3d_backward.
#
# Two verified TritonXPU platform defects make the generic implementation
# (src/flag_gems/ops/grid_sampler_3d_backward.py) silently wrong on this
# backend; both are fixed here without changing the mathematics:
#
# 1. `tl.atomic_add(..., mask=<data-dependent composite mask>)` does NOT honour
#    the mask, so out-of-bounds trilinear corners are accumulated at their
#    clamped (in-bounds) addresses.  Single-variable evidence on
#    (2,3,8,8,8)->(4,4,4) fp32, bilinear/zeros/align_corners=False:
#    HEAD form bad=373/3072 maxdiff 8.47e-1, `tl.where`-gated payload form
#    bad=0/3072 maxdiff 1.19e-7 (BLOCK 64 and 256 alike).
#    Fix: gate the *payload* with `tl.where`, keep only the tail mask on the
#    atomic (an atomic with no mask at all is also broken on this backend).
#
# 2. `tl.atomic_add` loses updates when several *programs* touch the same
#    address concurrently.  On the official benchmark cell
#    (2,64,32,32,32)->(16,16,16) the generic kernel (32 programs) is off by
#    2.8e-1 .. 9.0e-1 against the CPU reference, non-deterministically.
#    Fix: give every program exclusive ownership of one `grad_input[n, c]`
#    slice (grid = (N*C,), runtime loop over output-point tiles), so no two
#    programs ever target the same address.
#
# `other=` is never trusted (it is not honoured on this backend); every guarded
# load uses a clamped, always-legal address plus an explicit `tl.where`.
#
# grad_grid is computed by a separate kernel that puts the channel axis in the
# tile (one program per output point, `tl.sum` over C).  That removes the
# per-lane discrete gathers of the generic kernel and is 18-43x faster; it also
# side-steps two compile walls: the generic grad_grid loop cannot be lowered on
# its own (`output_mask=[False, True]` fails with `uni_sram ... 0/0` even on
# HEAD), and fusing an atomic with a `tl.where`-in-reduce gives
# `Failed to tune buffer size`.

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# Verified-safe channel tile widths on this backend are 64 and 128; 4 and 16
# both produce `CUDA error: unspecified launch failure`.  Wider widths are not
# validated, so large C is handled by looping the launch on the host.
_MAX_BLOCK_C = 128
_MIN_BLOCK_C = 64


@triton.jit
def _src_index(coord, size, align_corners: tl.constexpr):
    if align_corners:
        idx = (coord + 1.0) * 0.5 * (size - 1)
    else:
        idx = ((coord + 1.0) * size - 1.0) * 0.5
    return idx


@triton.jit
def _clip(coord, size):
    return tl.minimum(tl.maximum(coord, 0.0), size - 1.0)


@triton.jit
def _clip_grad(coord, size):
    return tl.where((coord >= 0.0) & (coord <= size - 1.0), 1.0, 0.0)


@triton.jit
def _reflect(coord, size, align_corners: tl.constexpr):
    if align_corners:
        span = size - 1.0
        c = tl.abs(coord)
        c = c - span
        c = span - tl.abs(c)
    else:
        span = size * 1.0
        c = tl.abs(coord + 0.5)
        c = c - span
        c = span - tl.abs(c)
        c = c - 0.5
    return c


@triton.jit
def _reflect_grad(coord_raw, size, align_corners: tl.constexpr):
    if align_corners:
        u = coord_raw
        v = tl.abs(u) - (size - 1.0)
    else:
        u = coord_raw + 0.5
        v = tl.abs(u) - size * 1.0
    sign_u = tl.where(u >= 0.0, 1.0, -1.0)
    sign_v = tl.where(v >= 0.0, 1.0, -1.0)
    reflected = _reflect(coord_raw, size, align_corners)
    return (-sign_v * sign_u) * _clip_grad(reflected, size)


@triton.jit
def _pad(coord, size, padding_mode: tl.constexpr, align_corners: tl.constexpr):
    if padding_mode == 1:
        c = _clip(coord, size)
    elif padding_mode == 2:
        c = _clip(_reflect(coord, size, align_corners), size)
    else:
        c = coord
    return c


@triton.jit
def _grad_input_bilinear_kernel(
    go_ptr,
    grid_ptr,
    gi_ptr,
    C,
    iD,
    iH,
    iW,
    oH,
    oW,
    go_sN,
    go_sC,
    gi_sN,
    gi_sC,
    grid_sN,
    grid_sD,
    grid_sH,
    grid_sW,
    n_per_batch,
    n_tiles,
    padding_mode: tl.constexpr,
    align_corners: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    go_base = n * go_sN + c * go_sC
    gi_base = n * gi_sN + c * gi_sC
    isD = iH * iW
    isH = iW
    for t in range(n_tiles):
        p = t * BLOCK_P + tl.arange(0, BLOCK_P)
        pm = p < n_per_batch
        ow = p % oW
        t1 = p // oW
        oh = t1 % oH
        od = t1 // oH
        gb = n * grid_sN + od * grid_sD + oh * grid_sH + ow * grid_sW
        gx = tl.where(pm, tl.load(grid_ptr + gb, mask=pm), 0.0)
        gy = tl.where(pm, tl.load(grid_ptr + gb + 1, mask=pm), 0.0)
        gz = tl.where(pm, tl.load(grid_ptr + gb + 2, mask=pm), 0.0)
        ix = _pad(_src_index(gx, iW, align_corners), iW, padding_mode, align_corners)
        iy = _pad(_src_index(gy, iH, align_corners), iH, padding_mode, align_corners)
        iz = _pad(_src_index(gz, iD, align_corners), iD, padding_mode, align_corners)
        ix0 = tl.floor(ix).to(tl.int32)
        iy0 = tl.floor(iy).to(tl.int32)
        iz0 = tl.floor(iz).to(tl.int32)
        tx = ix - ix0.to(tl.float32)
        ty = iy - iy0.to(tl.float32)
        tz = iz - iz0.to(tl.float32)
        ix1 = ix0 + 1
        iy1 = iy0 + 1
        iz1 = iz0 + 1
        x0ok = (ix0 >= 0) & (ix0 < iW)
        x1ok = (ix1 >= 0) & (ix1 < iW)
        y0ok = (iy0 >= 0) & (iy0 < iH)
        y1ok = (iy1 >= 0) & (iy1 < iH)
        z0ok = (iz0 >= 0) & (iz0 < iD)
        z1ok = (iz1 >= 0) & (iz1 < iD)
        x0s = tl.maximum(tl.minimum(ix0, iW - 1), 0)
        x1s = tl.maximum(tl.minimum(ix1, iW - 1), 0)
        y0s = tl.maximum(tl.minimum(iy0, iH - 1), 0)
        y1s = tl.maximum(tl.minimum(iy1, iH - 1), 0)
        z0s = tl.maximum(tl.minimum(iz0, iD - 1), 0)
        z1s = tl.maximum(tl.minimum(iz1, iD - 1), 0)
        g = tl.where(pm, tl.load(go_ptr + go_base + p, mask=pm), 0.0)
        zero = tl.zeros([BLOCK_P], dtype=tl.float32)
        a0 = gi_base + z0s * isD
        a1 = gi_base + z1s * isD
        b0 = y0s * isH
        b1 = y1s * isH
        tl.atomic_add(
            gi_ptr + a0 + b0 + x0s,
            tl.where(
                x0ok & y0ok & z0ok, g * (1.0 - tx) * (1.0 - ty) * (1.0 - tz), zero
            ),
            mask=pm,
        )
        tl.atomic_add(
            gi_ptr + a1 + b0 + x0s,
            tl.where(x0ok & y0ok & z1ok, g * (1.0 - tx) * (1.0 - ty) * tz, zero),
            mask=pm,
        )
        tl.atomic_add(
            gi_ptr + a0 + b1 + x0s,
            tl.where(x0ok & y1ok & z0ok, g * (1.0 - tx) * ty * (1.0 - tz), zero),
            mask=pm,
        )
        tl.atomic_add(
            gi_ptr + a1 + b1 + x0s,
            tl.where(x0ok & y1ok & z1ok, g * (1.0 - tx) * ty * tz, zero),
            mask=pm,
        )
        tl.atomic_add(
            gi_ptr + a0 + b0 + x1s,
            tl.where(x1ok & y0ok & z0ok, g * tx * (1.0 - ty) * (1.0 - tz), zero),
            mask=pm,
        )
        tl.atomic_add(
            gi_ptr + a1 + b0 + x1s,
            tl.where(x1ok & y0ok & z1ok, g * tx * (1.0 - ty) * tz, zero),
            mask=pm,
        )
        tl.atomic_add(
            gi_ptr + a0 + b1 + x1s,
            tl.where(x1ok & y1ok & z0ok, g * tx * ty * (1.0 - tz), zero),
            mask=pm,
        )
        tl.atomic_add(
            gi_ptr + a1 + b1 + x1s,
            tl.where(x1ok & y1ok & z1ok, g * tx * ty * tz, zero),
            mask=pm,
        )


@triton.jit
def _grad_input_nearest_kernel(
    go_ptr,
    grid_ptr,
    gi_ptr,
    C,
    iD,
    iH,
    iW,
    oH,
    oW,
    go_sN,
    go_sC,
    gi_sN,
    gi_sC,
    grid_sN,
    grid_sD,
    grid_sH,
    grid_sW,
    n_per_batch,
    n_tiles,
    padding_mode: tl.constexpr,
    align_corners: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    go_base = n * go_sN + c * go_sC
    gi_base = n * gi_sN + c * gi_sC
    isD = iH * iW
    isH = iW
    for t in range(n_tiles):
        p = t * BLOCK_P + tl.arange(0, BLOCK_P)
        pm = p < n_per_batch
        ow = p % oW
        t1 = p // oW
        oh = t1 % oH
        od = t1 // oH
        gb = n * grid_sN + od * grid_sD + oh * grid_sH + ow * grid_sW
        gx = tl.where(pm, tl.load(grid_ptr + gb, mask=pm), 0.0)
        gy = tl.where(pm, tl.load(grid_ptr + gb + 1, mask=pm), 0.0)
        gz = tl.where(pm, tl.load(grid_ptr + gb + 2, mask=pm), 0.0)
        ix = _pad(_src_index(gx, iW, align_corners), iW, padding_mode, align_corners)
        iy = _pad(_src_index(gy, iH, align_corners), iH, padding_mode, align_corners)
        iz = _pad(_src_index(gz, iD, align_corners), iD, padding_mode, align_corners)
        xn = tl.floor(ix + 0.5).to(tl.int32)
        yn = tl.floor(iy + 0.5).to(tl.int32)
        zn = tl.floor(iz + 0.5).to(tl.int32)
        ok = (
            (xn >= 0)
            & (xn < iW)
            & (yn >= 0)
            & (yn < iH)
            & (zn >= 0)
            & (zn < iD)
        )
        xs = tl.maximum(tl.minimum(xn, iW - 1), 0)
        ys = tl.maximum(tl.minimum(yn, iH - 1), 0)
        zs = tl.maximum(tl.minimum(zn, iD - 1), 0)
        g = tl.where(pm, tl.load(go_ptr + go_base + p, mask=pm), 0.0)
        zero = tl.zeros([BLOCK_P], dtype=tl.float32)
        tl.atomic_add(
            gi_ptr + gi_base + zs * isD + ys * isH + xs,
            tl.where(ok, g, zero),
            mask=pm,
        )


@triton.jit
def _grad_grid_bilinear_kernel(
    go_ptr,
    inp_ptr,
    grid_ptr,
    gg_ptr,
    C,
    c_begin,
    c_end,
    iD,
    iH,
    iW,
    oD,
    oH,
    oW,
    go_sN,
    go_sC,
    inp_sN,
    inp_sC,
    grid_sN,
    grid_sD,
    grid_sH,
    grid_sW,
    gg_sN,
    gg_sD,
    gg_sH,
    gg_sW,
    padding_mode: tl.constexpr,
    align_corners: tl.constexpr,
    ACCUMULATE: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    ow = pid % oW
    t1 = pid // oW
    oh = t1 % oH
    t2 = t1 // oH
    od = t2 % oD
    n = t2 // oD
    gb = n * grid_sN + od * grid_sD + oh * grid_sH + ow * grid_sW
    gx = tl.load(grid_ptr + gb)
    gy = tl.load(grid_ptr + gb + 1)
    gz = tl.load(grid_ptr + gb + 2)

    rx = _src_index(gx, iW, align_corners)
    ry = _src_index(gy, iH, align_corners)
    rz = _src_index(gz, iD, align_corners)

    if align_corners:
        mx = 0.5 * (iW - 1)
        my = 0.5 * (iH - 1)
        mz = 0.5 * (iD - 1)
    else:
        mx = 0.5 * iW
        my = 0.5 * iH
        mz = 0.5 * iD
    if padding_mode == 1:
        px = _clip_grad(rx, iW)
        py = _clip_grad(ry, iH)
        pz = _clip_grad(rz, iD)
    elif padding_mode == 2:
        px = _reflect_grad(rx, iW, align_corners)
        py = _reflect_grad(ry, iH, align_corners)
        pz = _reflect_grad(rz, iD, align_corners)
    else:
        px = 1.0
        py = 1.0
        pz = 1.0

    ix = _pad(rx, iW, padding_mode, align_corners)
    iy = _pad(ry, iH, padding_mode, align_corners)
    iz = _pad(rz, iD, padding_mode, align_corners)

    ix0 = tl.floor(ix).to(tl.int32)
    iy0 = tl.floor(iy).to(tl.int32)
    iz0 = tl.floor(iz).to(tl.int32)
    tx = ix - ix0.to(tl.float32)
    ty = iy - iy0.to(tl.float32)
    tz = iz - iz0.to(tl.float32)
    ix1 = ix0 + 1
    iy1 = iy0 + 1
    iz1 = iz0 + 1
    x0ok = (ix0 >= 0) & (ix0 < iW)
    x1ok = (ix1 >= 0) & (ix1 < iW)
    y0ok = (iy0 >= 0) & (iy0 < iH)
    y1ok = (iy1 >= 0) & (iy1 < iH)
    z0ok = (iz0 >= 0) & (iz0 < iD)
    z1ok = (iz1 >= 0) & (iz1 < iD)
    x0s = tl.maximum(tl.minimum(ix0, iW - 1), 0)
    x1s = tl.maximum(tl.minimum(ix1, iW - 1), 0)
    y0s = tl.maximum(tl.minimum(iy0, iH - 1), 0)
    y1s = tl.maximum(tl.minimum(iy1, iH - 1), 0)
    z0s = tl.maximum(tl.minimum(iz0, iD - 1), 0)
    z1s = tl.maximum(tl.minimum(iz1, iD - 1), 0)
    isD = iH * iW
    isH = iW
    q000 = z0s * isD + y0s * isH + x0s
    q001 = z1s * isD + y0s * isH + x0s
    q010 = z0s * isD + y1s * isH + x0s
    q011 = z1s * isD + y1s * isH + x0s
    q100 = z0s * isD + y0s * isH + x1s
    q101 = z1s * isD + y0s * isH + x1s
    q110 = z0s * isD + y1s * isH + x1s
    q111 = z1s * isD + y1s * isH + x1s

    cs = c_begin + tl.arange(0, BLOCK_C)
    cm = cs < c_end
    # Clamp the channel index so every lane addresses legal memory; the payload
    # is zeroed afterwards with tl.where (masks are not honoured here).
    csf = tl.where(cm, cs, 0)
    g = tl.load(go_ptr + n * go_sN + csf * go_sC + (od * oH + oh) * oW + ow)
    g = tl.where(cm, g, 0.0)
    ib = n * inp_sN + csf * inp_sC
    i000 = tl.load(inp_ptr + ib + q000)
    i001 = tl.load(inp_ptr + ib + q001)
    i010 = tl.load(inp_ptr + ib + q010)
    i011 = tl.load(inp_ptr + ib + q011)
    i100 = tl.load(inp_ptr + ib + q100)
    i101 = tl.load(inp_ptr + ib + q101)
    i110 = tl.load(inp_ptr + ib + q110)
    i111 = tl.load(inp_ptr + ib + q111)
    zero = tl.zeros([BLOCK_C], dtype=tl.float32)
    i000 = tl.where(x0ok & y0ok & z0ok, i000, zero)
    i001 = tl.where(x0ok & y0ok & z1ok, i001, zero)
    i010 = tl.where(x0ok & y1ok & z0ok, i010, zero)
    i011 = tl.where(x0ok & y1ok & z1ok, i011, zero)
    i100 = tl.where(x1ok & y0ok & z0ok, i100, zero)
    i101 = tl.where(x1ok & y0ok & z1ok, i101, zero)
    i110 = tl.where(x1ok & y1ok & z0ok, i110, zero)
    i111 = tl.where(x1ok & y1ok & z1ok, i111, zero)
    ax = g * (
        (1.0 - ty) * (1.0 - tz) * (i100 - i000)
        + (1.0 - ty) * tz * (i101 - i001)
        + ty * (1.0 - tz) * (i110 - i010)
        + ty * tz * (i111 - i011)
    )
    ay = g * (
        (1.0 - tx) * (1.0 - tz) * (i010 - i000)
        + (1.0 - tx) * tz * (i011 - i001)
        + tx * (1.0 - tz) * (i110 - i100)
        + tx * tz * (i111 - i101)
    )
    az = g * (
        (1.0 - tx) * (1.0 - ty) * (i001 - i000)
        + (1.0 - tx) * ty * (i011 - i010)
        + tx * (1.0 - ty) * (i101 - i100)
        + tx * ty * (i111 - i110)
    )
    sx = tl.sum(ax) * mx * px
    sy = tl.sum(ay) * my * py
    sz = tl.sum(az) * mz * pz
    ggb = n * gg_sN + od * gg_sD + oh * gg_sH + ow * gg_sW
    if ACCUMULATE:
        sx = sx + tl.load(gg_ptr + ggb)
        sy = sy + tl.load(gg_ptr + ggb + 1)
        sz = sz + tl.load(gg_ptr + ggb + 2)
    tl.store(gg_ptr + ggb, sx)
    tl.store(gg_ptr + ggb + 1, sy)
    tl.store(gg_ptr + ggb + 2, sz)


def _block_p(n_per_batch: int) -> int:
    # 64 / 256 / 1024 are the tile widths validated on this backend for the
    # grad_input kernel; latency is insensitive to the choice (306.8 / 306.8 /
    # 307.1 ms on the largest official cell), so pick the smallest that is not
    # wasteful.
    if n_per_batch <= 64:
        return 64
    return 256


def grid_sampler_3d_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    grid: torch.Tensor,
    interpolation_mode: int,
    padding_mode: int,
    align_corners: bool,
    output_mask: list,
) -> tuple:
    logger.debug(
        "GEMS_KUNLUNXIN GRID_SAMPLER_3D_BACKWARD: mode=%d pad=%d align=%s",
        interpolation_mode,
        padding_mode,
        align_corners,
    )

    assert grad_output.dtype in (
        torch.float32,
        torch.float64,
    ), f"grid_sampler_3d_backward only supports float32/float64, got {grad_output.dtype}"
    assert (
        input.dtype == grad_output.dtype
    ), "input and grad_output must have same dtype"
    assert grid.dtype == grad_output.dtype, "grid and grad_output must have same dtype"

    N, C, iD, iH, iW = input.shape
    _, oD, oH, oW, _ = grid.shape

    need_grad_input = bool(output_mask[0])
    need_grad_grid = bool(output_mask[1])

    grad_output = grad_output.contiguous()
    input = input.contiguous()
    grid = grid.contiguous()

    grad_input = torch.zeros(
        (N, C, iD, iH, iW), dtype=torch.float32, device=input.device
    )
    grad_grid = torch.zeros(
        (N, oD, oH, oW, 3), dtype=torch.float32, device=grid.device
    )

    n_per_batch = oD * oH * oW
    num_elements = N * n_per_batch
    if num_elements == 0 or C == 0:
        return grad_input.to(input.dtype), grad_grid.to(grid.dtype)

    grad_output_f32 = grad_output.float()
    input_f32 = input.float()
    grid_f32 = grid.float()

    if need_grad_input:
        block_p = _block_p(n_per_batch)
        n_tiles = triton.cdiv(n_per_batch, block_p)
        kernel = (
            _grad_input_bilinear_kernel
            if interpolation_mode == 0
            else _grad_input_nearest_kernel
        )
        kernel[(N * C,)](
            grad_output_f32,
            grid_f32,
            grad_input,
            C,
            iD,
            iH,
            iW,
            oH,
            oW,
            grad_output_f32.stride(0),
            grad_output_f32.stride(1),
            grad_input.stride(0),
            grad_input.stride(1),
            grid_f32.stride(0),
            grid_f32.stride(1),
            grid_f32.stride(2),
            grid_f32.stride(3),
            n_per_batch,
            n_tiles,
            padding_mode=padding_mode,
            align_corners=align_corners,
            BLOCK_P=block_p,
        )

    # Nearest interpolation has zero gradient w.r.t. the grid, and grad_grid is
    # already zero-filled, so only bilinear needs the second kernel.
    if need_grad_grid and interpolation_mode == 0:
        block_c = max(_MIN_BLOCK_C, min(_MAX_BLOCK_C, triton.next_power_of_2(C)))
        for c_begin in range(0, C, block_c):
            _grad_grid_bilinear_kernel[(num_elements,)](
                grad_output_f32,
                input_f32,
                grid_f32,
                grad_grid,
                C,
                c_begin,
                min(c_begin + block_c, C),
                iD,
                iH,
                iW,
                oD,
                oH,
                oW,
                grad_output_f32.stride(0),
                grad_output_f32.stride(1),
                input_f32.stride(0),
                input_f32.stride(1),
                grid_f32.stride(0),
                grid_f32.stride(1),
                grid_f32.stride(2),
                grid_f32.stride(3),
                grad_grid.stride(0),
                grad_grid.stride(1),
                grad_grid.stride(2),
                grad_grid.stride(3),
                padding_mode=padding_mode,
                align_corners=align_corners,
                ACCUMULATE=(c_begin > 0),
                BLOCK_C=block_c,
            )

    return grad_input.to(input.dtype), grad_grid.to(grid.dtype)
