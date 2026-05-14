"""
Grid sample backward operators for FlagGems.

Implements grid_sampler_2d_backward and grid_sampler_3d_backward to match
PyTorch's autograd contract.  Forward lives in :mod:`grid_sample` and only
computes the output tensor; this module provides the gradient w.r.t. input
and grid that PyTorch's autograd dispatches when ``output.backward()`` is
called.

Kernel design:
    * Each program handles ONE output pixel ``(n, h_out, w_out)`` for 2D, or
      one output voxel ``(n, d_out, h_out, w_out)`` for 3D, and loops over
      channels internally in ``BLOCK_C`` lanes.  This way the grid-coord
      arithmetic + denormalization + padding logic is done once per spatial
      location and reused over all channels.
    * ``grad_input`` is accumulated with ``tl.atomic_add`` (multiple output
      pixels can map to the same input pixel).
    * ``grad_grid`` is written directly (the program uniquely owns its
      ``(n, h_out, w_out)`` so no atomic needed); the channel reduction is
      accumulated in registers.

Padding semantics:
    * zeros:      sample == 0 if sample index is out of bounds.
    * border:     clip the *continuous* coordinate to ``[0, size-1]``; the
                  gradient through that clip is killed when active.
    * reflection: triangle-wave reflection of the continuous coordinate; the
                  gradient picks up the local sign (+1 or -1) of the wave.

All three padding modes (zeros / border / reflection) are implemented directly
inside every kernel via a ``padding_mode_id: tl.constexpr`` switch; bicubic
applies the padding per individual 4x4-stencil sample index to match
PyTorch's per-corner clip semantics.  No PyTorch autograd fallback is used.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


_INTERP_BILINEAR = 0
_INTERP_NEAREST = 1
_INTERP_BICUBIC = 2

_PAD_ZEROS = 0
_PAD_BORDER = 1
_PAD_REFLECTION = 2


# ---------------------------------------------------------------------------
# Tiny helpers used by every kernel.
# ---------------------------------------------------------------------------
@triton.jit
def _denorm_with_grad(g, size, align_corners: tl.constexpr):
    """Normalized grid coord g in [-1, 1] -> pixel-space coord; return (x, dx/dg)."""
    if align_corners:
        scale = (size - 1) / 2.0
        x = (g + 1.0) * scale
    else:
        scale = size / 2.0
        x = (g + 1.0) * scale - 0.5
    return x, scale


@triton.jit
def _reflect_with_grad(x, low, high):
    """Triangle-wave reflection of x into [low, high]; return (x_refl, sign)."""
    span = high - low
    period = 2.0 * span
    x_shift = x - low
    n = tl.floor(x_shift / period)
    x_mod = x_shift - n * period
    folded = x_mod > span
    x_norm = tl.where(folded, period - x_mod, x_mod)
    sign = tl.where(folded, -1.0, 1.0)
    return x_norm + low, sign


# ---------------------------------------------------------------------------
# 2D bilinear backward.
# ---------------------------------------------------------------------------
@libentry()
@triton.jit
def grid_sample_2d_bilinear_bwd_kernel(
    grad_output_ptr,
    input_ptr,
    grid_ptr,
    grad_input_ptr,
    grad_grid_ptr,
    N,
    C,
    H_in,
    W_in,
    H_out,
    W_out,
    align_corners: tl.constexpr,
    padding_mode_id: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    HW = H_out * W_out
    n = pid // HW
    rem = pid % HW
    h_out = rem // W_out
    w_out = rem % W_out

    grid_base = n * H_out * W_out * 2 + h_out * W_out * 2 + w_out * 2
    gx = tl.load(grid_ptr + grid_base).to(tl.float32)
    gy = tl.load(grid_ptr + grid_base + 1).to(tl.float32)

    x, x_scale = _denorm_with_grad(gx, W_in, align_corners)
    y, y_scale = _denorm_with_grad(gy, H_in, align_corners)

    if padding_mode_id == 2:
        if align_corners:
            x, sx = _reflect_with_grad(x, 0.0, W_in - 1.0)
            y, sy = _reflect_with_grad(y, 0.0, H_in - 1.0)
        else:
            x, sx = _reflect_with_grad(x, -0.5, W_in * 1.0 - 0.5)
            y, sy = _reflect_with_grad(y, -0.5, H_in * 1.0 - 0.5)
        mul_gx = x_scale * sx
        mul_gy = y_scale * sy
        cx = (x < 0.0) | (x > (W_in - 1))
        cy = (y < 0.0) | (y > (H_in - 1))
        x = tl.maximum(0.0, tl.minimum(x, W_in - 1.0))
        y = tl.maximum(0.0, tl.minimum(y, H_in - 1.0))
        mul_gx = tl.where(cx, 0.0, mul_gx)
        mul_gy = tl.where(cy, 0.0, mul_gy)
    elif padding_mode_id == 1:
        cx = (x < 0.0) | (x > (W_in - 1))
        cy = (y < 0.0) | (y > (H_in - 1))
        x = tl.maximum(0.0, tl.minimum(x, W_in - 1.0))
        y = tl.maximum(0.0, tl.minimum(y, H_in - 1.0))
        mul_gx = tl.where(cx, 0.0, x_scale)
        mul_gy = tl.where(cy, 0.0, y_scale)
    else:  # zeros
        mul_gx = x_scale
        mul_gy = y_scale

    x0 = tl.floor(x)
    y0 = tl.floor(y)
    x0i = tl.cast(x0, tl.int32)
    y0i = tl.cast(y0, tl.int32)
    x1i = x0i + 1
    y1i = y0i + 1
    wx = x - x0
    wy = y - y0

    x0_in = (x0i >= 0) & (x0i < W_in)
    x1_in = (x1i >= 0) & (x1i < W_in)
    y0_in = (y0i >= 0) & (y0i < H_in)
    y1_in = (y1i >= 0) & (y1i < H_in)
    in00 = x0_in & y0_in
    in01 = x1_in & y0_in
    in10 = x0_in & y1_in
    in11 = x1_in & y1_in

    w00 = (1.0 - wx) * (1.0 - wy)
    w01 = wx * (1.0 - wy)
    w10 = (1.0 - wx) * wy
    w11 = wx * wy

    g_gx_acc = 0.0
    g_gy_acc = 0.0

    c_offs = tl.arange(0, BLOCK_C)
    for c_base in tl.range(0, C, BLOCK_C):
        cs = c_base + c_offs
        c_mask = cs < C

        go_off = n * C * H_out * W_out + cs * H_out * W_out + h_out * W_out + w_out
        go = tl.load(grad_output_ptr + go_off, mask=c_mask, other=0.0).to(tl.float32)

        in_base = n * C * H_in * W_in + cs * H_in * W_in
        off00 = in_base + y0i * W_in + x0i
        off01 = in_base + y0i * W_in + x1i
        off10 = in_base + y1i * W_in + x0i
        off11 = in_base + y1i * W_in + x1i

        p00 = tl.load(input_ptr + off00, mask=c_mask & in00, other=0.0).to(tl.float32)
        p01 = tl.load(input_ptr + off01, mask=c_mask & in01, other=0.0).to(tl.float32)
        p10 = tl.load(input_ptr + off10, mask=c_mask & in10, other=0.0).to(tl.float32)
        p11 = tl.load(input_ptr + off11, mask=c_mask & in11, other=0.0).to(tl.float32)

        tl.atomic_add(grad_input_ptr + off00, go * w00, mask=c_mask & in00)
        tl.atomic_add(grad_input_ptr + off01, go * w01, mask=c_mask & in01)
        tl.atomic_add(grad_input_ptr + off10, go * w10, mask=c_mask & in10)
        tl.atomic_add(grad_input_ptr + off11, go * w11, mask=c_mask & in11)

        dout_dx = (p01 - p00) * (1.0 - wy) + (p11 - p10) * wy
        dout_dy = (p10 - p00) * (1.0 - wx) + (p11 - p01) * wx

        g_gx_acc += tl.sum(go * dout_dx, axis=0)
        g_gy_acc += tl.sum(go * dout_dy, axis=0)

    tl.store(grad_grid_ptr + grid_base, g_gx_acc * mul_gx)
    tl.store(grad_grid_ptr + grid_base + 1, g_gy_acc * mul_gy)


# ---------------------------------------------------------------------------
# 2D nearest backward.  grad_grid is identically zero.
# ---------------------------------------------------------------------------
@libentry()
@triton.jit
def grid_sample_2d_nearest_bwd_kernel(
    grad_output_ptr,
    grid_ptr,
    grad_input_ptr,
    N,
    C,
    H_in,
    W_in,
    H_out,
    W_out,
    align_corners: tl.constexpr,
    padding_mode_id: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    HW = H_out * W_out
    n = pid // HW
    rem = pid % HW
    h_out = rem // W_out
    w_out = rem % W_out

    grid_base = n * H_out * W_out * 2 + h_out * W_out * 2 + w_out * 2
    gx = tl.load(grid_ptr + grid_base).to(tl.float32)
    gy = tl.load(grid_ptr + grid_base + 1).to(tl.float32)

    x, _ = _denorm_with_grad(gx, W_in, align_corners)
    y, _ = _denorm_with_grad(gy, H_in, align_corners)

    if padding_mode_id == 2:
        if align_corners:
            x, _ = _reflect_with_grad(x, 0.0, W_in - 1.0)
            y, _ = _reflect_with_grad(y, 0.0, H_in - 1.0)
        else:
            x, _ = _reflect_with_grad(x, -0.5, W_in * 1.0 - 0.5)
            y, _ = _reflect_with_grad(y, -0.5, H_in * 1.0 - 0.5)
        x = tl.maximum(0.0, tl.minimum(x, W_in - 1.0))
        y = tl.maximum(0.0, tl.minimum(y, H_in - 1.0))
    elif padding_mode_id == 1:
        x = tl.maximum(0.0, tl.minimum(x, W_in - 1.0))
        y = tl.maximum(0.0, tl.minimum(y, H_in - 1.0))

    xr = tl.cast(tl.floor(x + 0.5), tl.int32)
    yr = tl.cast(tl.floor(y + 0.5), tl.int32)
    in_bounds = (xr >= 0) & (xr < W_in) & (yr >= 0) & (yr < H_in)

    c_offs = tl.arange(0, BLOCK_C)
    for c_base in tl.range(0, C, BLOCK_C):
        cs = c_base + c_offs
        c_mask = cs < C

        go_off = n * C * H_out * W_out + cs * H_out * W_out + h_out * W_out + w_out
        go = tl.load(grad_output_ptr + go_off, mask=c_mask, other=0.0).to(tl.float32)

        gi_off = n * C * H_in * W_in + cs * H_in * W_in + yr * W_in + xr
        tl.atomic_add(grad_input_ptr + gi_off, go, mask=c_mask & in_bounds)


# ---------------------------------------------------------------------------
# 2D bicubic backward (zeros padding only -- border/reflection delegated to
# torch in the launcher).  Cubic kernel with A=-0.75 (PyTorch convention).
# ---------------------------------------------------------------------------
@triton.jit
def _cubic_w_and_dw(wx):
    t0 = wx + 1.0
    t1 = wx
    t2 = 1.0 - wx
    t3 = 2.0 - wx
    W0 = -0.75 * t0 * t0 * t0 + 3.75 * t0 * t0 - 6.0 * t0 + 3.0
    W1 = 1.25 * t1 * t1 * t1 - 2.25 * t1 * t1 + 1.0
    W2 = 1.25 * t2 * t2 * t2 - 2.25 * t2 * t2 + 1.0
    W3 = -0.75 * t3 * t3 * t3 + 3.75 * t3 * t3 - 6.0 * t3 + 3.0
    dW0 = -2.25 * t0 * t0 + 7.5 * t0 - 6.0
    dW1 = 3.75 * t1 * t1 - 4.5 * t1
    dW2 = -(3.75 * t2 * t2 - 4.5 * t2)
    dW3 = -(-2.25 * t3 * t3 + 7.5 * t3 - 6.0)
    return W0, W1, W2, W3, dW0, dW1, dW2, dW3


@libentry()
@triton.jit
def grid_sample_2d_bicubic_bwd_kernel(
    grad_output_ptr,
    input_ptr,
    grid_ptr,
    grad_input_ptr,
    grad_grid_ptr,
    N,
    C,
    H_in,
    W_in,
    H_out,
    W_out,
    align_corners: tl.constexpr,
    padding_mode_id: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Bicubic backward.

    Padding semantics for bicubic match PyTorch's per-sample-index rule:
      * zeros:      out-of-bounds samples contribute 0; corresponding
                    grad_input write is masked off.
      * border:     each sample index is clipped to [0, size-1] *before* the
                    read/atomic_add; the continuous x/y are NOT clipped, so
                    the cubic weights still come from the raw fractional offset.
      * reflection: each sample index is integer-reflected via the standard
                    triangle wave with period 2*(size-1) or 2*size depending
                    on align_corners.
    The gradient through the discrete index-padding is 0, so grad_grid uses
    the unmodified x_scale / y_scale (no `mul_gx` clipping like bilinear).
    """
    pid = tl.program_id(0)
    HW = H_out * W_out
    n = pid // HW
    rem = pid % HW
    h_out = rem // W_out
    w_out = rem % W_out

    grid_base = n * H_out * W_out * 2 + h_out * W_out * 2 + w_out * 2
    gx = tl.load(grid_ptr + grid_base).to(tl.float32)
    gy = tl.load(grid_ptr + grid_base + 1).to(tl.float32)

    x, x_scale = _denorm_with_grad(gx, W_in, align_corners)
    y, y_scale = _denorm_with_grad(gy, H_in, align_corners)

    x0 = tl.floor(x)
    y0 = tl.floor(y)
    x0i = tl.cast(x0, tl.int32)
    y0i = tl.cast(y0, tl.int32)
    wx = x - x0
    wy = y - y0

    # Constants for the per-sample reflection arithmetic (only used when
    # padding_mode_id == 2, but cheap to compute either way).
    if align_corners:
        period_x = 2 * (W_in - 1)
        period_y = 2 * (H_in - 1)
    else:
        period_x = 2 * W_in
        period_y = 2 * H_in

    Wx0, Wx1, Wx2, Wx3, dWx0, dWx1, dWx2, dWx3 = _cubic_w_and_dw(wx)
    Wy0, Wy1, Wy2, Wy3, dWy0, dWy1, dWy2, dWy3 = _cubic_w_and_dw(wy)

    g_gx_acc = 0.0
    g_gy_acc = 0.0

    c_offs = tl.arange(0, BLOCK_C)
    for c_base in tl.range(0, C, BLOCK_C):
        cs = c_base + c_offs
        c_mask = cs < C

        go_off = n * C * H_out * W_out + cs * H_out * W_out + h_out * W_out + w_out
        go = tl.load(grad_output_ptr + go_off, mask=c_mask, other=0.0).to(tl.float32)

        in_base = n * C * H_in * W_in + cs * H_in * W_in

        dout_dx_acc = tl.zeros_like(go)
        dout_dy_acc = tl.zeros_like(go)

        # 16 neighbors hand-unrolled via Python range loop.  Each iteration's
        # `di`/`dj` is a Python int at trace time, so the if-elif weight
        # selection + per-sample padding expand to specialized code paths.
        for di_off in range(4):
            di = di_off - 1
            if di_off == 0:
                Wyi = Wy0
                dWyi = dWy0
            elif di_off == 1:
                Wyi = Wy1
                dWyi = dWy1
            elif di_off == 2:
                Wyi = Wy2
                dWyi = dWy2
            else:
                Wyi = Wy3
                dWyi = dWy3
            yy_raw = y0i + di

            # Per-sample y padding.
            if padding_mode_id == 1:  # border
                yy_eff = tl.maximum(0, tl.minimum(yy_raw, H_in - 1))
                yy_in = yy_raw == yy_raw  # always True
            elif padding_mode_id == 2:  # reflection
                ym = (yy_raw % period_y + period_y) % period_y
                if align_corners:
                    yy_eff = tl.where(ym <= H_in - 1, ym, period_y - ym)
                else:
                    yy_eff = tl.where(ym <= H_in - 1, ym, period_y - 1 - ym)
                yy_in = yy_raw == yy_raw  # always True
            else:  # zeros
                yy_eff = yy_raw
                yy_in = (yy_raw >= 0) & (yy_raw < H_in)

            for dj_off in range(4):
                dj = dj_off - 1
                if dj_off == 0:
                    Wxj = Wx0
                    dWxj = dWx0
                elif dj_off == 1:
                    Wxj = Wx1
                    dWxj = dWx1
                elif dj_off == 2:
                    Wxj = Wx2
                    dWxj = dWx2
                else:
                    Wxj = Wx3
                    dWxj = dWx3
                xx_raw = x0i + dj

                if padding_mode_id == 1:  # border
                    xx_eff = tl.maximum(0, tl.minimum(xx_raw, W_in - 1))
                    in_ij = yy_in
                elif padding_mode_id == 2:  # reflection
                    xm = (xx_raw % period_x + period_x) % period_x
                    if align_corners:
                        xx_eff = tl.where(xm <= W_in - 1, xm, period_x - xm)
                    else:
                        xx_eff = tl.where(xm <= W_in - 1, xm, period_x - 1 - xm)
                    in_ij = yy_in
                else:  # zeros
                    xx_eff = xx_raw
                    in_ij = yy_in & (xx_raw >= 0) & (xx_raw < W_in)

                p_off = in_base + yy_eff * W_in + xx_eff
                p = tl.load(input_ptr + p_off, mask=c_mask & in_ij, other=0.0).to(
                    tl.float32
                )
                tl.atomic_add(
                    grad_input_ptr + p_off, go * (Wxj * Wyi), mask=c_mask & in_ij
                )
                dout_dx_acc += p * (dWxj * Wyi)
                dout_dy_acc += p * (Wxj * dWyi)

        g_gx_acc += tl.sum(go * dout_dx_acc, axis=0)
        g_gy_acc += tl.sum(go * dout_dy_acc, axis=0)

    tl.store(grad_grid_ptr + grid_base, g_gx_acc * x_scale)
    tl.store(grad_grid_ptr + grid_base + 1, g_gy_acc * y_scale)


# ---------------------------------------------------------------------------
# 3D trilinear backward.
# ---------------------------------------------------------------------------
@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("grid_sample_3d_trilinear_bwd"),
    key=["C", "D_in", "H_in", "W_in"],
    # grad_input is atomic-accumulated; the autotune timing harness reruns
    # each config N times so we must zero grad_input between trials or the
    # accumulation explodes (saw ~1626x overshoot on fp32 zeros tests).
    reset_to_zero=["grad_input_ptr"],
)
@triton.jit
def grid_sample_3d_trilinear_bwd_kernel(
    grad_output_ptr,
    input_ptr,
    grid_ptr,
    grad_input_ptr,
    grad_grid_ptr,
    N,
    C,
    D_in,
    H_in,
    W_in,
    D_out,
    H_out,
    W_out,
    align_corners: tl.constexpr,
    padding_mode_id: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    DHW = D_out * H_out * W_out
    n = pid // DHW
    rem = pid % DHW
    d_out = rem // (H_out * W_out)
    rem2 = rem % (H_out * W_out)
    h_out = rem2 // W_out
    w_out = rem2 % W_out

    grid_base = (
        n * D_out * H_out * W_out * 3
        + d_out * H_out * W_out * 3
        + h_out * W_out * 3
        + w_out * 3
    )
    gx = tl.load(grid_ptr + grid_base).to(tl.float32)
    gy = tl.load(grid_ptr + grid_base + 1).to(tl.float32)
    gz = tl.load(grid_ptr + grid_base + 2).to(tl.float32)

    x, x_scale = _denorm_with_grad(gx, W_in, align_corners)
    y, y_scale = _denorm_with_grad(gy, H_in, align_corners)
    z, z_scale = _denorm_with_grad(gz, D_in, align_corners)

    if padding_mode_id == 2:
        if align_corners:
            x, sx = _reflect_with_grad(x, 0.0, W_in - 1.0)
            y, sy = _reflect_with_grad(y, 0.0, H_in - 1.0)
            z, sz = _reflect_with_grad(z, 0.0, D_in - 1.0)
        else:
            x, sx = _reflect_with_grad(x, -0.5, W_in * 1.0 - 0.5)
            y, sy = _reflect_with_grad(y, -0.5, H_in * 1.0 - 0.5)
            z, sz = _reflect_with_grad(z, -0.5, D_in * 1.0 - 0.5)
        mul_gx = x_scale * sx
        mul_gy = y_scale * sy
        mul_gz = z_scale * sz
        cx = (x < 0.0) | (x > (W_in - 1))
        cy = (y < 0.0) | (y > (H_in - 1))
        cz = (z < 0.0) | (z > (D_in - 1))
        x = tl.maximum(0.0, tl.minimum(x, W_in - 1.0))
        y = tl.maximum(0.0, tl.minimum(y, H_in - 1.0))
        z = tl.maximum(0.0, tl.minimum(z, D_in - 1.0))
        mul_gx = tl.where(cx, 0.0, mul_gx)
        mul_gy = tl.where(cy, 0.0, mul_gy)
        mul_gz = tl.where(cz, 0.0, mul_gz)
    elif padding_mode_id == 1:
        cx = (x < 0.0) | (x > (W_in - 1))
        cy = (y < 0.0) | (y > (H_in - 1))
        cz = (z < 0.0) | (z > (D_in - 1))
        x = tl.maximum(0.0, tl.minimum(x, W_in - 1.0))
        y = tl.maximum(0.0, tl.minimum(y, H_in - 1.0))
        z = tl.maximum(0.0, tl.minimum(z, D_in - 1.0))
        mul_gx = tl.where(cx, 0.0, x_scale)
        mul_gy = tl.where(cy, 0.0, y_scale)
        mul_gz = tl.where(cz, 0.0, z_scale)
    else:
        mul_gx = x_scale
        mul_gy = y_scale
        mul_gz = z_scale

    x0 = tl.floor(x)
    y0 = tl.floor(y)
    z0 = tl.floor(z)
    x0i = tl.cast(x0, tl.int32)
    y0i = tl.cast(y0, tl.int32)
    z0i = tl.cast(z0, tl.int32)
    x1i = x0i + 1
    y1i = y0i + 1
    z1i = z0i + 1
    wx = x - x0
    wy = y - y0
    wz = z - z0
    mx = 1.0 - wx
    my = 1.0 - wy
    mz = 1.0 - wz

    w000 = mz * my * mx
    w001 = mz * my * wx
    w010 = mz * wy * mx
    w011 = mz * wy * wx
    w100 = wz * my * mx
    w101 = wz * my * wx
    w110 = wz * wy * mx
    w111 = wz * wy * wx

    x0_in = (x0i >= 0) & (x0i < W_in)
    x1_in = (x1i >= 0) & (x1i < W_in)
    y0_in = (y0i >= 0) & (y0i < H_in)
    y1_in = (y1i >= 0) & (y1i < H_in)
    z0_in = (z0i >= 0) & (z0i < D_in)
    z1_in = (z1i >= 0) & (z1i < D_in)
    in000 = z0_in & y0_in & x0_in
    in001 = z0_in & y0_in & x1_in
    in010 = z0_in & y1_in & x0_in
    in011 = z0_in & y1_in & x1_in
    in100 = z1_in & y0_in & x0_in
    in101 = z1_in & y0_in & x1_in
    in110 = z1_in & y1_in & x0_in
    in111 = z1_in & y1_in & x1_in

    g_gx_acc = 0.0
    g_gy_acc = 0.0
    g_gz_acc = 0.0

    c_offs = tl.arange(0, BLOCK_C)
    for c_base in tl.range(0, C, BLOCK_C):
        cs = c_base + c_offs
        c_mask = cs < C

        go_off = (
            n * C * D_out * H_out * W_out
            + cs * D_out * H_out * W_out
            + d_out * H_out * W_out
            + h_out * W_out
            + w_out
        )
        go = tl.load(grad_output_ptr + go_off, mask=c_mask, other=0.0).to(tl.float32)

        in_base = n * C * D_in * H_in * W_in + cs * D_in * H_in * W_in
        sZ = H_in * W_in
        sY = W_in
        off000 = in_base + z0i * sZ + y0i * sY + x0i
        off001 = in_base + z0i * sZ + y0i * sY + x1i
        off010 = in_base + z0i * sZ + y1i * sY + x0i
        off011 = in_base + z0i * sZ + y1i * sY + x1i
        off100 = in_base + z1i * sZ + y0i * sY + x0i
        off101 = in_base + z1i * sZ + y0i * sY + x1i
        off110 = in_base + z1i * sZ + y1i * sY + x0i
        off111 = in_base + z1i * sZ + y1i * sY + x1i

        p000 = tl.load(input_ptr + off000, mask=c_mask & in000, other=0.0).to(
            tl.float32
        )
        p001 = tl.load(input_ptr + off001, mask=c_mask & in001, other=0.0).to(
            tl.float32
        )
        p010 = tl.load(input_ptr + off010, mask=c_mask & in010, other=0.0).to(
            tl.float32
        )
        p011 = tl.load(input_ptr + off011, mask=c_mask & in011, other=0.0).to(
            tl.float32
        )
        p100 = tl.load(input_ptr + off100, mask=c_mask & in100, other=0.0).to(
            tl.float32
        )
        p101 = tl.load(input_ptr + off101, mask=c_mask & in101, other=0.0).to(
            tl.float32
        )
        p110 = tl.load(input_ptr + off110, mask=c_mask & in110, other=0.0).to(
            tl.float32
        )
        p111 = tl.load(input_ptr + off111, mask=c_mask & in111, other=0.0).to(
            tl.float32
        )

        tl.atomic_add(grad_input_ptr + off000, go * w000, mask=c_mask & in000)
        tl.atomic_add(grad_input_ptr + off001, go * w001, mask=c_mask & in001)
        tl.atomic_add(grad_input_ptr + off010, go * w010, mask=c_mask & in010)
        tl.atomic_add(grad_input_ptr + off011, go * w011, mask=c_mask & in011)
        tl.atomic_add(grad_input_ptr + off100, go * w100, mask=c_mask & in100)
        tl.atomic_add(grad_input_ptr + off101, go * w101, mask=c_mask & in101)
        tl.atomic_add(grad_input_ptr + off110, go * w110, mask=c_mask & in110)
        tl.atomic_add(grad_input_ptr + off111, go * w111, mask=c_mask & in111)

        dout_dx = (
            (p001 - p000) * (mz * my)
            + (p011 - p010) * (mz * wy)
            + (p101 - p100) * (wz * my)
            + (p111 - p110) * (wz * wy)
        )
        dout_dy = (
            (p010 - p000) * (mz * mx)
            + (p011 - p001) * (mz * wx)
            + (p110 - p100) * (wz * mx)
            + (p111 - p101) * (wz * wx)
        )
        dout_dz = (
            (p100 - p000) * (my * mx)
            + (p101 - p001) * (my * wx)
            + (p110 - p010) * (wy * mx)
            + (p111 - p011) * (wy * wx)
        )

        g_gx_acc += tl.sum(go * dout_dx, axis=0)
        g_gy_acc += tl.sum(go * dout_dy, axis=0)
        g_gz_acc += tl.sum(go * dout_dz, axis=0)

    tl.store(grad_grid_ptr + grid_base, g_gx_acc * mul_gx)
    tl.store(grad_grid_ptr + grid_base + 1, g_gy_acc * mul_gy)
    tl.store(grad_grid_ptr + grid_base + 2, g_gz_acc * mul_gz)


# ---------------------------------------------------------------------------
# 3D nearest backward.
# ---------------------------------------------------------------------------
@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("grid_sample_3d_nearest_bwd"),
    key=["C", "D_in", "H_in", "W_in"],
    # Same atomic-accumulation reset as the trilinear bwd kernel above.
    reset_to_zero=["grad_input_ptr"],
)
@triton.jit
def grid_sample_3d_nearest_bwd_kernel(
    grad_output_ptr,
    grid_ptr,
    grad_input_ptr,
    N,
    C,
    D_in,
    H_in,
    W_in,
    D_out,
    H_out,
    W_out,
    align_corners: tl.constexpr,
    padding_mode_id: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    DHW = D_out * H_out * W_out
    n = pid // DHW
    rem = pid % DHW
    d_out = rem // (H_out * W_out)
    rem2 = rem % (H_out * W_out)
    h_out = rem2 // W_out
    w_out = rem2 % W_out

    grid_base = (
        n * D_out * H_out * W_out * 3
        + d_out * H_out * W_out * 3
        + h_out * W_out * 3
        + w_out * 3
    )
    gx = tl.load(grid_ptr + grid_base).to(tl.float32)
    gy = tl.load(grid_ptr + grid_base + 1).to(tl.float32)
    gz = tl.load(grid_ptr + grid_base + 2).to(tl.float32)

    x, _ = _denorm_with_grad(gx, W_in, align_corners)
    y, _ = _denorm_with_grad(gy, H_in, align_corners)
    z, _ = _denorm_with_grad(gz, D_in, align_corners)

    if padding_mode_id == 2:
        if align_corners:
            x, _ = _reflect_with_grad(x, 0.0, W_in - 1.0)
            y, _ = _reflect_with_grad(y, 0.0, H_in - 1.0)
            z, _ = _reflect_with_grad(z, 0.0, D_in - 1.0)
        else:
            x, _ = _reflect_with_grad(x, -0.5, W_in * 1.0 - 0.5)
            y, _ = _reflect_with_grad(y, -0.5, H_in * 1.0 - 0.5)
            z, _ = _reflect_with_grad(z, -0.5, D_in * 1.0 - 0.5)
        x = tl.maximum(0.0, tl.minimum(x, W_in - 1.0))
        y = tl.maximum(0.0, tl.minimum(y, H_in - 1.0))
        z = tl.maximum(0.0, tl.minimum(z, D_in - 1.0))
    elif padding_mode_id == 1:
        x = tl.maximum(0.0, tl.minimum(x, W_in - 1.0))
        y = tl.maximum(0.0, tl.minimum(y, H_in - 1.0))
        z = tl.maximum(0.0, tl.minimum(z, D_in - 1.0))

    xr = tl.cast(tl.floor(x + 0.5), tl.int32)
    yr = tl.cast(tl.floor(y + 0.5), tl.int32)
    zr = tl.cast(tl.floor(z + 0.5), tl.int32)
    in_bounds = (
        (xr >= 0) & (xr < W_in) & (yr >= 0) & (yr < H_in) & (zr >= 0) & (zr < D_in)
    )

    c_offs = tl.arange(0, BLOCK_C)
    for c_base in tl.range(0, C, BLOCK_C):
        cs = c_base + c_offs
        c_mask = cs < C

        go_off = (
            n * C * D_out * H_out * W_out
            + cs * D_out * H_out * W_out
            + d_out * H_out * W_out
            + h_out * W_out
            + w_out
        )
        go = tl.load(grad_output_ptr + go_off, mask=c_mask, other=0.0).to(tl.float32)

        gi_off = (
            n * C * D_in * H_in * W_in
            + cs * D_in * H_in * W_in
            + zr * H_in * W_in
            + yr * W_in
            + xr
        )
        tl.atomic_add(grad_input_ptr + gi_off, go, mask=c_mask & in_bounds)


# ---------------------------------------------------------------------------
# Python launchers matching torch aten signatures.
#
#   grid_sampler_2d_backward(grad_output, input, grid, interpolation_mode,
#                            padding_mode, align_corners, output_mask)
#   -> (grad_input, grad_grid)
# ---------------------------------------------------------------------------
def _choose_block_c(C):
    """Pick BLOCK_C as the next power of 2 ≥ min(C, 64)."""
    target = min(C, 64)
    bc = 1
    while bc < target:
        bc *= 2
    return max(bc, 8)


def grid_sampler_2d_backward(
    grad_output,
    input,
    grid,
    interpolation_mode,
    padding_mode,
    align_corners,
    output_mask=(True, True),
):
    """grad_output: (N, C, H_out, W_out); input: (N, C, H_in, W_in);
    grid: (N, H_out, W_out, 2).  Returns (grad_input, grad_grid)."""
    N, C, H_in, W_in = input.shape
    _, H_out, W_out, _ = grid.shape

    grad_output_c = grad_output.contiguous()
    input_c = input.contiguous()
    grid_c = grid.contiguous()

    # We accumulate gradients in fp32 then cast back to input/grid dtype.
    grad_input_f32 = torch.zeros(
        (N, C, H_in, W_in), dtype=torch.float32, device=input.device
    )
    grad_grid_f32 = torch.empty(
        (N, H_out, W_out, 2), dtype=torch.float32, device=grid.device
    )
    if interpolation_mode == _INTERP_NEAREST:
        grad_grid_f32.zero_()

    BLOCK_C = _choose_block_c(C)
    grid_launch = (N * H_out * W_out,)
    ac = bool(align_corners)

    if interpolation_mode == _INTERP_BILINEAR:
        grid_sample_2d_bilinear_bwd_kernel[grid_launch](
            grad_output_c.float(),
            input_c.float(),
            grid_c.float(),
            grad_input_f32,
            grad_grid_f32,
            N,
            C,
            H_in,
            W_in,
            H_out,
            W_out,
            ac,
            padding_mode,
            BLOCK_C,
        )
    elif interpolation_mode == _INTERP_NEAREST:
        grid_sample_2d_nearest_bwd_kernel[grid_launch](
            grad_output_c.float(),
            grid_c.float(),
            grad_input_f32,
            N,
            C,
            H_in,
            W_in,
            H_out,
            W_out,
            ac,
            padding_mode,
            BLOCK_C,
        )
    elif interpolation_mode == _INTERP_BICUBIC:
        grid_sample_2d_bicubic_bwd_kernel[grid_launch](
            grad_output_c.float(),
            input_c.float(),
            grid_c.float(),
            grad_input_f32,
            grad_grid_f32,
            N,
            C,
            H_in,
            W_in,
            H_out,
            W_out,
            ac,
            padding_mode,
            BLOCK_C,
        )
    else:
        raise NotImplementedError(f"interpolation_mode={interpolation_mode}")

    gi = grad_input_f32.to(input.dtype) if output_mask[0] else None
    gg = grad_grid_f32.to(grid.dtype) if output_mask[1] else None
    return gi, gg


def grid_sampler_3d_backward(
    grad_output,
    input,
    grid,
    interpolation_mode,
    padding_mode,
    align_corners,
    output_mask=(True, True),
):
    """grad_output: (N, C, D_out, H_out, W_out); input: (N, C, D_in, H_in, W_in);
    grid: (N, D_out, H_out, W_out, 3)."""
    if interpolation_mode == _INTERP_BICUBIC:
        raise NotImplementedError("grid_sample 3D does not support bicubic")

    N, C, D_in, H_in, W_in = input.shape
    _, D_out, H_out, W_out, _ = grid.shape

    grad_output_c = grad_output.contiguous()
    input_c = input.contiguous()
    grid_c = grid.contiguous()

    grad_input_f32 = torch.zeros(
        (N, C, D_in, H_in, W_in), dtype=torch.float32, device=input.device
    )
    grad_grid_f32 = torch.empty(
        (N, D_out, H_out, W_out, 3), dtype=torch.float32, device=grid.device
    )
    if interpolation_mode == _INTERP_NEAREST:
        grad_grid_f32.zero_()

    grid_launch = (N * D_out * H_out * W_out,)
    ac = bool(align_corners)

    # BLOCK_C is now selected by @triton.autotune over (BLOCK_C, num_warps,
    # num_stages); see tune_configs.yaml::grid_sample_3d_{trilinear,nearest}_bwd.
    if interpolation_mode == _INTERP_BILINEAR:  # 3D bilinear = trilinear
        grid_sample_3d_trilinear_bwd_kernel[grid_launch](
            grad_output_c.float(),
            input_c.float(),
            grid_c.float(),
            grad_input_f32,
            grad_grid_f32,
            N,
            C,
            D_in,
            H_in,
            W_in,
            D_out,
            H_out,
            W_out,
            ac,
            padding_mode,
        )
    elif interpolation_mode == _INTERP_NEAREST:
        grid_sample_3d_nearest_bwd_kernel[grid_launch](
            grad_output_c.float(),
            grid_c.float(),
            grad_input_f32,
            N,
            C,
            D_in,
            H_in,
            W_in,
            D_out,
            H_out,
            W_out,
            ac,
            padding_mode,
        )
    else:
        raise NotImplementedError(f"interpolation_mode={interpolation_mode}")

    gi = grad_input_f32.to(input.dtype) if output_mask[0] else None
    gg = grad_grid_f32.to(grid.dtype) if output_mask[1] else None
    return gi, gg
