import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# Interpolation mode constant used in validation (matches PyTorch ATen)
_INTERP_BICUBIC = 2


@triton.jit
def _cubic_weight(d):
    """Bicubic interpolation kernel weight, a=-0.75 (matches PyTorch)."""
    a = -0.75
    ad = tl.abs(d)
    ad2 = ad * ad
    ad3 = ad2 * ad
    w1 = (a + 2.0) * ad3 - (a + 3.0) * ad2 + 1.0
    w2 = a * ad3 - 5.0 * a * ad2 + 8.0 * a * ad - 4.0 * a
    return tl.where(ad <= 1.0, w1, tl.where(ad < 2.0, w2, 0.0))


@triton.jit
def _reflect_coord(x, size, align_corners: tl.constexpr):
    """
    Reflect coordinate x to lie within valid input range.

    align_corners=True:  reflects in [0, size-1]
    align_corners=False: reflects in [-0.5, size-0.5]
    """
    if align_corners:
        # twice_low=0, twice_high=2*(size-1) → half_span = size-1
        half_span = tl.cast(size - 1, tl.float32)
        # Edge case: size == 1 → half_span == 0, only valid coord is 0
        half_span = tl.maximum(half_span, 1e-10)
        span = half_span * 2.0
        x = tl.abs(x)
        floored = tl.floor(x / span)
        extra = x - floored * span
        return tl.where(extra <= half_span, extra, span - extra)
    else:
        # twice_low=-1, twice_high=2*size-1 → span = size, min = -0.5
        span = tl.cast(size, tl.float32)
        x = x + 0.5  # shift so [−0.5, size−0.5] → [0, size]
        x = tl.abs(x)
        dspan = span * 2.0
        floored = tl.floor(x / dspan)
        extra = x - floored * dspan
        return tl.where(extra <= span, extra, dspan - extra) - 0.5


@triton.jit
def _nearbyint_coord(x):
    flo = tl.floor(x)
    frac = x - flo
    rounded = tl.floor(x + 0.5)
    flo_i = flo.to(tl.int32)
    is_half = tl.abs(frac - 0.5) < 1e-6
    even_lower = (flo_i % 2) == 0
    return tl.where(is_half & even_lower, flo, rounded).to(tl.int32)


@triton.jit
def _clamp_and_mask_2d(
    xi,
    yi,
    W_in,
    H_in,
    stride_H,
    stride_W,
    hw_mask,
    padding_mode: tl.constexpr,
    align_corners: tl.constexpr,
):
    """
    Given integer pixel coordinates xi, yi of shape [BLOCK_HW], compute the
    address offset into a single channel plane and a validity mask.

    Returns (offset [BLOCK_HW], valid [BLOCK_HW]).
    """
    if padding_mode == 0:  # zeros
        xc = tl.maximum(0, tl.minimum(xi, W_in - 1))
        yc = tl.maximum(0, tl.minimum(yi, H_in - 1))
        valid = hw_mask & (xi >= 0) & (xi < W_in) & (yi >= 0) & (yi < H_in)
    elif padding_mode == 1:  # border
        xc = tl.maximum(0, tl.minimum(xi, W_in - 1))
        yc = tl.maximum(0, tl.minimum(yi, H_in - 1))
        valid = hw_mask
    else:  # reflection
        xf = _reflect_coord(xi.to(tl.float32), W_in, align_corners)
        yf = _reflect_coord(yi.to(tl.float32), H_in, align_corners)
        xc = tl.maximum(0, tl.minimum(xf.to(tl.int32), W_in - 1))
        yc = tl.maximum(0, tl.minimum(yf.to(tl.int32), H_in - 1))
        valid = hw_mask
    return yc * stride_H + xc * stride_W, valid


@triton.jit
def _clamp_and_mask_3d(
    xi,
    yi,
    zi,
    W_in,
    H_in,
    D_in,
    stride_D,
    stride_H,
    stride_W,
    dhw_mask,
    padding_mode: tl.constexpr,
    align_corners: tl.constexpr,
):
    """
    Given integer voxel coordinates xi, yi, zi of shape [BLOCK_DHW], compute the
    address offset into a single channel volume and a validity mask.

    Returns (offset [BLOCK_DHW], valid [BLOCK_DHW]).
    """
    if padding_mode == 0:  # zeros
        xc = tl.maximum(0, tl.minimum(xi, W_in - 1))
        yc = tl.maximum(0, tl.minimum(yi, H_in - 1))
        zc = tl.maximum(0, tl.minimum(zi, D_in - 1))
        valid = (
            dhw_mask
            & (xi >= 0)
            & (xi < W_in)
            & (yi >= 0)
            & (yi < H_in)
            & (zi >= 0)
            & (zi < D_in)
        )
    elif padding_mode == 1:  # border
        xc = tl.maximum(0, tl.minimum(xi, W_in - 1))
        yc = tl.maximum(0, tl.minimum(yi, H_in - 1))
        zc = tl.maximum(0, tl.minimum(zi, D_in - 1))
        valid = dhw_mask
    else:  # reflection
        xf = _reflect_coord(xi.to(tl.float32), W_in, align_corners)
        yf = _reflect_coord(yi.to(tl.float32), H_in, align_corners)
        zf = _reflect_coord(zi.to(tl.float32), D_in, align_corners)
        xc = tl.maximum(0, tl.minimum(xf.to(tl.int32), W_in - 1))
        yc = tl.maximum(0, tl.minimum(yf.to(tl.int32), H_in - 1))
        zc = tl.maximum(0, tl.minimum(zf.to(tl.int32), D_in - 1))
        valid = dhw_mask
    return zc * stride_D + yc * stride_H + xc * stride_W, valid


@triton.jit
def _cubic_weight_grad(t):
    a = -0.75
    x0 = -1.0 - t
    x1 = -t
    x2 = 1.0 - t
    x3 = 2.0 - t
    # Each g_i = dw_i/dt = dw_i/d(x_i) * d(x_i)/dt, where d(x_i)/dt = -1
    g0 = -((-3.0 * a * x0 - 10.0 * a) * x0 - 8.0 * a)
    g1 = -((-3.0 * (a + 2.0) * x1 - 2.0 * (a + 3.0)) * x1)
    g2 = -((3.0 * (a + 2.0) * x2 - 2.0 * (a + 3.0)) * x2)
    g3 = -((3.0 * a * x3 - 10.0 * a) * x3 + 8.0 * a)
    return g0, g1, g2, g3


_GRID_SAMPLER_2D_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_HW": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_HW": 256}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_HW": 512}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_HW": 1024}, num_warps=8, num_stages=2),
]
_GRID_SAMPLER_2D_AUTOTUNE_KEY = [
    "C",
    "HW_out",
    "interpolation_mode",
    "padding_mode",
]
_GRID_SAMPLER_2D_BACKWARD_AUTOTUNE_KEY = ["C", "HW_out"]

_GRID_SAMPLER_3D_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_DHW": 32}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_DHW": 64}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_DHW": 128}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_DHW": 128}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_DHW": 256}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_DHW": 512}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_DHW": 1024}, num_warps=16, num_stages=2),
    triton.Config({"BLOCK_DHW": 2048}, num_warps=32, num_stages=2),
]
_GRID_SAMPLER_3D_AUTOTUNE_KEY = [
    "C",
    "DHW_out",
    "interpolation_mode",
    "padding_mode",
]
_GRID_SAMPLER_3D_BACKWARD_AUTOTUNE_KEY = ["C", "DHW_out"]


@triton.autotune(
    configs=_GRID_SAMPLER_2D_AUTOTUNE_CONFIGS,
    key=_GRID_SAMPLER_2D_AUTOTUNE_KEY,
)
@triton.jit
def _grid_sampler_2d_kernel(
    input_ptr,
    grid_ptr,
    output_ptr,
    N,
    C,
    H_in,
    W_in,
    W_out,
    HW_out,
    stride_in_N,
    stride_in_C,
    stride_in_H,
    stride_in_W,
    stride_g_N,
    stride_g_H,
    stride_g_W,
    stride_out_N,
    stride_out_C,
    stride_out_H,
    stride_out_W,
    interpolation_mode: tl.constexpr,
    padding_mode: tl.constexpr,
    align_corners: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    """
    Unified 2D grid_sample forward kernel.  Each program handles BLOCK_HW
    output spatial positions (flattened H_out*W_out) and loops over C
    internally.  Geometry is computed once and reused across all channels.
    """
    pid = tl.program_id(0)
    n_idx = tl.program_id(1)

    hw_offs = pid * BLOCK_HW + tl.arange(0, BLOCK_HW)
    hw_mask = hw_offs < HW_out
    h_idx = hw_offs // W_out
    w_idx = hw_offs % W_out

    # ── Load grid coordinates (once for all channels) ─────────────────
    g_base = grid_ptr + n_idx * stride_g_N + h_idx * stride_g_H + w_idx * stride_g_W
    gx = tl.load(g_base + 0, mask=hw_mask, other=0.0).to(tl.float32)
    gy = tl.load(g_base + 1, mask=hw_mask, other=0.0).to(tl.float32)

    # ── Non-finite handling (vectorized) ────────────────────────────────
    nan_x = gx != gx
    nan_y = gy != gy
    pos_inf_x = gx == float("inf")
    neg_inf_x = gx == -float("inf")
    pos_inf_y = gy == float("inf")
    neg_inf_y = gy == -float("inf")
    any_nonfinite = nan_x | nan_y | pos_inf_x | neg_inf_x | pos_inf_y | neg_inf_y
    nan_val = (gx - gx) + (gy - gy)

    if padding_mode == 1:
        gx = tl.where(nan_x | neg_inf_x, -1.0, tl.where(pos_inf_x, 1.0, gx))
        gy = tl.where(nan_y | neg_inf_y, -1.0, tl.where(pos_inf_y, 1.0, gy))
    elif padding_mode == 2:
        gx = tl.where(nan_x | pos_inf_x | neg_inf_x, -1.0, gx)
        gy = tl.where(nan_y | pos_inf_y | neg_inf_y, -1.0, gy)
    else:
        gx = tl.where(any_nonfinite, -1.0, gx)
        gy = tl.where(any_nonfinite, -1.0, gy)

    # For zeros padding, non-finite positions must produce 0 output.
    if padding_mode == 0:
        valid_mask = hw_mask & ~any_nonfinite
    else:
        valid_mask = hw_mask

    # ── Unnormalize from [-1, 1] to pixel coordinates ───────────────────
    if align_corners:
        ix = (gx + 1.0) * 0.5 * (W_in - 1)
        iy = (gy + 1.0) * 0.5 * (H_in - 1)
    else:
        ix = ((gx + 1.0) * W_in - 1.0) * 0.5
        iy = ((gy + 1.0) * H_in - 1.0) * 0.5

    in_n = input_ptr + n_idx * stride_in_N
    out_n = (
        output_ptr + n_idx * stride_out_N + h_idx * stride_out_H + w_idx * stride_out_W
    )

    if interpolation_mode == 0:  # bilinear
        x0 = tl.floor(ix).to(tl.int32)
        y0 = tl.floor(iy).to(tl.int32)
        tx = ix - tl.floor(ix)
        ty = iy - tl.floor(iy)

        w00 = (1.0 - tx) * (1.0 - ty)
        w01 = tx * (1.0 - ty)
        w10 = (1.0 - tx) * ty
        w11 = tx * ty

        off00, m00 = _clamp_and_mask_2d(
            x0,
            y0,
            W_in,
            H_in,
            stride_in_H,
            stride_in_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off01, m01 = _clamp_and_mask_2d(
            x0 + 1,
            y0,
            W_in,
            H_in,
            stride_in_H,
            stride_in_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off10, m10 = _clamp_and_mask_2d(
            x0,
            y0 + 1,
            W_in,
            H_in,
            stride_in_H,
            stride_in_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off11, m11 = _clamp_and_mask_2d(
            x0 + 1,
            y0 + 1,
            W_in,
            H_in,
            stride_in_H,
            stride_in_W,
            valid_mask,
            padding_mode,
            align_corners,
        )

        for c in range(C):
            cb = in_n + c * stride_in_C
            v00 = tl.load(cb + off00, mask=m00, other=0.0).to(tl.float32)
            v01 = tl.load(cb + off01, mask=m01, other=0.0).to(tl.float32)
            v10 = tl.load(cb + off10, mask=m10, other=0.0).to(tl.float32)
            v11 = tl.load(cb + off11, mask=m11, other=0.0).to(tl.float32)
            result = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11
            tl.store(
                out_n + c * stride_out_C,
                result.to(output_ptr.dtype.element_ty),
                mask=hw_mask,
            )

    elif interpolation_mode == 1:  # nearest
        x_near = _nearbyint_coord(ix)
        y_near = _nearbyint_coord(iy)
        off_near, m_near = _clamp_and_mask_2d(
            x_near,
            y_near,
            W_in,
            H_in,
            stride_in_H,
            stride_in_W,
            valid_mask,
            padding_mode,
            align_corners,
        )

        for c in range(C):
            val = tl.load(in_n + c * stride_in_C + off_near, mask=m_near, other=0.0).to(
                tl.float32
            )
            tl.store(
                out_n + c * stride_out_C,
                val.to(output_ptr.dtype.element_ty),
                mask=hw_mask,
            )

    else:  # bicubic (interpolation_mode == 2)
        x0 = tl.floor(ix).to(tl.int32)
        y0 = tl.floor(iy).to(tl.int32)
        tx = ix - tl.floor(ix)
        ty = iy - tl.floor(iy)

        wx0 = _cubic_weight(-1.0 - tx)
        wx1 = _cubic_weight(0.0 - tx)
        wx2 = _cubic_weight(1.0 - tx)
        wx3 = _cubic_weight(2.0 - tx)
        wy0 = _cubic_weight(-1.0 - ty)
        wy1 = _cubic_weight(0.0 - ty)
        wy2 = _cubic_weight(1.0 - ty)
        wy3 = _cubic_weight(2.0 - ty)

        # Precompute 4x4 neighbor offsets and masks
        off_r0c0, m_r0c0 = _clamp_and_mask_2d(
            x0 - 1,
            y0 - 1,
            W_in,
            H_in,
            stride_in_H,
            stride_in_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off_r0c1, m_r0c1 = _clamp_and_mask_2d(
            x0,
            y0 - 1,
            W_in,
            H_in,
            stride_in_H,
            stride_in_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off_r0c2, m_r0c2 = _clamp_and_mask_2d(
            x0 + 1,
            y0 - 1,
            W_in,
            H_in,
            stride_in_H,
            stride_in_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off_r0c3, m_r0c3 = _clamp_and_mask_2d(
            x0 + 2,
            y0 - 1,
            W_in,
            H_in,
            stride_in_H,
            stride_in_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off_r1c0, m_r1c0 = _clamp_and_mask_2d(
            x0 - 1,
            y0,
            W_in,
            H_in,
            stride_in_H,
            stride_in_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off_r1c1, m_r1c1 = _clamp_and_mask_2d(
            x0,
            y0,
            W_in,
            H_in,
            stride_in_H,
            stride_in_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off_r1c2, m_r1c2 = _clamp_and_mask_2d(
            x0 + 1,
            y0,
            W_in,
            H_in,
            stride_in_H,
            stride_in_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off_r1c3, m_r1c3 = _clamp_and_mask_2d(
            x0 + 2,
            y0,
            W_in,
            H_in,
            stride_in_H,
            stride_in_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off_r2c0, m_r2c0 = _clamp_and_mask_2d(
            x0 - 1,
            y0 + 1,
            W_in,
            H_in,
            stride_in_H,
            stride_in_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off_r2c1, m_r2c1 = _clamp_and_mask_2d(
            x0,
            y0 + 1,
            W_in,
            H_in,
            stride_in_H,
            stride_in_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off_r2c2, m_r2c2 = _clamp_and_mask_2d(
            x0 + 1,
            y0 + 1,
            W_in,
            H_in,
            stride_in_H,
            stride_in_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off_r2c3, m_r2c3 = _clamp_and_mask_2d(
            x0 + 2,
            y0 + 1,
            W_in,
            H_in,
            stride_in_H,
            stride_in_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off_r3c0, m_r3c0 = _clamp_and_mask_2d(
            x0 - 1,
            y0 + 2,
            W_in,
            H_in,
            stride_in_H,
            stride_in_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off_r3c1, m_r3c1 = _clamp_and_mask_2d(
            x0,
            y0 + 2,
            W_in,
            H_in,
            stride_in_H,
            stride_in_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off_r3c2, m_r3c2 = _clamp_and_mask_2d(
            x0 + 1,
            y0 + 2,
            W_in,
            H_in,
            stride_in_H,
            stride_in_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off_r3c3, m_r3c3 = _clamp_and_mask_2d(
            x0 + 2,
            y0 + 2,
            W_in,
            H_in,
            stride_in_H,
            stride_in_W,
            valid_mask,
            padding_mode,
            align_corners,
        )

        for c in range(C):
            cb = in_n + c * stride_in_C
            r0 = (
                wx0 * tl.load(cb + off_r0c0, mask=m_r0c0, other=0.0).to(tl.float32)
                + wx1 * tl.load(cb + off_r0c1, mask=m_r0c1, other=0.0).to(tl.float32)
                + wx2 * tl.load(cb + off_r0c2, mask=m_r0c2, other=0.0).to(tl.float32)
                + wx3 * tl.load(cb + off_r0c3, mask=m_r0c3, other=0.0).to(tl.float32)
            )
            r1 = (
                wx0 * tl.load(cb + off_r1c0, mask=m_r1c0, other=0.0).to(tl.float32)
                + wx1 * tl.load(cb + off_r1c1, mask=m_r1c1, other=0.0).to(tl.float32)
                + wx2 * tl.load(cb + off_r1c2, mask=m_r1c2, other=0.0).to(tl.float32)
                + wx3 * tl.load(cb + off_r1c3, mask=m_r1c3, other=0.0).to(tl.float32)
            )
            r2 = (
                wx0 * tl.load(cb + off_r2c0, mask=m_r2c0, other=0.0).to(tl.float32)
                + wx1 * tl.load(cb + off_r2c1, mask=m_r2c1, other=0.0).to(tl.float32)
                + wx2 * tl.load(cb + off_r2c2, mask=m_r2c2, other=0.0).to(tl.float32)
                + wx3 * tl.load(cb + off_r2c3, mask=m_r2c3, other=0.0).to(tl.float32)
            )
            r3 = (
                wx0 * tl.load(cb + off_r3c0, mask=m_r3c0, other=0.0).to(tl.float32)
                + wx1 * tl.load(cb + off_r3c1, mask=m_r3c1, other=0.0).to(tl.float32)
                + wx2 * tl.load(cb + off_r3c2, mask=m_r3c2, other=0.0).to(tl.float32)
                + wx3 * tl.load(cb + off_r3c3, mask=m_r3c3, other=0.0).to(tl.float32)
            )
            result = wy0 * r0 + wy1 * r1 + wy2 * r2 + wy3 * r3
            # Bicubic with NaN grid must propagate NaN
            result = tl.where(any_nonfinite & hw_mask, nan_val, result)
            tl.store(
                out_n + c * stride_out_C,
                result.to(output_ptr.dtype.element_ty),
                mask=hw_mask,
            )


@triton.autotune(
    configs=_GRID_SAMPLER_3D_AUTOTUNE_CONFIGS,
    key=_GRID_SAMPLER_3D_AUTOTUNE_KEY,
)
@triton.jit
def _grid_sampler_3d_kernel(
    input_ptr,
    grid_ptr,
    output_ptr,
    N,
    C,
    D_in,
    H_in,
    W_in,
    W_out,
    HW_out,
    DHW_out,
    stride_in_N,
    stride_in_C,
    stride_in_D,
    stride_in_H,
    stride_in_W,
    stride_g_N,
    stride_g_D,
    stride_g_H,
    stride_g_W,
    stride_out_N,
    stride_out_C,
    stride_out_D,
    stride_out_H,
    stride_out_W,
    interpolation_mode: tl.constexpr,
    padding_mode: tl.constexpr,
    align_corners: tl.constexpr,
    BLOCK_DHW: tl.constexpr,
):
    """
    Unified 3D grid_sample forward kernel.  Each program handles BLOCK_DHW
    output spatial positions (flattened D_out*H_out*W_out) and loops over C
    internally.  Geometry is computed once and reused across all channels.
    """
    pid = tl.program_id(0)
    n_idx = tl.program_id(1)

    dhw_offs = pid * BLOCK_DHW + tl.arange(0, BLOCK_DHW)
    dhw_mask = dhw_offs < DHW_out
    d_idx = dhw_offs // HW_out
    hw_rem = dhw_offs % HW_out
    h_idx = hw_rem // W_out
    w_idx = hw_rem % W_out

    # ── Load grid coordinates (once for all channels) ─────────────────
    g_base = (
        grid_ptr
        + n_idx * stride_g_N
        + d_idx * stride_g_D
        + h_idx * stride_g_H
        + w_idx * stride_g_W
    )
    gx = tl.load(g_base + 0, mask=dhw_mask, other=0.0).to(tl.float32)
    gy = tl.load(g_base + 1, mask=dhw_mask, other=0.0).to(tl.float32)
    gz = tl.load(g_base + 2, mask=dhw_mask, other=0.0).to(tl.float32)

    # ── NaN handling (vectorized) ───────────────────────────────────────
    nan_x = gx != gx
    nan_y = gy != gy
    nan_z = gz != gz
    pos_inf_x = gx == float("inf")
    neg_inf_x = gx == -float("inf")
    pos_inf_y = gy == float("inf")
    neg_inf_y = gy == -float("inf")
    pos_inf_z = gz == float("inf")
    neg_inf_z = gz == -float("inf")
    any_nonfinite = (
        nan_x
        | nan_y
        | nan_z
        | pos_inf_x
        | neg_inf_x
        | pos_inf_y
        | neg_inf_y
        | pos_inf_z
        | neg_inf_z
    )

    if padding_mode == 1:
        gx = tl.where(nan_x | neg_inf_x, -1.0, tl.where(pos_inf_x, 1.0, gx))
        gy = tl.where(nan_y | neg_inf_y, -1.0, tl.where(pos_inf_y, 1.0, gy))
        gz = tl.where(nan_z | neg_inf_z, -1.0, tl.where(pos_inf_z, 1.0, gz))
    elif padding_mode == 2:
        gx = tl.where(nan_x | pos_inf_x | neg_inf_x, -1.0, gx)
        gy = tl.where(nan_y | pos_inf_y | neg_inf_y, -1.0, gy)
        gz = tl.where(nan_z | pos_inf_z | neg_inf_z, -1.0, gz)
    else:
        gx = tl.where(any_nonfinite, -1.0, gx)
        gy = tl.where(any_nonfinite, -1.0, gy)
        gz = tl.where(any_nonfinite, -1.0, gz)

    if padding_mode == 0:
        valid_mask = dhw_mask & ~any_nonfinite
    else:
        valid_mask = dhw_mask

    # ── Unnormalize from [-1, 1] to pixel coordinates ───────────────────
    if align_corners:
        ix = (gx + 1.0) * 0.5 * (W_in - 1)
        iy = (gy + 1.0) * 0.5 * (H_in - 1)
        iz = (gz + 1.0) * 0.5 * (D_in - 1)
    else:
        ix = ((gx + 1.0) * W_in - 1.0) * 0.5
        iy = ((gy + 1.0) * H_in - 1.0) * 0.5
        iz = ((gz + 1.0) * D_in - 1.0) * 0.5

    in_n = input_ptr + n_idx * stride_in_N
    out_n = (
        output_ptr
        + n_idx * stride_out_N
        + d_idx * stride_out_D
        + h_idx * stride_out_H
        + w_idx * stride_out_W
    )

    if interpolation_mode == 0:  # trilinear
        x0 = tl.floor(ix).to(tl.int32)
        y0 = tl.floor(iy).to(tl.int32)
        z0 = tl.floor(iz).to(tl.int32)
        tx = ix - tl.floor(ix)
        ty = iy - tl.floor(iy)
        tz = iz - tl.floor(iz)

        w000 = (1.0 - tx) * (1.0 - ty) * (1.0 - tz)
        w001 = tx * (1.0 - ty) * (1.0 - tz)
        w010 = (1.0 - tx) * ty * (1.0 - tz)
        w011 = tx * ty * (1.0 - tz)
        w100 = (1.0 - tx) * (1.0 - ty) * tz
        w101 = tx * (1.0 - ty) * tz
        w110 = (1.0 - tx) * ty * tz
        w111 = tx * ty * tz

        off000, m000 = _clamp_and_mask_3d(
            x0,
            y0,
            z0,
            W_in,
            H_in,
            D_in,
            stride_in_D,
            stride_in_H,
            stride_in_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off001, m001 = _clamp_and_mask_3d(
            x0 + 1,
            y0,
            z0,
            W_in,
            H_in,
            D_in,
            stride_in_D,
            stride_in_H,
            stride_in_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off010, m010 = _clamp_and_mask_3d(
            x0,
            y0 + 1,
            z0,
            W_in,
            H_in,
            D_in,
            stride_in_D,
            stride_in_H,
            stride_in_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off011, m011 = _clamp_and_mask_3d(
            x0 + 1,
            y0 + 1,
            z0,
            W_in,
            H_in,
            D_in,
            stride_in_D,
            stride_in_H,
            stride_in_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off100, m100 = _clamp_and_mask_3d(
            x0,
            y0,
            z0 + 1,
            W_in,
            H_in,
            D_in,
            stride_in_D,
            stride_in_H,
            stride_in_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off101, m101 = _clamp_and_mask_3d(
            x0 + 1,
            y0,
            z0 + 1,
            W_in,
            H_in,
            D_in,
            stride_in_D,
            stride_in_H,
            stride_in_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off110, m110 = _clamp_and_mask_3d(
            x0,
            y0 + 1,
            z0 + 1,
            W_in,
            H_in,
            D_in,
            stride_in_D,
            stride_in_H,
            stride_in_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off111, m111 = _clamp_and_mask_3d(
            x0 + 1,
            y0 + 1,
            z0 + 1,
            W_in,
            H_in,
            D_in,
            stride_in_D,
            stride_in_H,
            stride_in_W,
            valid_mask,
            padding_mode,
            align_corners,
        )

        for c in range(C):
            cb = in_n + c * stride_in_C
            v000 = tl.load(cb + off000, mask=m000, other=0.0).to(tl.float32)
            v001 = tl.load(cb + off001, mask=m001, other=0.0).to(tl.float32)
            v010 = tl.load(cb + off010, mask=m010, other=0.0).to(tl.float32)
            v011 = tl.load(cb + off011, mask=m011, other=0.0).to(tl.float32)
            v100 = tl.load(cb + off100, mask=m100, other=0.0).to(tl.float32)
            v101 = tl.load(cb + off101, mask=m101, other=0.0).to(tl.float32)
            v110 = tl.load(cb + off110, mask=m110, other=0.0).to(tl.float32)
            v111 = tl.load(cb + off111, mask=m111, other=0.0).to(tl.float32)
            result = (
                w000 * v000
                + w001 * v001
                + w010 * v010
                + w011 * v011
                + w100 * v100
                + w101 * v101
                + w110 * v110
                + w111 * v111
            )
            tl.store(
                out_n + c * stride_out_C,
                result.to(output_ptr.dtype.element_ty),
                mask=dhw_mask,
            )

    else:  # nearest
        x_near = _nearbyint_coord(ix)
        y_near = _nearbyint_coord(iy)
        z_near = _nearbyint_coord(iz)
        off_near, m_near = _clamp_and_mask_3d(
            x_near,
            y_near,
            z_near,
            W_in,
            H_in,
            D_in,
            stride_in_D,
            stride_in_H,
            stride_in_W,
            valid_mask,
            padding_mode,
            align_corners,
        )

        for c in range(C):
            val = tl.load(in_n + c * stride_in_C + off_near, mask=m_near, other=0.0).to(
                tl.float32
            )
            tl.store(
                out_n + c * stride_out_C,
                val.to(output_ptr.dtype.element_ty),
                mask=dhw_mask,
            )


@triton.autotune(
    configs=_GRID_SAMPLER_2D_AUTOTUNE_CONFIGS,
    key=_GRID_SAMPLER_2D_BACKWARD_AUTOTUNE_KEY,
    reset_to_zero=["grad_input_ptr"],
)
@triton.jit
def _grid_sampler_2d_backward_input_kernel(
    grad_output_ptr,
    grid_ptr,
    grad_input_ptr,
    N,
    C,
    H_in,
    W_in,
    W_out,
    HW_out,
    stride_go_N,
    stride_go_C,
    stride_go_H,
    stride_go_W,
    stride_g_N,
    stride_g_H,
    stride_g_W,
    stride_gi_N,
    stride_gi_C,
    stride_gi_H,
    stride_gi_W,
    interpolation_mode: tl.constexpr,
    padding_mode: tl.constexpr,
    align_corners: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    n_idx = tl.program_id(1)

    hw_offs = pid * BLOCK_HW + tl.arange(0, BLOCK_HW)
    hw_mask = hw_offs < HW_out
    h_idx = hw_offs // W_out
    w_idx = hw_offs % W_out

    g_base = grid_ptr + n_idx * stride_g_N + h_idx * stride_g_H + w_idx * stride_g_W
    gx = tl.load(g_base + 0, mask=hw_mask, other=0.0).to(tl.float32)
    gy = tl.load(g_base + 1, mask=hw_mask, other=0.0).to(tl.float32)

    nan_x = gx != gx
    nan_y = gy != gy
    any_nan = nan_x | nan_y
    gx = tl.where(nan_x, -1.0, gx)
    gy = tl.where(nan_y, -1.0, gy)

    if padding_mode == 0:
        valid_mask = hw_mask & ~any_nan
    else:
        valid_mask = hw_mask

    if align_corners:
        ix = (gx + 1.0) * 0.5 * (W_in - 1)
        iy = (gy + 1.0) * 0.5 * (H_in - 1)
    else:
        ix = ((gx + 1.0) * W_in - 1.0) * 0.5
        iy = ((gy + 1.0) * H_in - 1.0) * 0.5

    go_base = (
        grad_output_ptr
        + n_idx * stride_go_N
        + h_idx * stride_go_H
        + w_idx * stride_go_W
    )
    gi_n = grad_input_ptr + n_idx * stride_gi_N

    if interpolation_mode == 0:  # bilinear
        x0 = tl.floor(ix).to(tl.int32)
        y0 = tl.floor(iy).to(tl.int32)
        tx = ix - tl.floor(ix)
        ty = iy - tl.floor(iy)

        w00 = (1.0 - tx) * (1.0 - ty)
        w01 = tx * (1.0 - ty)
        w10 = (1.0 - tx) * ty
        w11 = tx * ty

        off00, m00 = _clamp_and_mask_2d(
            x0,
            y0,
            W_in,
            H_in,
            stride_gi_H,
            stride_gi_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off01, m01 = _clamp_and_mask_2d(
            x0 + 1,
            y0,
            W_in,
            H_in,
            stride_gi_H,
            stride_gi_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off10, m10 = _clamp_and_mask_2d(
            x0,
            y0 + 1,
            W_in,
            H_in,
            stride_gi_H,
            stride_gi_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off11, m11 = _clamp_and_mask_2d(
            x0 + 1,
            y0 + 1,
            W_in,
            H_in,
            stride_gi_H,
            stride_gi_W,
            valid_mask,
            padding_mode,
            align_corners,
        )

        for c in range(C):
            grad = tl.load(go_base + c * stride_go_C, mask=hw_mask, other=0.0).to(
                tl.float32
            )
            cb = gi_n + c * stride_gi_C
            tl.atomic_add(
                cb + off00, (grad * w00).to(grad_input_ptr.dtype.element_ty), mask=m00
            )
            tl.atomic_add(
                cb + off01, (grad * w01).to(grad_input_ptr.dtype.element_ty), mask=m01
            )
            tl.atomic_add(
                cb + off10, (grad * w10).to(grad_input_ptr.dtype.element_ty), mask=m10
            )
            tl.atomic_add(
                cb + off11, (grad * w11).to(grad_input_ptr.dtype.element_ty), mask=m11
            )

    elif interpolation_mode == 1:  # nearest
        x_near = _nearbyint_coord(ix)
        y_near = _nearbyint_coord(iy)
        off_near, m_near = _clamp_and_mask_2d(
            x_near,
            y_near,
            W_in,
            H_in,
            stride_gi_H,
            stride_gi_W,
            valid_mask,
            padding_mode,
            align_corners,
        )

        for c in range(C):
            grad = tl.load(go_base + c * stride_go_C, mask=hw_mask, other=0.0).to(
                tl.float32
            )
            tl.atomic_add(
                gi_n + c * stride_gi_C + off_near,
                grad.to(grad_input_ptr.dtype.element_ty),
                mask=m_near,
            )

    else:  # bicubic
        x0 = tl.floor(ix).to(tl.int32)
        y0 = tl.floor(iy).to(tl.int32)
        tx = ix - tl.floor(ix)
        ty = iy - tl.floor(iy)

        wx0 = _cubic_weight(-1.0 - tx)
        wx1 = _cubic_weight(-tx)
        wx2 = _cubic_weight(1.0 - tx)
        wx3 = _cubic_weight(2.0 - tx)
        wy0 = _cubic_weight(-1.0 - ty)
        wy1 = _cubic_weight(-ty)
        wy2 = _cubic_weight(1.0 - ty)
        wy3 = _cubic_weight(2.0 - ty)

        # Precompute 4x4 neighbor offsets and masks
        off_r0c0, m_r0c0 = _clamp_and_mask_2d(
            x0 - 1,
            y0 - 1,
            W_in,
            H_in,
            stride_gi_H,
            stride_gi_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off_r0c1, m_r0c1 = _clamp_and_mask_2d(
            x0,
            y0 - 1,
            W_in,
            H_in,
            stride_gi_H,
            stride_gi_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off_r0c2, m_r0c2 = _clamp_and_mask_2d(
            x0 + 1,
            y0 - 1,
            W_in,
            H_in,
            stride_gi_H,
            stride_gi_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off_r0c3, m_r0c3 = _clamp_and_mask_2d(
            x0 + 2,
            y0 - 1,
            W_in,
            H_in,
            stride_gi_H,
            stride_gi_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off_r1c0, m_r1c0 = _clamp_and_mask_2d(
            x0 - 1,
            y0,
            W_in,
            H_in,
            stride_gi_H,
            stride_gi_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off_r1c1, m_r1c1 = _clamp_and_mask_2d(
            x0,
            y0,
            W_in,
            H_in,
            stride_gi_H,
            stride_gi_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off_r1c2, m_r1c2 = _clamp_and_mask_2d(
            x0 + 1,
            y0,
            W_in,
            H_in,
            stride_gi_H,
            stride_gi_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off_r1c3, m_r1c3 = _clamp_and_mask_2d(
            x0 + 2,
            y0,
            W_in,
            H_in,
            stride_gi_H,
            stride_gi_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off_r2c0, m_r2c0 = _clamp_and_mask_2d(
            x0 - 1,
            y0 + 1,
            W_in,
            H_in,
            stride_gi_H,
            stride_gi_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off_r2c1, m_r2c1 = _clamp_and_mask_2d(
            x0,
            y0 + 1,
            W_in,
            H_in,
            stride_gi_H,
            stride_gi_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off_r2c2, m_r2c2 = _clamp_and_mask_2d(
            x0 + 1,
            y0 + 1,
            W_in,
            H_in,
            stride_gi_H,
            stride_gi_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off_r2c3, m_r2c3 = _clamp_and_mask_2d(
            x0 + 2,
            y0 + 1,
            W_in,
            H_in,
            stride_gi_H,
            stride_gi_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off_r3c0, m_r3c0 = _clamp_and_mask_2d(
            x0 - 1,
            y0 + 2,
            W_in,
            H_in,
            stride_gi_H,
            stride_gi_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off_r3c1, m_r3c1 = _clamp_and_mask_2d(
            x0,
            y0 + 2,
            W_in,
            H_in,
            stride_gi_H,
            stride_gi_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off_r3c2, m_r3c2 = _clamp_and_mask_2d(
            x0 + 1,
            y0 + 2,
            W_in,
            H_in,
            stride_gi_H,
            stride_gi_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off_r3c3, m_r3c3 = _clamp_and_mask_2d(
            x0 + 2,
            y0 + 2,
            W_in,
            H_in,
            stride_gi_H,
            stride_gi_W,
            valid_mask,
            padding_mode,
            align_corners,
        )

        for c in range(C):
            grad = tl.load(go_base + c * stride_go_C, mask=hw_mask, other=0.0).to(
                tl.float32
            )
            cb = gi_n + c * stride_gi_C
            tl.atomic_add(
                cb + off_r0c0,
                (grad * wy0 * wx0).to(grad_input_ptr.dtype.element_ty),
                mask=m_r0c0,
            )
            tl.atomic_add(
                cb + off_r0c1,
                (grad * wy0 * wx1).to(grad_input_ptr.dtype.element_ty),
                mask=m_r0c1,
            )
            tl.atomic_add(
                cb + off_r0c2,
                (grad * wy0 * wx2).to(grad_input_ptr.dtype.element_ty),
                mask=m_r0c2,
            )
            tl.atomic_add(
                cb + off_r0c3,
                (grad * wy0 * wx3).to(grad_input_ptr.dtype.element_ty),
                mask=m_r0c3,
            )
            tl.atomic_add(
                cb + off_r1c0,
                (grad * wy1 * wx0).to(grad_input_ptr.dtype.element_ty),
                mask=m_r1c0,
            )
            tl.atomic_add(
                cb + off_r1c1,
                (grad * wy1 * wx1).to(grad_input_ptr.dtype.element_ty),
                mask=m_r1c1,
            )
            tl.atomic_add(
                cb + off_r1c2,
                (grad * wy1 * wx2).to(grad_input_ptr.dtype.element_ty),
                mask=m_r1c2,
            )
            tl.atomic_add(
                cb + off_r1c3,
                (grad * wy1 * wx3).to(grad_input_ptr.dtype.element_ty),
                mask=m_r1c3,
            )
            tl.atomic_add(
                cb + off_r2c0,
                (grad * wy2 * wx0).to(grad_input_ptr.dtype.element_ty),
                mask=m_r2c0,
            )
            tl.atomic_add(
                cb + off_r2c1,
                (grad * wy2 * wx1).to(grad_input_ptr.dtype.element_ty),
                mask=m_r2c1,
            )
            tl.atomic_add(
                cb + off_r2c2,
                (grad * wy2 * wx2).to(grad_input_ptr.dtype.element_ty),
                mask=m_r2c2,
            )
            tl.atomic_add(
                cb + off_r2c3,
                (grad * wy2 * wx3).to(grad_input_ptr.dtype.element_ty),
                mask=m_r2c3,
            )
            tl.atomic_add(
                cb + off_r3c0,
                (grad * wy3 * wx0).to(grad_input_ptr.dtype.element_ty),
                mask=m_r3c0,
            )
            tl.atomic_add(
                cb + off_r3c1,
                (grad * wy3 * wx1).to(grad_input_ptr.dtype.element_ty),
                mask=m_r3c1,
            )
            tl.atomic_add(
                cb + off_r3c2,
                (grad * wy3 * wx2).to(grad_input_ptr.dtype.element_ty),
                mask=m_r3c2,
            )
            tl.atomic_add(
                cb + off_r3c3,
                (grad * wy3 * wx3).to(grad_input_ptr.dtype.element_ty),
                mask=m_r3c3,
            )


@triton.autotune(
    configs=_GRID_SAMPLER_2D_AUTOTUNE_CONFIGS,
    key=_GRID_SAMPLER_2D_BACKWARD_AUTOTUNE_KEY,
)
@triton.jit
def _grid_sampler_2d_backward_grid_kernel(
    grad_output_ptr,
    input_ptr,
    grid_ptr,
    grad_grid_ptr,
    N,
    C,
    H_in,
    W_in,
    W_out,
    HW_out,
    stride_go_N,
    stride_go_C,
    stride_go_H,
    stride_go_W,
    stride_in_N,
    stride_in_C,
    stride_in_H,
    stride_in_W,
    stride_g_N,
    stride_g_H,
    stride_g_W,
    stride_gg_N,
    stride_gg_H,
    stride_gg_W,
    interpolation_mode: tl.constexpr,
    padding_mode: tl.constexpr,
    align_corners: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    n_idx = tl.program_id(1)

    hw_offs = pid * BLOCK_HW + tl.arange(0, BLOCK_HW)
    hw_mask = hw_offs < HW_out
    h_idx = hw_offs // W_out
    w_idx = hw_offs % W_out

    g_base = grid_ptr + n_idx * stride_g_N + h_idx * stride_g_H + w_idx * stride_g_W
    gx = tl.load(g_base + 0, mask=hw_mask, other=0.0).to(tl.float32)
    gy = tl.load(g_base + 1, mask=hw_mask, other=0.0).to(tl.float32)

    nan_x = gx != gx
    nan_y = gy != gy
    any_nan = nan_x | nan_y
    gx = tl.where(nan_x, -1.0, gx)
    gy = tl.where(nan_y, -1.0, gy)

    if padding_mode == 0:
        valid_mask = hw_mask & ~any_nan
    else:
        valid_mask = hw_mask

    gg_base = (
        grad_grid_ptr + n_idx * stride_gg_N + h_idx * stride_gg_H + w_idx * stride_gg_W
    )

    if interpolation_mode == 1:
        tl.store(
            gg_base + 0,
            tl.zeros((BLOCK_HW,), dtype=grad_grid_ptr.dtype.element_ty),
            mask=hw_mask,
        )
        tl.store(
            gg_base + 1,
            tl.zeros((BLOCK_HW,), dtype=grad_grid_ptr.dtype.element_ty),
            mask=hw_mask,
        )
        return

    if align_corners:
        gix_mult = tl.cast(W_in - 1, tl.float32) * 0.5
        giy_mult = tl.cast(H_in - 1, tl.float32) * 0.5
        ix = (gx + 1.0) * gix_mult
        iy = (gy + 1.0) * giy_mult
    else:
        gix_mult = tl.cast(W_in, tl.float32) * 0.5
        giy_mult = tl.cast(H_in, tl.float32) * 0.5
        ix = ((gx + 1.0) * W_in - 1.0) * 0.5
        iy = ((gy + 1.0) * H_in - 1.0) * 0.5

    acc_x = tl.zeros((BLOCK_HW,), dtype=tl.float32)
    acc_y = tl.zeros((BLOCK_HW,), dtype=tl.float32)

    go_base = (
        grad_output_ptr
        + n_idx * stride_go_N
        + h_idx * stride_go_H
        + w_idx * stride_go_W
    )
    in_n = input_ptr + n_idx * stride_in_N

    if interpolation_mode == 0:  # bilinear
        x0 = tl.floor(ix).to(tl.int32)
        y0 = tl.floor(iy).to(tl.int32)
        tx = ix - tl.floor(ix)
        ty = iy - tl.floor(iy)

        off00, m00 = _clamp_and_mask_2d(
            x0,
            y0,
            W_in,
            H_in,
            stride_in_H,
            stride_in_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off01, m01 = _clamp_and_mask_2d(
            x0 + 1,
            y0,
            W_in,
            H_in,
            stride_in_H,
            stride_in_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off10, m10 = _clamp_and_mask_2d(
            x0,
            y0 + 1,
            W_in,
            H_in,
            stride_in_H,
            stride_in_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off11, m11 = _clamp_and_mask_2d(
            x0 + 1,
            y0 + 1,
            W_in,
            H_in,
            stride_in_H,
            stride_in_W,
            valid_mask,
            padding_mode,
            align_corners,
        )

        for c in range(C):
            cb = in_n + c * stride_in_C
            go = tl.load(go_base + c * stride_go_C, mask=hw_mask, other=0.0).to(
                tl.float32
            )
            v00 = tl.load(cb + off00, mask=m00, other=0.0).to(tl.float32)
            v01 = tl.load(cb + off01, mask=m01, other=0.0).to(tl.float32)
            v10 = tl.load(cb + off10, mask=m10, other=0.0).to(tl.float32)
            v11 = tl.load(cb + off11, mask=m11, other=0.0).to(tl.float32)
            dix = (v01 - v00) * (1.0 - ty) + (v11 - v10) * ty
            diy = (v10 - v00) * (1.0 - tx) + (v11 - v01) * tx
            acc_x += go * dix
            acc_y += go * diy

    else:  # bicubic
        x0 = tl.floor(ix).to(tl.int32)
        y0 = tl.floor(iy).to(tl.int32)
        tx = ix - tl.floor(ix)
        ty = iy - tl.floor(iy)
        wx0 = _cubic_weight(-1.0 - tx)
        wx1 = _cubic_weight(-tx)
        wx2 = _cubic_weight(1.0 - tx)
        wx3 = _cubic_weight(2.0 - tx)
        wy0 = _cubic_weight(-1.0 - ty)
        wy1 = _cubic_weight(-ty)
        wy2 = _cubic_weight(1.0 - ty)
        wy3 = _cubic_weight(2.0 - ty)
        dwx0, dwx1, dwx2, dwx3 = _cubic_weight_grad(tx)
        dwy0, dwy1, dwy2, dwy3 = _cubic_weight_grad(ty)

        # Precompute 4x4 offsets
        off_r0c0, m_r0c0 = _clamp_and_mask_2d(
            x0 - 1,
            y0 - 1,
            W_in,
            H_in,
            stride_in_H,
            stride_in_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off_r0c1, m_r0c1 = _clamp_and_mask_2d(
            x0,
            y0 - 1,
            W_in,
            H_in,
            stride_in_H,
            stride_in_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off_r0c2, m_r0c2 = _clamp_and_mask_2d(
            x0 + 1,
            y0 - 1,
            W_in,
            H_in,
            stride_in_H,
            stride_in_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off_r0c3, m_r0c3 = _clamp_and_mask_2d(
            x0 + 2,
            y0 - 1,
            W_in,
            H_in,
            stride_in_H,
            stride_in_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off_r1c0, m_r1c0 = _clamp_and_mask_2d(
            x0 - 1,
            y0,
            W_in,
            H_in,
            stride_in_H,
            stride_in_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off_r1c1, m_r1c1 = _clamp_and_mask_2d(
            x0,
            y0,
            W_in,
            H_in,
            stride_in_H,
            stride_in_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off_r1c2, m_r1c2 = _clamp_and_mask_2d(
            x0 + 1,
            y0,
            W_in,
            H_in,
            stride_in_H,
            stride_in_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off_r1c3, m_r1c3 = _clamp_and_mask_2d(
            x0 + 2,
            y0,
            W_in,
            H_in,
            stride_in_H,
            stride_in_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off_r2c0, m_r2c0 = _clamp_and_mask_2d(
            x0 - 1,
            y0 + 1,
            W_in,
            H_in,
            stride_in_H,
            stride_in_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off_r2c1, m_r2c1 = _clamp_and_mask_2d(
            x0,
            y0 + 1,
            W_in,
            H_in,
            stride_in_H,
            stride_in_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off_r2c2, m_r2c2 = _clamp_and_mask_2d(
            x0 + 1,
            y0 + 1,
            W_in,
            H_in,
            stride_in_H,
            stride_in_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off_r2c3, m_r2c3 = _clamp_and_mask_2d(
            x0 + 2,
            y0 + 1,
            W_in,
            H_in,
            stride_in_H,
            stride_in_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off_r3c0, m_r3c0 = _clamp_and_mask_2d(
            x0 - 1,
            y0 + 2,
            W_in,
            H_in,
            stride_in_H,
            stride_in_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off_r3c1, m_r3c1 = _clamp_and_mask_2d(
            x0,
            y0 + 2,
            W_in,
            H_in,
            stride_in_H,
            stride_in_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off_r3c2, m_r3c2 = _clamp_and_mask_2d(
            x0 + 1,
            y0 + 2,
            W_in,
            H_in,
            stride_in_H,
            stride_in_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off_r3c3, m_r3c3 = _clamp_and_mask_2d(
            x0 + 2,
            y0 + 2,
            W_in,
            H_in,
            stride_in_H,
            stride_in_W,
            valid_mask,
            padding_mode,
            align_corners,
        )

        for c in range(C):
            cb = in_n + c * stride_in_C
            go = tl.load(go_base + c * stride_go_C, mask=hw_mask, other=0.0).to(
                tl.float32
            )

            v_r0c0 = tl.load(cb + off_r0c0, mask=m_r0c0, other=0.0).to(tl.float32)
            v_r0c1 = tl.load(cb + off_r0c1, mask=m_r0c1, other=0.0).to(tl.float32)
            v_r0c2 = tl.load(cb + off_r0c2, mask=m_r0c2, other=0.0).to(tl.float32)
            v_r0c3 = tl.load(cb + off_r0c3, mask=m_r0c3, other=0.0).to(tl.float32)
            v_r1c0 = tl.load(cb + off_r1c0, mask=m_r1c0, other=0.0).to(tl.float32)
            v_r1c1 = tl.load(cb + off_r1c1, mask=m_r1c1, other=0.0).to(tl.float32)
            v_r1c2 = tl.load(cb + off_r1c2, mask=m_r1c2, other=0.0).to(tl.float32)
            v_r1c3 = tl.load(cb + off_r1c3, mask=m_r1c3, other=0.0).to(tl.float32)
            v_r2c0 = tl.load(cb + off_r2c0, mask=m_r2c0, other=0.0).to(tl.float32)
            v_r2c1 = tl.load(cb + off_r2c1, mask=m_r2c1, other=0.0).to(tl.float32)
            v_r2c2 = tl.load(cb + off_r2c2, mask=m_r2c2, other=0.0).to(tl.float32)
            v_r2c3 = tl.load(cb + off_r2c3, mask=m_r2c3, other=0.0).to(tl.float32)
            v_r3c0 = tl.load(cb + off_r3c0, mask=m_r3c0, other=0.0).to(tl.float32)
            v_r3c1 = tl.load(cb + off_r3c1, mask=m_r3c1, other=0.0).to(tl.float32)
            v_r3c2 = tl.load(cb + off_r3c2, mask=m_r3c2, other=0.0).to(tl.float32)
            v_r3c3 = tl.load(cb + off_r3c3, mask=m_r3c3, other=0.0).to(tl.float32)

            row0 = wx0 * v_r0c0 + wx1 * v_r0c1 + wx2 * v_r0c2 + wx3 * v_r0c3
            row1 = wx0 * v_r1c0 + wx1 * v_r1c1 + wx2 * v_r1c2 + wx3 * v_r1c3
            row2 = wx0 * v_r2c0 + wx1 * v_r2c1 + wx2 * v_r2c2 + wx3 * v_r2c3
            row3 = wx0 * v_r3c0 + wx1 * v_r3c1 + wx2 * v_r3c2 + wx3 * v_r3c3
            drow0 = dwx0 * v_r0c0 + dwx1 * v_r0c1 + dwx2 * v_r0c2 + dwx3 * v_r0c3
            drow1 = dwx0 * v_r1c0 + dwx1 * v_r1c1 + dwx2 * v_r1c2 + dwx3 * v_r1c3
            drow2 = dwx0 * v_r2c0 + dwx1 * v_r2c1 + dwx2 * v_r2c2 + dwx3 * v_r2c3
            drow3 = dwx0 * v_r3c0 + dwx1 * v_r3c1 + dwx2 * v_r3c2 + dwx3 * v_r3c3

            dix = wy0 * drow0 + wy1 * drow1 + wy2 * drow2 + wy3 * drow3
            diy = dwy0 * row0 + dwy1 * row1 + dwy2 * row2 + dwy3 * row3
            acc_x += go * dix
            acc_y += go * diy

    # For NaN positions in zeros padding, grad_grid should be 0 (acc started at 0 and valid_mask excluded them)
    tl.store(
        gg_base + 0, (acc_x * gix_mult).to(grad_grid_ptr.dtype.element_ty), mask=hw_mask
    )
    tl.store(
        gg_base + 1, (acc_y * giy_mult).to(grad_grid_ptr.dtype.element_ty), mask=hw_mask
    )


@triton.autotune(
    configs=_GRID_SAMPLER_3D_AUTOTUNE_CONFIGS,
    key=_GRID_SAMPLER_3D_BACKWARD_AUTOTUNE_KEY,
    reset_to_zero=["grad_input_ptr"],
)
@triton.jit
def _grid_sampler_3d_backward_input_kernel(
    grad_output_ptr,
    grid_ptr,
    grad_input_ptr,
    N,
    C,
    D_in,
    H_in,
    W_in,
    W_out,
    HW_out,
    DHW_out,
    stride_go_N,
    stride_go_C,
    stride_go_D,
    stride_go_H,
    stride_go_W,
    stride_g_N,
    stride_g_D,
    stride_g_H,
    stride_g_W,
    stride_gi_N,
    stride_gi_C,
    stride_gi_D,
    stride_gi_H,
    stride_gi_W,
    interpolation_mode: tl.constexpr,
    padding_mode: tl.constexpr,
    align_corners: tl.constexpr,
    BLOCK_DHW: tl.constexpr,
):
    pid = tl.program_id(0)
    n_idx = tl.program_id(1)

    dhw_offs = pid * BLOCK_DHW + tl.arange(0, BLOCK_DHW)
    dhw_mask = dhw_offs < DHW_out
    d_idx = dhw_offs // HW_out
    hw_rem = dhw_offs % HW_out
    h_idx = hw_rem // W_out
    w_idx = hw_rem % W_out

    g_base = (
        grid_ptr
        + n_idx * stride_g_N
        + d_idx * stride_g_D
        + h_idx * stride_g_H
        + w_idx * stride_g_W
    )
    gx = tl.load(g_base + 0, mask=dhw_mask, other=0.0).to(tl.float32)
    gy = tl.load(g_base + 1, mask=dhw_mask, other=0.0).to(tl.float32)
    gz = tl.load(g_base + 2, mask=dhw_mask, other=0.0).to(tl.float32)

    nan_x = gx != gx
    nan_y = gy != gy
    nan_z = gz != gz
    any_nan = nan_x | nan_y | nan_z
    gx = tl.where(nan_x, -1.0, gx)
    gy = tl.where(nan_y, -1.0, gy)
    gz = tl.where(nan_z, -1.0, gz)

    if padding_mode == 0:
        valid_mask = dhw_mask & ~any_nan
    else:
        valid_mask = dhw_mask

    if align_corners:
        ix = (gx + 1.0) * 0.5 * (W_in - 1)
        iy = (gy + 1.0) * 0.5 * (H_in - 1)
        iz = (gz + 1.0) * 0.5 * (D_in - 1)
    else:
        ix = ((gx + 1.0) * W_in - 1.0) * 0.5
        iy = ((gy + 1.0) * H_in - 1.0) * 0.5
        iz = ((gz + 1.0) * D_in - 1.0) * 0.5

    go_base = (
        grad_output_ptr
        + n_idx * stride_go_N
        + d_idx * stride_go_D
        + h_idx * stride_go_H
        + w_idx * stride_go_W
    )
    gi_n = grad_input_ptr + n_idx * stride_gi_N

    if interpolation_mode == 0:  # trilinear
        x0 = tl.floor(ix).to(tl.int32)
        y0 = tl.floor(iy).to(tl.int32)
        z0 = tl.floor(iz).to(tl.int32)
        tx = ix - tl.floor(ix)
        ty = iy - tl.floor(iy)
        tz = iz - tl.floor(iz)

        w000 = (1.0 - tx) * (1.0 - ty) * (1.0 - tz)
        w001 = tx * (1.0 - ty) * (1.0 - tz)
        w010 = (1.0 - tx) * ty * (1.0 - tz)
        w011 = tx * ty * (1.0 - tz)
        w100 = (1.0 - tx) * (1.0 - ty) * tz
        w101 = tx * (1.0 - ty) * tz
        w110 = (1.0 - tx) * ty * tz
        w111 = tx * ty * tz

        off000, m000 = _clamp_and_mask_3d(
            x0,
            y0,
            z0,
            W_in,
            H_in,
            D_in,
            stride_gi_D,
            stride_gi_H,
            stride_gi_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off001, m001 = _clamp_and_mask_3d(
            x0 + 1,
            y0,
            z0,
            W_in,
            H_in,
            D_in,
            stride_gi_D,
            stride_gi_H,
            stride_gi_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off010, m010 = _clamp_and_mask_3d(
            x0,
            y0 + 1,
            z0,
            W_in,
            H_in,
            D_in,
            stride_gi_D,
            stride_gi_H,
            stride_gi_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off011, m011 = _clamp_and_mask_3d(
            x0 + 1,
            y0 + 1,
            z0,
            W_in,
            H_in,
            D_in,
            stride_gi_D,
            stride_gi_H,
            stride_gi_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off100, m100 = _clamp_and_mask_3d(
            x0,
            y0,
            z0 + 1,
            W_in,
            H_in,
            D_in,
            stride_gi_D,
            stride_gi_H,
            stride_gi_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off101, m101 = _clamp_and_mask_3d(
            x0 + 1,
            y0,
            z0 + 1,
            W_in,
            H_in,
            D_in,
            stride_gi_D,
            stride_gi_H,
            stride_gi_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off110, m110 = _clamp_and_mask_3d(
            x0,
            y0 + 1,
            z0 + 1,
            W_in,
            H_in,
            D_in,
            stride_gi_D,
            stride_gi_H,
            stride_gi_W,
            valid_mask,
            padding_mode,
            align_corners,
        )
        off111, m111 = _clamp_and_mask_3d(
            x0 + 1,
            y0 + 1,
            z0 + 1,
            W_in,
            H_in,
            D_in,
            stride_gi_D,
            stride_gi_H,
            stride_gi_W,
            valid_mask,
            padding_mode,
            align_corners,
        )

        for c in range(C):
            grad = tl.load(go_base + c * stride_go_C, mask=dhw_mask, other=0.0).to(
                tl.float32
            )
            cb = gi_n + c * stride_gi_C
            tl.atomic_add(
                cb + off000,
                (grad * w000).to(grad_input_ptr.dtype.element_ty),
                mask=m000,
            )
            tl.atomic_add(
                cb + off001,
                (grad * w001).to(grad_input_ptr.dtype.element_ty),
                mask=m001,
            )
            tl.atomic_add(
                cb + off010,
                (grad * w010).to(grad_input_ptr.dtype.element_ty),
                mask=m010,
            )
            tl.atomic_add(
                cb + off011,
                (grad * w011).to(grad_input_ptr.dtype.element_ty),
                mask=m011,
            )
            tl.atomic_add(
                cb + off100,
                (grad * w100).to(grad_input_ptr.dtype.element_ty),
                mask=m100,
            )
            tl.atomic_add(
                cb + off101,
                (grad * w101).to(grad_input_ptr.dtype.element_ty),
                mask=m101,
            )
            tl.atomic_add(
                cb + off110,
                (grad * w110).to(grad_input_ptr.dtype.element_ty),
                mask=m110,
            )
            tl.atomic_add(
                cb + off111,
                (grad * w111).to(grad_input_ptr.dtype.element_ty),
                mask=m111,
            )

    else:  # nearest
        x_near = _nearbyint_coord(ix)
        y_near = _nearbyint_coord(iy)
        z_near = _nearbyint_coord(iz)
        off_near, m_near = _clamp_and_mask_3d(
            x_near,
            y_near,
            z_near,
            W_in,
            H_in,
            D_in,
            stride_gi_D,
            stride_gi_H,
            stride_gi_W,
            valid_mask,
            padding_mode,
            align_corners,
        )

        for c in range(C):
            grad = tl.load(go_base + c * stride_go_C, mask=dhw_mask, other=0.0).to(
                tl.float32
            )
            tl.atomic_add(
                gi_n + c * stride_gi_C + off_near,
                grad.to(grad_input_ptr.dtype.element_ty),
                mask=m_near,
            )


@triton.autotune(
    configs=_GRID_SAMPLER_3D_AUTOTUNE_CONFIGS,
    key=_GRID_SAMPLER_3D_BACKWARD_AUTOTUNE_KEY,
)
@triton.jit
def _grid_sampler_3d_backward_grid_kernel(
    grad_output_ptr,
    input_ptr,
    grid_ptr,
    grad_grid_ptr,
    N,
    C,
    D_in,
    H_in,
    W_in,
    W_out,
    HW_out,
    DHW_out,
    stride_go_N,
    stride_go_C,
    stride_go_D,
    stride_go_H,
    stride_go_W,
    stride_in_N,
    stride_in_C,
    stride_in_D,
    stride_in_H,
    stride_in_W,
    stride_g_N,
    stride_g_D,
    stride_g_H,
    stride_g_W,
    stride_gg_N,
    stride_gg_D,
    stride_gg_H,
    stride_gg_W,
    interpolation_mode: tl.constexpr,
    padding_mode: tl.constexpr,
    align_corners: tl.constexpr,
    BLOCK_DHW: tl.constexpr,
):
    pid = tl.program_id(0)
    n_idx = tl.program_id(1)

    dhw_offs = pid * BLOCK_DHW + tl.arange(0, BLOCK_DHW)
    dhw_mask = dhw_offs < DHW_out
    d_idx = dhw_offs // HW_out
    hw_rem = dhw_offs % HW_out
    h_idx = hw_rem // W_out
    w_idx = hw_rem % W_out

    g_base = (
        grid_ptr
        + n_idx * stride_g_N
        + d_idx * stride_g_D
        + h_idx * stride_g_H
        + w_idx * stride_g_W
    )
    gx = tl.load(g_base + 0, mask=dhw_mask, other=0.0).to(tl.float32)
    gy = tl.load(g_base + 1, mask=dhw_mask, other=0.0).to(tl.float32)
    gz = tl.load(g_base + 2, mask=dhw_mask, other=0.0).to(tl.float32)

    nan_x = gx != gx
    nan_y = gy != gy
    nan_z = gz != gz
    any_nan = nan_x | nan_y | nan_z
    gx = tl.where(nan_x, -1.0, gx)
    gy = tl.where(nan_y, -1.0, gy)
    gz = tl.where(nan_z, -1.0, gz)

    if padding_mode == 0:
        valid_mask = dhw_mask & ~any_nan
    else:
        valid_mask = dhw_mask

    gg_base = (
        grad_grid_ptr
        + n_idx * stride_gg_N
        + d_idx * stride_gg_D
        + h_idx * stride_gg_H
        + w_idx * stride_gg_W
    )
    zero_vec = tl.zeros((BLOCK_DHW,), dtype=grad_grid_ptr.dtype.element_ty)

    if interpolation_mode == 1:
        tl.store(gg_base + 0, zero_vec, mask=dhw_mask)
        tl.store(gg_base + 1, zero_vec, mask=dhw_mask)
        tl.store(gg_base + 2, zero_vec, mask=dhw_mask)
        return

    if align_corners:
        gix_mult = tl.cast(W_in - 1, tl.float32) * 0.5
        giy_mult = tl.cast(H_in - 1, tl.float32) * 0.5
        giz_mult = tl.cast(D_in - 1, tl.float32) * 0.5
        ix = (gx + 1.0) * gix_mult
        iy = (gy + 1.0) * giy_mult
        iz = (gz + 1.0) * giz_mult
    else:
        gix_mult = tl.cast(W_in, tl.float32) * 0.5
        giy_mult = tl.cast(H_in, tl.float32) * 0.5
        giz_mult = tl.cast(D_in, tl.float32) * 0.5
        ix = ((gx + 1.0) * W_in - 1.0) * 0.5
        iy = ((gy + 1.0) * H_in - 1.0) * 0.5
        iz = ((gz + 1.0) * D_in - 1.0) * 0.5

    x0 = tl.floor(ix).to(tl.int32)
    y0 = tl.floor(iy).to(tl.int32)
    z0 = tl.floor(iz).to(tl.int32)
    tx = ix - tl.floor(ix)
    ty = iy - tl.floor(iy)
    tz = iz - tl.floor(iz)

    wx0 = 1.0 - tx
    wx1 = tx
    wy0 = 1.0 - ty
    wy1 = ty
    wz0 = 1.0 - tz
    wz1 = tz

    acc_x = tl.zeros((BLOCK_DHW,), dtype=tl.float32)
    acc_y = tl.zeros((BLOCK_DHW,), dtype=tl.float32)
    acc_z = tl.zeros((BLOCK_DHW,), dtype=tl.float32)

    off000, m000 = _clamp_and_mask_3d(
        x0,
        y0,
        z0,
        W_in,
        H_in,
        D_in,
        stride_in_D,
        stride_in_H,
        stride_in_W,
        valid_mask,
        padding_mode,
        align_corners,
    )
    off001, m001 = _clamp_and_mask_3d(
        x0 + 1,
        y0,
        z0,
        W_in,
        H_in,
        D_in,
        stride_in_D,
        stride_in_H,
        stride_in_W,
        valid_mask,
        padding_mode,
        align_corners,
    )
    off010, m010 = _clamp_and_mask_3d(
        x0,
        y0 + 1,
        z0,
        W_in,
        H_in,
        D_in,
        stride_in_D,
        stride_in_H,
        stride_in_W,
        valid_mask,
        padding_mode,
        align_corners,
    )
    off011, m011 = _clamp_and_mask_3d(
        x0 + 1,
        y0 + 1,
        z0,
        W_in,
        H_in,
        D_in,
        stride_in_D,
        stride_in_H,
        stride_in_W,
        valid_mask,
        padding_mode,
        align_corners,
    )
    off100, m100 = _clamp_and_mask_3d(
        x0,
        y0,
        z0 + 1,
        W_in,
        H_in,
        D_in,
        stride_in_D,
        stride_in_H,
        stride_in_W,
        valid_mask,
        padding_mode,
        align_corners,
    )
    off101, m101 = _clamp_and_mask_3d(
        x0 + 1,
        y0,
        z0 + 1,
        W_in,
        H_in,
        D_in,
        stride_in_D,
        stride_in_H,
        stride_in_W,
        valid_mask,
        padding_mode,
        align_corners,
    )
    off110, m110 = _clamp_and_mask_3d(
        x0,
        y0 + 1,
        z0 + 1,
        W_in,
        H_in,
        D_in,
        stride_in_D,
        stride_in_H,
        stride_in_W,
        valid_mask,
        padding_mode,
        align_corners,
    )
    off111, m111 = _clamp_and_mask_3d(
        x0 + 1,
        y0 + 1,
        z0 + 1,
        W_in,
        H_in,
        D_in,
        stride_in_D,
        stride_in_H,
        stride_in_W,
        valid_mask,
        padding_mode,
        align_corners,
    )

    go_base = (
        grad_output_ptr
        + n_idx * stride_go_N
        + d_idx * stride_go_D
        + h_idx * stride_go_H
        + w_idx * stride_go_W
    )
    in_n = input_ptr + n_idx * stride_in_N

    for c in range(C):
        cb = in_n + c * stride_in_C
        go = tl.load(go_base + c * stride_go_C, mask=dhw_mask, other=0.0).to(tl.float32)

        v000 = tl.load(cb + off000, mask=m000, other=0.0).to(tl.float32)
        v001 = tl.load(cb + off001, mask=m001, other=0.0).to(tl.float32)
        v010 = tl.load(cb + off010, mask=m010, other=0.0).to(tl.float32)
        v011 = tl.load(cb + off011, mask=m011, other=0.0).to(tl.float32)
        v100 = tl.load(cb + off100, mask=m100, other=0.0).to(tl.float32)
        v101 = tl.load(cb + off101, mask=m101, other=0.0).to(tl.float32)
        v110 = tl.load(cb + off110, mask=m110, other=0.0).to(tl.float32)
        v111 = tl.load(cb + off111, mask=m111, other=0.0).to(tl.float32)

        dix = wz0 * (wy0 * (v001 - v000) + wy1 * (v011 - v010)) + wz1 * (
            wy0 * (v101 - v100) + wy1 * (v111 - v110)
        )
        diy = wz0 * (wx0 * (v010 - v000) + wx1 * (v011 - v001)) + wz1 * (
            wx0 * (v110 - v100) + wx1 * (v111 - v101)
        )
        diz = wy0 * (wx0 * (v100 - v000) + wx1 * (v101 - v001)) + wy1 * (
            wx0 * (v110 - v010) + wx1 * (v111 - v011)
        )
        acc_x += go * dix
        acc_y += go * diy
        acc_z += go * diz

    tl.store(
        gg_base + 0,
        (acc_x * gix_mult).to(grad_grid_ptr.dtype.element_ty),
        mask=dhw_mask,
    )
    tl.store(
        gg_base + 1,
        (acc_y * giy_mult).to(grad_grid_ptr.dtype.element_ty),
        mask=dhw_mask,
    )
    tl.store(
        gg_base + 2,
        (acc_z * giz_mult).to(grad_grid_ptr.dtype.element_ty),
        mask=dhw_mask,
    )


def _check_grid_sampler_common(input: torch.Tensor, grid: torch.Tensor):
    if input.device != grid.device:
        raise RuntimeError(
            "grid_sampler(): expected input and grid to be on same device, but "
            f"input is on {input.device} and grid is on {grid.device}"
        )
    if input.layout != torch.strided or grid.layout != torch.strided:
        raise RuntimeError(
            "grid_sampler(): expected input and grid to have torch.strided layout, "
            f"but input has {input.layout} and grid has {grid.layout}"
        )
    if input.shape[0] != grid.shape[0]:
        raise RuntimeError(
            "grid_sampler(): expected grid and input to have same batch size, but "
            f"got input with sizes {list(input.shape)} and grid with sizes {list(grid.shape)}"
        )
    if grid.shape[-1] != input.dim() - 2:
        raise RuntimeError(
            "grid_sampler(): expected grid to have size "
            f"{input.dim() - 2} in last dimension, but got grid with sizes "
            f"{list(grid.shape)}"
        )
    for dim in range(2, input.dim()):
        if input.shape[dim] <= 0:
            raise RuntimeError(
                "grid_sampler(): expected input to have non-empty spatial "
                f"dimensions, but input has sizes {list(input.shape)} with "
                f"dimension {dim} being empty"
            )


def _check_grid_sampler_2d(input: torch.Tensor, grid: torch.Tensor):
    if input.dim() != 4 or input.dim() != grid.dim():
        raise RuntimeError(
            "grid_sampler(): expected 4D input and grid with same number of "
            f"dimensions, but got input with sizes {list(input.shape)} and grid "
            f"with sizes {list(grid.shape)}"
        )


def _check_grid_sampler_3d(
    input: torch.Tensor, grid: torch.Tensor, interpolation_mode: int
):
    if input.dim() != 5 or input.dim() != grid.dim():
        raise RuntimeError(
            "grid_sampler(): expected 5D input and grid with same number of "
            f"dimensions, but got input with sizes {list(input.shape)} and grid "
            f"with sizes {list(grid.shape)}"
        )
    if interpolation_mode == _INTERP_BICUBIC:
        raise RuntimeError(
            "grid_sampler(): bicubic interpolation only supports 4D input"
        )


def grid_sampler_2d(
    input: torch.Tensor,
    grid: torch.Tensor,
    interpolation_mode: int,
    padding_mode: int,
    align_corners: bool,
) -> torch.Tensor:
    logger.debug("GEMS GRID_SAMPLER_2D")

    if not input.is_cuda:
        raise ValueError("grid_sampler_2d: expected CUDA tensor for input")
    _check_grid_sampler_common(input, grid)
    _check_grid_sampler_2d(input, grid)

    N, C, H_in, W_in = input.shape
    H_out, W_out = grid.shape[1], grid.shape[2]

    # Ensure contiguous for predictable strides
    input = input.contiguous()
    grid = grid.contiguous()

    output = torch.empty((N, C, H_out, W_out), dtype=input.dtype, device=input.device)

    sN, sC, sH, sW = input.stride()
    gN, gH, gW, _ = grid.stride()
    oN, oC, oH, oW = output.stride()

    HW_out = H_out * W_out
    grid_fn = lambda meta: (triton.cdiv(HW_out, meta["BLOCK_HW"]), N)

    _grid_sampler_2d_kernel[grid_fn](
        input,
        grid,
        output,
        N,
        C,
        H_in,
        W_in,
        W_out,
        HW_out,
        sN,
        sC,
        sH,
        sW,
        gN,
        gH,
        gW,
        oN,
        oC,
        oH,
        oW,
        interpolation_mode,
        padding_mode,
        align_corners,
    )

    return output


def grid_sampler_3d(
    input: torch.Tensor,
    grid: torch.Tensor,
    interpolation_mode: int,
    padding_mode: int,
    align_corners: bool,
) -> torch.Tensor:
    logger.debug("GEMS GRID_SAMPLER_3D")

    if not input.is_cuda:
        raise ValueError("grid_sampler_3d: expected CUDA tensor for input")

    _check_grid_sampler_common(input, grid)
    _check_grid_sampler_3d(input, grid, interpolation_mode)

    N, C, D_in, H_in, W_in = input.shape
    D_out, H_out, W_out = grid.shape[1], grid.shape[2], grid.shape[3]

    input = input.contiguous()
    grid = grid.contiguous()

    output = torch.empty(
        (N, C, D_out, H_out, W_out), dtype=input.dtype, device=input.device
    )

    sN, sC, sD, sH, sW = input.stride()
    gN, gD, gH, gW, _ = grid.stride()
    oN, oC, oD, oH, oW = output.stride()

    HW_out = H_out * W_out
    DHW_out = D_out * HW_out
    grid_fn = lambda meta: (triton.cdiv(DHW_out, meta["BLOCK_DHW"]), N)

    _grid_sampler_3d_kernel[grid_fn](
        input,
        grid,
        output,
        N,
        C,
        D_in,
        H_in,
        W_in,
        W_out,
        HW_out,
        DHW_out,
        sN,
        sC,
        sD,
        sH,
        sW,
        gN,
        gD,
        gH,
        gW,
        oN,
        oC,
        oD,
        oH,
        oW,
        interpolation_mode,
        padding_mode,
        align_corners,
    )

    return output


def grid_sampler_2d_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    grid: torch.Tensor,
    interpolation_mode: int,
    padding_mode: int,
    align_corners: bool,
    output_mask,
):
    logger.debug("GEMS GRID_SAMPLER_2D_BACKWARD")

    if not input.is_cuda or not grad_output.is_cuda:
        raise ValueError("grid_sampler_2d_backward: expected CUDA tensors")

    _check_grid_sampler_common(input, grid)
    _check_grid_sampler_2d(input, grid)

    grad_output = grad_output.contiguous()
    input = input.contiguous()
    grid = grid.contiguous()

    N, C, H_in, W_in = input.shape
    H_out, W_out = grid.shape[1], grid.shape[2]

    sgoN, sgoC, sgoH, sgoW = grad_output.stride()
    sN, sC, sH, sW = input.stride()
    gN, gH, gW, _ = grid.stride()

    HW_out = H_out * W_out

    grad_input = None
    if output_mask[0]:
        grad_input = torch.zeros_like(input, memory_format=torch.contiguous_format)
        sgiN, sgiC, sgiH, sgiW = grad_input.stride()
        input_grid = lambda meta: (triton.cdiv(HW_out, meta["BLOCK_HW"]), N)
        _grid_sampler_2d_backward_input_kernel[input_grid](
            grad_output,
            grid,
            grad_input,
            N,
            C,
            H_in,
            W_in,
            W_out,
            HW_out,
            sgoN,
            sgoC,
            sgoH,
            sgoW,
            gN,
            gH,
            gW,
            sgiN,
            sgiC,
            sgiH,
            sgiW,
            interpolation_mode,
            padding_mode,
            align_corners,
        )

    # Match ATen's default overload, which always materializes grad_grid.
    grad_grid = torch.empty_like(grid, memory_format=torch.contiguous_format)
    sggN, sggH, sggW, _ = grad_grid.stride()
    grid_grid = lambda meta: (triton.cdiv(HW_out, meta["BLOCK_HW"]), N)
    _grid_sampler_2d_backward_grid_kernel[grid_grid](
        grad_output,
        input,
        grid,
        grad_grid,
        N,
        C,
        H_in,
        W_in,
        W_out,
        HW_out,
        sgoN,
        sgoC,
        sgoH,
        sgoW,
        sN,
        sC,
        sH,
        sW,
        gN,
        gH,
        gW,
        sggN,
        sggH,
        sggW,
        interpolation_mode,
        padding_mode,
        align_corners,
    )

    return grad_input, grad_grid


def grid_sampler_3d_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    grid: torch.Tensor,
    interpolation_mode: int,
    padding_mode: int,
    align_corners: bool,
    output_mask,
):
    logger.debug("GEMS GRID_SAMPLER_3D_BACKWARD")

    if not input.is_cuda or not grad_output.is_cuda:
        raise ValueError("grid_sampler_3d_backward: expected CUDA tensors")

    _check_grid_sampler_common(input, grid)
    _check_grid_sampler_3d(input, grid, interpolation_mode)

    grad_output = grad_output.contiguous()
    input = input.contiguous()
    grid = grid.contiguous()

    N, C, D_in, H_in, W_in = input.shape
    D_out, H_out, W_out = grid.shape[1], grid.shape[2], grid.shape[3]

    sgoN, sgoC, sgoD, sgoH, sgoW = grad_output.stride()
    sN, sC, sD, sH, sW = input.stride()
    gN, gD, gH, gW, _ = grid.stride()

    HW_out = H_out * W_out
    DHW_out = D_out * HW_out

    grad_input = None
    if output_mask[0]:
        grad_input = torch.zeros_like(input, memory_format=torch.contiguous_format)
        sgiN, sgiC, sgiD, sgiH, sgiW = grad_input.stride()
        input_grid = lambda meta: (triton.cdiv(DHW_out, meta["BLOCK_DHW"]), N)
        _grid_sampler_3d_backward_input_kernel[input_grid](
            grad_output,
            grid,
            grad_input,
            N,
            C,
            D_in,
            H_in,
            W_in,
            W_out,
            HW_out,
            DHW_out,
            sgoN,
            sgoC,
            sgoD,
            sgoH,
            sgoW,
            gN,
            gD,
            gH,
            gW,
            sgiN,
            sgiC,
            sgiD,
            sgiH,
            sgiW,
            interpolation_mode,
            padding_mode,
            align_corners,
        )

    # Match ATen's default overload, which always materializes grad_grid.
    grad_grid = torch.empty_like(grid, memory_format=torch.contiguous_format)
    sggN, sggD, sggH, sggW, _ = grad_grid.stride()
    grid_grid = lambda meta: (triton.cdiv(DHW_out, meta["BLOCK_DHW"]), N)
    _grid_sampler_3d_backward_grid_kernel[grid_grid](
        grad_output,
        input,
        grid,
        grad_grid,
        N,
        C,
        D_in,
        H_in,
        W_in,
        W_out,
        HW_out,
        DHW_out,
        sgoN,
        sgoC,
        sgoD,
        sgoH,
        sgoW,
        sN,
        sC,
        sD,
        sH,
        sW,
        gN,
        gD,
        gH,
        gW,
        sggN,
        sggD,
        sggH,
        sggW,
        interpolation_mode,
        padding_mode,
        align_corners,
    )

    return grad_input, grad_grid


def grid_sampler(
    input: torch.Tensor,
    grid: torch.Tensor,
    interpolation_mode: int,
    padding_mode: int,
    align_corners: bool,
) -> torch.Tensor:
    if input.dim() == 4:
        return grid_sampler_2d(
            input, grid, interpolation_mode, padding_mode, align_corners
        )
    if input.dim() == 5:
        return grid_sampler_3d(
            input, grid, interpolation_mode, padding_mode, align_corners
        )
    if input.dim() != grid.dim():
        raise RuntimeError(
            "grid_sampler(): expected input and grid with same number of "
            f"dimensions, but got input with sizes {list(input.shape)} and grid "
            f"with sizes {list(grid.shape)}"
        )
    raise RuntimeError(
        "grid_sampler(): expected 4D or 5D input and grid with same number of "
        f"dimensions, but got input with sizes {list(input.shape)} and grid "
        f"with sizes {list(grid.shape)}"
    )
