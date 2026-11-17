# Kunlunxin (XPU) specialization of aten::grid_sampler_2d_backward.
#
# The generic implementation (src/flag_gems/ops/grid_sampler_2d_backward.py)
# scatters the bilinear corner contributions back to ``grad_input`` and
# accumulates the per-channel ``grad_grid`` terms with ``tl.atomic_add``.  On
# XPU3 ``tt.atomic_rmw`` is not a correct read-modify-write (lost / duplicated
# updates), which is the observed root cause of the 49 numeric accuracy
# failures of tests/test_grid_sampler_2d_backward.py (see
# harness/KernelOpt_Knowledge/wiki/patterns/xpu3-atomic-add-misaccumulates.md).
#
# This override removes atomics completely; every output element has exactly
# one writer:
#
# * ``grad_input`` is computed **input-centric**.  One program owns one input
#   pixel ``(n, c, h, w)`` and scans the whole output plane ``p = 0 .. P-1``
#   with a 1-D tile, matching each output pixel's four bilinear corners against
#   ``(h, w)``.  The many-to-one output -> input mapping therefore becomes an
#   in-register masked reduction (``tl.sum`` over the pixel axis) instead of a
#   scatter.  Out-of-range corners need no explicit bound masks under zeros
#   padding: an out-of-range integer index simply never equals a valid ``w`` /
#   ``h``.  Border/reflection fold the continuous coordinate like PyTorch's
#   ``grid_sampler_compute_source_index``.
# * ``grad_grid`` is already a reduction over channels at a fixed output pixel,
#   so one program owns one output pixel and reduces the channel axis in
#   registers (``tl.sum``).  Gradients are accumulated in fp32 across channel
#   blocks on the host (load + add), so no loop-carried scalar accumulator and
#   no atomics are needed.  Nearest interpolation has zero ``grad_grid``.
#
# Platform constraints honoured here (see KernelOpt_Knowledge/wiki/patterns):
#   * no ``tl.atomic_*`` anywhere;
#   * no masked loads: every address is clamped to a legal range and the value
#     is zeroed afterwards with ``tl.where`` (masks are not honoured reliably);
#   * no 2-D tile reductions that feed from broadcast tiles, and no loop-carried
#     ``tl.sum``: reductions are 1-D and happen once *after* the (optional)
#     runtime loop, on a vector accumulator;
#   * stores are scalar (one value per lane-0 element) so the "narrow store
#     lane axis is silently widened" trap does not apply.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

_MAX_SINGLE_P = 256  # largest output plane folded into one loop-free tile
_LOOP_BLOCK_P = 128  # tile width inside a runtime loop (wider hits unroll-control)
_MAX_BLOCK_C = 128  # grad_grid channel tile (64/128 verified on this backend)
_MIN_BLOCK_C = 64


@triton.jit
def _src_index(g, n: tl.constexpr, ALIGN_CORNERS: tl.constexpr):
    """Normalised grid coordinate ([-1, 1]) -> pixel-space coordinate."""
    if ALIGN_CORNERS:
        return (g + 1.0) * (0.5 * (n - 1.0))
    return (g + 1.0) * (0.5 * n) - 0.5


@triton.jit
def _reflect_coord(c, n: tl.constexpr, ALIGN_CORNERS: tl.constexpr):
    """Reflect ``c`` into ``[0, n-1]`` (arbitrary number of periods), then clip.

    Mirrors PyTorch ``reflect_coordinates`` + ``clip_coordinates``.
    """
    if ALIGN_CORNERS:
        lo = 0.0
        span = n - 1.0
    else:
        lo = -0.5
        span = n * 1.0
    dist = tl.abs(c - lo)
    flips = tl.floor(dist / span)
    extra = dist - flips * span
    par = (flips.to(tl.int32) & 1) == 0
    ref = tl.where(par, extra + lo, span - extra + lo)
    ref = tl.where(span == 0.0, 0.0, ref)
    return tl.minimum(tl.maximum(ref, 0.0), n - 1.0)


@triton.jit
def _pad_coord(c, n: tl.constexpr, PADDING: tl.constexpr, ALIGN_CORNERS: tl.constexpr):
    """Apply the padding mode to a continuous pixel-space coordinate."""
    if PADDING == 1:  # border: clip the continuous coordinate
        return tl.minimum(tl.maximum(c, 0.0), n - 1.0)
    elif PADDING == 2:  # reflection: reflect then clip
        return _reflect_coord(c, n, ALIGN_CORNERS)
    return c


@triton.jit
def _reflect_grad(c, n: tl.constexpr, ALIGN_CORNERS: tl.constexpr):
    """d/dc of ``clip(reflect(c))`` (PyTorch ``reflect_coordinates_set_grad``).

    RevInt ``clip_coordinates_set_grad`` returns 0 at/over the borders; the sign
    flips once per reflection period and once for ``c < lo``.
    """
    if ALIGN_CORNERS:
        lo = 0.0
        span = n - 1.0
        u = c
    else:
        lo = -0.5
        span = n * 1.0
        u = c + 0.5
    dist = tl.abs(c - lo)
    flips = tl.floor(dist / span)
    extra = dist - flips * span
    par = (flips.to(tl.int32) & 1) == 0
    fold = tl.where(par, extra + lo, span - extra + lo)
    sign = tl.where(u >= 0.0, 1.0, -1.0)
    refl = tl.where(par, 1.0, -1.0) * sign
    clip = tl.where((fold > 0.0) & (fold < n - 1.0), 1.0, 0.0)
    return tl.where(span == 0.0, 0.0, refl * clip)


@triton.jit
def _pad_grad(c, n: tl.constexpr, PADDING: tl.constexpr, ALIGN_CORNERS: tl.constexpr):
    """d(padded coordinate)/d(raw coordinate) for ``grad_grid``."""
    if PADDING == 1:  # border: derivative of clip
        return tl.where((c > 0.0) & (c < n - 1.0), 1.0, 0.0)
    elif PADDING == 2:  # reflection: derivative of clip(reflect(c))
        return _reflect_grad(c, n, ALIGN_CORNERS)
    return 1.0


@triton.jit
def _nearest_idx(floor_c, frac):
    """Round half to even, matching ``std::nearbyint`` in PyTorch."""
    up = (frac > 0.5) | ((frac == 0.5) & ((floor_c & 1) == 1))
    return floor_c + up.to(tl.int32)


@triton.jit
def _grad_input_pixel(
    grad_output_ptr,
    grid_ptr,
    pid_n,
    pid_c,
    GO_BASE,
    GRID_BASE,
    go_pix_stride,
    H: tl.constexpr,
    W: tl.constexpr,
    OW: tl.constexpr,
    P: tl.constexpr,
    OFFS,
    VALID,
    MODE: tl.constexpr,
    PADDING: tl.constexpr,
    ALIGN_CORNERS: tl.constexpr,
):
    """Contribution of one pixel-tile to input pixel ``(hh, ww)``.

    ``hh`` / ``ww`` are the owning input pixel; ``OFFS``/``VALID`` are the
    output-pixel tile lanes.
    """
    pid_q = tl.program_id(2)
    hh = pid_q // W
    ww = pid_q - hh * W

    ps = tl.where(VALID, OFFS, 0)
    gx = tl.load(grid_ptr + GRID_BASE + ps * 2).to(tl.float32)
    gy = tl.load(grid_ptr + GRID_BASE + ps * 2 + 1).to(tl.float32)
    gx = tl.where(VALID, gx, 0.0)
    gy = tl.where(VALID, gy, 0.0)

    x = _pad_coord(_src_index(gx, W, ALIGN_CORNERS), W, PADDING, ALIGN_CORNERS)
    y = _pad_coord(_src_index(gy, H, ALIGN_CORNERS), H, PADDING, ALIGN_CORNERS)
    # Keep the float -> int conversion representable; out-of-range values never
    # match a valid input index, so clamping the payload is harmless.
    x = tl.minimum(tl.maximum(x, -1.0e8), 1.0e8)
    y = tl.minimum(tl.maximum(y, -1.0e8), 1.0e8)

    x0f = tl.floor(x)
    y0f = tl.floor(y)
    x0 = x0f.to(tl.int32)
    y0 = y0f.to(tl.int32)

    if MODE == 1:  # nearest: piecewise constant, a single corner
        xn = _nearest_idx(x0, x - x0f)
        yn = _nearest_idx(y0, y - y0f)
        weight = tl.where((xn == ww) & (yn == hh), 1.0, 0.0)
    else:  # bilinear: four corners, weights keyed on exact index equality
        tx = x - x0f
        ty = y - y0f
        x1 = x0 + 1
        y1 = y0 + 1
        w00 = tl.where((x0 == ww) & (y0 == hh), (1.0 - tx) * (1.0 - ty), 0.0)
        w10 = tl.where((x1 == ww) & (y0 == hh), tx * (1.0 - ty), 0.0)
        w01 = tl.where((x0 == ww) & (y1 == hh), (1.0 - tx) * ty, 0.0)
        w11 = tl.where((x1 == ww) & (y1 == hh), tx * ty, 0.0)
        weight = w00 + w10 + w01 + w11

    w = tl.where(VALID, weight, 0.0)
    g = tl.load(grad_output_ptr + GO_BASE + ps * go_pix_stride).to(tl.float32)
    g = tl.where(VALID, g, 0.0)
    return g * w


@triton.jit
def _grad_input_kernel(
    grad_output_ptr,
    grid_ptr,
    grad_input_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    OW: tl.constexpr,
    P: tl.constexpr,
    GRID_BASE_N,
    go_sN,
    go_sC,
    go_pix_stride,
    gi_sN,
    gi_sC,
    NB,
    MODE: tl.constexpr,
    PADDING: tl.constexpr,
    ALIGN_CORNERS: tl.constexpr,
    BLOCK_P: tl.constexpr,
    SINGLE: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    go_base = pid_n * go_sN + pid_c * go_sC
    grid_base = pid_n * GRID_BASE_N

    if SINGLE:
        offs = tl.arange(0, BLOCK_P)
        valid = offs < P
        val = tl.sum(
            _grad_input_pixel(
                grad_output_ptr,
                grid_ptr,
                pid_n,
                pid_c,
                go_base,
                grid_base,
                go_pix_stride,
                H,
                W,
                OW,
                P,
                offs,
                valid,
                MODE,
                PADDING,
                ALIGN_CORNERS,
            ),
            axis=0,
        )
    else:
        acc = tl.zeros([BLOCK_P], dtype=tl.float32)
        for i in tl.range(0, NB):
            offs = i * BLOCK_P + tl.arange(0, BLOCK_P)
            valid = offs < P
            acc += _grad_input_pixel(
                grad_output_ptr,
                grid_ptr,
                pid_n,
                pid_c,
                go_base,
                grid_base,
                go_pix_stride,
                H,
                W,
                OW,
                P,
                offs,
                valid,
                MODE,
                PADDING,
                ALIGN_CORNERS,
            )
        val = tl.sum(acc, axis=0)

    pid_q = tl.program_id(2)
    tl.store(
        grad_input_ptr + pid_n * gi_sN + pid_c * gi_sC + pid_q,
        val.to(grad_input_ptr.dtype.element_ty),
    )


@triton.jit
def _grad_grid_kernel(
    grad_output_ptr,
    input_ptr,
    grid_ptr,
    grad_grid_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    OW: tl.constexpr,
    P: tl.constexpr,
    GRID_BASE_N,
    go_sN,
    go_sC,
    in_sN,
    in_sC,
    gg_sN,
    c_begin,
    c_end,
    MODE: tl.constexpr,
    PADDING: tl.constexpr,
    ALIGN_CORNERS: tl.constexpr,
    ACCUMULATE: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """grad_grid for one output pixel (program 0) of one batch element."""
    p = tl.program_id(0)
    pid_n = tl.program_id(1)

    oh = p // OW
    ow = p - oh * OW
    grid_off = pid_n * GRID_BASE_N + p * 2
    gx = tl.load(grid_ptr + grid_off)
    gy = tl.load(grid_ptr + grid_off + 1)
    gx = gx.to(tl.float32)
    gy = gy.to(tl.float32)

    if ALIGN_CORNERS:
        scale_x = 0.5 * (W - 1.0)
        scale_y = 0.5 * (H - 1.0)
    else:
        scale_x = 0.5 * W
        scale_y = 0.5 * H

    rx = _src_index(gx, W, ALIGN_CORNERS)
    ry = _src_index(gy, H, ALIGN_CORNERS)
    pad_x = _pad_grad(rx, W, PADDING, ALIGN_CORNERS)
    pad_y = _pad_grad(ry, H, PADDING, ALIGN_CORNERS)

    x = _pad_coord(rx, W, PADDING, ALIGN_CORNERS)
    y = _pad_coord(ry, H, PADDING, ALIGN_CORNERS)
    x = tl.minimum(tl.maximum(x, -1.0e8), 1.0e8)
    y = tl.minimum(tl.maximum(y, -1.0e8), 1.0e8)
    x0f = tl.floor(x)
    y0f = tl.floor(y)
    x0 = x0f.to(tl.int32)
    y0 = y0f.to(tl.int32)
    tx = x - x0f
    ty = y - y0f
    x1 = x0 + 1
    y1 = y0 + 1

    # Clamped addresses keep every load legal; value masks gate the corners.
    ix0 = tl.minimum(tl.maximum(x0, 0), W - 1)
    ix1 = tl.minimum(tl.maximum(x1, 0), W - 1)
    iy0 = tl.minimum(tl.maximum(y0, 0), H - 1)
    iy1 = tl.minimum(tl.maximum(y1, 0), H - 1)
    cs = c_begin + tl.arange(0, BLOCK_C)
    cm = cs < c_end
    if PADDING == 1:  # border: clamped coordinate, every corner contributes
        keep00 = cm
        keep10 = cm
        keep01 = cm
        keep11 = cm
    else:  # zeros / reflection: out-of-range corners contribute nothing
        keep00 = cm & (x0 >= 0) & (x0 < W) & (y0 >= 0) & (y0 < H)
        keep10 = cm & (x1 >= 0) & (x1 < W) & (y0 >= 0) & (y0 < H)
        keep01 = cm & (x0 >= 0) & (x0 < W) & (y1 >= 0) & (y1 < H)
        keep11 = cm & (x1 >= 0) & (x1 < W) & (y1 >= 0) & (y1 < H)
    q00 = iy0 * W + ix0
    q10 = iy0 * W + ix1
    q01 = iy1 * W + ix0
    q11 = iy1 * W + ix1

    cs = c_begin + tl.arange(0, BLOCK_C)
    cm = cs < c_end
    cc = tl.where(cm, cs, 0)

    go = tl.load(grad_output_ptr + pid_n * go_sN + cc * go_sC + oh * OW + ow)
    go = tl.where(cm, go, 0.0).to(tl.float32)
    ib = pid_n * in_sN + cc * in_sC
    i00 = tl.load(input_ptr + ib + q00).to(tl.float32)
    i10 = tl.load(input_ptr + ib + q10).to(tl.float32)
    i01 = tl.load(input_ptr + ib + q01).to(tl.float32)
    i11 = tl.load(input_ptr + ib + q11).to(tl.float32)
    i00 = tl.where(keep00, i00, 0.0)
    i10 = tl.where(keep10, i10, 0.0)
    i01 = tl.where(keep01, i01, 0.0)
    i11 = tl.where(keep11, i11, 0.0)

    diff_x0 = i10 - i00
    diff_x1 = i11 - i01
    gxs = go * (scale_x * (diff_x0 * (1.0 - ty) + diff_x1 * ty)) * pad_x
    diff_y0 = i01 - i00
    diff_y1 = i11 - i10
    gys = go * (scale_y * (diff_y0 * (1.0 - tx) + diff_y1 * tx)) * pad_y

    sx = tl.sum(gxs, axis=0)
    sy = tl.sum(gys, axis=0)

    gg_base = pid_n * gg_sN + p * 2
    if ACCUMULATE:
        sx = sx + tl.load(grad_grid_ptr + gg_base)
        sy = sy + tl.load(grad_grid_ptr + gg_base + 1)
    tl.store(grad_grid_ptr + gg_base, sx.to(grad_grid_ptr.dtype.element_ty))
    tl.store(grad_grid_ptr + gg_base + 1, sy.to(grad_grid_ptr.dtype.element_ty))


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def grid_sampler_2d_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    grid: torch.Tensor,
    interpolation_mode: int,
    padding_mode: int,
    align_corners: bool,
    output_mask: tuple,
):
    """Backward of ``aten::grid_sampler_2d`` (atomic-free XPU3 override)."""
    logger.debug(
        "GEMS_KUNLUNXIN GRID_SAMPLER_2D_BACKWARD: mode=%d pad=%d align=%s",
        interpolation_mode,
        padding_mode,
        align_corners,
    )

    if interpolation_mode == 2:
        raise NotImplementedError(
            "grid_sampler_2d_backward: bicubic interpolation is not supported yet"
        )
    if interpolation_mode not in (0, 1) or padding_mode not in (0, 1, 2):
        raise ValueError(
            f"grid_sampler_2d_backward: invalid interpolation_mode "
            f"{interpolation_mode} or padding_mode {padding_mode}"
        )

    need_grad_input, need_grad_grid = bool(output_mask[0]), bool(output_mask[1])

    # Operands are contiguous so the kernels can use plain index arithmetic.
    grad_output = grad_output.contiguous()
    input = input.contiguous()
    grid = grid.contiguous()

    N, C, H, W = input.shape
    OH, OW = grid.shape[1:3]
    P = OH * OW

    grad_input = None
    grad_grid = None
    if not need_grad_input and not need_grad_grid:
        return (grad_input, grad_grid)

    if need_grad_input:
        grad_input = torch.empty((N, C, H, W), dtype=input.dtype, device=input.device)
    if need_grad_grid:
        # fp32 staging so multi-block channel accumulation is exact; cast once.
        grad_grid = torch.zeros((N, OH, OW, 2), dtype=torch.float32, device=grid.device)

    if N == 0 or C == 0 or H == 0 or W == 0 or P == 0:
        if need_grad_input:
            grad_input = grad_input.to(input.dtype)
        if need_grad_grid:
            grad_grid = grad_grid.to(grid.dtype)
        return (grad_input, grad_grid)

    align = bool(align_corners)

    if need_grad_input:
        if P <= _MAX_SINGLE_P:
            block_p = max(64, _next_pow2(P))
            single = True
            nb = 1
        else:
            block_p = _LOOP_BLOCK_P
            single = False
            nb = triton.cdiv(P, block_p)
        grid_launch = (N, C, H * W)
        with torch_device_fn.device(input.device):
            _grad_input_kernel[grid_launch](
                grad_output,
                grid,
                grad_input,
                N,
                C,
                H,
                W,
                OW,
                P,
                grid.stride(0),
                grad_output.stride(0),
                grad_output.stride(1),
                grad_output.stride(3),
                grad_input.stride(0),
                grad_input.stride(1),
                nb,
                MODE=interpolation_mode,
                PADDING=padding_mode,
                ALIGN_CORNERS=align,
                BLOCK_P=block_p,
                SINGLE=single,
                num_warps=4,
                num_stages=1,
            )

    # Nearest interpolation is piecewise constant in the sampling coordinate,
    # so grad_grid is identically zero (already allocated with torch.zeros).
    if need_grad_grid and interpolation_mode == 0:
        block_c = max(_MIN_BLOCK_C, min(_MAX_BLOCK_C, _next_pow2(C)))
        for c_begin in range(0, C, block_c):
            with torch_device_fn.device(input.device):
                _grad_grid_kernel[(P, N)](
                    grad_output,
                    input,
                    grid,
                    grad_grid,
                    C,
                    H,
                    W,
                    OW,
                    P,
                    grid.stride(0),
                    grad_output.stride(0),
                    grad_output.stride(1),
                    input.stride(0),
                    input.stride(1),
                    grad_grid.stride(0),
                    c_begin,
                    min(c_begin + block_c, C),
                    MODE=interpolation_mode,
                    PADDING=padding_mode,
                    ALIGN_CORNERS=align,
                    ACCUMULATE=(c_begin > 0),
                    BLOCK_C=block_c,
                    num_warps=4,
                    num_stages=1,
                )

    if need_grad_grid:
        grad_grid = grad_grid.to(grid.dtype)

    return (grad_input, grad_grid)
