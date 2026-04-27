import logging
import math

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _cubic_aa_filter(x):
    """Keys cubic filter with a = -0.5 (PIL-compatible).  x must be >= 0."""
    return tl.where(
        x < 1.0,
        (1.5 * x - 2.5) * x * x + 1.0,
        tl.where(
            x < 2.0,
            ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0,
            0.0,
        ),
    )


@triton.jit
def _f2i(x):
    """float -> int32 with clamping to avoid undefined overflow."""
    _LO: tl.constexpr = -2147483648.0
    _HI: tl.constexpr = 2147483520.0
    return tl.minimum(tl.maximum(x, _LO), _HI).to(tl.int32)


@triton.jit
def _fused_backward_kernel(
    grad_out_ptr,  # [NC, H_out, W_out] flat
    grad_in_ptr,  # [NC, H_in,  W_in]  flat (output)
    # H params
    H_in,
    H_out,
    h_scale,
    support_h,
    invscale_h,
    inv_h_scale,
    # W params
    W_in,
    W_out,
    w_scale,
    support_w,
    invscale_w,
    inv_w_scale,
    # Stride
    stride_go_nc,  # = H_out * W_out
    # Compile-time constants
    BLOCK_IW: tl.constexpr,
    MAX_OH: tl.constexpr,
    MAX_OW: tl.constexpr,
    MAX_KSIZE_H: tl.constexpr,
    MAX_KSIZE_W: tl.constexpr,
):
    pid_row = tl.program_id(0)  # nc * H_in + ih
    pid_col = tl.program_id(1)  # iw tile

    nc = pid_row // H_in
    ih = pid_row % H_in
    ih_f = ih.to(tl.float32)

    iw_base = pid_col * BLOCK_IW
    iws = iw_base + tl.arange(0, BLOCK_IW)
    iw_mask = iws < W_in
    iw_f = iws.to(tl.float32)

    # Scalar: which oh values contribute to this ih
    oh_start = tl.maximum(_f2i((ih_f + 0.5 - support_h) * inv_h_scale - 0.5), 0)

    # Vector: which ow values contribute to each iw
    ow_starts = tl.maximum(_f2i((iw_f + 0.5 - support_w) * inv_w_scale - 0.5), 0)

    go_nc_base = nc.to(tl.int64) * stride_go_nc

    accum = tl.zeros([BLOCK_IW], dtype=tl.float32)

    # --- d_ow OUTER loop: wx computed once per d_ow, reused across d_oh ---
    for d_ow in tl.static_range(MAX_OW):
        ow = ow_starts + d_ow  # vector
        ow_valid_base = iw_mask & (ow >= 0) & (ow < W_out)

        # Compute wx (vector) — only once per d_ow
        center_w = w_scale * (ow.to(tl.float32) + 0.5)
        xmin_w = tl.maximum(_f2i(center_w - support_w + 0.5), 0)
        xsize_w = tl.minimum(_f2i(center_w + support_w + 0.5), W_in) - xmin_w
        xsize_w_pos = tl.maximum(xsize_w, 0)
        iw_in_range = ow_valid_base & (iws >= xmin_w) & (iws < xmin_w + xsize_w_pos)

        # Inline total_wx computation (vector)
        xmin_w_f = xmin_w.to(tl.float32)
        total_wx = tl.zeros([BLOCK_IW], dtype=tl.float32)
        for j_w in tl.static_range(MAX_KSIZE_W):
            arg_w = tl.abs((j_w + xmin_w_f - center_w + 0.5) * invscale_w)
            w_w = _cubic_aa_filter(arg_w)
            total_wx += tl.where(j_w < xsize_w_pos, w_w, 0.0)

        raw_wx = _cubic_aa_filter(tl.abs((iw_f - center_w + 0.5) * invscale_w))
        wx = tl.where(iw_in_range & (total_wx != 0.0), raw_wx / total_wx, 0.0)

        ow_safe = tl.maximum(tl.minimum(ow, W_out - 1), 0)

        # --- d_oh INNER loop: wy is scalar, cheap to recompute ---
        for d_oh in tl.static_range(MAX_OH):
            oh = oh_start + d_oh  # scalar
            oh_valid = (oh >= 0) & (oh < H_out)

            # Compute wy (scalar)
            center_h = h_scale * (oh + 0.5)
            ymin_h = tl.maximum(_f2i(center_h - support_h + 0.5), 0)
            ysize_h = tl.minimum(_f2i(center_h + support_h + 0.5), H_in) - ymin_h
            ysize_h_pos = tl.maximum(ysize_h, 0)
            ih_in_range = oh_valid & (ih >= ymin_h) & (ih < ymin_h + ysize_h_pos)

            # Inline total_wy computation (scalar, very cheap)
            ymin_h_f = ymin_h.to(tl.float32)
            total_wy = 0.0
            for j_h in tl.static_range(MAX_KSIZE_H):
                arg_h = tl.abs((j_h + ymin_h_f - center_h + 0.5) * invscale_h)
                w_h = _cubic_aa_filter(arg_h)
                total_wy += tl.where(j_h < ysize_h_pos, w_h, 0.0)

            raw_wy = _cubic_aa_filter(tl.abs((ih_f - center_h + 0.5) * invscale_h))
            wy = tl.where(ih_in_range & (total_wy != 0.0), raw_wy / total_wy, 0.0)

            # Load grad_out and accumulate
            valid = iw_in_range & ih_in_range
            oh_safe = tl.maximum(tl.minimum(oh, H_out - 1), 0)
            g = tl.load(
                grad_out_ptr
                + go_nc_base
                + oh_safe.to(tl.int64) * W_out
                + ow_safe.to(tl.int64),
                mask=valid,
                other=0.0,
            )
            accum += wy * wx * g

    gi_off = pid_row.to(tl.int64) * W_in + iws.to(tl.int64)
    tl.store(
        grad_in_ptr + gi_off,
        accum.to(grad_in_ptr.dtype.element_ty),
        mask=iw_mask,
    )


@triton.jit
def _precompute_weight_sums_kernel(
    total_w_ptr,
    output_size,
    input_size,
    scale,
    support,
    invscale,
    MAX_KSIZE: tl.constexpr,
):
    oi = tl.program_id(0)
    if oi >= output_size:
        return
    center = scale * (oi + 0.5)
    xmin = tl.maximum(_f2i(center - support + 0.5), 0)
    xsize = tl.minimum(_f2i(center + support + 0.5), input_size) - xmin
    xsize = tl.minimum(tl.maximum(xsize, 0), MAX_KSIZE)
    xmin_f = xmin.to(tl.float32)
    total = 0.0
    for j in tl.static_range(MAX_KSIZE):
        arg = tl.abs((j + xmin_f - center + 0.5) * invscale)
        w = _cubic_aa_filter(arg)
        total += tl.where(j < xsize, w, 0.0)
    tl.store(total_w_ptr + oi, total)


@triton.jit
def _pass1_w_gather_nchw_kernel(
    grad_out_ptr,  # [NC, H_out, W_out] flat
    buf_ptr,  # [NC, H_out, W_in]  flat (output)
    total_wx_ptr,  # [W_out]
    W_in,
    W_out,
    w_scale,
    support_w,
    invscale_w,
    inv_w_scale,
    BLOCK_IW: tl.constexpr,
    MAX_OW: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    iw_base = pid_col * BLOCK_IW
    iws = iw_base + tl.arange(0, BLOCK_IW)
    iw_mask = iws < W_in
    iw_f = iws.to(tl.float32)

    go_base = pid_row.to(tl.int64) * W_out
    buf_base = pid_row.to(tl.int64) * W_in

    ow_starts = tl.maximum(_f2i((iw_f + 0.5 - support_w) * inv_w_scale - 0.5), 0)

    accum = tl.zeros([BLOCK_IW], dtype=tl.float32)

    for d_ow in tl.static_range(MAX_OW):
        ow = ow_starts + d_ow
        ow_valid = iw_mask & (ow >= 0) & (ow < W_out)

        center_w = w_scale * (ow.to(tl.float32) + 0.5)
        xmin = tl.maximum(_f2i(center_w - support_w + 0.5), 0)
        xsize = tl.minimum(_f2i(center_w + support_w + 0.5), W_in) - xmin
        in_range = ow_valid & (iws >= xmin) & (iws < xmin + tl.maximum(xsize, 0))

        raw_wx = _cubic_aa_filter(tl.abs((iw_f - center_w + 0.5) * invscale_w))
        ow_safe = tl.maximum(tl.minimum(ow, W_out - 1), 0)
        tw_x = tl.load(total_wx_ptr + ow_safe, mask=in_range, other=1.0)
        wx = tl.where(in_range & (tw_x != 0.0), raw_wx / tw_x, 0.0)

        g = tl.load(
            grad_out_ptr + go_base + ow_safe.to(tl.int64), mask=in_range, other=0.0
        )
        accum += wx * g

    tl.store(buf_ptr + buf_base + iws.to(tl.int64), accum, mask=iw_mask)


@triton.jit
def _pass2_h_gather_nchw_kernel(
    buf_ptr,  # [NC, H_out, W_in] flat (input)
    grad_in_ptr,  # [NC, H_in,  W_in] flat (output)
    total_wy_ptr,  # [H_out]
    H_in,
    W_in,
    H_out,
    h_scale,
    support_h,
    invscale_h,
    inv_h_scale,
    stride_buf_hw,  # = H_out * W_in
    BLOCK_IW: tl.constexpr,
    MAX_OH: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    nc = pid_row // H_in
    ih = pid_row % H_in
    ih_f = ih.to(tl.float32)

    iw_base = pid_col * BLOCK_IW
    iws = iw_base + tl.arange(0, BLOCK_IW)
    iw_mask = iws < W_in

    oh_start = tl.maximum(_f2i((ih_f + 0.5 - support_h) * inv_h_scale - 0.5), 0)

    buf_nc_base = nc.to(tl.int64) * stride_buf_hw

    accum = tl.zeros([BLOCK_IW], dtype=tl.float32)

    for d_oh in tl.static_range(MAX_OH):
        oh = oh_start + d_oh
        oh_valid = (oh >= 0) & (oh < H_out)

        center_h = h_scale * (oh + 0.5)
        ymin = tl.maximum(_f2i(center_h - support_h + 0.5), 0)
        ysize = tl.minimum(_f2i(center_h + support_h + 0.5), H_in) - ymin
        ih_in_range = oh_valid & (ih >= ymin) & (ih < ymin + tl.maximum(ysize, 0))

        raw_wy = _cubic_aa_filter(tl.abs((ih_f - center_h + 0.5) * invscale_h))
        oh_safe = tl.maximum(tl.minimum(oh, H_out - 1), 0)
        tw_y = tl.load(total_wy_ptr + oh_safe)
        wy = tl.where(ih_in_range & (tw_y != 0.0), raw_wy / tw_y, 0.0)

        buf_off = buf_nc_base + oh_safe.to(tl.int64) * W_in + iws.to(tl.int64)
        b = tl.load(buf_ptr + buf_off, mask=iw_mask & ih_in_range, other=0.0)

        accum += wy * b

    gi_off = pid_row.to(tl.int64) * W_in + iws.to(tl.int64)
    tl.store(
        grad_in_ptr + gi_off,
        accum.to(grad_in_ptr.dtype.element_ty),
        mask=iw_mask,
    )


def _compute_scale(input_size, output_size, align_corners, scale=None):
    if align_corners:
        return float(input_size - 1) / (output_size - 1) if output_size > 1 else 0.0
    else:
        return (
            (1.0 / scale)
            if (scale is not None and scale > 0)
            else float(input_size) / output_size
        )


# Threshold: when total elements (across the larger of input / output spatial)
# is below this, the fused single-kernel path is used (1 launch instead of 4).
# Above this, the 2-pass separable path is more memory-bandwidth efficient.
_FUSE_THRESHOLD = 1 << 20  # 1M elements


def _upsample_bicubic2d_aa_backward(
    grad_output: torch.Tensor,
    output_size,  # [H_out, W_out]
    input_size,  # [N, C, H_in, W_in]
    align_corners: bool,
    scales_h=None,
    scales_w=None,
) -> torch.Tensor:
    N, C, H_in, W_in = input_size
    H_out, W_out = output_size

    assert grad_output.shape == (N, C, H_out, W_out), (
        f"grad_output shape {grad_output.shape} != "
        f"expected ({N}, {C}, {H_out}, {W_out})"
    )

    NC = N * C
    if NC == 0 or H_in == 0 or W_in == 0 or H_out == 0 or W_out == 0:
        return grad_output.new_zeros(input_size)

    # ---- Work in NCHW — zero-copy reshape to [NC, H, W] ----
    grad_out_flat = grad_output.contiguous().reshape(NC, H_out, W_out)

    # ---- Scales & filter parameters ----
    h_scale = _compute_scale(H_in, H_out, align_corners, scales_h)
    w_scale = _compute_scale(W_in, W_out, align_corners, scales_w)

    INTERP_SIZE = 4
    support_h = (INTERP_SIZE * 0.5) * h_scale if h_scale >= 1.0 else INTERP_SIZE * 0.5
    support_w = (INTERP_SIZE * 0.5) * w_scale if w_scale >= 1.0 else INTERP_SIZE * 0.5
    invscale_h = 1.0 / h_scale if h_scale >= 1.0 else 1.0
    invscale_w = 1.0 / w_scale if w_scale >= 1.0 else 1.0

    MAX_KSIZE_H = math.ceil(support_h) * 2 + 1
    MAX_KSIZE_W = math.ceil(support_w) * 2 + 1

    _EPS = 1e-10
    inv_h_scale = 1.0 / max(h_scale, _EPS)
    inv_w_scale = 1.0 / max(w_scale, _EPS)

    MAX_OH = min(math.ceil(2 * support_h * inv_h_scale) + 2, max(H_out, 1))
    MAX_OW = min(math.ceil(2 * support_w * inv_w_scale) + 2, max(W_out, 1))

    # ---- BLOCK_IW & num_warps ----
    BLOCK_IW = min(triton.next_power_of_2(max(W_in, 1)), 256)
    if BLOCK_IW < 32:
        BLOCK_IW = 32
    nw = 1 if BLOCK_IW <= 32 else (2 if BLOCK_IW <= 64 else 4)

    # ---- Choose fused vs 2-pass ----
    total_elems = NC * max(H_in * W_in, H_out * W_out)
    use_fused = total_elems <= _FUSE_THRESHOLD

    if use_fused:
        # ============================================================
        # FUSED PATH — single kernel launch, no intermediate buffer
        # ============================================================
        grad_in_flat = torch.empty(
            NC, H_in, W_in, dtype=grad_output.dtype, device=grad_output.device
        )
        grid = (NC * H_in, triton.cdiv(W_in, BLOCK_IW))
        _fused_backward_kernel[grid](
            grad_out_flat,
            grad_in_flat,
            H_in,
            H_out,
            h_scale,
            support_h,
            invscale_h,
            inv_h_scale,
            W_in,
            W_out,
            w_scale,
            support_w,
            invscale_w,
            inv_w_scale,
            H_out * W_out,  # stride_go_nc
            BLOCK_IW=BLOCK_IW,
            MAX_OH=MAX_OH,
            MAX_OW=MAX_OW,
            MAX_KSIZE_H=MAX_KSIZE_H,
            MAX_KSIZE_W=MAX_KSIZE_W,
            num_warps=nw,
        )
        return grad_in_flat.reshape(N, C, H_in, W_in)

    else:
        # ============================================================
        # 2-PASS PATH — separable, memory-bandwidth efficient for big tensors
        # ============================================================

        # Phase 0: precompute weight sums
        total_wy = torch.empty(
            max(H_out, 1), dtype=torch.float32, device=grad_output.device
        )
        total_wx = torch.empty(
            max(W_out, 1), dtype=torch.float32, device=grad_output.device
        )
        if H_out > 0:
            _precompute_weight_sums_kernel[(H_out,)](
                total_wy,
                H_out,
                H_in,
                h_scale,
                support_h,
                invscale_h,
                MAX_KSIZE=MAX_KSIZE_H,
            )
        if W_out > 0:
            _precompute_weight_sums_kernel[(W_out,)](
                total_wx,
                W_out,
                W_in,
                w_scale,
                support_w,
                invscale_w,
                MAX_KSIZE=MAX_KSIZE_W,
            )

        # Phase 1: W-gather -> buf [NC, H_out, W_in]
        buf = torch.empty(
            NC, H_out, W_in, dtype=torch.float32, device=grad_output.device
        )
        grid1 = (NC * H_out, triton.cdiv(W_in, BLOCK_IW))
        _pass1_w_gather_nchw_kernel[grid1](
            grad_out_flat,
            buf,
            total_wx,
            W_in,
            W_out,
            w_scale,
            support_w,
            invscale_w,
            inv_w_scale,
            BLOCK_IW=BLOCK_IW,
            MAX_OW=MAX_OW,
            num_warps=nw,
        )

        # Phase 2: H-gather -> grad_in [NC, H_in, W_in]
        grad_in_flat = torch.empty(
            NC, H_in, W_in, dtype=grad_output.dtype, device=grad_output.device
        )
        grid2 = (NC * H_in, triton.cdiv(W_in, BLOCK_IW))
        _pass2_h_gather_nchw_kernel[grid2](
            buf,
            grad_in_flat,
            total_wy,
            H_in,
            W_in,
            H_out,
            h_scale,
            support_h,
            invscale_h,
            inv_h_scale,
            H_out * W_in,  # stride_buf_hw
            BLOCK_IW=BLOCK_IW,
            MAX_OH=MAX_OH,
            num_warps=nw,
        )

        return grad_in_flat.reshape(N, C, H_in, W_in)
