import torch
import triton
import triton.language as tl


@triton.jit
def im2col_kernel(
    x_ptr,  # *Pointer* to input tensor [N, C, H, W]
    out_ptr,  # *Pointer* to output tensor [N, C*kH*kW, outH*outW]
    N,
    C,
    H,
    W,
    kH,
    kW,
    dH,
    dW,
    pH,
    pW,
    sH,
    sW,
    outH,
    outW,
    rows_total,  # C * kH * kW
    L,  # outH * outW
    num_row_tiles,  # ceil_div(rows_total, BLOCK_M)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    n = pid0 // num_row_tiles
    row_tile = pid0 % num_row_tiles

    row_offsets = row_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = pid1 * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_rows = row_offsets < rows_total
    mask_cols = col_offsets < L

    k_area = kH * kW

    c_idx = row_offsets // k_area
    rem = row_offsets % k_area
    kh_idx = rem // kW
    kw_idx = rem % kW

    oh_vec = col_offsets // outW
    ow_vec = col_offsets % outW

    # Broadcast to [BLOCK_M, BLOCK_N]
    oh = oh_vec[None, :]
    ow = ow_vec[None, :]
    kh = kh_idx[:, None]
    kw = kw_idx[:, None]
    c = c_idx[:, None]

    ih = oh * sH - pH + kh * dH
    iw = ow * sW - pW + kw * dW

    in_h = (ih >= 0) & (ih < H)
    in_w = (iw >= 0) & (iw < W)
    in_bounds = in_h & in_w

    # Base offsets
    base_in = (n.to(tl.int64) * C * H * W).to(tl.int64)
    base_out = (n.to(tl.int64) * rows_total * L).to(tl.int64)

    # Compute input pointers
    ptrs_in = (
        x_ptr + base_in + ((c.to(tl.int64) * H + ih.to(tl.int64)) * W + iw.to(tl.int64))
    )

    # Compute output pointers
    ptrs_out = (
        out_ptr
        + base_out
        + (row_offsets[:, None].to(tl.int64) * L + col_offsets[None, :].to(tl.int64))
    )

    mask = mask_rows[:, None] & mask_cols[None, :] & in_bounds

    vals = tl.load(ptrs_in, mask=mask, other=0)
    tl.store(ptrs_out, vals, mask=(mask_rows[:, None] & mask_cols[None, :]))


def _parse_2tuple(x, name):
    if isinstance(x, int):
        return (x, x)
    if (
        isinstance(x, (list, tuple))
        and len(x) == 2
        and all(isinstance(v, int) for v in x)
    ):
        return (int(x[0]), int(x[1]))
    raise ValueError(f"{name} must be an int or a tuple/list of two ints")


def _compute_output_dims(H, W, kH, kW, dH, dW, pH, pW, sH, sW):
    outH = (H + 2 * pH - (dH * (kH - 1) + 1)) // sH + 1
    outW = (W + 2 * pW - (dW * (kW - 1) + 1)) // sW + 1
    return outH, outW


def _launch_im2col_kernel(x, out, kH, kW, dH, dW, pH, pW, sH, sW):
    assert x.is_cuda and out.is_cuda, "Inputs must be CUDA tensors"
    x = x.contiguous()
    out = out.contiguous()

    N, C, H, W = x.shape
    outH, outW = _compute_output_dims(H, W, kH, kW, dH, dW, pH, pW, sH, sW)
    rows_total = C * kH * kW
    L = outH * outW

    if rows_total == 0 or L == 0 or N == 0:
        return  # Nothing to do

    BLOCK_M = 64
    BLOCK_N = 128

    num_row_tiles = triton.cdiv(rows_total, BLOCK_M)
    num_col_tiles = triton.cdiv(L, BLOCK_N)
    grid = (N * num_row_tiles, num_col_tiles)

    im2col_kernel[grid](
        x,
        out,
        N,
        C,
        H,
        W,
        kH,
        kW,
        dH,
        dW,
        pH,
        pW,
        sH,
        sW,
        outH,
        outW,
        rows_total,
        L,
        num_row_tiles,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2,
    )


def im2col(input, kernel_size, dilation=1, padding=0, stride=1):
    x = input
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.ndim != 4:
        raise ValueError("im2col expects input of shape (N, C, H, W) or (C, H, W)")
    kH, kW = _parse_2tuple(kernel_size, "kernel_size")
    dH, dW = _parse_2tuple(dilation, "dilation")
    pH, pW = _parse_2tuple(padding, "padding")
    sH, sW = _parse_2tuple(stride, "stride")

    N, C, H, W = x.shape
    outH, outW = _compute_output_dims(H, W, kH, kW, dH, dW, pH, pW, sH, sW)
    rows_total = C * kH * kW
    L = outH * outW

    out = torch.empty((N, rows_total, L), device=x.device, dtype=x.dtype)
    if L == 0 or rows_total == 0 or N == 0:
        return out if input.ndim == 4 else out.squeeze(0)

    _launch_im2col_kernel(x, out, kH, kW, dH, dW, pH, pW, sH, sW)
    return out if input.ndim == 4 else out.squeeze(0)


def im2col_out(input, kernel_size, dilation=1, padding=0, stride=1, out=None):
    x = input
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.ndim != 4:
        raise ValueError("im2col_out expects input of shape (N, C, H, W) or (C, H, W)")
    kH, kW = _parse_2tuple(kernel_size, "kernel_size")
    dH, dW = _parse_2tuple(dilation, "dilation")
    pH, pW = _parse_2tuple(padding, "padding")
    sH, sW = _parse_2tuple(stride, "stride")

    N, C, H, W = x.shape
    outH, outW = _compute_output_dims(H, W, kH, kW, dH, dW, pH, pW, sH, sW)
    rows_total = C * kH * kW
    L = outH * outW

    if out is None:
        out = torch.empty((N, rows_total, L), device=x.device, dtype=x.dtype)
    else:
        if out.ndim == 2 and N == 1:
            # Allow (C*kH*kW, L) for single batch for convenience
            expected = (rows_total, L)
        else:
            expected = (N, rows_total, L)
        if tuple(out.shape) != expected:
            raise ValueError(f"out has shape {tuple(out.shape)}, expected {expected}")
        if out.device != x.device or out.dtype != x.dtype:
            raise ValueError("out must have same device and dtype as input")

    if L == 0 or rows_total == 0 or N == 0:
        return out

    # If out was provided as 2D for N=1, make it 3D view for kernel, then restore
    squeeze_after = False
    if out.ndim == 2 and N == 1:
        out_3d = out.view(1, rows_total, L)
        squeeze_after = True
    else:
        out_3d = out

    _launch_im2col_kernel(x, out_3d, kH, kW, dH, dW, pH, pW, sH, sW)

    return out_3d.view(rows_total, L) if squeeze_after else out
