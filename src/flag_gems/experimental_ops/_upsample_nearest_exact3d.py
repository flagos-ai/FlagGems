import math

import torch
import triton
import triton.language as tl


@triton.jit
def _upsample_nearest_exact3d_kernel(
    in_ptr,
    out_ptr,
    N,
    C,
    D_in,
    H_in,
    W_in,
    D_out,
    H_out,
    W_out,
    in_sN,
    in_sC,
    in_sD,
    in_sH,
    in_sW,
    out_sN,
    out_sC,
    out_sD,
    out_sH,
    out_sW,
    scale_d,
    scale_h,
    scale_w,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    offs_i64 = offs.to(tl.int64)

    W_out_i64 = tl.full([1], W_out, dtype=tl.int64)
    H_out_i64 = tl.full([1], H_out, dtype=tl.int64)
    D_out_i64 = tl.full([1], D_out, dtype=tl.int64)
    C_i64 = tl.full([1], C, dtype=tl.int64)

    w = offs_i64 % W_out_i64
    tmp = offs_i64 // W_out_i64
    h = tmp % H_out_i64
    tmp = tmp // H_out_i64
    d = tmp % D_out_i64
    tmp = tmp // D_out_i64
    c = tmp % C_i64
    n = tmp // C_i64

    # Compute source indices with "nearest-exact" rule:
    # src = round(((out_idx + 0.5) * scale) - 0.5), then clamped to [0, in_size-1]
    d_f = d.to(tl.float32)
    h_f = h.to(tl.float32)
    w_f = w.to(tl.float32)

    src_d_f = (d_f + 0.5) * scale_d - 0.5
    src_h_f = (h_f + 0.5) * scale_h - 0.5
    src_w_f = (w_f + 0.5) * scale_w - 0.5

    src_d = tl.floor(src_d_f + 0.5).to(tl.int64)
    src_h = tl.floor(src_h_f + 0.5).to(tl.int64)
    src_w = tl.floor(src_w_f + 0.5).to(tl.int64)

    zero_i64 = tl.full([1], 0, dtype=tl.int64)
    D_in_i64 = tl.full([1], D_in, dtype=tl.int64)
    H_in_i64 = tl.full([1], H_in, dtype=tl.int64)
    W_in_i64 = tl.full([1], W_in, dtype=tl.int64)

    src_d = tl.maximum(zero_i64, tl.minimum(src_d, D_in_i64 - 1))
    src_h = tl.maximum(zero_i64, tl.minimum(src_h, H_in_i64 - 1))
    src_w = tl.maximum(zero_i64, tl.minimum(src_w, W_in_i64 - 1))

    in_sN_i64 = tl.full([1], in_sN, dtype=tl.int64)
    in_sC_i64 = tl.full([1], in_sC, dtype=tl.int64)
    in_sD_i64 = tl.full([1], in_sD, dtype=tl.int64)
    in_sH_i64 = tl.full([1], in_sH, dtype=tl.int64)
    in_sW_i64 = tl.full([1], in_sW, dtype=tl.int64)

    out_sN_i64 = tl.full([1], out_sN, dtype=tl.int64)
    out_sC_i64 = tl.full([1], out_sC, dtype=tl.int64)
    out_sD_i64 = tl.full([1], out_sD, dtype=tl.int64)
    out_sH_i64 = tl.full([1], out_sH, dtype=tl.int64)
    out_sW_i64 = tl.full([1], out_sW, dtype=tl.int64)

    in_offset = (
        n * in_sN_i64
        + c * in_sC_i64
        + src_d * in_sD_i64
        + src_h * in_sH_i64
        + src_w * in_sW_i64
    )
    out_offset = (
        n * out_sN_i64
        + c * out_sC_i64
        + d * out_sD_i64
        + h * out_sH_i64
        + w * out_sW_i64
    )

    val = tl.load(in_ptr + in_offset, mask=mask, other=0)
    tl.store(out_ptr + out_offset, val, mask=mask)


def _compute_output_size_and_scales(
    input,
    output_size=None,
    scales_d=None,
    scales_h=None,
    scales_w=None,
    scales_vec=None,
):
    assert input.dim() == 5, "Input must be a 5D tensor (N, C, D, H, W)"
    N, C, D_in, H_in, W_in = input.shape

    # Determine output size
    if output_size is not None:
        assert (
            len(output_size) == 3
        ), "output_size must be a sequence of length 3 for (D_out, H_out, W_out)"
        D_out, H_out, W_out = (
            int(output_size[0]),
            int(output_size[1]),
            int(output_size[2]),
        )
    else:
        if scales_vec is not None:
            assert (
                len(scales_vec) == 3
            ), "scales_vec must have length 3 [scale_d, scale_h, scale_w]"
            s_d, s_h, s_w = (
                float(scales_vec[0]),
                float(scales_vec[1]),
                float(scales_vec[2]),
            )
        else:
            s_d = None if scales_d is None else float(scales_d)
            s_h = None if scales_h is None else float(scales_h)
            s_w = None if scales_w is None else float(scales_w)
        assert (
            s_d is not None and s_h is not None and s_w is not None
        ), "Either output_size or all scale factors must be provided"
        D_out = max(1, int(math.floor(D_in * s_d)))
        H_out = max(1, int(math.floor(H_in * s_h)))
        W_out = max(1, int(math.floor(W_in * s_w)))

    # Determine mapping scales for nearest-exact:
    # scale_map = D_in / D_out if output_size provided, else 1.0 / scale_factor
    if output_size is not None:
        scale_d_map = float(D_in) / float(D_out) if D_out > 0 else 0.0
        scale_h_map = float(H_in) / float(H_out) if H_out > 0 else 0.0
        scale_w_map = float(W_in) / float(W_out) if W_out > 0 else 0.0
    else:
        if scales_vec is not None:
            s_d, s_h, s_w = (
                float(scales_vec[0]),
                float(scales_vec[1]),
                float(scales_vec[2]),
            )
        else:
            s_d = float(scales_d)
            s_h = float(scales_h)
            s_w = float(scales_w)
        assert s_d > 0 and s_h > 0 and s_w > 0, "scale factors must be positive"
        scale_d_map = 1.0 / s_d
        scale_h_map = 1.0 / s_h
        scale_w_map = 1.0 / s_w

    return (
        N,
        C,
        D_in,
        H_in,
        W_in,
        D_out,
        H_out,
        W_out,
        scale_d_map,
        scale_h_map,
        scale_w_map,
    )


def _launch_upsample_nearest_exact3d(
    input, output, D_out, H_out, W_out, scale_d_map, scale_h_map, scale_w_map
):
    assert input.is_cuda and output.is_cuda, "Tensors must be on CUDA device"
    assert input.dtype == output.dtype, "Input and output dtypes must match"

    N, C, D_in, H_in, W_in = input.shape

    in_strides = input.stride()
    out_strides = output.stride()

    in_sN, in_sC, in_sD, in_sH, in_sW = [int(s) for s in in_strides]
    out_sN, out_sC, out_sD, out_sH, out_sW = [int(s) for s in out_strides]

    n_elements = int(N) * int(C) * int(D_out) * int(H_out) * int(W_out)

    if n_elements == 0:
        return

    BLOCK_SIZE = 2048
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    _upsample_nearest_exact3d_kernel[grid](
        input,
        output,
        int(N),
        int(C),
        int(D_in),
        int(H_in),
        int(W_in),
        int(D_out),
        int(H_out),
        int(W_out),
        in_sN,
        in_sC,
        in_sD,
        in_sH,
        in_sW,
        out_sN,
        out_sC,
        out_sD,
        out_sH,
        out_sW,
        float(scale_d_map),
        float(scale_h_map),
        float(scale_w_map),
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )


def _upsample_nearest_exact3d(
    input, output_size=None, scales_d=None, scales_h=None, scales_w=None
):
    (
        N,
        C,
        D_in,
        H_in,
        W_in,
        D_out,
        H_out,
        W_out,
        sd,
        sh,
        sw,
    ) = _compute_output_size_and_scales(
        input,
        output_size=output_size,
        scales_d=scales_d,
        scales_h=scales_h,
        scales_w=scales_w,
        scales_vec=None,
    )
    out = torch.empty(
        (N, C, D_out, H_out, W_out), device=input.device, dtype=input.dtype
    )
    _launch_upsample_nearest_exact3d(input, out, D_out, H_out, W_out, sd, sh, sw)
    return out


def _upsample_nearest_exact3d_out(
    input, output_size=None, scales_d=None, scales_h=None, scales_w=None, out=None
):
    (
        N,
        C,
        D_in,
        H_in,
        W_in,
        D_out,
        H_out,
        W_out,
        sd,
        sh,
        sw,
    ) = _compute_output_size_and_scales(
        input,
        output_size=output_size,
        scales_d=scales_d,
        scales_h=scales_h,
        scales_w=scales_w,
        scales_vec=None,
    )
    assert out is not None, "out tensor must be provided"
    assert out.shape == (
        N,
        C,
        D_out,
        H_out,
        W_out,
    ), f"out tensor has wrong shape, expected {(N, C, D_out, H_out, W_out)}, got {tuple(out.shape)}"
    _launch_upsample_nearest_exact3d(input, out, D_out, H_out, W_out, sd, sh, sw)
    return out


def _upsample_nearest_exact3d_vec(
    input, output_size=None, scales=None, scale_factors=None
):
    # Accept either 'scales' or 'scale_factors' as a 3-element sequence
    scales_vec = scales if scales is not None else scale_factors
    (
        N,
        C,
        D_in,
        H_in,
        W_in,
        D_out,
        H_out,
        W_out,
        sd,
        sh,
        sw,
    ) = _compute_output_size_and_scales(
        input,
        output_size=output_size,
        scales_d=None,
        scales_h=None,
        scales_w=None,
        scales_vec=scales_vec,
    )
    out = torch.empty(
        (N, C, D_out, H_out, W_out), device=input.device, dtype=input.dtype
    )
    _launch_upsample_nearest_exact3d(input, out, D_out, H_out, W_out, sd, sh, sw)
    return out
