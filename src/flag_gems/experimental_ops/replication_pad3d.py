import torch
import triton
import triton.language as tl


@triton.jit
def replication_pad3d_kernel(
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
    pad_d_before,
    pad_h_before,
    pad_w_before,
    in_stride_n,
    in_stride_c,
    in_stride_d,
    in_stride_h,
    in_stride_w,
    out_stride_n,
    out_stride_c,
    out_stride_d,
    out_stride_h,
    out_stride_w,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Unravel linear indices into (n, c, d_out, h_out, w_out)
    w_out = offs % W_out
    tmp = offs // W_out
    h_out = tmp % H_out
    tmp = tmp // H_out
    d_out = tmp % D_out
    tmp = tmp // D_out
    c = tmp % C
    n = tmp // C

    # Compute clamped input indices
    w_in = w_out - pad_w_before
    w_in = tl.maximum(w_in, 0)
    w_in = tl.minimum(w_in, W_in - 1)

    h_in = h_out - pad_h_before
    h_in = tl.maximum(h_in, 0)
    h_in = tl.minimum(h_in, H_in - 1)

    d_in = d_out - pad_d_before
    d_in = tl.maximum(d_in, 0)
    d_in = tl.minimum(d_in, D_in - 1)

    # Compute input and output pointers (strided)
    in_offset = (
        n * in_stride_n
        + c * in_stride_c
        + d_in * in_stride_d
        + h_in * in_stride_h
        + w_in * in_stride_w
    )
    out_offset = (
        n * out_stride_n
        + c * out_stride_c
        + d_out * out_stride_d
        + h_out * out_stride_h
        + w_out * out_stride_w
    )

    vals = tl.load(in_ptr + in_offset, mask=mask, other=0)
    tl.store(out_ptr + out_offset, vals, mask=mask)


def _normalize_3d_pad(padding):
    if isinstance(padding, (list, tuple)) and len(padding) == 6:
        return tuple(int(x) for x in padding)
    raise ValueError(
        "padding must be a sequence of 6 integers: (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom, pad_d_front, pad_d_back)"
    )


def _get_5d_shape_and_strides(t: torch.Tensor):
    # Returns (N, C, D, H, W), (sN, sC, sD, sH, sW), and a flag indicating if original was 4D
    if t.dim() == 5:
        N, C, D, H, W = t.shape
        sN, sC, sD, sH, sW = t.stride()
        was_4d = False
        return (N, C, D, H, W), (sN, sC, sD, sH, sW), was_4d
    elif t.dim() == 4:
        C, D, H, W = t.shape
        sC, sD, sH, sW = t.stride()
        # Emulate leading N=1 dimension with stride 0 for indexing
        N = 1
        sN = 0
        was_4d = True
        return (N, C, D, H, W), (sN, sC, sD, sH, sW), was_4d
    else:
        raise ValueError("Input must be 4D (C, D, H, W) or 5D (N, C, D, H, W).")


def _launch_replication_pad3d_kernel(x: torch.Tensor, padding, out: torch.Tensor):
    assert x.is_cuda and out.is_cuda, "Tensors must be on CUDA device"
    assert x.dtype == out.dtype, "Input and output dtypes must match"
    assert x.device == out.device, "Input and output must be on the same device"
    assert x.is_contiguous(
        memory_format=torch.contiguous_format
    ), "Input must be contiguous"
    # Output can be non-contiguous; we handle via strides

    (
        pad_w_before,
        pad_w_after,
        pad_h_before,
        pad_h_after,
        pad_d_before,
        pad_d_after,
    ) = _normalize_3d_pad(padding)

    (
        (N, C, D_in, H_in, W_in),
        (in_sN, in_sC, in_sD, in_sH, in_sW),
        x_was_4d,
    ) = _get_5d_shape_and_strides(x)
    (
        (N_o, C_o, D_out, H_out, W_out),
        (out_sN, out_sC, out_sD, out_sH, out_sW),
        out_was_4d,
    ) = _get_5d_shape_and_strides(out)

    # Validate shapes
    assert N_o == N and C_o == C, "Output N and C must match input"
    expected_D_out = D_in + pad_d_before + pad_d_after
    expected_H_out = H_in + pad_h_before + pad_h_after
    expected_W_out = W_in + pad_w_before + pad_w_after
    assert (D_out, H_out, W_out) == (
        expected_D_out,
        expected_H_out,
        expected_W_out,
    ), "Output spatial shape mismatch"

    n_elements = out.numel()
    if n_elements == 0:
        return out

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    replication_pad3d_kernel[grid](
        x,
        out,
        N,
        C,
        D_in,
        H_in,
        W_in,
        D_out,
        H_out,
        W_out,
        pad_d_before,
        pad_h_before,
        pad_w_before,
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
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def replication_pad3d(input: torch.Tensor, padding):
    (
        pad_w_before,
        pad_w_after,
        pad_h_before,
        pad_h_after,
        pad_d_before,
        pad_d_after,
    ) = _normalize_3d_pad(padding)
    (N, C, D_in, H_in, W_in), _, was_4d = _get_5d_shape_and_strides(input)

    D_out = D_in + pad_d_before + pad_d_after
    H_out = H_in + pad_h_before + pad_h_after
    W_out = W_in + pad_w_before + pad_w_after

    if was_4d:
        out_shape = (C, D_out, H_out, W_out)
    else:
        out_shape = (N, C, D_out, H_out, W_out)

    out = torch.empty(out_shape, device=input.device, dtype=input.dtype)
    _launch_replication_pad3d_kernel(
        input,
        (
            pad_w_before,
            pad_w_after,
            pad_h_before,
            pad_h_after,
            pad_d_before,
            pad_d_after,
        ),
        out,
    )
    return out


def replication_pad3d_out(input: torch.Tensor, padding, out: torch.Tensor):
    (
        pad_w_before,
        pad_w_after,
        pad_h_before,
        pad_h_after,
        pad_d_before,
        pad_d_after,
    ) = _normalize_3d_pad(padding)
    (N, C, D_in, H_in, W_in), _, was_4d_in = _get_5d_shape_and_strides(input)

    D_out = D_in + pad_d_before + pad_d_after
    H_out = H_in + pad_h_before + pad_h_after
    W_out = W_in + pad_w_before + pad_w_after

    # Validate provided out shape
    if out.dim() == 5:
        expected_out_shape = (N, C, D_out, H_out, W_out)
    elif out.dim() == 4:
        expected_out_shape = (C, D_out, H_out, W_out)
    else:
        raise ValueError("out tensor must be 4D or 5D")
    assert (
        tuple(out.shape) == expected_out_shape
    ), f"out has incorrect shape, expected {expected_out_shape}, got {tuple(out.shape)}"

    _launch_replication_pad3d_kernel(
        input,
        (
            pad_w_before,
            pad_w_after,
            pad_h_before,
            pad_h_after,
            pad_d_before,
            pad_d_after,
        ),
        out,
    )
    return out
