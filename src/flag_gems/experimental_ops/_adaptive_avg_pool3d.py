import torch
import triton
import triton.language as tl


@triton.jit
def adaptive_avg_pool3d_kernel(
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
    stride_in_n,
    stride_in_c,
    stride_in_d,
    stride_in_h,
    stride_in_w,
    stride_out_n,
    stride_out_c,
    stride_out_d,
    stride_out_h,
    stride_out_w,
):
    pid = tl.program_id(axis=0)

    # Unravel pid -> (n, c, d_o, h_o, w_o)
    W_out_i64 = tl.full((), W_out, tl.int64)
    H_out_i64 = tl.full((), H_out, tl.int64)
    D_out_i64 = tl.full((), D_out, tl.int64)
    C_i64 = tl.full((), C, tl.int64)

    idx = tl.cast(pid, tl.int64)
    w_o = idx % W_out_i64
    idx = idx // W_out_i64
    h_o = idx % H_out_i64
    idx = idx // H_out_i64
    d_o = idx % D_out_i64
    idx = idx // D_out_i64
    c = idx % C_i64
    n = idx // C_i64

    # Compute start/end indices for each dimension (integer arithmetic)
    D_in_i64 = tl.full((), D_in, tl.int64)
    H_in_i64 = tl.full((), H_in, tl.int64)
    W_in_i64 = tl.full((), W_in, tl.int64)

    d0 = (d_o * D_in_i64) // D_out_i64
    d1 = ((d_o + 1) * D_in_i64 + D_out_i64 - 1) // D_out_i64
    h0 = (h_o * H_in_i64) // H_out_i64
    h1 = ((h_o + 1) * H_in_i64 + H_out_i64 - 1) // H_out_i64
    w0 = (w_o * W_in_i64) // W_out_i64
    w1 = ((w_o + 1) * W_in_i64 + W_out_i64 - 1) // W_out_i64

    dd = d1 - d0
    hh = h1 - h0
    ww = w1 - w0
    denom = dd * hh * ww

    # Base offsets and strides (int64)
    stride_in_n_i64 = tl.full((), stride_in_n, tl.int64)
    stride_in_c_i64 = tl.full((), stride_in_c, tl.int64)
    stride_in_d_i64 = tl.full((), stride_in_d, tl.int64)
    stride_in_h_i64 = tl.full((), stride_in_h, tl.int64)
    stride_in_w_i64 = tl.full((), stride_in_w, tl.int64)

    stride_out_n_i64 = tl.full((), stride_out_n, tl.int64)
    stride_out_c_i64 = tl.full((), stride_out_c, tl.int64)
    stride_out_d_i64 = tl.full((), stride_out_d, tl.int64)
    stride_out_h_i64 = tl.full((), stride_out_h, tl.int64)
    stride_out_w_i64 = tl.full((), stride_out_w, tl.int64)

    base_nc = n * stride_in_n_i64 + c * stride_in_c_i64

    acc = tl.zeros((), dtype=tl.float32)

    di = d0
    while di < d1:
        hi = h0
        while hi < h1:
            wi = w0
            while wi < w1:
                in_idx = (
                    base_nc
                    + di * stride_in_d_i64
                    + hi * stride_in_h_i64
                    + wi * stride_in_w_i64
                )
                val = tl.load(in_ptr + in_idx)
                acc += tl.cast(val, tl.float32)
                wi += 1
            hi += 1
        di += 1

    denom_f = tl.cast(denom, tl.float32)
    out_val = acc / denom_f

    out_idx = (
        n * stride_out_n_i64
        + c * stride_out_c_i64
        + d_o * stride_out_d_i64
        + h_o * stride_out_h_i64
        + w_o * stride_out_w_i64
    )
    tl.store(out_ptr + out_idx, out_val)


def _normalize_output_size_3d(output_size):
    if isinstance(output_size, torch.Size):
        output_size = tuple(output_size)
    if isinstance(output_size, (list, tuple)):
        if len(output_size) != 3:
            raise ValueError(
                "output_size for _adaptive_avg_pool3d must have 3 elements (D_out, H_out, W_out)"
            )
        return tuple(int(x) for x in output_size)
    raise TypeError("output_size must be a sequence of three integers")


def _prepare_5d_input(t):
    if t.dim() == 5:
        return t, False
    if t.dim() == 4:
        return t.unsqueeze(0), True  # add N=1
    raise ValueError(
        "input for _adaptive_avg_pool3d must be 4D (C,D,H,W) or 5D (N,C,D,H,W)"
    )


def _launch_adaptive_avg_pool3d_kernel(x, out):
    assert x.is_cuda and out.is_cuda, "Tensors must be CUDA tensors"
    N, C, D_in, H_in, W_in = x.shape
    D_out, H_out, W_out = out.shape[-3], out.shape[-2], out.shape[-1]

    stride_in_n, stride_in_c, stride_in_d, stride_in_h, stride_in_w = x.stride()
    stride_out_n, stride_out_c, stride_out_d, stride_out_h, stride_out_w = out.stride()

    total = N * C * D_out * H_out * W_out
    if total == 0:
        return

    grid = (total,)
    adaptive_avg_pool3d_kernel[grid](
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
        stride_in_n,
        stride_in_c,
        stride_in_d,
        stride_in_h,
        stride_in_w,
        stride_out_n,
        stride_out_c,
        stride_out_d,
        stride_out_h,
        stride_out_w,
        num_warps=4,
    )


def _adaptive_avg_pool3d(input: torch.Tensor, output_size):
    x5d, squeezed = _prepare_5d_input(input)
    D_out, H_out, W_out = _normalize_output_size_3d(output_size)

    N, C, D_in, H_in, W_in = x5d.shape
    out_shape_5d = (N, C, D_out, H_out, W_out)
    out5d = torch.empty(
        out_shape_5d, device=x5d.device, dtype=x5d.dtype, layout=x5d.layout
    )

    _launch_adaptive_avg_pool3d_kernel(x5d, out5d)

    if squeezed:
        return out5d.squeeze(0)
    return out5d


def _adaptive_avg_pool3d_out(input: torch.Tensor, output_size, out: torch.Tensor):
    x5d, squeezed = _prepare_5d_input(input)
    D_out, H_out, W_out = _normalize_output_size_3d(output_size)

    # Prepare out to be 5D if needed
    if squeezed:
        if out.dim() == 4:
            out5d = out.unsqueeze(0)
        elif out.dim() == 5 and out.size(0) == 1:
            out5d = out
        else:
            raise ValueError("Provided 'out' must be 4D (C,D,H,W) when input is 4D")
    else:
        out5d = out
        if out5d.dim() != 5:
            raise ValueError("Provided 'out' must be 5D (N,C,D,H,W) when input is 5D")

    # Validate shape
    expected_shape = (x5d.size(0), x5d.size(1), D_out, H_out, W_out)
    if tuple(out5d.shape) != expected_shape:
        raise ValueError(
            f"out has incorrect shape. Expected {expected_shape}, got {tuple(out5d.shape)}"
        )

    if out5d.device != x5d.device or out5d.dtype != x5d.dtype:
        raise ValueError(
            "out must be on the same device and have the same dtype as input"
        )

    _launch_adaptive_avg_pool3d_kernel(x5d, out5d)

    return out
