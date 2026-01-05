import torch
import triton
import triton.language as tl


@triton.jit
def adaptive_avg_pool2d_kernel(
    x_ptr,  # *Pointer* to input tensor (contiguous NCHW)
    out_ptr,  # *Pointer* to output tensor (contiguous NCHW)
    N,
    C,
    H_in,
    W_in,
    H_out,
    W_out,
    OUT_DTYPE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    # Unravel pid (flattened index over N*C*H_out*W_out) into (n, c, oh, ow)
    ow = pid % W_out
    tmp = pid // W_out
    oh = tmp % H_out
    tmp = tmp // H_out
    c = tmp % C
    n = tmp // C

    # Compute input pooling window [start_h, end_h) x [start_w, end_w)
    start_h = (oh * H_in) // H_out
    end_h = ((oh + 1) * H_in + H_out - 1) // H_out
    start_w = (ow * W_in) // W_out
    end_w = ((ow + 1) * W_in + W_out - 1) // W_out

    # Base offset for (n, c, 0, 0)
    base_nc = (n * C + c) * H_in * W_in

    # Accumulate sum in fp32 for numerical stability
    acc = 0.0
    h = start_h
    while h < end_h:
        row_base = base_nc + h * W_in
        w = start_w
        while w < end_w:
            idx = row_base + w
            val = tl.load(x_ptr + idx)
            acc += tl.cast(val, tl.float32)
            w += 1
        h += 1

    pool_h = end_h - start_h
    pool_w = end_w - start_w
    denom = pool_h * pool_w
    denom_f = tl.cast(denom, tl.float32)
    avg = acc / denom_f

    # Store result at the corresponding output index; out is also flattened contiguous
    out_val = tl.cast(avg, OUT_DTYPE)
    tl.store(out_ptr + pid, out_val)


def _get_out_hw_from_output_size(output_size, h_in, w_in):
    # Allow various forms: None, int, (int,), (int, int), torch.Size, list
    if output_size is None:
        return h_in, w_in
    if isinstance(output_size, (int,)):
        return int(output_size), int(output_size)
    if isinstance(output_size, (list, tuple, torch.Size)):
        if len(output_size) == 0:
            return h_in, w_in
        if len(output_size) == 1:
            return int(output_size[0]), int(output_size[0])
        return int(output_size[0]), int(output_size[1])
    # Fallback
    return h_in, w_in


def _torch_to_triton_dtype(dtype: torch.dtype):
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.bfloat16:
        return tl.bfloat16
    if dtype == torch.float32:
        return tl.float32
    raise TypeError(
        f"Unsupported dtype for _adaptive_avg_pool2d Triton kernel: {dtype}"
    )


def _launch_adaptive_avg_pool2d_kernel(x4d: torch.Tensor, out4d: torch.Tensor):
    # x4d, out4d should be contiguous NCHW
    N, C, H_in, W_in = x4d.shape
    H_out, W_out = out4d.shape[-2], out4d.shape[-1]
    total = N * C * H_out * W_out

    grid = (total,)
    OUT_DTYPE = _torch_to_triton_dtype(out4d.dtype)

    adaptive_avg_pool2d_kernel[grid](
        x4d,
        out4d,
        N,
        C,
        H_in,
        W_in,
        H_out,
        W_out,
        OUT_DTYPE=OUT_DTYPE,
    )


def _adaptive_avg_pool2d(input: torch.Tensor, output_size):
    # Supports input shapes [N, C, H, W] or [C, H, W]
    assert input.is_cuda, "Input must be on CUDA device for Triton kernel."
    orig_3d = False
    if input.dim() == 3:
        orig_3d = True
        x = input.unsqueeze(0)
    elif input.dim() == 4:
        x = input
    else:
        raise ValueError(
            f"_adaptive_avg_pool2d expects 3D or 4D input, got {input.dim()}D"
        )

    N, C, H_in, W_in = x.shape
    H_out, W_out = _get_out_hw_from_output_size(output_size, H_in, W_in)

    # Allocate output
    out = torch.empty((N, C, H_out, W_out), device=x.device, dtype=input.dtype)

    # Ensure contiguous
    x_contig = x.contiguous()
    out_contig = out.contiguous()

    _launch_adaptive_avg_pool2d_kernel(x_contig, out_contig)

    if out_contig.data_ptr() != out.data_ptr():
        out.copy_(out_contig)

    if orig_3d:
        return out.squeeze(0)
    return out


def _adaptive_avg_pool2d_out(input: torch.Tensor, output_size, out: torch.Tensor):
    # Writes result into provided 'out' tensor; supports 3D and 4D cases
    assert input.is_cuda, "Input must be on CUDA device for Triton kernel."
    assert out.is_cuda, "Out tensor must be on CUDA device for Triton kernel."
    if input.dim() == 3:
        x = input.unsqueeze(0)
        # Determine out shape
        C = input.shape[0]
        H_in, W_in = input.shape[-2], input.shape[-1]
        H_out, W_out = _get_out_hw_from_output_size(output_size, H_in, W_in)
        expected_out_shape_4d = (1, C, H_out, W_out)
        # Prepare out as 4D
        if out.dim() == 3:
            # [C, H_out, W_out]
            if out.shape != (C, H_out, W_out):
                raise ValueError(
                    f"out has incorrect shape {out.shape}, expected {(C, H_out, W_out)}"
                )
            out4d = out.unsqueeze(0)
        elif out.dim() == 4:
            if out.shape != expected_out_shape_4d:
                raise ValueError(
                    f"out has incorrect shape {out.shape}, expected {expected_out_shape_4d}"
                )
            out4d = out
        else:
            raise ValueError(f"out must be 3D or 4D when input is 3D, got {out.dim()}D")
    elif input.dim() == 4:
        x = input
        N, C, H_in, W_in = x.shape
        H_out, W_out = _get_out_hw_from_output_size(output_size, H_in, W_in)
        expected_out_shape_4d = (N, C, H_out, W_out)
        if out.dim() != 4 or tuple(out.shape) != expected_out_shape_4d:
            raise ValueError(
                f"out has incorrect shape {tuple(out.shape)}, expected {expected_out_shape_4d}"
            )
        out4d = out
    else:
        raise ValueError(
            f"_adaptive_avg_pool2d_out expects 3D or 4D input, got {input.dim()}D"
        )

    if out4d.dtype != input.dtype:
        raise TypeError(f"out dtype {out4d.dtype} must match input dtype {input.dtype}")
    if input.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(
            f"Unsupported dtype for _adaptive_avg_pool2d Triton kernel: {input.dtype}"
        )

    x_contig = x.contiguous()
    out4d_contig = out4d.contiguous()

    _launch_adaptive_avg_pool2d_kernel(x_contig, out4d_contig)

    if out4d_contig.data_ptr() != out4d.data_ptr():
        out4d.copy_(out4d_contig)

    return out4d if input.dim() == 4 else out4d.squeeze(0)
