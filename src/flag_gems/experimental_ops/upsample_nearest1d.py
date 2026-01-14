import math

import torch
import triton
import triton.language as tl


@triton.jit
def _upsample_nearest1d_kernel(
    in_ptr,
    out_ptr,
    N,
    C,
    W_IN,
    W_OUT,
    in_stride_n,
    in_stride_c,
    in_stride_w,
    out_stride_n,
    out_stride_c,
    out_stride_w,
    use_scale,
    inv_scale,
    BLOCK_W: tl.constexpr,
):
    pid_w = tl.program_id(0)  # along W_OUT
    pid_nc = tl.program_id(1)  # along N*C

    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    nc = pid_nc

    n = nc // C
    c = nc % C

    mask = (offs_w < W_OUT) & (n < N) & (c < C)

    # Compute source indices
    # Using integer math when output_size is provided: j = floor(offs_w * W_IN / W_OUT)
    j_from_output = tl.minimum((offs_w * W_IN) // W_OUT, W_IN - 1)

    # Using explicit scale factor when provided: j = floor(offs_w / scale) = floor(offs_w * inv_scale)
    j_from_scale = tl.minimum(
        (offs_w.to(tl.float32) * inv_scale).to(tl.int32), W_IN - 1
    )

    cond = use_scale != 0
    j = tl.where(cond, j_from_scale, j_from_output)

    base_in = n * in_stride_n + c * in_stride_c
    base_out = n * out_stride_n + c * out_stride_c

    in_idx = base_in + j * in_stride_w
    out_idx = base_out + offs_w * out_stride_w

    val = tl.load(in_ptr + in_idx, mask=mask, other=0)
    tl.store(out_ptr + out_idx, val, mask=mask)


def _upsample_nearest1d_impl(
    input: torch.Tensor, output_size=None, scales=None, out: torch.Tensor = None
):
    if not input.is_cuda:
        raise ValueError("Input tensor must be on CUDA device.")
    if input.dim() != 3:
        raise ValueError("upsample_nearest1d expects a 3D tensor of shape (N, C, W).")
    N, C, W_in = input.shape

    use_scale = False
    inv_scale = 0.0

    if output_size is not None:
        if not isinstance(output_size, (list, tuple)) or len(output_size) != 1:
            raise ValueError(
                "output_size must be a sequence of length 1 for 1D upsampling."
            )
        W_out = int(output_size[0])
    else:
        # derive from scales
        if scales is None:
            raise ValueError("Either output_size or scales must be provided.")
        if isinstance(scales, (list, tuple)):
            if len(scales) == 0 or scales[0] is None:
                raise ValueError("Invalid scales for 1D upsampling.")
            s = float(scales[0])
        else:
            s = float(scales)
        if s <= 0:
            raise ValueError("Scale factor must be positive.")
        W_out = int(math.floor(W_in * s))
        use_scale = True
        inv_scale = 1.0 / s

    if W_out <= 0:
        raise ValueError("Computed output width must be positive.")

    # Prepare output
    if out is None:
        out = torch.empty((N, C, W_out), device=input.device, dtype=input.dtype)
    else:
        if not out.is_cuda:
            raise ValueError("Output tensor must be on CUDA device.")
        if list(out.shape) != [N, C, W_out]:
            raise ValueError(
                f"Output tensor has incorrect shape, expected ({N}, {C}, {W_out})."
            )
        if out.dtype != input.dtype:
            raise ValueError("Output tensor must have the same dtype as input.")

    # Extract strides
    in_stride_n, in_stride_c, in_stride_w = input.stride()
    out_stride_n, out_stride_c, out_stride_w = out.stride()

    # Launch kernel
    BLOCK_W = 256
    grid = (triton.cdiv(W_out, BLOCK_W), N * C)
    _upsample_nearest1d_kernel[grid](
        input,
        out,
        N,
        C,
        W_in,
        W_out,
        in_stride_n,
        in_stride_c,
        in_stride_w,
        out_stride_n,
        out_stride_c,
        out_stride_w,
        int(use_scale),
        float(inv_scale),
        BLOCK_W=BLOCK_W,
    )
    return out


def upsample_nearest1d(input: torch.Tensor, output_size=None, scales=None):
    return _upsample_nearest1d_impl(
        input, output_size=output_size, scales=scales, out=None
    )


def upsample_nearest1d_vec(input: torch.Tensor, output_size=None, scales=None):
    # scales expected to be a sequence; pass through as-is
    return _upsample_nearest1d_impl(
        input, output_size=output_size, scales=scales, out=None
    )


def upsample_nearest1d_out(
    input: torch.Tensor, output_size=None, scales=None, *, out: torch.Tensor
):
    _upsample_nearest1d_impl(input, output_size=output_size, scales=scales, out=out)
    return out
