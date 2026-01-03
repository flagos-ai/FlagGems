import torch
import triton
import triton.language as tl


@triton.jit
def pixel_unshuffle_kernel(
    in_ptr,            # *Pointer* to input tensor (contiguous NCHW)
    out_ptr,           # *Pointer* to output tensor (contiguous NCHW)
    n_elements,        # total number of elements (N*C*H*W)
    N, C, H, W,        # input dimensions
    R,                 # downscale factor
    C_out, H_out, W_out,  # output dimensions
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Strides for contiguous NCHW
    sN_in = C * H * W
    sC_in = H * W
    sH_in = W
    sW_in = 1

    sN_out = C_out * H_out * W_out
    sC_out = H_out * W_out
    sH_out = W_out
    sW_out = 1

    # Decode output linear index into (n, c_out, h_out, w_out)
    n = offsets // sN_out
    rem1 = offsets - n * sN_out
    c_out = rem1 // sC_out
    rem2 = rem1 - c_out * sC_out
    h_out = rem2 // sH_out
    w_out = rem2 - h_out * sH_out

    # Map output channel to input channel and spatial offsets
    r2 = R * R
    c_in = c_out // r2
    remc = c_out - c_in * r2
    dh = remc // R
    dw = remc - dh * R

    # Compute input spatial indices
    h_in = h_out * R + dh
    w_in = w_out * R + dw

    # Compute input linear index
    in_index = n * sN_in + c_in * sC_in + h_in * sH_in + w_in * sW_in

    x = tl.load(in_ptr + in_index, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)


def _launch_pixel_unshuffle_kernel(inp: torch.Tensor, downscale_factor: int, out: torch.Tensor):
    assert inp.is_cuda and out.is_cuda, "Input and output must be CUDA tensors"
    assert inp.is_contiguous(), "Input must be contiguous (NCHW)"
    assert out.is_contiguous(), "Output must be contiguous (NCHW)"
    assert inp.ndim == 4, "Input must be a 4D tensor (N, C, H, W)"
    N, C, H, W = inp.shape
    r = int(downscale_factor)
    assert r > 0, "downscale_factor must be > 0"
    assert (H % r == 0) and (W % r == 0), "H and W must be divisible by downscale_factor"
    C_out = C * r * r
    H_out = H // r
    W_out = W // r
    assert out.shape == (N, C_out, H_out, W_out), "Output has incorrect shape"

    n_elements = inp.numel()
    if n_elements == 0:
        return

    BLOCK_SIZE = 1024
    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
    pixel_unshuffle_kernel[grid](
        inp,
        out,
        n_elements,
        N, C, H, W,
        r,
        C_out, H_out, W_out,
        BLOCK_SIZE=BLOCK_SIZE,
    )


def pixel_unshuffle(input, downscale_factor, *, layout=None):
    """
    Wrapper for aten::pixel_unshuffle
    Args:
      input: Tensor[N, C, H, W] (contiguous)
      downscale_factor: int
      layout: unused (for API parity)
    """
    x = input
    if not x.is_contiguous():
        x = x.contiguous()
    assert x.ndim == 4, "Input must be a 4D tensor (N, C, H, W)"
    N, C, H, W = x.shape
    r = int(downscale_factor)
    assert r > 0, "downscale_factor must be > 0"
    assert (H % r == 0) and (W % r == 0), "H and W must be divisible by downscale_factor"

    out_shape = (N, C * r * r, H // r, W // r)
    out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
    _launch_pixel_unshuffle_kernel(x, r, out)
    return out


def pixel_unshuffle_out(input, downscale_factor, out):
    """
    Wrapper for aten::pixel_unshuffle.out
    Args:
      input: Tensor[N, C, H, W] (contiguous)
      downscale_factor: int
      out: preallocated Tensor[N, C*r*r, H//r, W//r] (contiguous)
    """
    x = input
    if not x.is_contiguous():
        x = x.contiguous()
    assert x.ndim == 4, "Input must be a 4D tensor (N, C, H, W)"
    N, C, H, W = x.shape
    r = int(downscale_factor)
    assert r > 0, "downscale_factor must be > 0"
    assert (H % r == 0) and (W % r == 0), "H and W must be divisible by downscale_factor"
    expected_shape = (N, C * r * r, H // r, W // r)
    assert out.shape == expected_shape, f"out must have shape {expected_shape}"
    assert out.dtype == x.dtype, "out dtype must match input dtype"
    assert out.device == x.device, "out device must match input device"
    if not out.is_contiguous():
        raise ValueError("out must be contiguous")

    _launch_pixel_unshuffle_kernel(x, r, out)
    return out