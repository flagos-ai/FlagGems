import torch
import triton
import triton.language as tl


@triton.jit
def _reflection_pad2d_kernel(
    in_ptr,
    out_ptr,
    N,
    C,
    H_in,
    W_in,
    H_out,
    W_out,
    pad_left,
    pad_top,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    offs = offs.to(tl.int64)

    W_out = tl.full([], W_out, tl.int64)
    H_out = tl.full([], H_out, tl.int64)
    C = tl.full([], C, tl.int64)
    W_in = tl.full([], W_in, tl.int64)
    H_in = tl.full([], H_in, tl.int64)
    pad_left = tl.full([], pad_left, tl.int64)
    pad_top = tl.full([], pad_top, tl.int64)

    w_out = offs % W_out
    tmp = offs // W_out
    h_out = tmp % H_out
    tmp = tmp // H_out
    c = tmp % C
    n = tmp // C

    w_in = w_out - pad_left
    h_in = h_out - pad_top

    # reflection index for width
    m_w = W_in - 1
    period_w = m_w * 2
    abs_w = tl.abs(w_in)
    r_w = tl.where(period_w != 0, abs_w % period_w, 0)
    w_idx = tl.where(r_w <= m_w, r_w, period_w - r_w)

    # reflection index for height
    m_h = H_in - 1
    period_h = m_h * 2
    abs_h = tl.abs(h_in)
    r_h = tl.where(period_h != 0, abs_h % period_h, 0)
    h_idx = tl.where(r_h <= m_h, r_h, period_h - r_h)

    in_index = (((n * C + c) * H_in) + h_idx) * W_in + w_idx

    x = tl.load(in_ptr + in_index, mask=mask)
    tl.store(out_ptr + offs, x, mask=mask)


def _parse_padding(padding):
    if not isinstance(padding, (list, tuple)) or len(padding) != 4:
        raise ValueError(
            "padding must be a sequence of four integers: (pad_left, pad_right, pad_top, pad_bottom)"
        )
    pad_left, pad_right, pad_top, pad_bottom = map(int, padding)
    if pad_left < 0 or pad_right < 0 or pad_top < 0 or pad_bottom < 0:
        raise ValueError("reflection_pad2d does not support negative padding")
    return pad_left, pad_right, pad_top, pad_bottom


def _launch_reflection_pad2d_kernel(inp, out, padding):
    assert inp.is_cuda and out.is_cuda, "reflection_pad2d requires CUDA tensors"
    assert inp.dtype == out.dtype, "input and output dtype must match"
    assert inp.is_contiguous(), "input must be contiguous"
    assert out.is_contiguous(), "output must be contiguous"

    pad_left, pad_right, pad_top, pad_bottom = _parse_padding(padding)

    if inp.dim() == 4:
        N, C, H_in, W_in = inp.shape
        H_out = H_in + pad_top + pad_bottom
        W_out = W_in + pad_left + pad_right
        expected_out_shape = (N, C, H_out, W_out)
    elif inp.dim() == 3:
        C, H_in, W_in = inp.shape
        N = 1
        H_out = H_in + pad_top + pad_bottom
        W_out = W_in + pad_left + pad_right
        expected_out_shape = (C, H_out, W_out)
    else:
        raise ValueError("reflection_pad2d expects 3D (C,H,W) or 4D (N,C,H,W) input")

    # Validate output shape
    if tuple(out.shape) != expected_out_shape:
        raise ValueError(
            f"output has incorrect shape, expected {expected_out_shape}, got {tuple(out.shape)}"
        )

    # PyTorch constraints for reflection padding
    if H_in <= 0 or W_in <= 0:
        raise ValueError("input spatial dimensions must be positive")
    if (pad_left >= W_in) or (pad_right >= W_in):
        raise ValueError(
            "Padding size should be less than the corresponding input dimension (width)."
        )
    if (pad_top >= H_in) or (pad_bottom >= H_in):
        raise ValueError(
            "Padding size should be less than the corresponding input dimension (height)."
        )
    if (H_in == 1 and (pad_top > 0 or pad_bottom > 0)) or (
        W_in == 1 and (pad_left > 0 or pad_right > 0)
    ):
        raise ValueError(
            "For reflection padding, input size must be at least 2 in each dimension being padded."
        )

    # Prepare parameters for kernel
    if inp.dim() == 3:
        N = 1
        C = inp.shape[0]

    n_elements = out.numel()

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _reflection_pad2d_kernel[grid](
        inp,
        out,
        N,
        C,
        H_in,
        W_in,
        H_out,
        W_out,
        pad_left,
        pad_top,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )


def reflection_pad2d(input: torch.Tensor, padding):
    pad_left, pad_right, pad_top, pad_bottom = _parse_padding(padding)

    if input.dim() == 4:
        N, C, H_in, W_in = input.shape
        H_out = H_in + pad_top + pad_bottom
        W_out = W_in + pad_left + pad_right
        out = torch.empty((N, C, H_out, W_out), device=input.device, dtype=input.dtype)
    elif input.dim() == 3:
        C, H_in, W_in = input.shape
        H_out = H_in + pad_top + pad_bottom
        W_out = W_in + pad_left + pad_right
        out = torch.empty((C, H_out, W_out), device=input.device, dtype=input.dtype)
    else:
        raise ValueError("reflection_pad2d expects 3D (C,H,W) or 4D (N,C,H,W) input")

    _launch_reflection_pad2d_kernel(
        input.contiguous(), out, (pad_left, pad_right, pad_top, pad_bottom)
    )
    return out


def reflection_pad2d_out(input: torch.Tensor, padding, out: torch.Tensor):
    pad_left, pad_right, pad_top, pad_bottom = _parse_padding(padding)

    if input.dim() == 4:
        N, C, H_in, W_in = input.shape
        H_out = H_in + pad_top + pad_bottom
        W_out = W_in + pad_left + pad_right
        expected_shape = (N, C, H_out, W_out)
    elif input.dim() == 3:
        C, H_in, W_in = input.shape
        H_out = H_in + pad_top + pad_bottom
        W_out = W_in + pad_left + pad_right
        expected_shape = (C, H_out, W_out)
    else:
        raise ValueError(
            "reflection_pad2d_out expects 3D (C,H,W) or 4D (N,C,H,W) input"
        )

    if tuple(out.shape) != expected_shape:
        raise ValueError(
            f"out tensor has incorrect shape, expected {expected_shape}, got {tuple(out.shape)}"
        )
    if out.device != input.device:
        raise ValueError("input and out must be on the same device")
    if out.dtype != input.dtype:
        raise ValueError("input and out must have the same dtype")

    _launch_reflection_pad2d_kernel(
        input.contiguous(), out, (pad_left, pad_right, pad_top, pad_bottom)
    )
    return out
