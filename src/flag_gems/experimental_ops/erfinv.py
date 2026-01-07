import torch
import triton
import triton.language as tl


@triton.jit
def erfinv_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    one = 1.0
    absx = tl.abs(xf)
    w = -tl.log((one - xf) * (one + xf))

    use_low = w < 5.0

    wl = w - 2.5
    pl = 2.81022636e-08
    pl = 3.43273939e-07 + pl * wl
    pl = -3.5233877e-06 + pl * wl
    pl = -4.39150654e-06 + pl * wl
    pl = 2.1858087e-04 + pl * wl
    pl = -1.25372503e-03 + pl * wl
    pl = -4.17768164e-03 + pl * wl
    pl = 2.46640727e-01 + pl * wl
    pl = 1.50140941e00 + pl * wl

    wh = tl.sqrt(w) - 3.0
    ph = -2.00214257e-04
    ph = 1.00950558e-04 + ph * wh
    ph = 1.34934322e-03 + ph * wh
    ph = -3.67342844e-03 + ph * wh
    ph = 5.73950773e-03 + ph * wh
    ph = -7.62246130e-03 + ph * wh
    ph = 9.43887047e-03 + ph * wh
    ph = 1.00167406e00 + ph * wh
    ph = 2.83297682e00 + ph * wh

    p = tl.where(use_low, pl, ph)
    res = p * xf

    nan_vec = tl.full([BLOCK_SIZE], float("nan"), dtype=tl.float32)
    inf_vec = tl.full([BLOCK_SIZE], float("inf"), dtype=tl.float32)

    mask_nan = xf != xf
    mask_oob = absx > 1.0
    mask_pos1 = xf == 1.0
    mask_neg1 = xf == -1.0

    res = tl.where(mask_nan, nan_vec, res)
    res = tl.where(mask_oob, nan_vec, res)
    res = tl.where(mask_pos1, inf_vec, res)
    res = tl.where(mask_neg1, -inf_vec, res)

    y = res.to(x.dtype)
    tl.store(out_ptr + offsets, y, mask=mask)


def _launch_erfinv_kernel(x: torch.Tensor, out: torch.Tensor):
    assert x.is_cuda and out.is_cuda, "Inputs must be CUDA tensors"
    assert (
        x.numel() == out.numel()
    ), "Input and output must have the same number of elements"
    assert x.dtype == out.dtype, "Input and output must have the same dtype"
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    erfinv_kernel[grid](
        x,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )


def erfinv(x: torch.Tensor):
    x_in = x
    if not x_in.is_contiguous():
        x_in = x_in.contiguous()
    out = torch.empty_like(x_in)
    _launch_erfinv_kernel(x_in, out)
    # Match original shape/strides of input if needed
    if out.shape != x.shape or out.stride() != x.stride():
        out = out.reshape(x.shape).as_strided(x.size(), x.stride())
    return out


def erfinv_out(x: torch.Tensor, out: torch.Tensor):
    # Resize out to match input shape if necessary
    if out.shape != x.shape:
        out.resize_(x.shape)
    # Ensure dtype matches input dtype for aten out semantics
    assert out.dtype == x.dtype, "out tensor must have the same dtype as input"
    x_in = x if x.is_contiguous() else x.contiguous()
    if out.is_contiguous():
        _launch_erfinv_kernel(x_in, out)
        return out
    else:
        tmp = torch.empty_like(out, memory_format=torch.contiguous_format)
        _launch_erfinv_kernel(x_in, tmp)
        out.copy_(tmp)
        return out
