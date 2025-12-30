import torch
import triton
import triton.language as tl


@triton.jit
def special_i1_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)
    ax = tl.abs(x_f32)

    # Small region: |x| <= 3.75
    y = x_f32 / 3.75
    y2 = y * y
    # Horner polynomial for small |x|
    p = 0.00032411
    p = 0.00301532 + y2 * p
    p = 0.02658733 + y2 * p
    p = 0.15084934 + y2 * p
    p = 0.51498869 + y2 * p
    p = 0.87890594 + y2 * p
    p = 0.5 + y2 * p
    ans_small = x_f32 * p

    # Large region: |x| > 3.75
    # Use asymptotic expansion: I1(x) ~ exp(|x|)/sqrt(|x|) * poly(3.75/|x|)
    # Coefficients from Cephes
    t = 3.75 / tl.maximum(ax, 1e-20)
    q = -0.00420059
    q = 0.01787654 + t * q
    q = -0.02895312 + t * q
    q = 0.02282967 + t * q
    q = -0.01031555 + t * q
    q = 0.00163801 + t * q
    q = -0.00362018 + t * q
    q = -0.03988024 + t * q
    q = 0.39894228 + t * q
    pref = tl.exp(ax) / tl.sqrt(tl.maximum(ax, 1e-20))
    ans_large = pref * q
    # I1 is odd
    ans_large = tl.where(x_f32 < 0, -ans_large, ans_large)

    is_small = ax <= 3.75
    ans = tl.where(is_small, ans_small, ans_large)

    # Cast back to input dtype and store
    tl.store(out_ptr + offsets, ans.to(x.dtype), mask=mask)


def _launch_special_i1(x: torch.Tensor, out: torch.Tensor):
    assert x.is_cuda and out.is_cuda, "Tensors must be CUDA tensors"
    assert x.numel() == out.numel(), "Input and output must have the same number of elements"
    assert x.dtype == out.dtype, "Input and output must have the same dtype"

    n_elements = x.numel()
    if n_elements == 0:
        return

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    special_i1_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)


def special_i1(self: torch.Tensor):
    x = self
    x_c = x.contiguous()
    out = torch.empty_like(x_c)
    _launch_special_i1(x_c, out)
    # If original was non-contiguous, return view with same shape
    if x.layout == torch.strided and x.is_contiguous():
        return out
    else:
        return out.view_as(x)


def special_i1_out(self: torch.Tensor, out: torch.Tensor):
    x = self
    # Ensure dtypes and devices match expectations
    if out.dtype != x.dtype:
        raise TypeError("out dtype must match input dtype")
    if out.device != x.device:
        raise TypeError("out device must match input device")

    x_c = x.contiguous()
    out_c = out.contiguous()
    _launch_special_i1(x_c, out_c)
    if out_c.data_ptr() != out.data_ptr():
        out.copy_(out_c)
    return out