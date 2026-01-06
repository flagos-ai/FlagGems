import torch
import triton
import triton.language as tl


@triton.jit
def lgamma_(x_ptr, n_elements, IS_FP64: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x_raw = tl.load(x_ptr + offsets, mask=mask)

    if IS_FP64:
        xf = x_raw.to(tl.float64)
    else:
        xf = x_raw.to(tl.float32)

    pi = 3.141592653589793
    half_log_two_pi = 0.9189385332046727
    g = 7.0
    c0 = 0.99999999999980993
    c1 = 676.5203681218851
    c2 = -1259.1392167224028
    c3 = 771.32342877765313
    c4 = -176.61502916214059
    c5 = 12.507343278686905
    c6 = -0.13857109526572012
    c7 = 9.9843695780195716e-6
    c8 = 1.5056327351493116e-7

    # Lanczos evaluation for xf
    u = xf
    u1 = u - 1.0
    a = c0
    a = a + c1 / (u1 + 1.0)
    a = a + c2 / (u1 + 2.0)
    a = a + c3 / (u1 + 3.0)
    a = a + c4 / (u1 + 4.0)
    a = a + c5 / (u1 + 5.0)
    a = a + c6 / (u1 + 6.0)
    a = a + c7 / (u1 + 7.0)
    a = a + c8 / (u1 + 8.0)
    t = u1 + g + 0.5
    res_std = half_log_two_pi + (u1 + 0.5) * tl.log(t) - t + tl.log(a)

    # Lanczos evaluation for (1 - xf)
    u_ref = 1.0 - xf
    ur1 = u_ref - 1.0
    ar = c0
    ar = ar + c1 / (ur1 + 1.0)
    ar = ar + c2 / (ur1 + 2.0)
    ar = ar + c3 / (ur1 + 3.0)
    ar = ar + c4 / (ur1 + 4.0)
    ar = ar + c5 / (ur1 + 5.0)
    ar = ar + c6 / (ur1 + 6.0)
    ar = ar + c7 / (ur1 + 7.0)
    ar = ar + c8 / (ur1 + 8.0)
    tr_ = ur1 + g + 0.5
    lgn_1mz = half_log_two_pi + (ur1 + 0.5) * tl.log(tr_) - tr_ + tl.log(ar)

    s = tl.sin(pi * xf)
    res_ref = tl.log(pi) - tl.log(tl.abs(s)) - lgn_1mz

    res = tl.where(xf < 0.5, res_ref, res_std)

    y = res.to(x_raw.dtype)
    tl.store(x_ptr + offsets, y, mask=mask)


_lgamma_kernel = lgamma_


def lgamma_(x: torch.Tensor):
    assert isinstance(x, torch.Tensor), "Input must be a torch.Tensor"
    assert x.is_cuda, "Input tensor must be on CUDA device"
    assert x.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ), "Unsupported dtype for lgamma_"

    needs_copy_back = False
    target = x
    if not x.is_contiguous():
        target = x.contiguous()
        needs_copy_back = True

    n_elements = target.numel()
    if n_elements == 0:
        return x

    IS_FP64 = target.dtype == torch.float64
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _lgamma_kernel[grid](target, n_elements, IS_FP64=IS_FP64, BLOCK_SIZE=BLOCK_SIZE)

    if needs_copy_back:
        x.copy_(target)

    return x
