import torch
import triton
import triton.language as tl


@triton.jit
def swiglu_kernel(x_ptr, out_ptr, half_n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < half_n

    x1 = tl.load(x_ptr + offs, mask=mask)
    x2 = tl.load(x_ptr + offs + half_n, mask=mask)
    gate = x1 * tl.sigmoid(x1)
    res = gate * x2
    tl.store(out_ptr + offs, res, mask=mask)


def swiglu(x: torch.Tensor):
    assert x.shape[-1] % 2 == 0
    out_shape = list(x.shape)
    out_shape[-1] = out_shape[-1] // 2
    out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
    half_dim = out_shape[-1]
    n_elem = out.numel()
    BLOCK = 1024
    grid = (triton.cdiv(n_elem, BLOCK),)
    swiglu_kernel[grid](x, out, half_dim, BLOCK)
    return out
