import torch
import triton
import triton.language as tl


@triton.jit
def sign_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    one = tl.full([BLOCK_SIZE], 1, x.dtype)
    neg_one = tl.full([BLOCK_SIZE], -1, x.dtype)

    res = tl.where(x > 0, one, tl.where(x < 0, neg_one, x))
    tl.store(out_ptr + offsets, res, mask=mask)


def _launch_sign_kernel(x: torch.Tensor, out: torch.Tensor):
    n_elements = out.numel()
    if n_elements == 0:
        return
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    sign_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)


def sign(x: torch.Tensor):
    if x.is_complex():
        raise NotImplementedError(
            "Complex dtypes are not supported by this Triton sign kernel."
        )
    out = torch.empty_like(x)
    _launch_sign_kernel(x.contiguous(), out.contiguous())
    return out


def sign_out(x: torch.Tensor, out: torch.Tensor):
    if x.is_complex() or out.is_complex():
        raise NotImplementedError(
            "Complex dtypes are not supported by this Triton sign kernel."
        )
    if out.shape != x.shape:
        raise ValueError("Output tensor must have the same shape as input tensor.")
    if out.dtype != x.dtype:
        raise ValueError("Output tensor must have the same dtype as input tensor.")
    if out.device != x.device:
        raise ValueError("Output tensor must be on the same device as input tensor.")
    _launch_sign_kernel(x.contiguous(), out.contiguous())
    return out
