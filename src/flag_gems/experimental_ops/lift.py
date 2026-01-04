import torch
import triton
import triton.language as tl


@triton.jit
def _copy_kernel(src_ptr, dst_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(src_ptr + offsets, mask=mask)
    tl.store(dst_ptr + offsets, x, mask=mask)


def lift(x: torch.Tensor):
    if not isinstance(x, torch.Tensor):
        raise TypeError("lift expects a single Tensor argument")
    if x.device.type != "cuda":
        raise RuntimeError("lift: input tensor must be on a CUDA device")
    out = torch.empty_like(x)
    n_elements = out.numel()
    if n_elements > 0:
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _copy_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)
    return out


def lift_out(x: torch.Tensor, out: torch.Tensor):
    if not isinstance(x, torch.Tensor) or not isinstance(out, torch.Tensor):
        raise TypeError("lift_out expects (Tensor x, Tensor out)")
    if x.device.type != "cuda" or out.device.type != "cuda":
        raise RuntimeError("lift_out: both input and out tensors must be on a CUDA device")
    if out.device != x.device:
        raise RuntimeError("lift_out: out tensor must be on the same device as input")
    if out.dtype != x.dtype:
        raise RuntimeError("lift_out: out tensor must have the same dtype as input")
    # Resize out to match shape; this ensures a contiguous layout and correct size.
    if tuple(out.shape) != tuple(x.shape):
        out.resize_(x.shape)
    n_elements = x.numel()
    if n_elements > 0:
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _copy_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)
    return out