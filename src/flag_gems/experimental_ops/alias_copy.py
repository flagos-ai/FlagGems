import torch
import triton
import triton.language as tl


@triton.jit
def _alias_copy_kernel(src_ptr, dst_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    vals = tl.load(src_ptr + offsets, mask=mask)
    tl.store(dst_ptr + offsets, vals, mask=mask)


def alias_copy(x: torch.Tensor):
    """
    Wrapper for aten::alias_copy
    Creates and returns a copy of `x` with identical content.
    """
    if not x.is_cuda:
        raise RuntimeError("alias_copy: Triton kernel requires CUDA tensors.")
    out = torch.empty_like(x)
    n_elements = out.numel()
    if n_elements == 0:
        return out
    # Ensure contiguous memory for efficient linear copy
    src = x.contiguous() if not x.is_contiguous() else x
    if not out.is_contiguous():
        out = out.contiguous()
    if src.dtype != out.dtype:
        raise RuntimeError("alias_copy: dtype mismatch between input and output.")
    if src.device != out.device:
        raise RuntimeError("alias_copy: input and output must be on the same device.")
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _alias_copy_kernel[grid](src, out, n_elements, BLOCK_SIZE=1024)
    return out


def alias_copy_out(x: torch.Tensor, out: torch.Tensor):
    """
    Wrapper for aten::alias_copy.out
    Copies `x` into `out` and returns `out`.
    """
    if not x.is_cuda or not out.is_cuda:
        raise RuntimeError("alias_copy_out: Triton kernel requires CUDA tensors.")
    if x.dtype != out.dtype:
        raise RuntimeError("alias_copy_out: dtype of input and output must match.")
    if x.numel() != out.numel():
        raise RuntimeError(
            "alias_copy_out: input and output must have the same number of elements."
        )
    if x.device != out.device:
        raise RuntimeError(
            "alias_copy_out: input and output must be on the same device."
        )
    if not out.is_contiguous():
        raise RuntimeError("alias_copy_out: output tensor must be contiguous.")
    src = x.contiguous() if not x.is_contiguous() else x
    n_elements = out.numel()
    if n_elements == 0:
        return out
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _alias_copy_kernel[grid](src, out, n_elements, BLOCK_SIZE=1024)
    return out
