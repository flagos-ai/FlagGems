import torch
import triton
import triton.language as tl


@triton.jit
def log2_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # log2(x) = ln(x) * (1/ln(2))
    y = tl.log(x) * 1.4426950408889634074  # 1/ln(2)
    tl.store(y_ptr + offsets, y, mask=mask)


def _launch_log2_kernel(x_fp32: torch.Tensor, y_fp32: torch.Tensor):
    assert x_fp32.is_cuda and y_fp32.is_cuda, "Inputs must be CUDA tensors."
    assert x_fp32.dtype == torch.float32 and y_fp32.dtype == torch.float32, "Kernel expects float32 tensors."
    assert x_fp32.numel() == y_fp32.numel(), "Input and output must have the same number of elements."
    n_elements = x_fp32.numel()
    if n_elements == 0:
        return
    x_flat = x_fp32.contiguous().view(-1)
    y_flat = y_fp32.contiguous().view(-1)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    log2_kernel[grid](x_flat, y_flat, n_elements, BLOCK_SIZE=1024)


def log2(input: torch.Tensor):
    if not isinstance(input, torch.Tensor):
        raise TypeError("log2: input must be a torch.Tensor")
    if not input.is_cuda:
        raise RuntimeError("log2: input must be a CUDA tensor for Triton execution.")
    if input.is_complex():
        raise NotImplementedError("log2: complex dtypes are not supported by this Triton implementation.")

    # Determine output dtype (follow PyTorch behavior: integers -> float32)
    if input.is_floating_point():
        out_dtype = input.dtype
    else:
        out_dtype = torch.float32

    # Compute in float32 for broad device support, then cast to desired dtype if needed
    x_fp32 = input.to(torch.float32)
    y_fp32 = torch.empty_like(x_fp32, dtype=torch.float32)
    _launch_log2_kernel(x_fp32, y_fp32)

    if out_dtype != torch.float32:
        return y_fp32.to(dtype=out_dtype)
    return y_fp32


def log2_out(input: torch.Tensor, out: torch.Tensor):
    if not isinstance(input, torch.Tensor) or not isinstance(out, torch.Tensor):
        raise TypeError("log2_out: both input and out must be torch.Tensors")
    if not input.is_cuda or not out.is_cuda:
        raise RuntimeError("log2_out: input and out must be CUDA tensors for Triton execution.")
    if input.is_complex() or out.is_complex():
        raise NotImplementedError("log2_out: complex dtypes are not supported by this Triton implementation.")
    if not out.is_floating_point():
        raise TypeError("log2_out: out tensor must have a floating-point dtype.")
    if out.shape != input.shape:
        raise ValueError("log2_out: out tensor must have the same shape as input.")

    # Compute in float32 and cast/copy to out
    x_fp32 = input.to(torch.float32)
    if out.dtype == torch.float32 and out.is_contiguous():
        y_fp32 = out
    else:
        y_fp32 = torch.empty_like(x_fp32, dtype=torch.float32)

    _launch_log2_kernel(x_fp32, y_fp32)

    if y_fp32.data_ptr() != out.data_ptr() or out.dtype != torch.float32:
        out.copy_(y_fp32.to(dtype=out.dtype))
    return out