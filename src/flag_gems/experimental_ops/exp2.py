import torch
import triton
import triton.language as tl


@triton.jit
def exp2_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    ln2 = 0.693147180559945309417232121458176568
    y = tl.exp(x * ln2)
    tl.store(out_ptr + offsets, y, mask=mask)


def exp2(x: torch.Tensor) -> torch.Tensor:
    if not x.is_cuda:
        raise ValueError("exp2: input tensor must be on CUDA device")
    supported_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    if x.dtype not in supported_dtypes:
        raise TypeError(
            f"exp2: unsupported dtype {x.dtype}. Supported: {supported_dtypes}"
        )
    x_contig = x.contiguous()
    out = torch.empty_like(x_contig)
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    exp2_kernel[grid](x_contig, out, n_elements, BLOCK_SIZE=1024)
    return out


def exp2_out(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    if not x.is_cuda or not out.is_cuda:
        raise ValueError("exp2_out: both input and out tensors must be on CUDA device")
    if x.shape != out.shape:
        raise ValueError("exp2_out: input and out must have the same shape")
    if x.dtype != out.dtype:
        raise TypeError("exp2_out: input and out must have the same dtype")
    supported_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    if x.dtype not in supported_dtypes:
        raise TypeError(
            f"exp2_out: unsupported dtype {x.dtype}. Supported: {supported_dtypes}"
        )
    x_contig = x.contiguous()
    out_contig = out.contiguous()
    n_elements = out_contig.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    exp2_kernel[grid](x_contig, out_contig, n_elements, BLOCK_SIZE=1024)
    if out_contig.data_ptr() != out.data_ptr():
        out.copy_(out_contig)
    return out
