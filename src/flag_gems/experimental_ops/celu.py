import torch
import triton
import triton.language as tl


@triton.jit
def celu_kernel(x_ptr, out_ptr, n_elements, alpha, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    x_fp = x.to(tl.float32)
    y_fp = tl.where(x_fp > 0.0, x_fp, alpha * (tl.exp(x_fp / alpha) - 1.0))
    y = y_fp.to(x.dtype)

    tl.store(out_ptr + offsets, y, mask=mask)


def _parse_alpha(alpha):
    if isinstance(alpha, torch.Tensor):
        if alpha.numel() != 1:
            raise ValueError("alpha tensor must be a scalar (numel() == 1)")
        alpha = float(alpha.item())
    else:
        alpha = float(alpha)
    if alpha == 0.0:
        raise ValueError("alpha must be non-zero")
    return alpha


def celu(input: torch.Tensor, alpha: float = 1.0):
    alpha = _parse_alpha(alpha)
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")
    if not input.is_cuda:
        raise ValueError("input must be on CUDA device")
    if not torch.is_floating_point(input):
        raise TypeError("input must be a floating point tensor")

    x_contig = input.contiguous()
    out = torch.empty_like(x_contig)

    n_elements = out.numel()
    if n_elements == 0:
        return out

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    celu_kernel[grid](x_contig, out, n_elements, alpha, BLOCK_SIZE=1024)
    return out


def celu_out(input: torch.Tensor, alpha: float = 1.0, out: torch.Tensor = None):
    alpha = _parse_alpha(alpha)
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")
    if out is None or not isinstance(out, torch.Tensor):
        raise TypeError("out must be a preallocated torch.Tensor")
    if not input.is_cuda or not out.is_cuda:
        raise ValueError("input and out must be on CUDA device")
    if not torch.is_floating_point(input) or not torch.is_floating_point(out):
        raise TypeError("input and out must be floating point tensors")
    if out.shape != input.shape:
        raise ValueError("out must have the same shape as input")
    if out.dtype != input.dtype:
        raise ValueError("out must have the same dtype as input")

    x_contig = input.contiguous()
    if out.is_contiguous():
        out_contig = out
    else:
        out_contig = torch.empty_like(x_contig)

    n_elements = x_contig.numel()
    if n_elements == 0:
        if out_contig is not out:
            out.copy_(out_contig)
        return out

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    celu_kernel[grid](x_contig, out_contig, n_elements, alpha, BLOCK_SIZE=1024)

    if out_contig is not out:
        out.copy_(out_contig)
    return out