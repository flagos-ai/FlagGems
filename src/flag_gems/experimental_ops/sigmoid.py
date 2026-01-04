import torch
import triton
import triton.language as tl


@triton.jit
def sigmoid_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)
    y = 1.0 / (1.0 + tl.exp(-x_f32))
    y = y.to(x.dtype)
    tl.store(out_ptr + offsets, y, mask=mask)


def _sigmoid_common(x: torch.Tensor, out: torch.Tensor = None):
    if not isinstance(x, torch.Tensor):
        raise TypeError("sigmoid: expected a torch.Tensor as input")
    if not x.is_cuda:
        raise ValueError("sigmoid: input tensor must be on CUDA device")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise NotImplementedError(f"sigmoid: dtype {x.dtype} is not supported (supported: float16, bfloat16, float32)")

    n_elements = x.numel()
    if out is None:
        out = torch.empty_like(x)
    else:
        if not isinstance(out, torch.Tensor):
            raise TypeError("sigmoid.out: 'out' must be a torch.Tensor")
        if not out.is_cuda:
            raise ValueError("sigmoid.out: 'out' tensor must be on CUDA device")
        if out.shape != x.shape:
            raise ValueError(f"sigmoid.out: 'out' shape {out.shape} does not match input shape {x.shape}")
        if out.dtype != x.dtype:
            raise ValueError(f"sigmoid.out: 'out' dtype {out.dtype} must match input dtype {x.dtype}")

    if n_elements == 0:
        return out

    x_contig = x.contiguous()
    out_contig = out if out.is_contiguous() else torch.empty_like(out, memory_format=torch.contiguous_format)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    sigmoid_kernel[grid](x_contig, out_contig, n_elements, BLOCK_SIZE=1024)

    if out_contig.data_ptr() != out.data_ptr():
        out.copy_(out_contig)
    return out


def sigmoid(self: torch.Tensor):
    return _sigmoid_common(self, out=None)


def sigmoid_out(self: torch.Tensor, out: torch.Tensor):
    return _sigmoid_common(self, out=out)