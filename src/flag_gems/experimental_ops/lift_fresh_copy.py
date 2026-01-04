import torch
import triton
import triton.language as tl


@triton.jit
def _copy_kernel(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)


def lift_fresh_copy(*args, **kwargs):
    # Attempt to find the input tensor from args/kwargs
    x = None
    if len(args) > 0 and isinstance(args[0], torch.Tensor):
        x = args[0]
    elif "self" in kwargs and isinstance(kwargs["self"], torch.Tensor):
        x = kwargs["self"]
    else:
        for v in list(args) + list(kwargs.values()):
            if isinstance(v, torch.Tensor):
                x = v
                break
    if x is None:
        raise ValueError("lift_fresh_copy expects a Tensor argument")

    if not x.is_cuda:
        raise ValueError("lift_fresh_copy Triton kernel requires a CUDA tensor")

    x_contig = x.contiguous()
    out = torch.empty_like(x_contig, memory_format=torch.contiguous_format)

    n_elements = x_contig.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _copy_kernel[grid](x_contig, out, n_elements, BLOCK_SIZE=1024)

    return out.view_as(x_contig)


def lift_fresh_copy_out(x: torch.Tensor, out: torch.Tensor = None):
    if x is None or not isinstance(x, torch.Tensor):
        raise ValueError("lift_fresh_copy_out expects 'x' to be a Tensor")
    if not x.is_cuda:
        raise ValueError("lift_fresh_copy_out Triton kernel requires CUDA tensors")

    x_contig = x.contiguous()

    if out is None:
        out = torch.empty_like(x_contig, memory_format=torch.contiguous_format)
    else:
        if not out.is_cuda:
            raise ValueError("Output tensor 'out' must be on CUDA")
        if out.dtype != x_contig.dtype:
            raise ValueError("Output tensor 'out' must have the same dtype as input")
        # Resize to match input shape and ensure contiguous layout
        if out.numel() != x_contig.numel() or not out.is_contiguous():
            out.resize_(x_contig.shape)
            if not out.is_contiguous():
                out = out.contiguous()

    n_elements = x_contig.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _copy_kernel[grid](x_contig, out, n_elements, BLOCK_SIZE=1024)

    return out.view_as(x_contig)