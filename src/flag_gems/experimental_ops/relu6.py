import torch
import triton
import triton.language as tl


@triton.jit
def relu6(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.maximum(x, 0)
    y = tl.minimum(y, 6)
    tl.store(out_ptr + offsets, y, mask=mask)


relu6_kernel = relu6


def relu6(*args, **kwargs):
    x = (
        args[0]
        if len(args) > 0
        else kwargs.get("input", kwargs.get("self", kwargs.get("x")))
    )
    if x is None:
        raise TypeError(
            "relu6 expects a tensor as the first positional argument or keyword 'input'/'self'/'x'."
        )

    x_contig = x.contiguous()

    if not x_contig.is_cuda:
        return torch.clamp(x_contig, min=0, max=6)

    out = torch.empty_like(x_contig)
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    relu6_kernel[grid](x_contig, out, n_elements, BLOCK_SIZE=1024)
    return out
