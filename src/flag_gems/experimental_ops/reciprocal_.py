import torch
import triton
import triton.language as tl


@triton.jit
def reciprocal_(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    out = 1.0 / x
    tl.store(x_ptr + offsets, out, mask=mask)


# Preserve a reference to the Triton kernel before defining the Python wrapper with the same name.
reciprocal___kernel = reciprocal_


def reciprocal_(x: torch.Tensor):
    # Fallback for unsupported cases
    supported_dtypes = {torch.float16, torch.bfloat16, torch.float32}
    if (
        (not isinstance(x, torch.Tensor))
        or (not x.is_cuda)
        or (not x.is_contiguous())
        or (x.dtype not in supported_dtypes)
    ):
        return torch.ops.aten.reciprocal_(x)

    n_elements = x.numel()
    if n_elements == 0:
        return x

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # noqa: E731
    reciprocal___kernel[grid](x, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return x
