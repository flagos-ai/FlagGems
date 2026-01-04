import torch
import triton
import triton.language as tl


@triton.jit
def exp2_(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x_f32 = x.to(tl.float32)
    ln2 = 0.693147180559945309417232121458176568
    y_f32 = tl.exp(x_f32 * ln2)
    y = y_f32.to(x.dtype)
    tl.store(x_ptr + offsets, y, mask=mask)


# Preserve reference to the Triton kernel before defining the Python wrapper with the same name.
exp2__kernel = exp2_


def exp2_(*args, **kwargs):
    x = None
    if len(args) > 0:
        x = args[0]
    else:
        x = kwargs.get("input", None)
        if x is None:
            x = kwargs.get("x", None)
    assert isinstance(x, torch.Tensor), "exp2_ expects a torch.Tensor as its first argument"
    assert x.is_cuda, "exp2_ Triton kernel requires a CUDA tensor"
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    exp2__kernel[grid](x, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return x