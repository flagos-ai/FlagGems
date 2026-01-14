import torch
import triton
import triton.language as tl


@triton.jit
def selu_(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    x_f32 = x.to(tl.float32)
    alpha = 1.6732632423543772
    scale = 1.0507009873554805
    y_f32 = scale * tl.where(x_f32 > 0, x_f32, alpha * (tl.exp(x_f32) - 1.0))
    y = y_f32.to(x.dtype)

    tl.store(x_ptr + offsets, y, mask=mask)


# Keep a handle to the Triton kernel before defining the Python wrapper with the same name.
selu__kernel = selu_


def selu_(*args, **kwargs):
    # Extract the input tensor from positional or keyword arguments
    x = None
    if len(args) > 0 and torch.is_tensor(args[0]):
        x = args[0]
    elif "input" in kwargs and torch.is_tensor(kwargs["input"]):
        x = kwargs["input"]
    elif "self" in kwargs and torch.is_tensor(kwargs["self"]):
        x = kwargs["self"]
    elif "x" in kwargs and torch.is_tensor(kwargs["x"]):
        x = kwargs["x"]
    else:
        raise ValueError(
            "selu_ expects a Tensor as the first argument or under 'input'/'self'/'x' keyword."
        )

    # Fallback for unsupported cases
    supported_dtypes = {torch.float16, torch.bfloat16, torch.float32}
    if (not x.is_cuda) or (not x.is_contiguous()) or (x.dtype not in supported_dtypes):
        torch.ops.aten.selu_(x)
        return x

    n_elements = x.numel()
    if n_elements == 0:
        return x

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    selu__kernel[grid](x, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return x
