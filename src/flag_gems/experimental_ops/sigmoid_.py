import torch
import triton
import triton.language as tl


@triton.jit
def sigmoid_(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_dtype = x.dtype
    x_fp32 = x.to(tl.float32)

    exp_neg = tl.exp(-x_fp32)
    exp_pos = tl.exp(x_fp32)
    out_pos = 1.0 / (1.0 + exp_neg)
    out_neg = exp_pos / (1.0 + exp_pos)
    cond = x_fp32 >= 0
    y_fp32 = tl.where(cond, out_pos, out_neg)
    y = y_fp32.to(x_dtype)

    tl.store(x_ptr + offsets, y, mask=mask)


# Keep a reference to the Triton kernel before defining the Python wrapper with the same name.
sigmoid___kernel = sigmoid_


def sigmoid_(*args, **kwargs):
    # Extract the input tensor following aten.sigmoid_ schema (self is the tensor)
    x = None
    if args:
        x = args[0]
    else:
        x = kwargs.get("self", kwargs.get("input", None))

    if not isinstance(x, torch.Tensor):
        raise TypeError("sigmoid_ expects a torch.Tensor as the first argument")

    # Fallback for unsupported cases
    if x.numel() == 0:
        return x
    if (
        (not x.is_cuda)
        or (not x.is_contiguous())
        or x.dtype not in (torch.float16, torch.bfloat16, torch.float32)
    ):
        return torch.ops.aten.sigmoid_(x)

    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    sigmoid___kernel[grid](x, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return x
