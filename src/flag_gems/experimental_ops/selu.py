import torch
import triton
import triton.language as tl


@triton.jit
def selu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    x_f32 = x.to(tl.float32)

    # SELU constants from PyTorch
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946

    zero = 0.0
    x_neg = tl.minimum(x_f32, zero)  # clamp to non-positive to avoid exp overflow
    neg_part = alpha * (tl.exp(x_neg) - 1.0)
    out_f32 = tl.where(x_f32 > 0.0, x_f32, neg_part)
    out_f32 = scale * out_f32

    y = out_f32.to(x.dtype)
    tl.store(y_ptr + offsets, y, mask=mask)


def selu(*args, **kwargs):
    # Resolve input tensor from args/kwargs
    x = None
    if len(args) > 0:
        x = args[0]
    elif "input" in kwargs:
        x = kwargs["input"]
    elif "self" in kwargs:
        x = kwargs["self"]
    else:
        raise TypeError("selu() missing required argument 'input' (pos 1)")

    if not isinstance(x, torch.Tensor):
        raise TypeError("selu() expected a torch.Tensor as input")

    # Fallback to PyTorch if not on CUDA
    if x.device.type != "cuda":
        return torch.ops.aten.selu(x)

    if not x.is_floating_point():
        raise TypeError("selu() expected a floating point tensor")

    x_contig = x.contiguous()
    y = torch.empty_like(x_contig)

    n_elements = y.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    selu_kernel[grid](x_contig, y, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return y
