import torch
import triton
import triton.language as tl


@triton.jit
def zero_kernel(
    out_ptr,  # *Pointer* to tensor to be zeroed
    n_elements,  # Number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Create a zero value with the correct dtype using a dummy load to infer dtype
    dummy = tl.load(out_ptr + offsets, mask=mask, other=0)
    z = tl.zeros([BLOCK_SIZE], dtype=dummy.dtype)
    tl.store(out_ptr + offsets, z, mask=mask)


def _launch_zero_kernel(tensor: torch.Tensor):
    assert isinstance(tensor, torch.Tensor), "Expected a torch.Tensor"
    assert tensor.is_cuda, "Tensor must be on CUDA device"
    assert tensor.is_contiguous(), "Tensor must be contiguous"
    assert tensor.numel() >= 0
    n_elements = tensor.numel()
    if n_elements == 0:
        return tensor
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    zero_kernel[grid](tensor, n_elements, BLOCK_SIZE=1024)
    return tensor


def zero(*args, **kwargs):
    # Accept common conventions: first positional as target, or 'self'/'input'/'out' in kwargs
    target = None
    if len(args) >= 1 and isinstance(args[0], torch.Tensor):
        target = args[0]
    elif "self" in kwargs and isinstance(kwargs["self"], torch.Tensor):
        target = kwargs["self"]
    elif "input" in kwargs and isinstance(kwargs["input"], torch.Tensor):
        target = kwargs["input"]
    elif "out" in kwargs and isinstance(kwargs["out"], torch.Tensor):
        target = kwargs["out"]
    else:
        raise ValueError(
            "zero expects a Tensor as the first argument or in kwargs as 'self', 'input', or 'out'"
        )
    return _launch_zero_kernel(target)


def zero_out(*args, **kwargs):
    # Out variant: prefer 'out' kwarg; else first positional
    out = None
    if "out" in kwargs and isinstance(kwargs["out"], torch.Tensor):
        out = kwargs["out"]
    elif len(args) >= 1 and isinstance(args[0], torch.Tensor):
        out = args[0]
    else:
        raise ValueError(
            "zero_out expects an output Tensor as the first positional argument or 'out' kwarg"
        )
    return _launch_zero_kernel(out)
