import torch
import triton
import triton.language as tl


@triton.jit
def _fix_trunc_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    # Upcast to fp32 for stable math, then cast back to input dtype
    x_fp32 = x.to(tl.float32)
    res_pos = tl.floor(x_fp32)
    res_neg = tl.ceil(x_fp32)
    res_fp32 = tl.where(x_fp32 >= 0, res_pos, res_neg)
    res = res_fp32.to(x.dtype)

    tl.store(out_ptr + offsets, res, mask=mask)


@triton.jit
def _copy_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)


def _launch_fix_kernel(x: torch.Tensor, out: torch.Tensor, block_size: int = 1024):
    assert x.is_cuda and out.is_cuda, "Input and output must be on CUDA device"
    assert (
        x.numel() == out.numel()
    ), "Input and output must have the same number of elements"
    assert x.device == out.device, "Input and output must be on the same device"
    assert (
        x.is_contiguous() and out.is_contiguous()
    ), "Only contiguous tensors are supported"

    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # For floating point types, perform truncation toward zero.
    # For non-floating types, it's effectively a no-op (copy).
    if x.is_floating_point():
        _fix_trunc_kernel[grid](x, out, n_elements, BLOCK_SIZE=block_size)
    else:
        _copy_kernel[grid](x, out, n_elements, BLOCK_SIZE=block_size)


def fix(self: torch.Tensor):
    """
    Wrapper for ATen operator: ('fix', <Autograd.disable: False>)
    Rounds elements toward zero (like trunc) for floating tensors.
    Leaves integer tensors unchanged.
    """
    if self.is_complex():
        # Fallback for complex dtypes not supported by Triton: use PyTorch
        return torch.trunc(self)

    out = torch.empty_like(self)
    _launch_fix_kernel(self, out)
    return out


def fix_out(self: torch.Tensor, out: torch.Tensor):
    """
    Wrapper for ATen operator: ('fix.out', <Autograd.disable: False>)
    Writes the result into 'out'.
    """
    if self.is_complex():
        # Fallback for complex dtypes not supported by Triton: use PyTorch
        out.copy_(torch.trunc(self))
        return out

    _launch_fix_kernel(self, out)
    return out
