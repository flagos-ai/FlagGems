import torch
import triton
import triton.language as tl


@triton.jit
def hardtanh_kernel(
    x_ptr, out_ptr, n_elements, min_val, max_val, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    x_fp32 = x.to(tl.float32)

    min_v = min_val  # expected to be float32 scalar
    max_v = max_val  # expected to be float32 scalar
    x_clamped = tl.maximum(tl.minimum(x_fp32, max_v), min_v)

    y = x_clamped.to(x.dtype)
    tl.store(out_ptr + offsets, y, mask=mask)


def _launch_hardtanh(
    input: torch.Tensor, output: torch.Tensor, min_val: float, max_val: float
):
    assert input.is_cuda and output.is_cuda, "Tensors must be on CUDA device"
    assert input.device == output.device, "Input and output must be on the same device"
    assert input.dtype == output.dtype, "Input and output must have the same dtype"
    n_elements = input.numel()
    if n_elements == 0:
        return output
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    hardtanh_kernel[grid](
        input,
        output,
        n_elements,
        float(min_val),
        float(max_val),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


def hardtanh(self: torch.Tensor, min_val: float = -1.0, max_val: float = 1.0):
    x = self
    assert x.is_cuda, "Input tensor must be on CUDA device"
    x_contig = x.contiguous()
    out = torch.empty_like(x_contig)
    _launch_hardtanh(x_contig, out, min_val, max_val)
    # If original tensor wasn't contiguous, we still return a tensor matching input's shape and dtype
    return out.view_as(x)


def hardtanh_out(
    self: torch.Tensor,
    min_val: float = -1.0,
    max_val: float = 1.0,
    out: torch.Tensor = None,
):
    x = self
    assert x.is_cuda, "Input tensor must be on CUDA device"
    if out is None:
        out = torch.empty_like(x)
    assert out.is_cuda, "Output tensor must be on CUDA device"
    assert out.shape == x.shape, "Output tensor must have the same shape as input"
    assert out.dtype == x.dtype, "Output tensor must have the same dtype as input"
    if not out.is_contiguous():
        # For non-contiguous out, compute into a contiguous buffer then copy back
        out_contig = torch.empty_like(out.contiguous())
        _launch_hardtanh(x.contiguous(), out_contig, min_val, max_val)
        out.copy_(out_contig)
        return out
    _launch_hardtanh(x.contiguous(), out, min_val, max_val)
    return out
