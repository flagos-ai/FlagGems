import torch
import triton
import triton.language as tl


@triton.jit
def _native_dropout_backward_kernel(
    grad_ptr,        # *Pointer* to grad_output tensor
    mask_ptr,        # *Pointer* to mask tensor (cast to same dtype as grad)
    out_ptr,         # *Pointer* to output grad_input tensor
    n_elements,      # Number of elements
    scale,           # Scaling factor (float)
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    in_bounds = offsets < n_elements

    g = tl.load(grad_ptr + offsets, mask=in_bounds, other=0)
    m = tl.load(mask_ptr + offsets, mask=in_bounds, other=0)

    # grad_input = grad_output * mask * scale
    out = g * m * scale
    tl.store(out_ptr + offsets, out, mask=in_bounds)


def _launch_native_dropout_backward(grad_output: torch.Tensor, mask: torch.Tensor, scale: float, out: torch.Tensor):
    assert grad_output.is_cuda and mask.is_cuda and out.is_cuda, "All tensors must be CUDA tensors"
    assert grad_output.numel() == mask.numel() == out.numel(), "grad_output, mask, and out must have the same number of elements"
    assert grad_output.dtype in (torch.float16, torch.bfloat16, torch.float32), "Supported dtypes: float16, bfloat16, float32"
    assert out.dtype == grad_output.dtype, "Output dtype must match grad_output dtype"
    assert grad_output.device == mask.device == out.device, "All tensors must be on the same device"

    go = grad_output.contiguous()
    m = mask.contiguous()
    if m.dtype != go.dtype:
        m = m.to(dtype=go.dtype)

    out_contig = out if out.is_contiguous() else torch.empty_like(go)

    n_elements = go.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _native_dropout_backward_kernel[grid](
        go, m, out_contig, n_elements, float(scale), BLOCK_SIZE=BLOCK_SIZE
    )

    if out_contig.data_ptr() != out.data_ptr():
        out.copy_(out_contig)
    return out


def native_dropout_backward(grad_output: torch.Tensor, mask: torch.Tensor, scale: float):
    """
    Wrapper for aten::native_dropout_backward
    Computes grad_input = grad_output * mask.to(grad_output.dtype) * scale
    """
    out = torch.empty_like(grad_output)
    return _launch_native_dropout_backward(grad_output, mask, scale, out)


def native_dropout_backward_out(grad_output: torch.Tensor, mask: torch.Tensor, scale: float, out: torch.Tensor):
    """
    Wrapper for aten::native_dropout_backward.out
    Writes result into 'out'
    """
    _launch_native_dropout_backward(grad_output, mask, scale, out)
    return out