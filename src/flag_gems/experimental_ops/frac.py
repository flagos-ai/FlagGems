import torch
import triton
import triton.language as tl


@triton.jit
def frac_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr, IS_FP16: tl.constexpr, IS_BF16: tl.constexpr, IS_FP64: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0)

    # Choose compute dtype
    if IS_FP64:
        x_comp = x.to(tl.float64)
    elif IS_FP16 or IS_BF16:
        x_comp = x.to(tl.float32)
    else:
        x_comp = x  # float32

    trunc_val = tl.where(x_comp >= 0, tl.floor(x_comp), tl.ceil(x_comp))
    y_comp = x_comp - trunc_val

    # Cast back to output dtype
    if IS_FP64:
        y = y_comp.to(tl.float64)
    elif IS_FP16:
        y = y_comp.to(tl.float16)
    elif IS_BF16:
        y = y_comp.to(tl.bfloat16)
    else:
        y = y_comp.to(tl.float32)

    tl.store(out_ptr + offsets, y, mask=mask)


def _launch_frac(x: torch.Tensor, out: torch.Tensor):
    assert x.is_cuda and out.is_cuda, "Inputs must be CUDA tensors"
    assert x.numel() == out.numel(), "Input and output must have the same number of elements"
    assert x.dtype == out.dtype, "Input and output must have the same dtype"
    if not x.is_floating_point():
        raise NotImplementedError("frac is only implemented for floating point dtypes")
    if x.is_complex():
        raise NotImplementedError("frac is not implemented for complex dtypes in this Triton kernel")

    n_elements = x.numel()
    if n_elements == 0:
        return out

    # Use contiguous buffers for kernel execution
    x_contig = x.contiguous()
    out_contig = out.contiguous()

    is_fp16 = x_contig.dtype == torch.float16
    is_bf16 = x_contig.dtype == torch.bfloat16
    is_fp64 = x_contig.dtype == torch.float64

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    frac_kernel[grid](
        x_contig, out_contig, n_elements,
        BLOCK_SIZE=1024,
        IS_FP16=is_fp16,
        IS_BF16=is_bf16,
        IS_FP64=is_fp64,
    )

    # If out was non-contiguous, copy results back
    if out_contig.data_ptr() != out.data_ptr():
        out.copy_(out_contig)
    return out


def frac(input: torch.Tensor):
    out = torch.empty_like(input)
    _launch_frac(input, out)
    return out


def frac_out(input: torch.Tensor, out: torch.Tensor):
    # Ensure shape and dtype match per .out contract
    assert out.shape == input.shape, "out must have the same shape as input"
    assert out.dtype == input.dtype, "out must have the same dtype as input"
    _launch_frac(input, out)
    return out