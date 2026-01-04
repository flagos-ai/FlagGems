import torch
import triton
import triton.language as tl


@triton.jit
def relu_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    n_elements,  # Number of elements
    COMPUTE_FP32: tl.constexpr,  # Whether to upcast to fp32 for computation
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0)

    if COMPUTE_FP32:
        x_f32 = x.to(tl.float32)
        y_f32 = tl.maximum(x_f32, 0.0)
        y = y_f32.to(x.dtype)
    else:
        y = tl.maximum(x, 0)

    tl.store(output_ptr + offsets, y, mask=mask)


def relu(input: torch.Tensor) -> torch.Tensor:
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")

    if input.is_complex():
        raise TypeError("relu does not support complex tensors.")

    if not input.is_cuda:
        raise RuntimeError(
            "Triton kernels require CUDA tensors. Move the tensor to a CUDA device."
        )

    dtype = input.dtype

    # Handle boolean tensors: ReLU is identity
    if dtype == torch.bool:
        return input.clone()

    # Determine computation path
    compute_in_fp32 = False
    if input.is_floating_point():
        if dtype in (torch.float16, torch.bfloat16):
            compute_in_fp32 = True
        else:
            compute_in_fp32 = False
    else:
        # Integer tensors handled in native dtype (no fp32 upcast)
        compute_in_fp32 = False

    x = input.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    relu_kernel[grid](
        x,
        out,
        n_elements,
        COMPUTE_FP32=compute_in_fp32,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out
