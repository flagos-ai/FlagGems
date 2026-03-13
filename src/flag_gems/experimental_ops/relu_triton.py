import torch
import triton
import triton.language as tl

@triton.jit
def relu(input_ptr,  # Pointer to input tensor
         output_ptr,  # Pointer to output tensor (float32)
         n_elements,  # Total number of elements
         BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0).to(tl.float32)
    y = tl.maximum(x, 0.0)
    tl.store(output_ptr + offsets, y, mask=mask)

# Preserve reference to the Triton kernel before defining the Python wrapper with the same name.
relu_kernel = relu

def relu(input: torch.Tensor) -> torch.Tensor:
    assert input.is_cuda, "Input tensor must be on CUDA device."
    input_contig = input.contiguous()
    n_elements = input_contig.numel()

    # Use float32 for computation
    out_fp32 = torch.empty_like(input_contig, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    relu_kernel[grid](input_contig, out_fp32, n_elements, BLOCK_SIZE=1024)

    # Convert back to original dtype if needed
    if input.dtype != torch.float32:
        return out_fp32.to(dtype=input.dtype)
    return out_fp32