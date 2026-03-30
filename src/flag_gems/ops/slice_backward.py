import torch
import triton
import triton.language as tl


@triton.jit
def slice_backward_kernel(
    grad_output_ptr,
    grad_input_ptr,
    numel,
    inner,
    slice_len,
    dim_size,
    start,
    step,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = offsets < numel

    grad = tl.load(grad_output_ptr + offsets, mask=mask)

    outer_idx = offsets // (slice_len * inner)

    slice_idx = (offsets // inner) % slice_len

    inner_idx = offsets % inner

    dim_index = start + slice_idx * step

    input_offset = outer_idx * dim_size * inner + dim_index * inner + inner_idx

    tl.store(grad_input_ptr + input_offset, grad, mask=mask)


def slice_backward(
    grad_output,
    input_sizes,
    dim,
    start,
    end,
    step,
):
    grad_input = torch.zeros(
        input_sizes,
        device=grad_output.device,
        dtype=grad_output.dtype,
    )

    shape = list(input_sizes)

    outer = 1
    for i in range(dim):
        outer *= shape[i]

    inner = 1
    for i in range(dim + 1, len(shape)):
        inner *= shape[i]

    dim_size = shape[dim]

    actual_start = max(0, min(start, dim_size))
    actual_end = max(0, min(end, dim_size))
    slice_len = max(0, (actual_end - actual_start + step - 1) // step)
    start = actual_start

    numel = grad_output.numel()

    BLOCK = 1024

    grid = lambda meta: (triton.cdiv(numel, meta["BLOCK_SIZE"]),)

    slice_backward_kernel[grid](
        grad_output,
        grad_input,
        numel,
        inner,
        slice_len,
        dim_size,
        start,
        step,
        BLOCK_SIZE=BLOCK,
    )

    return grad_input
