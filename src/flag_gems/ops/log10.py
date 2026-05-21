
import torch
import triton
import triton.language as tl


@triton.jit
def log10_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):

    pid = tl.program_id(0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    x = tl.maximum(x, 1e-12)

    y = tl.log(x) / 2.302585092994046

    tl.store(y_ptr + offsets, y, mask=mask)


def log10(input: torch.Tensor):

    assert input.is_cuda, "Input must be CUDA tensor"

    output = torch.empty_like(input)

    n_elements = input.numel()

    BLOCK_SIZE = 1024

    grid = lambda meta: (
        triton.cdiv(n_elements, meta["BLOCK_SIZE"]),
    )

    log10_kernel[grid](
        input,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return output
