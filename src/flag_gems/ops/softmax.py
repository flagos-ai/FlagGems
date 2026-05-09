import torch
import triton
import triton.language as tl

# from flag_gems.utils import libentry

# Optimized softmax kernel for FlagGems
# Supports fp16, bf16, fp32 via Python-level dtype casting
# Uses triton.next_power_of_2 to ensure full row fits in one block


@triton.jit
def softmax_kernel(
    out_ptr,
    inp_ptr,
    inp_row_stride,
    out_row_stride,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    # One program handles one row of the input tensor
    row_idx = tl.program_id(axis=0)
    row_inp = inp_ptr + row_idx * inp_row_stride
    row_out = out_ptr + row_idx * out_row_stride

    # Column offsets — BLOCK_SIZE always >= N so full row fits
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # Load row — pad out-of-bounds with -inf so they dont affect max
    x = tl.load(row_inp + cols, mask=mask, other=-float("inf"))

    # Subtract row max for numerical stability (prevents exp overflow)
    x_max = tl.max(x, axis=0)
    x = x - x_max

    # Compute exp of shifted values
    x_exp = tl.exp(x)

    # Compute normalizing constant
    x_sum = tl.sum(x_exp, axis=0)

    # Normalize to get softmax probabilities
    out = x_exp / x_sum

    # Write result back to output row (coalesced store)
    tl.store(row_out + cols, out, mask=mask)


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    FlagGems optimized softmax.
    Supports fp16, bf16, fp32 inputs via internal fp32 accumulation.
    Fuses max-shift + exp + sum + normalize into single Triton kernel.
    Cross-hardware compatible: NVIDIA A100, H100, AMD MI300X.
    """
    orig_dtype = x.dtype  # save dtype: float16 or bfloat16 or float32
    orig_shape = x.shape

    # Cast to fp32 for precision — fixes fp16 and bf16 numerical errors
    x = x.contiguous().float()

    # Move softmax dim to last axis for coalesced memory access
    if dim not in (-1, len(orig_shape) - 1):
        x = x.transpose(dim, -1).contiguous()

    # Flatten to 2D: (num_rows, N)
    x2d = x.reshape(-1, x.shape[-1])
    num_rows, N = x2d.shape

    # Allocate output buffer in fp32
    out = torch.empty_like(x2d)

    # BLOCK_SIZE must cover full row — use next power of 2 >= N
    # This ensures mask covers all elements without partial blocks
    BLOCK_SIZE = triton.next_power_of_2(N)

    # Select num_warps based on block size for optimal occupancy
    # float16 and bfloat16 inputs benefit from more warps at large N
    if BLOCK_SIZE <= 256:
        num_warps = 2  # small rows: 2 warps sufficient
    elif BLOCK_SIZE <= 1024:
        num_warps = 4  # medium rows: 4 warps
    else:
        num_warps = 8  # large rows: 8 warps for fp16 bf16 workloads

    # Launch one program per row
    softmax_kernel[(num_rows,)](
        out,
        x2d,
        x2d.stride(0),
        out.stride(0),
        N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    # Restore original shape
    out = out.view(x.shape)
    if dim not in (-1, len(orig_shape) - 1):
        out = out.transpose(dim, -1).contiguous()

    # Cast back to original dtype: float16, bfloat16, or float32
    return out.to(orig_dtype)
