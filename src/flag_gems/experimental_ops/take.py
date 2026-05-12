import torch
import triton
import triton.language as tl


@triton.jit
def take_kernel(
    in_ptr,  # pointer to input flattened tensor
    idx_ptr,  # pointer to flattened indices (int32)
    out_ptr,  # pointer to flattened output tensor
    n_index,  # number of indices (int32)
    in_numel,  # number of elements in input (int32)
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_index

    idx = tl.load(idx_ptr + offs, mask=mask, other=0).to(tl.int32)

    # Bounds check to avoid OOB memory access; assumes valid indices in normal use.
    in_range = (idx >= 0) & (idx < in_numel) & mask
    idx_safe = tl.maximum(0, tl.minimum(idx, in_numel - 1))

    vals = tl.load(in_ptr + idx_safe, mask=mask, other=0)
    # Zero out values for invalid indices (shouldn't happen if inputs are valid)
    vals = tl.where(in_range, vals, 0)
    tl.store(out_ptr + offs, vals, mask=mask)


def _launch_take(input: torch.Tensor, index: torch.Tensor, out_flat: torch.Tensor):
    assert (
        input.is_cuda and index.is_cuda and out_flat.is_cuda
    ), "All tensors must be CUDA tensors"
    # Flatten input as per torch.take semantics (use contiguous flattened memory)
    input_flat = input.contiguous().view(-1)
    # Indices flattened and converted to int32 for kernel
    index_flat = index.contiguous().view(-1).to(torch.int32)
    n_index = index_flat.numel()
    if n_index == 0:
        return
    grid = lambda meta: (triton.cdiv(n_index, meta["BLOCK_SIZE"]),)
    take_kernel[grid](
        input_flat,
        index_flat,
        out_flat,
        n_index,
        input_flat.numel(),
        BLOCK_SIZE=1024,
    )


def take(input: torch.Tensor, index: torch.Tensor):
    """
    Wrapper for aten::take
    Returns a 1-D tensor with elements of input at the given flat indices in index.
    """
    assert input.device == index.device, "input and index must be on the same device"
    out_flat = torch.empty(index.numel(), device=input.device, dtype=input.dtype)
    _launch_take(input, index, out_flat)
    return out_flat.view(index.shape)


def take_out(input: torch.Tensor, index: torch.Tensor, out: torch.Tensor):
    """
    Wrapper for aten::take.out
    Writes result into 'out' and returns it.
    """
    assert (
        input.device == index.device == out.device
    ), "All tensors must be on the same device"
    # Ensure output has correct dtype and shape; resize if needed
    if out.dtype != input.dtype:
        raise TypeError(
            f"out dtype {out.dtype} does not match input dtype {input.dtype}"
        )
    if out.numel() != index.numel() or tuple(out.shape) != tuple(index.shape):
        out.resize_(index.shape)

    # Use a temporary contiguous flat buffer to ensure correctness even if 'out' is non-contiguous
    tmp_flat = torch.empty(index.numel(), device=input.device, dtype=input.dtype)
    _launch_take(input, index, tmp_flat)
    out.copy_(tmp_flat.view(index.shape))
    return out
