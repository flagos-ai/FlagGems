from typing import Sequence

import torch
import triton
import triton.language as tl


@triton.jit
def _stack_copy_kernel(
    src_ptr,  # *Pointer* to source tensor (flattened, contiguous)
    out_ptr,  # *Pointer* to output tensor (flattened, contiguous)
    numel_in,  # Number of elements in one input tensor (P * Q)
    Q,  # Inner product size
    K,  # Number of tensors to stack
    k_idx,  # Which tensor index [0..K-1] this launch is copying
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel_in

    # Map input flat index -> output flat index
    # Given pos in [0, P*Q):
    #   outer = pos // Q
    #   inner = pos % Q
    # Output flat index = outer * (K*Q) + k_idx * Q + inner
    outer = offsets // Q
    inner = offsets % Q
    out_index = outer * (K * Q) + k_idx * Q + inner

    vals = tl.load(src_ptr + offsets, mask=mask)
    tl.store(out_ptr + out_index, vals, mask=mask)


def _prepare_stack_shapes(tensors: Sequence[torch.Tensor], dim: int):
    if len(tensors) == 0:
        raise ValueError("stack() expects a non-empty sequence of tensors")

    # Ensure all tensors are on the same device, dtype, and have same shape
    ref = tensors[0]
    device = ref.device
    dtype = ref.dtype
    shape = tuple(ref.shape)
    for t in tensors:
        if t.device != device:
            raise ValueError("All tensors must be on the same device")
        if t.dtype != dtype:
            raise ValueError("All tensors must have the same dtype")
        if tuple(t.shape) != shape:
            raise ValueError("All tensors must have the same shape")

    D = len(shape)
    dim = dim if dim >= 0 else dim + (D + 1)
    if not (0 <= dim <= D):
        raise ValueError(
            f"dim out of range (expected to be in range [{-D-1}, {D}], but got {dim})"
        )

    # Compute P and Q
    P = 1
    for i in range(0, dim):
        P *= shape[i] if D > 0 else 1
    Q = 1
    for i in range(dim, D):
        Q *= shape[i] if D > 0 else 1

    K = len(tensors)
    out_shape = list(shape[:dim]) + [K] + list(shape[dim:])
    return device, dtype, shape, out_shape, P, Q, K, dim


def _launch_stack_kernels(
    tensors: Sequence[torch.Tensor], out: torch.Tensor, P: int, Q: int, K: int
):
    # Input must be contiguous; output must be contiguous
    if not out.is_contiguous():
        raise ValueError("Output tensor must be contiguous")

    # Number of elements per input tensor
    numel_in = P * Q
    if numel_in == 0:
        return

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(numel_in, meta["BLOCK_SIZE"]),)

    out_flat = out.view(-1)
    for k_idx, t in enumerate(tensors):
        src = t.contiguous().view(-1)
        # Launch Triton kernel to copy src into out at slice index k_idx along stacking dim
        _stack_copy_kernel[grid](
            src,
            out_flat,
            numel_in,
            Q,
            K,
            k_idx,
            BLOCK_SIZE=BLOCK_SIZE,
        )


def stack(tensors: Sequence[torch.Tensor], dim: int = 0):
    device, dtype, in_shape, out_shape, P, Q, K, dim = _prepare_stack_shapes(
        tensors, dim
    )
    out = torch.empty(out_shape, device=device, dtype=dtype)
    _launch_stack_kernels(tensors, out, P, Q, K)
    return out


def stack_out(tensors: Sequence[torch.Tensor], dim: int = 0, out: torch.Tensor = None):
    if out is None:
        raise ValueError("stack_out requires an 'out' tensor")
    device, dtype, in_shape, out_shape, P, Q, K, dim = _prepare_stack_shapes(
        tensors, dim
    )

    if out.device != device:
        raise ValueError("Output tensor device must match input tensors' device")
    if out.dtype != dtype:
        raise ValueError("Output tensor dtype must match input tensors' dtype")
    if list(out.shape) != list(out_shape):
        raise ValueError(
            f"Output tensor has incorrect shape. Expected {out_shape}, got {list(out.shape)}"
        )

    _launch_stack_kernels(tensors, out, P, Q, K)
    return out
