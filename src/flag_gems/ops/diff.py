import logging
from functools import reduce
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

Tensor = torch.Tensor


@libentry()
@triton.jit
def diff_kernel_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Compute diff along the inner (last) dimension.

    For each row m and output position n, computes:
    output[m, n] = input[m, n + 1] - input[m, n]

    Input shape: (M, N), Output shape: (M, N-1)
    """
    pid_m = tle.program_id(0)

    # Row indices this block handles
    row_start = pid_m * BLOCK_M
    row_offsets = row_start + tl.arange(0, BLOCK_M)
    row_mask = row_offsets < M

    # Output has N-1 elements per row
    output_N = N - 1

    # Process output elements in tiles
    for n_start in range(0, output_N, BLOCK_N):
        col_offsets = n_start + tl.arange(0, BLOCK_N)
        col_mask = col_offsets < output_N

        # Combined mask
        mask = row_mask[:, None] & col_mask[None, :]

        # Load input[m, n+1] and input[m, n]
        input_offsets_next = row_offsets[:, None] * N + (col_offsets[None, :] + 1)
        input_offsets_curr = row_offsets[:, None] * N + col_offsets[None, :]

        inp_next = tl.load(input_ptr + input_offsets_next, mask=mask, other=0.0)
        inp_curr = tl.load(input_ptr + input_offsets_curr, mask=mask, other=0.0)

        # Compute diff
        diff_val = inp_next - inp_curr

        # Store output
        output_offsets = row_offsets[:, None] * output_N + col_offsets[None, :]
        tl.store(output_ptr + output_offsets, diff_val, mask=mask)


@libentry()
@triton.jit
def diff_kernel_non_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute diff along a non-inner dimension.

    Input is viewed as (M, N, K) where we compute diff along dim 1 (size N).
    For each position (m, n, k), computes:
    output[m, n, k] = input[m, n + 1, k] - input[m, n, k]

    Input shape: (M, N, K), Output shape: (M, N-1, K)
    """
    pid_m = tle.program_id(0)
    pid_k = tle.program_id(1)

    # K indices this block handles
    k_start = pid_k * BLOCK_K
    k_offsets = k_start + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < K

    # Output has N-1 elements along dim 1
    output_N = N - 1

    # Process all n positions for this (m, k) block
    for n in range(output_N):
        # Load input[m, n+1, k] and input[m, n, k]
        input_offset_next = pid_m * N * K + (n + 1) * K + k_offsets
        input_offset_curr = pid_m * N * K + n * K + k_offsets

        inp_next = tl.load(input_ptr + input_offset_next, mask=k_mask, other=0.0)
        inp_curr = tl.load(input_ptr + input_offset_curr, mask=k_mask, other=0.0)

        # Compute diff
        diff_val = inp_next - inp_curr

        # Store output
        output_offset = pid_m * output_N * K + n * K + k_offsets
        tl.store(output_ptr + output_offset, diff_val, mask=k_mask)


def _diff_once(inp: Tensor, dim: int) -> Tensor:
    """Compute single forward difference along specified dimension.

    Args:
        inp: Input tensor (must be contiguous)
        dim: Dimension to compute difference along

    Returns:
        Tensor with shape reduced by 1 along dim
    """
    shape = list(inp.shape)
    ndim = inp.ndim
    dim = dim % ndim

    N = shape[dim]  # Size along diff dimension
    if N < 2:
        raise RuntimeError(
            f"diff requires at least 2 elements along dim {dim}, got {N}"
        )

    # Compute M (product of dims before dim) and K (product of dims after dim)
    M = reduce(lambda x, y: x * y, shape[:dim], 1)
    K = reduce(lambda x, y: x * y, shape[dim + 1 :], 1)

    # Output shape has dim reduced by 1
    out_shape = list(shape)
    out_shape[dim] = N - 1
    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    with torch_device_fn.device(inp.device):
        if K == 1:
            # Inner dimension case
            # Block sizes must be powers of 2 for triton
            BLOCK_M = triton.next_power_of_2(min(32, M))
            BLOCK_N = triton.next_power_of_2(min(256, N - 1))
            grid = (triton.cdiv(M, BLOCK_M),)
            diff_kernel_inner[grid](
                out,
                inp,
                M,
                N,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )
        else:
            # Non-inner dimension case
            BLOCK_K = triton.next_power_of_2(min(256, K))
            grid = (M, triton.cdiv(K, BLOCK_K))
            diff_kernel_non_inner[grid](
                out,
                inp,
                M,
                N,
                K,
                BLOCK_M=1,
                BLOCK_K=BLOCK_K,
            )

    return out


def diff(
    inp: Tensor,
    n: int = 1,
    dim: int = -1,
    prepend: Optional[Tensor] = None,
    append: Optional[Tensor] = None,
) -> Tensor:
    """Compute the n-th forward difference along the given dimension.

    The first-order differences are given by out[i] = input[i + 1] - input[i].
    Higher-order differences are calculated by using diff recursively.

    Args:
        inp: Input tensor
        n: Number of times to recursively compute the difference
        dim: Dimension to compute the difference along (default: -1)
        prepend: Values to prepend to input along dim before computing diff
        append: Values to append to input along dim before computing diff

    Returns:
        Tensor containing the n-th order differences
    """
    logger.debug("GEMS DIFF")

    if n == 0:
        return inp.clone()

    if n < 0:
        raise RuntimeError(f"diff expects n >= 0, got {n}")

    ndim = inp.ndim
    if ndim == 0:
        raise RuntimeError("diff requires input to be at least one-dimensional")

    dim = dim % ndim

    # Handle prepend and append by concatenating
    tensors_to_cat = []
    if prepend is not None:
        tensors_to_cat.append(prepend)
    tensors_to_cat.append(inp)
    if append is not None:
        tensors_to_cat.append(append)

    if len(tensors_to_cat) > 1:
        inp = torch.cat(tensors_to_cat, dim=dim)

    inp = inp.contiguous()

    # Apply diff n times
    result = inp
    for _ in range(n):
        if result.shape[dim] < 2:
            raise RuntimeError(
                f"diff requires at least 2 elements along dim {dim} for each iteration"
            )
        result = _diff_once(result, dim)

    return result
