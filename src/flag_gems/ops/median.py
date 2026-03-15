import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def median_dim_kernel(
    inp_ptr,
    val_ptr,
    idx_ptr,
    M,
    N,
    stride_m,
    stride_n,
    BLOCK_N: tl.constexpr,
):
    """Compute median along the last dimension of a 2D (M, N) view.

    Uses a simple selection approach: for each row, count how many elements
    are less than each element to find the median position.
    """
    pid = tle.program_id(0)
    if pid >= M:
        return

    row_start = pid * stride_m
    k = N // 2  # median index (lower median for even N)

    # Load entire row
    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < N
    row = tl.load(inp_ptr + row_start + offsets * stride_n, mask=mask, other=float("inf"))

    # Simple O(N^2) selection: for each position, count elements less than it
    # and elements equal but at lower index
    best_val = tl.load(inp_ptr + row_start)  # initialize
    best_idx = tl.zeros([], dtype=tl.int64)

    for candidate in range(N):
        if candidate < BLOCK_N:
            cand_val_vec = tl.load(
                inp_ptr + row_start + candidate * stride_n
            )

            # Count elements strictly less than candidate
            count_less = tl.sum(tl.where(mask & (row < cand_val_vec), 1, 0))
            # Count elements equal to candidate with index < candidate
            equal_mask = mask & (row == cand_val_vec) & (offsets < candidate)
            count_equal_before = tl.sum(tl.where(equal_mask, 1, 0))

            rank = count_less + count_equal_before

            if rank == k:
                best_val = cand_val_vec
                best_idx = tl.full([], candidate, dtype=tl.int64)

    tl.store(val_ptr + pid, best_val)
    tl.store(idx_ptr + pid, best_idx)


def median(inp):
    logger.debug("GEMS MEDIAN")
    # Global median: flatten and find median
    flat = inp.contiguous().view(-1)
    N = flat.numel()
    if N == 0:
        return flat.new_empty([])

    sorted_vals, _ = torch.sort(flat)
    return sorted_vals[N // 2].clone()


def median_dim(inp, dim, keepdim=False):
    logger.debug("GEMS MEDIAN DIM")

    dim = dim % inp.ndim

    # Transpose target dim to last, then flatten leading dims
    perm = list(range(inp.ndim))
    perm[dim], perm[-1] = perm[-1], perm[dim]
    inp_t = inp.permute(perm).contiguous()

    shape = inp_t.shape
    M = 1
    for s in shape[:-1]:
        M *= s
    N = shape[-1]

    if N == 0:
        out_shape = list(inp.shape)
        out_shape[dim] = 0
        values = inp.new_empty(out_shape)
        indices = torch.empty(out_shape, dtype=torch.int64, device=inp.device)
        if keepdim:
            return values, indices
        return values.squeeze(dim), indices.squeeze(dim)

    inp_flat = inp_t.view(M, N)

    values = torch.empty(M, dtype=inp.dtype, device=inp.device)
    indices = torch.empty(M, dtype=torch.int64, device=inp.device)

    BLOCK_N = triton.next_power_of_2(N)

    with torch_device_fn.device(inp.device):
        median_dim_kernel[(M,)](
            inp_flat, values, indices, M, N, N, 1, BLOCK_N
        )

    # Reshape output
    out_shape = list(shape[:-1])
    values = values.view(out_shape)
    indices = indices.view(out_shape)

    # Undo the permutation (without the last dim which was reduced)
    inv_perm = [0] * (inp.ndim - 1)
    out_perm = perm[:-1]
    for i, p in enumerate(out_perm):
        if p < inp.ndim - 1:
            inv_perm[p] = i
        else:
            inv_perm[dim] = i

    values = values.permute(inv_perm).contiguous()
    indices = indices.permute(inv_perm).contiguous()

    if keepdim:
        values = values.unsqueeze(dim)
        indices = indices.unsqueeze(dim)

    return values, indices
