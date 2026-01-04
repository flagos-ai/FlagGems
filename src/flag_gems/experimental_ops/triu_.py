import torch
import triton
import triton.language as tl


@triton.jit
def triu_(
    x_ptr,
    base_offsets_ptr,
    diagonal,
    M,
    N,
    stride_m,
    stride_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    pid_b = tl.program_id(axis=2)

    # Compute row/col indices this program will handle
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Masks for bounds
    mask_rows = rows < M
    mask_cols = cols < N
    mask = mask_rows[:, None] & mask_cols[None, :]

    # Load base offset for this batch
    off_b = tl.load(base_offsets_ptr + pid_b)

    # Compute pointers for the tile
    rows_2d = rows[:, None]
    cols_2d = cols[None, :]
    ptrs = x_ptr + off_b + rows_2d * stride_m + cols_2d * stride_n

    # Compute triangular mask: keep if (col - row) >= diagonal
    keep = (cols_2d - rows_2d) >= diagonal

    # Load, apply mask, and store
    vals = tl.load(ptrs, mask=mask, other=0)
    out = tl.where(keep, vals, 0)
    tl.store(ptrs, out, mask=mask)


# Preserve kernel object before defining the wrapper with the same name
triu___kernel = triu_


def triu_(*args, **kwargs):
    # Parse inputs similar to torch.ops.aten.triu_(input, diagonal=0)
    if len(args) == 0:
        x = kwargs.get("input", kwargs.get("self", None))
        if x is None:
            raise ValueError("triu_ expects a tensor as the first argument")
        diagonal = kwargs.get("diagonal", 0)
    else:
        x = args[0]
        if len(args) > 1:
            diagonal = args[1]
        else:
            diagonal = kwargs.get("diagonal", 0)

    if not isinstance(x, torch.Tensor):
        raise TypeError("triu_ expects a torch.Tensor as input")
    if not x.is_cuda:
        raise ValueError("triu_ expects a CUDA tensor")

    # No-op for empty tensors
    if x.numel() == 0:
        return x
    # Require at least 2D to form a matrix; mimic PyTorch behavior by early return for degenerate case
    if x.ndim < 2:
        return x

    device = x.device
    ndim = x.ndim
    M = x.size(-2)
    N = x.size(-1)
    if M == 0 or N == 0:
        return x

    # Compute base offsets for each batch instance (in elements)
    batch_ndim = max(0, ndim - 2)
    if batch_ndim == 0:
        base_offsets = torch.tensor(
            [x.storage_offset()], dtype=torch.int64, device=device
        )
    else:
        batch_sizes = list(x.shape[:batch_ndim])
        batch_strides = list(x.stride()[:batch_ndim])
        total_batches = 1
        for s in batch_sizes:
            total_batches *= s
        base_offsets = torch.empty(total_batches, dtype=torch.int64, device=device)
        base = x.storage_offset()
        # Convert linear batch id to multi-index and compute base offset
        for b in range(total_batches):
            rem = b
            off = base
            for d in range(batch_ndim - 1, -1, -1):
                size_d = batch_sizes[d]
                if size_d > 0:
                    idx_d = rem % size_d
                    rem //= size_d
                else:
                    idx_d = 0
                off += idx_d * batch_strides[d]
            base_offsets[b] = off

    stride_m = x.stride(-2)
    stride_n = x.stride(-1)

    # Launch kernel
    BLOCK_M = 32
    BLOCK_N = 32
    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
        base_offsets.numel(),
    )

    triu___kernel[grid](
        x,
        base_offsets,
        int(diagonal),
        M,
        N,
        stride_m,
        stride_n,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    return x
