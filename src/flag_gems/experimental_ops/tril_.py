import torch
import triton
import triton.language as tl


@triton.jit
def tril___kernel(
    x_ptr,  # *Pointer* to input tensor data (in-place)
    offsets_ptr,  # *Pointer* to base offsets per batch
    stride_m,  # stride for the row dimension (elements)
    stride_n,  # stride for the col dimension (elements)
    M,  # number of rows
    N,  # number of cols
    B,  # number of batch matrices
    diagonal,  # diagonal offset (int)
    BLOCK_M: tl.constexpr,  # block size along M
    BLOCK_N: tl.constexpr,  # block size along N
):
    pid_n = tl.program_id(axis=0)  # tile id along N
    pid_m = tl.program_id(axis=1)  # tile id along M
    pid_b = tl.program_id(axis=2)  # batch id

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = rows < M
    mask_n = cols < N
    mask = mask_m[:, None] & mask_n[None, :]

    # Load base offset for this batch
    base = tl.load(offsets_ptr + pid_b, mask=True, other=0).to(tl.int64)

    # Compute pointers for the tile
    row_offsets = rows.to(tl.int64)[:, None] * stride_m
    col_offsets = cols.to(tl.int64)[None, :] * stride_n
    ptrs = x_ptr + base + row_offsets + col_offsets

    # Load current values
    vals = tl.load(ptrs, mask=mask, other=0)

    # Compute keep mask for lower triangle: keep if (col - row) <= diagonal
    # Cast to int32 for the comparison
    r = rows[:, None].to(tl.int32)
    c = cols[None, :].to(tl.int32)
    k = tl.full((), diagonal, tl.int32)
    keep = (c - r) <= k

    # Zero out elements above the diagonal
    out = tl.where(keep, vals, 0)

    # Store back in-place
    tl.store(ptrs, out, mask=mask)


def tril_(*args, **kwargs):
    # Parse arguments similar to torch.ops.aten.tril_(x, diagonal=0)
    if len(args) == 0 and "input" not in kwargs and "self" not in kwargs:
        raise ValueError("tril_ expects at least a tensor argument")
    x = None
    if len(args) >= 1:
        x = args[0]
    else:
        x = kwargs.get("input", kwargs.get("self", None))
    if x is None:
        raise ValueError("tril_ could not find the input tensor argument")

    # Diagonal argument
    if "diagonal" in kwargs:
        diagonal = int(kwargs["diagonal"])
    elif len(args) >= 2:
        diagonal = int(args[1])
    else:
        diagonal = 0

    # Handle trivial cases or non-CUDA fallback
    if not x.is_cuda:
        # Fallback to PyTorch's implementation
        return torch.tril_(x, diagonal=diagonal)
    if x.ndim < 2 or x.numel() == 0:
        return x

    # Shapes and sizes
    M = x.size(-2)
    N = x.size(-1)
    batch_shape = x.shape[:-2]
    B = 1
    for s in batch_shape:
        B *= s

    # If there are zero batch matrices, nothing to do
    if B == 0 or M == 0 or N == 0:
        return x

    # Compute per-batch base offsets (in elements) from original strides
    device = x.device
    dtype = x.dtype
    strides = x.stride()
    stride_m = int(x.stride(-2))
    stride_n = int(x.stride(-1))

    # Build offsets for each batch index
    offsets = torch.zeros(B, dtype=torch.int64, device=device)
    if B > 1:
        sizes = list(batch_shape)
        batch_strides = list(strides[:-2])
        tmp = torch.arange(B, device=device, dtype=torch.int64)
        for d in range(len(sizes) - 1, -1, -1):
            sz = sizes[d]
            if sz == 0:
                # Shouldn't happen since B==0 would have returned early
                pass
            idx_d = tmp % sz
            offsets += idx_d * batch_strides[d]
            tmp = tmp // sz

    # Launch kernel
    BLOCK_M = 32
    BLOCK_N = 64
    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(M, BLOCK_M), B)

    tril___kernel[grid](
        x,  # x_ptr
        offsets,  # offsets_ptr
        stride_m,  # stride_m
        stride_n,  # stride_n
        M,  # M
        N,  # N
        B,  # B
        int(diagonal),  # diagonal
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    return x
