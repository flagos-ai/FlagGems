import torch
import triton
import triton.language as tl


@triton.jit
def eye_kernel(
    out_ptr,  # *Pointer* to output 2D tensor
    n_rows,  # number of rows (n)
    n_cols,  # number of cols (m)
    stride_row,  # stride for row dimension
    stride_col,  # stride for col dimension
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    row_idx = offs_m[:, None]
    col_idx = offs_n[None, :]

    in_bounds = (row_idx < n_rows) & (col_idx < n_cols)
    is_diag = row_idx == col_idx

    # Produce 1 on diagonal, 0 elsewhere; Triton will cast to the pointer dtype on store
    vals = tl.where(is_diag, 1, 0)
    ptrs = out_ptr + row_idx * stride_row + col_idx * stride_col
    tl.store(ptrs, vals, mask=in_bounds)


# Shared implementation
def _eye_impl(n, m=None, dtype=None, device=None, out: torch.Tensor = None):
    if m is None:
        m = n

    if out is None:
        if dtype is None:
            dtype = torch.get_default_dtype()
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        out = torch.empty((n, m), dtype=dtype, device=device)
    else:
        if out.dim() != 2:
            raise ValueError("out tensor must be 2D")
        # Resize to expected shape if necessary
        if out.shape[0] != n or out.shape[1] != m:
            out.resize_(n, m)

    # Handle empty tensors
    if n == 0 or m == 0:
        out.zero_()
        return out

    # CUDA path uses Triton
    if out.is_cuda:
        BLOCK_M = 64
        BLOCK_N = 64
        grid = (triton.cdiv(n, BLOCK_M), triton.cdiv(m, BLOCK_N))
        eye_kernel[grid](
            out,
            n,
            m,
            out.stride(0),
            out.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )
        return out

    # CPU fallback without calling torch.eye
    out.zero_()
    k = min(n, m)
    if k > 0:
        idx = torch.arange(k, device=out.device)
        one = torch.ones(k, dtype=out.dtype, device=out.device)
        out[idx, idx] = one
    return out


# Wrappers for ATen operator interfaces


def eye(n, m=None, dtype=None, device=None):
    return _eye_impl(n, m, dtype, device, out=None)


def eye_m(n, m, dtype=None, device=None):
    return _eye_impl(n, m, dtype, device, out=None)


def eye_out(n, out: torch.Tensor):
    # eye.out expects shape (n, n)
    return _eye_impl(n, n, dtype=out.dtype, device=out.device, out=out)


def eye_m_out(n, m, out: torch.Tensor):
    # eye.m_out expects shape (n, m)
    return _eye_impl(n, m, dtype=out.dtype, device=out.device, out=out)
