import torch
import triton
import triton.language as tl


@triton.jit
def _safe_softmax(input_ptr, output_ptr, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    row_offset = row_id * n_cols
    x = tl.load(input_ptr + row_offset + cols, mask=mask, other=-float("inf"))
    x_fp32 = x.to(tl.float32)

    x_max = tl.max(x_fp32, axis=0)
    all_neginf = x_max == -float("inf")

    x_shifted = x_fp32 - x_max
    exp_x = tl.exp(x_shifted)
    sum_exp = tl.sum(exp_x, axis=0)
    softmax = exp_x / sum_exp

    softmax = tl.where(all_neginf, tl.zeros([BLOCK_SIZE], dtype=tl.float32), softmax)

    tl.store(output_ptr + row_offset + cols, softmax, mask=mask)


# Preserve kernel handle before defining wrapper with the same name
_safe_softmax_kernel = _safe_softmax


def _safe_softmax(x: torch.Tensor, dim: int = -1, dtype: torch.dtype = None):
    assert x.is_cuda, "Input tensor must be on CUDA device"
    assert x.ndim >= 1, "Input tensor must have at least 1 dimension"

    dim = dim if dim >= 0 else x.ndim + dim
    assert 0 <= dim < x.ndim, "Invalid dim for softmax"

    if dim != x.ndim - 1:
        perm = list(range(x.ndim))
        perm[dim], perm[-1] = perm[-1], perm[dim]
        y = x.permute(perm).contiguous()
        inv_perm = [0] * x.ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i
    else:
        y = x.contiguous()
        inv_perm = None

    n_cols = y.shape[-1]
    n_rows = y.numel() // n_cols

    y_fp32 = y.float()
    out_fp32 = torch.empty_like(y_fp32)

    def _next_pow2(v: int) -> int:
        if v <= 1:
            return 1
        return 1 << (v - 1).bit_length()

    BLOCK_SIZE = min(4096, _next_pow2(n_cols))
    grid = lambda meta: (n_rows,)

    _safe_softmax_kernel[grid](y_fp32, out_fp32, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)

    out = out_fp32
    if dtype is not None:
        out = out.to(dtype)
    else:
        out = out.to(x.dtype)

    out = out.view(*y.shape)
    if inv_perm is not None:
        out = out.permute(inv_perm)

    return out
