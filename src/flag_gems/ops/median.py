"""
FlagGems median operator — Triton implementation.

torch.median(input) -> Tensor
torch.median(input, dim, keepdim) -> (values, indices)
"""
import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)
_MedianResult = torch.return_types.median

_MAX_BLOCK = 4096


@libentry()
@triton.jit
def _median_kernel(
    X,
    VAL_OUT,
    IDX_OUT,
    N,
    stride_batch,
    BLOCK_SIZE: tl.constexpr,
    WRITE_IDX: tl.constexpr,
):
    pid = ext.program_id(0)
    x_row = X + pid * stride_batch
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x_val = tl.load(x_row + offs, mask=mask, other=float("inf")).to(tl.float32)
    sorted_val = tl.sort(x_val, dim=0, descending=False)

    k = (N - 1) // 2
    k_mask = offs == k
    med_val = tl.sum(
        tl.where(k_mask, sorted_val, tl.zeros([BLOCK_SIZE], dtype=tl.float32)),
        axis=0,
    )
    tl.store(VAL_OUT + pid, med_val)

    if WRITE_IDX:
        diff = tl.abs(x_val - med_val)
        diff = tl.where(mask, diff, float("inf"))
        orig_idx = tl.argmin(diff, axis=0)
        tl.store(IDX_OUT + pid, orig_idx.to(tl.int64))


def _block_size_for(n):
    return min(_MAX_BLOCK, triton.next_power_of_2(max(n, 1)))


def _run_triton(x_2d, N, write_idx, orig_dtype):
    batch = x_2d.shape[0]
    BLOCK = _block_size_for(N)
    out_val = torch.empty(batch, device=x_2d.device, dtype=torch.float32)
    out_idx = torch.empty(batch, device=x_2d.device, dtype=torch.int64)

    if N <= _MAX_BLOCK:
        with torch_device_fn.device(x_2d.device):
            _median_kernel[(batch,)](
                x_2d,
                out_val,
                out_idx,
                N,
                x_2d.stride(0),
                BLOCK_SIZE=BLOCK,
                WRITE_IDX=write_idx,
            )
    else:
        sv, si = torch.sort(x_2d, dim=1, stable=True)
        k = (N - 1) // 2
        out_val = sv[:, k]
        out_idx = si[:, k].to(torch.int64)

    return out_val.to(orig_dtype), out_idx


def median(input: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS MEDIAN")
    if not input.is_cuda:
        raise RuntimeError("FlagGems median: CUDA tensor required")
    if input.numel() == 0:
        raise RuntimeError("median() input must be non-empty")

    orig_dtype = input.dtype
    N = input.numel()
    x_2d = input.contiguous().reshape(1, N).to(torch.float32)
    vals, _ = _run_triton(x_2d, N, False, orig_dtype)
    return vals[0]


def median_out(input: torch.Tensor, *, out: torch.Tensor) -> torch.Tensor:
    result = median(input)
    out.resize_(result.shape)
    out.copy_(result)
    return out


def median_dim(
    input: torch.Tensor,
    dim: int,
    keepdim: bool = False,
) -> "torch.return_types.median":
    logger.debug("GEMS MEDIAN DIM")
    if not input.is_cuda:
        raise RuntimeError("FlagGems median: CUDA tensor required")

    ndim = input.ndim
    if ndim == 0:
        return _MedianResult(
            (input.clone(), torch.zeros([], device=input.device, dtype=torch.int64))
        )

    if dim < 0:
        dim = dim + ndim
    if not (0 <= dim < ndim):
        raise IndexError(f"dim {dim} out of range [{-ndim}, {ndim - 1}]")

    N = input.shape[dim]
    if N == 0:
        raise RuntimeError("median() cannot reduce zero-size dimension")

    orig_dtype = input.dtype
    x = input.movedim(dim, -1).contiguous().to(torch.float32)
    batch_shape = x.shape[:-1]
    batch = math.prod(batch_shape) if batch_shape else 1
    x_2d = x.reshape(batch, N)

    vals_flat, idx_flat = _run_triton(x_2d, N, True, orig_dtype)

    values = vals_flat.reshape(batch_shape)
    indices = idx_flat.reshape(batch_shape)

    if keepdim:
        values = values.unsqueeze(dim)
        indices = indices.unsqueeze(dim)

    return _MedianResult((values, indices))


def median_dim_values(
    input: torch.Tensor,
    dim: int,
    keepdim: bool = False,
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    result = median_dim(input, dim, keepdim)
    out.resize_(result.values.shape)
    out.copy_(result.values)
    return out
