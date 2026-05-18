"""
FlagGems median operator.
Uses CPU-side stable sort for deterministic tie-breaking matching PyTorch's behavior.
"""
import logging

import torch

logger = logging.getLogger(__name__)
_MedianResult = torch.return_types.median


def median(input: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS MEDIAN")
    if not input.is_cuda:
        raise RuntimeError("FlagGems median: CUDA tensor required")
    if input.numel() == 0:
        raise RuntimeError("median() input must be non-empty")
    flat = input.contiguous().reshape(-1)
    k = (flat.numel() - 1) // 2
    return torch.sort(flat, stable=True)[0][k]


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

    k = (N - 1) // 2

    # CPU stable sort for deterministic tie-breaking
    sv, si = torch.sort(input, dim=dim, stable=True)
    med_vals = sv.select(dim, k).to(input.device)
    med_idxs = si.select(dim, k).to(input.device, dtype=torch.int64)

    if keepdim:
        med_vals = med_vals.unsqueeze(dim)
        med_idxs = med_idxs.unsqueeze(dim)

    return _MedianResult((med_vals, med_idxs))


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
