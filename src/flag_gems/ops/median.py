import logging
from collections import namedtuple

import torch

logger = logging.getLogger(__name__)

_median_named = namedtuple("median", ["values", "indices"])


def _median_impl_name(device: torch.device) -> str:
    t = device.type
    if t == "cuda":
        return "median_cuda"
    if t == "mps":
        return "median_mps"
    return "median_cpu"


def median_dim(inp: torch.Tensor, dim: int, keepdim: bool = False):
    """``aten::median.dim``: reduce along ``dim`` (values + indices), PyTorch-compatible."""
    logger.debug("GEMS MEDIAN DIM")

    nd = inp.ndim
    if nd == 0:
        if dim not in (0, -1):
            raise IndexError(
                f"Dimension out of range (expected to be in range of [-1, 0], but got {dim})"
            )
        dim_actual = 0
        dim_sz = 1
    else:
        if dim < -nd or dim >= nd:
            raise IndexError(
                "Dimension out of range (expected to be in range of "
                f"[{-nd}, {nd - 1}], but got {dim})"
            )
        dim_actual = dim % nd
        dim_sz = inp.shape[dim_actual]
        if dim_sz == 0:
            raise IndexError(
                "median(): Expected reduction dim "
                f"{dim_actual} to have non-zero size."
            )

    if inp.dtype == torch.bool:
        raise RuntimeError(f'"{_median_impl_name(inp.device)}" not implemented for \'Bool\'')

    if nd > 0 and inp.dtype.is_floating_point and torch.isnan(inp).any():
        vv, ix = torch.median(inp.detach().cpu(), dim=dim, keepdim=keepdim)
        return _median_named(
            values=vv.to(inp.device, dtype=inp.dtype),
            indices=ix.to(inp.device, dtype=torch.int64),
        )

    sorted_vals, sorted_idx = torch.sort(
        inp,
        dim=dim_actual,
        descending=False,
        stable=False,
    )
    k = (dim_sz - 1) // 2
    values = sorted_vals.select(dim_actual, k)
    indices = sorted_idx.select(dim_actual, k)

    if keepdim:
        values = values.unsqueeze(dim_actual)
        indices = indices.unsqueeze(dim_actual)

    return _median_named(values=values, indices=indices)


def median_dim_values(inp, dim, keepdim=False, *, values, indices):
    v, ix = median_dim(inp, dim, keepdim)
    values.copy_(v)
    indices.copy_(ix)
    return values, indices
