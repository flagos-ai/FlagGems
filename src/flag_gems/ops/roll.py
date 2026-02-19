import logging

import torch

logger = logging.getLogger(__name__)


def _normalize_shifts_dims(shifts, dims, ndim):
    if dims is None:
        if isinstance(shifts, (list, tuple)):
            if len(shifts) != 1:
                raise RuntimeError("shifts and dimensions must align for roll.")
            shifts = shifts[0]
        return (int(shifts),), (0,)

    if isinstance(dims, int):
        dims_tuple = (dims,)
    else:
        dims_tuple = tuple(int(d) for d in dims)

    if isinstance(shifts, int):
        shifts_tuple = (int(shifts),) * len(dims_tuple)
    else:
        shifts_tuple = tuple(int(s) for s in shifts)

    if len(shifts_tuple) != len(dims_tuple):
        raise RuntimeError("shifts and dimensions must align for roll.")

    dims_tuple = tuple(d % ndim for d in dims_tuple)
    if len(set(dims_tuple)) != len(dims_tuple):
        raise RuntimeError("dims cannot contain duplicate values.")

    return shifts_tuple, dims_tuple


def _roll_dim(x: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
    size = x.size(dim)
    if size == 0:
        return x
    shift = shift % size
    if shift == 0:
        return x
    left = x.narrow(dim, size - shift, shift)
    right = x.narrow(dim, 0, size - shift)
    return torch.cat((left, right), dim=dim)


def roll(self, shifts, dims=None):
    logger.debug("GEMS ROLL")
    shifts_tuple, dims_tuple = _normalize_shifts_dims(shifts, dims, self.dim())

    if dims is None:
        flat = self.reshape(-1)
        rolled = _roll_dim(flat, shifts_tuple[0], 0)
        return rolled.reshape(self.shape)

    out = self
    for shift, dim in zip(shifts_tuple, dims_tuple):
        out = _roll_dim(out, shift, dim)
    return out
