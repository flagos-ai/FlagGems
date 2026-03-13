import logging
from typing import Sequence, Union

import torch

logger = logging.getLogger(__name__)


def _normalize_shifts_dims(
    inp: torch.Tensor,
    shifts: Union[int, Sequence[int]],
    dims: Union[None, int, Sequence[int]],
) -> tuple:
    """Normalize shifts and dims to lists, handling the flatten case."""
    if isinstance(shifts, int):
        shifts = [shifts]
    else:
        shifts = list(shifts)

    if dims is None:
        # Flatten case: roll on flattened tensor
        return shifts, None

    if isinstance(dims, int):
        dims = [dims]
    else:
        dims = list(dims)

    if len(shifts) != len(dims):
        raise RuntimeError(
            f"shifts and dims must have the same size, got shifts={len(shifts)} and dims={len(dims)}"
        )

    # Normalize negative dims
    ndim = inp.ndim
    normalized_dims = []
    for d in dims:
        if d < -ndim or d >= ndim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of [{-ndim}, {ndim - 1}], but got {d})"
            )
        normalized_dims.append(d % ndim)

    return shifts, normalized_dims


def _roll_single_dim(inp: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
    """Roll along a single dimension using slicing and concatenation."""
    size = inp.shape[dim]
    if size == 0:
        return inp

    # Normalize shift to [0, size)
    shift = shift % size
    if shift == 0:
        return inp

    # torch.roll with shift=k along dim means element at position i goes to position (i+k)%size
    # So output[i] = input[(i - k) % size]
    # Using narrow: we want output = cat([input[size-shift:], input[:size-shift]], dim)
    first_part = inp.narrow(dim, size - shift, shift)
    second_part = inp.narrow(dim, 0, size - shift)
    return torch.cat([first_part, second_part], dim=dim)


def roll(
    inp: torch.Tensor,
    shifts: Union[int, Sequence[int]],
    dims: Union[None, int, Sequence[int]] = None,
) -> torch.Tensor:
    logger.debug("GEMS ROLL")

    shifts_list, dims_list = _normalize_shifts_dims(inp, shifts, dims)

    # Handle empty tensor
    if inp.numel() == 0:
        return inp.clone()

    # Handle flatten case (dims is None)
    if dims_list is None:
        # Flatten, roll along dim 0, reshape back
        original_shape = inp.shape
        flat = inp.contiguous().flatten()
        shift = shifts_list[0] if shifts_list else 0

        if flat.numel() == 0:
            return inp.clone()

        # Normalize shift
        shift = shift % flat.numel()
        if shift == 0:
            return inp.clone()

        # Roll the flattened tensor
        rolled_flat = _roll_single_dim(flat, shift, 0)
        return rolled_flat.view(original_shape)

    # Handle no-op case
    if not dims_list:
        return inp.clone()

    # Apply rolls for each dimension
    result = inp
    for shift, dim in zip(shifts_list, dims_list):
        result = _roll_single_dim(result, shift, dim)

    return result
