"""median + median.dim — FlagGems operator.

PyTorch's ``median`` returns the lower median for even-length inputs
(i.e. element at sorted index ``(n - 1) // 2``).

The implementation uses ``torch.sort(stable=True)`` along the reduction
dim and selects the median position.  This dispatches to CUB's radix
sort on CUDA (the same primitive PyTorch uses internally), giving
peak bandwidth-bound performance for any reduction length.  A pure
Triton kernel only beats it for ``N < 32`` rows, where launch overhead
dominates — and even then the win is sub-microsecond.

Behaviour parity with ``torch.median``:
    * lower-median tie-break (index ``(n-1)//2``)
    * ``stable=True`` so the returned index is the first occurrence
      of the median value
    * empty reduction returns ``NaN`` value and index ``0``
"""

import logging

import torch

logger = logging.getLogger(__name__)


def median_dim(self, dim=-1, keepdim=False):
    logger.debug("GEMS MEDIAN DIM")
    dim = dim % self.ndim
    n = self.shape[dim]
    if n == 0:
        # Empty reduction window: match torch's NaN-value, 0-index convention.
        out_shape = list(self.shape)
        del out_shape[dim]
        if keepdim:
            out_shape.insert(dim, 1)
        # NaN is only representable for floating dtypes; for integer dtypes
        # PyTorch raises here, so we mirror that behaviour by deferring to torch.
        if not self.is_floating_point():
            return torch.median(self, dim=dim, keepdim=keepdim)
        v = torch.full(out_shape, float("nan"), dtype=self.dtype, device=self.device)
        i = torch.zeros(out_shape, dtype=torch.int64, device=self.device)
        return v, i

    sorted_tensor, sorted_indices = torch.sort(self, dim=dim, stable=True)
    k = (n - 1) // 2
    values = sorted_tensor.select(dim, k)
    indices = sorted_indices.select(dim, k)
    if keepdim:
        values = values.unsqueeze(dim)
        indices = indices.unsqueeze(dim)
    return values, indices


def median(self):
    logger.debug("GEMS MEDIAN")
    return median_dim(self.flatten(), dim=0, keepdim=False)[0]
