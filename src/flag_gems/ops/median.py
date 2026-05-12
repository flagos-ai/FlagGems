"""median + median.dim — FlagGems operator.

Two execution paths:

1.  **Triton small-N fast path** — when the reduction is along the last
    dim and the reduced length ``N`` fits in a single register tile
    (≤ 1024 power-of-two elements), one program per row sorts the
    row with ``tl.sort`` and writes both the lower-median value and
    its first-occurrence index in a single kernel launch.  Saves the
    full-tensor ``torch.sort`` + ``select`` round-trip on small rows.

2.  **General fallback** — ``torch.sort(stable=True)`` + ``select``
    along the reduction dim.  Always correct; uses CUB radix on CUDA.
    Picked for non-last-dim reductions, integer dtypes, ``N > 1024``
    rows, and as a safety net.

Behaviour parity with ``torch.median``:
    * Lower-median tie-break (sorted index ``(n - 1) // 2``).
    * Returned index is the **first occurrence** of the median value
      when the value is repeated.
    * Empty reduction returns ``NaN`` value and index ``0`` for
      floating dtypes; defers to torch (which raises) for integer.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Triton kernel: lower-median of a single row that fits in registers.
# ---------------------------------------------------------------------------
@libentry()
@triton.jit
def _median_lastdim_kernel(
    in_ptr,
    out_val_ptr,
    out_idx_ptr,
    M,
    N,
    k,
    BLOCK_N: tl.constexpr,
):
    """One program per row of length ``N``.

    Pad with +inf so the padded positions sort to the end and never get
    picked as the k-th element (we always have ``k < N``).  After
    sorting, the k-th sorted value is the lower median.  The index of
    its first occurrence in the original row is found by min-reducing
    ``offs`` over the matching positions.
    """
    pid = tl.program_id(0)
    if pid >= M:
        return

    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    POSINF = float("inf")
    x = tl.load(in_ptr + pid * N + offs, mask=mask, other=POSINF).to(tl.float32)

    # Sort the (padded) row.  Padded entries are +inf so they sit at the
    # end; the k-th sorted element is the true lower median.
    x_sorted = tl.sort(x, dim=0)
    median_val = tl.sum(tl.where(tl.arange(0, BLOCK_N) == k, x_sorted, 0.0))

    # First-occurrence index of the median value in the original row.
    is_match = (x == median_val) & mask
    masked_offs = tl.where(is_match, offs, BLOCK_N)  # BLOCK_N > any valid idx
    first_idx = tl.min(masked_offs, axis=0)

    tl.store(out_val_ptr + pid, median_val.to(out_val_ptr.dtype.element_ty))
    tl.store(out_idx_ptr + pid, first_idx.to(out_idx_ptr.dtype.element_ty))


def _median_lastdim_triton(x: torch.Tensor, keepdim: bool):
    """Fast path: last-dim, floating dtype, small N.  Returns ``None`` to
    signal the caller to use the general fallback when not applicable."""
    N = x.shape[-1]
    if N == 0:
        return None
    BLOCK_N = triton.next_power_of_2(N)
    if BLOCK_N > 1024:  # tl.sort capacity on most archs
        return None
    if not x.is_floating_point():
        return None
    if not x.is_cuda:
        return None

    x_c = x.contiguous()
    M = x.numel() // N
    out_shape = x.shape[:-1]
    values = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    indices = torch.empty(out_shape, dtype=torch.int64, device=x.device)
    grid = (M,)
    k = (N - 1) // 2
    try:
        with torch_device_fn.device(x.device):
            _median_lastdim_kernel[grid](
                x_c.view(M, N),
                values.view(M),
                indices.view(M),
                M,
                N,
                k,
                BLOCK_N=BLOCK_N,
            )
    except Exception:
        return None

    if keepdim:
        values = values.unsqueeze(-1)
        indices = indices.unsqueeze(-1)
    return values, indices


# ---------------------------------------------------------------------------
# Reference / general fallback: torch.sort + select.
# ---------------------------------------------------------------------------
def _median_dim_torch(x, dim, keepdim):
    sorted_tensor, sorted_indices = torch.sort(x, dim=dim, stable=True)
    n = x.shape[dim]
    k = (n - 1) // 2
    values = sorted_tensor.select(dim, k)
    indices = sorted_indices.select(dim, k)
    if keepdim:
        values = values.unsqueeze(dim)
        indices = indices.unsqueeze(dim)
    return values, indices


# ---------------------------------------------------------------------------
# Public APIs.
# ---------------------------------------------------------------------------
def median_dim(self, dim=-1, keepdim=False):
    logger.debug("GEMS MEDIAN DIM")
    dim = dim % self.ndim
    n = self.shape[dim]

    if n == 0:
        out_shape = list(self.shape)
        del out_shape[dim]
        if keepdim:
            out_shape.insert(dim, 1)
        if not self.is_floating_point():
            return torch.median(self, dim=dim, keepdim=keepdim)
        v = torch.full(out_shape, float("nan"), dtype=self.dtype, device=self.device)
        i = torch.zeros(out_shape, dtype=torch.int64, device=self.device)
        return v, i

    # Triton small-N fast path on the last dim.
    if dim == self.ndim - 1:
        fast = _median_lastdim_triton(self, keepdim)
        if fast is not None:
            return fast

    return _median_dim_torch(self, dim, keepdim)


def median(self):
    logger.debug("GEMS MEDIAN")
    return median_dim(self.flatten(), dim=0, keepdim=False)[0]
