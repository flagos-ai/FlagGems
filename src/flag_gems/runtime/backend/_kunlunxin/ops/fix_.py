import logging

import torch
import triton

from .fix import _fix_trunc_kernel, _pick_block

logger = logging.getLogger(__name__)

# Same fast-path tiling as the vendored ``fix`` operator: fix_ and fix are both
# truncation toward zero, so the validated fix kernel (pure-FP helper, exact
# for every input) is reused for the in-place variant.
_FAST_BLOCK = 16384
_FAST_WARPS = 32


def _trunc_inplace(t: torch.Tensor) -> None:
    """In-place truncation toward zero for a *real* floating tensor."""
    numel = t.numel()
    if numel == 0:
        return

    if not t.is_contiguous():
        # Linear-index kernel: run on the contiguous copy, then write back
        # through the vendor copy_ override (no aten fallback).
        t_contig = t.contiguous()
        _trunc_inplace(t_contig)
        t.copy_(t_contig)
        return

    # Constant-tuple grid, mask-free when the block divides evenly (masked
    # memory is ~2x more expensive on this backend). Each element is loaded
    # and stored exactly once by its owning lane, so in-place is safe.
    block_size, num_warps = _pick_block(numel, t.dtype)
    even = numel % block_size == 0
    _fix_trunc_kernel[(triton.cdiv(numel, block_size),)](
        t, t, numel, BLOCK_SIZE=block_size, EVEN=even, num_warps=num_warps
    )


def fix_(self: torch.Tensor):
    """
    Wrapper for ATen operator: ('fix_', <Autograd.disable: False>)
    In-place truncation toward zero for floating tensors.
    Integer tensors are left unchanged (no-op).
    """
    logger.debug("GEMS_KUNLUNXIN FIX_")

    if self.is_complex():
        # Complex: truncate the real/imag components separately.  ATen has no
        # complex fix (torch.fix on a complex tensor raises
        # NotImplementedError), so view the buffer as a flat real tensor and
        # reuse the float in-place path.
        _trunc_inplace(torch.view_as_real(self).view(-1))
        return self

    # Integer tensors: truncation is a no-op.
    if not self.is_floating_point():
        return self

    _trunc_inplace(self)
    return self
