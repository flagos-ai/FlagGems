import logging

import torch

logger = logging.getLogger(__name__)


# Backward of reflection_pad1d WITHOUT atomics.
#
# ROOT CAUSE (generic ops/reflection_pad1d_backward.py): it parallelizes over
# OUTPUT positions, reflects each to an input position, and accumulates with
# `tl.atomic_add`. On KunlunXin XPU the masked atomic_add drops / double-counts
# updates at the reflected borders (baseline: res all 3.0 where ref is 1/3/1),
# so 48 cases fail.
#
# Fix: reflection padding backward is a "fold" -- each output element adds back
# to exactly one input element (identity for the interior, plus one reflected
# copy per padded border). A 1D fold along the last axis is:
#     grad_input = grad_out[interior]
#     grad_input[1 : p0+1]      += flip(grad_out[0 : p0])       # left border
#     grad_input[L-1-p1 : L-1]  += flip(grad_out[p0+L : Lo])    # right border
# All contiguous slice / flip / add ops -- no atomics, no data-dependent gather,
# and exact.
def _reflect_fold(g: torch.Tensor, dim: int, p0: int, p1: int) -> torch.Tensor:
    L = g.shape[dim] - p0 - p1
    gi = g.narrow(dim, p0, L).clone()  # interior (identity mapping)
    if p0 > 0:
        # output i in [0, p0) reflects to input index p0 - i in [1, p0]
        gi.narrow(dim, 1, p0).add_(g.narrow(dim, 0, p0).flip(dim))
    if p1 > 0:
        # output i in [p0+L, Lo) reflects to input index in [L-1-p1, L-1)
        gi.narrow(dim, L - 1 - p1, p1).add_(g.narrow(dim, p0 + L, p1).flip(dim))
    return gi


def reflection_pad1d_backward(grad_output, self, padding):
    logger.debug("GEMS_KUNLUNXIN REFLECTION_PAD1D_BACKWARD")

    if isinstance(padding, int):
        pad_w0 = pad_w1 = padding
    else:
        pad_w0, pad_w1 = padding

    if self.dim() not in (2, 3):
        raise ValueError("input must be a 2D or 3D tensor")

    W_in = self.shape[-1]
    W_out = W_in + pad_w0 + pad_w1

    if grad_output.shape[-1] != W_out:
        raise ValueError(
            f"grad_output last dim {grad_output.shape[-1]}, expected {W_out}"
        )

    if pad_w0 == 0 and pad_w1 == 0:
        return grad_output.clone()

    g = grad_output.contiguous()
    g = _reflect_fold(g, g.dim() - 1, pad_w0, pad_w1)

    return g.contiguous().to(self.dtype)
