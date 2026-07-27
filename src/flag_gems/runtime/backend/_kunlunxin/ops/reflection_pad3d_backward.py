import logging

import torch

logger = logging.getLogger(__name__)


# Backward of reflection_pad3d WITHOUT atomics.
#
# ROOT CAUSE of the old slowness (generic ops/reflection_pad3d_backward.py):
# it parallelized over OUTPUT positions, reflected each to an input position,
# and accumulated with `tl.atomic_add`. On KunlunXin XPU atomic_add is a
# structural throughput wall (~0.1 GB/s, same as scatter_add_), so the larger
# shapes ran at speedup 0.03-0.06. The float32-atomic accumulation was also
# imprecise enough to fail 9 accuracy cases at baseline.
#
# Fix: reflection padding is SEPARABLE per axis, and its backward is a "fold":
# each output element adds back to exactly one input element (identity for the
# interior, plus a single reflected copy for each padded border). A 1D fold
# along one axis is therefore:
#     grad_input = grad_out[interior]
#     grad_input[1 : p0+1]      += flip(grad_out[0 : p0])       # left border
#     grad_input[L-1-p1 : L-1]  += flip(grad_out[p0+L : Lo])    # right border
# All of these are contiguous slice / flip / add ops. Under use_gems they
# re-dispatch to fast gems elementwise kernels (no atomics, no data-dependent
# gather, and exact -- fixes the 9 baseline accuracy failures). Folding W then H
# then D reconstructs the full 3D gradient exactly. Large benchmark shapes get
# ~2.5-3x; tiny shapes are launch-bound (several small kernels vs one atomic
# launch) but remain correct and off the atomic wall.
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


def reflection_pad3d_backward(grad_output, self, padding):
    logger.debug("GEMS_KUNLUNXIN REFLECTION_PAD3D_BACKWARD")

    if isinstance(padding, int):
        pad_d0 = pad_d1 = pad_h0 = pad_h1 = pad_w0 = pad_w1 = padding
    else:
        pad_d0, pad_d1, pad_h0, pad_h1, pad_w0, pad_w1 = padding

    if self.dim() != 5:
        raise ValueError("input must be a 5D tensor")

    N, C, D_in, H_in, W_in = self.shape
    D_out, H_out, W_out = (
        D_in + pad_d0 + pad_d1,
        H_in + pad_h0 + pad_h1,
        W_in + pad_w0 + pad_w1,
    )

    expected_grad_shape = (N, C, D_out, H_out, W_out)
    if tuple(grad_output.shape) != expected_grad_shape:
        raise ValueError(
            f"grad_output has shape {tuple(grad_output.shape)}, expected {expected_grad_shape}"
        )

    if (
        pad_d0 == 0
        and pad_d1 == 0
        and pad_h0 == 0
        and pad_h1 == 0
        and pad_w0 == 0
        and pad_w1 == 0
    ):
        return grad_output.clone()

    g = grad_output.contiguous()
    # Fold each spatial axis (W=dim4, H=dim3, D=dim2). Separable, so order is
    # irrelevant to correctness.
    g = _reflect_fold(g, 4, pad_w0, pad_w1)
    g = _reflect_fold(g, 3, pad_h0, pad_h1)
    g = _reflect_fold(g, 2, pad_d0, pad_d1)

    return g.contiguous().to(self.dtype)
