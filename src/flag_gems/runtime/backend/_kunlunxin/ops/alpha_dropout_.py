import logging

import triton

from flag_gems.ops.alpha_dropout import _ALPHA_PRIME, _alpha_dropout_affine
from flag_gems.ops.alpha_dropout_ import _alpha_dropout_inplace_kernel
from flag_gems.ops.copy import copy_ as _triton_copy_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import philox_backend_seed_offset

logger = logging.getLogger("flag_gems.ops.alpha_dropout_")

_UNROLL = 4


def _alpha_dropout_grid(meta):
    return (triton.cdiv(meta["N"], meta["BLOCK"] * _UNROLL),)


def alpha_dropout_(input, p=0.5, train=True):
    """Inplace alpha dropout."""
    logger.debug("GEMS_KUNLUNXIN ALPHA_DROPOUT_ INPLACE FORWARD")

    if not train or p == 0:
        return input

    if p == 1.0:
        a, b = _alpha_dropout_affine(p)
        input.fill_(a * _ALPHA_PRIME + b)
        return input

    assert 0.0 < p < 1.0, "p must be in (0, 1)"

    device = input.device
    input_contig = input.contiguous()
    is_contiguous = input.is_contiguous()

    N = input_contig.numel()
    increment = triton.cdiv(N, _UNROLL)

    a, b = _alpha_dropout_affine(p)

    with torch_device_fn.device(device):
        philox_seed, philox_offset = philox_backend_seed_offset(increment)
        _alpha_dropout_inplace_kernel[_alpha_dropout_grid](
            input_contig, N, p, a, b, philox_seed, philox_offset
        )

    if not is_contiguous:
        _triton_copy_(input, input_contig)

    return input
