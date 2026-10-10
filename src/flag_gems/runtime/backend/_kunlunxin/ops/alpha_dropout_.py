import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.alpha_dropout import _ALPHA_PRIME, _alpha_dropout_affine
from flag_gems.ops.alpha_dropout_ import _alpha_dropout_inplace_kernel
from flag_gems.ops.copy import copy_ as _triton_copy_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import philox_backend_seed_offset

logger = logging.getLogger("flag_gems.ops.alpha_dropout_")

_UNROLL = 4
_FILL_BLOCK = 1024


def _alpha_dropout_grid(meta):
    return (triton.cdiv(meta["N"], meta["BLOCK"] * _UNROLL),)


@triton.jit
def _alpha_dropout_fill_kernel(X, N, value, BLOCK: tl.constexpr):
    """Write a constant `value` into the first `N` contiguous elements of X."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    vals = value + tl.zeros([BLOCK], tl.float32)
    tl.store(X + offs, vals, mask=mask)


def _alpha_dropout_fill_(input, value):
    """In-place fill of `input` with a scalar `value` using a gems Triton kernel.

    Replaces the previous ``input.fill_(...)`` torch-op fallback. For a
    non-contiguous ``input`` a contiguous staging buffer is filled and copied
    back through the gems Triton copy (dst keeps its original strides).
    """
    N = input.numel()
    if N == 0:
        return input

    if input.is_contiguous():
        target = input
    else:
        target = torch.empty_like(input, memory_format=torch.contiguous_format)

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK"]),)  # noqa: E731
    with torch_device_fn.device(input.device):
        _alpha_dropout_fill_kernel[grid](target, N, value, BLOCK=_FILL_BLOCK)

    if target is not input:
        _triton_copy_(input, target)
    return input


def _ensure_contiguous(t):
    """Return a contiguous tensor holding ``t``'s data without a torch-op fallback.

    Contiguous inputs are returned as-is (a no-op, no extra launch/alloc). For a
    strided input we allocate a contiguous buffer (``torch.empty`` forces the
    default contiguous layout -- ``empty_like`` would inherit ``t``'s strides) and
    fill it through the gems Triton copy instead of torch ``.contiguous()``. Real
    strided -> contiguous is the safe copy direction (``_can_use_triton`` keeps it
    on the Triton path for non-complex, non-float8, same-device strided tensors).
    """
    if t.is_contiguous():
        return t
    return _triton_copy_(torch.empty(t.shape, dtype=t.dtype, device=t.device), t)


def alpha_dropout_(input, p=0.5, train=True):
    """Inplace alpha dropout."""
    logger.debug("GEMS_KUNLUNXIN ALPHA_DROPOUT_ INPLACE FORWARD")

    if not train or p == 0:
        return input

    if p == 1.0:
        a, b = _alpha_dropout_affine(p)
        _alpha_dropout_fill_(input, a * _ALPHA_PRIME + b)
        return input

    assert 0.0 < p < 1.0, "p must be in (0, 1)"

    device = input.device
    is_contiguous = input.is_contiguous()
    input_contig = _ensure_contiguous(input)

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
