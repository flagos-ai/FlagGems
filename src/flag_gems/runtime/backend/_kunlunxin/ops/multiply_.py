import logging

import torch

from .mul import mul_

logger = logging.getLogger(__name__)


def _normalize_scalar(B):
    """Normalize a Python ``bool`` scalar to its integer equivalent (0/1).

    ``pointwise_dynamic`` emits the scalar variant of the generated kernel as
    ``@triton.jit(do_not_specialize=["val0"])``, so ``val0``'s element type is
    frozen by the *first* call in the process.  A Python ``bool`` makes ``val0``
    an ``i1`` parameter; every later ``int`` scalar is then silently truncated to
    its low bit, i.e. ``Tensor.multiply_(7)`` computes ``tensor * 1`` and
    ``Tensor.multiply_(2)`` computes ``tensor * 0`` (reproduced on XPU 7,
    2026-09-19).  ``bool`` is numerically identical to ``int`` 0/1 for
    multiplication, so normalizing here keeps the shared kernel typed as an
    integer and removes the trigger from the ``multiply_`` entry points.
    """
    if isinstance(B, bool):
        return int(B)
    return B


def multiply_(A, B):
    """In-place multiply (multiply_), an alias for the kunlunxin mul_.

    The generic `flag_gems.ops.multiply_` binds the *generic* `mul_` at import
    time, so it never reaches this backend's optimized `mul_` (contiguous
    pointwise kernel). Overriding `multiply_` here routes it to the fast
    kunlunxin `mul_`, matching `a.mul_(b)` (sp ~0.75-0.99 vs ~0.04 generic).
    """
    logger.debug("GEMS_KUNLUNXIN MULTIPLY_")
    if not isinstance(A, torch.Tensor):
        raise ValueError("Unreachable.")
    return mul_(A, _normalize_scalar(B))
