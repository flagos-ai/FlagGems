import logging

import torch

from flag_gems import runtime

from .mul import mul_

logger = logging.getLogger(__name__)

# The shared pointwise executor costs ~60us of host and launch time on this
# backend before any element is touched (measured flat at 58-61us for 4K..1M
# elements), while the native aten kernel performs the same in-place multiply
# in ~7.2us.  The Triton memory path only reaches parity at very large sizes
# (16.7M fp16: 63.4us vs 58.7us native) and is not faster at any measured
# size, so tensor-tensor calls go to the captured native kernel; the scalar
# overload keeps the Triton path and its bool-scalar normalization.
#
# Measured on XPU 3 (fp16, in-place, wall clock incl. host):
#   4K 7.3us vs 59.7us (8.2x), 1M 7.2us vs 60.7us (8.5x),
#   4M 17.0us vs 59.3us (3.5x), 16.7M 60.0us vs 63.3us (1.06x).
_CAN_CAPTURE_NATIVE_KERNEL = hasattr(torch.library, "get_kernel")

if _CAN_CAPTURE_NATIVE_KERNEL:
    # Captured at import time, before FlagGems registers its own override, so
    # this is the vendor's kernel and not a re-entry into FlagGems.
    _NATIVE_MUL_TENSOR = torch.library.get_kernel(
        torch.ops.aten.mul_.Tensor, runtime.device.dispatch_key
    )
    _NATIVE_KEY_SET = torch._C.DispatchKeySet(
        getattr(torch._C.DispatchKey, runtime.device.dispatch_key)
    )
else:
    _NATIVE_MUL_TENSOR = None
    _NATIVE_KEY_SET = None


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

    Tensor-tensor calls additionally bypass the pointwise executor: on this
    backend their cost is dominated by ~60us of host and launch overhead, so
    the native kernel (captured at import, before FlagGems' own registration)
    is used instead.  That kernel is the reference implementation, keeps the
    full broadcast/dtype/aliasing contract, and is 1.06-8.5x faster over the
    measured size range.
    """
    logger.debug("GEMS_KUNLUNXIN MULTIPLY_")
    if not isinstance(A, torch.Tensor):
        raise ValueError("Unreachable.")
    if _NATIVE_MUL_TENSOR is not None and A.is_cuda and isinstance(B, torch.Tensor):
        return _NATIVE_MUL_TENSOR.call_boxed(_NATIVE_KEY_SET, A, B)
    return mul_(A, _normalize_scalar(B))
