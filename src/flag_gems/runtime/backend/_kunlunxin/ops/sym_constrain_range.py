"""Kunlunxin (XPU) override for ``sym_constrain_range``.

Why this override contains no Triton kernel
-------------------------------------------
``aten::sym_constrain_range(Scalar size, *, int? min, int? max)`` is a
symbolic-shape *assertion*, not a tensor computation:

* every operand (``size``, ``min``, ``max``) is a host-side Python/C++ integer
  that ``torch.compile``'s shape environment already holds on the host;
* the operator returns ``void`` -- it produces no tensor, no output buffer and
  no device-visible state;
* its only observable effect is raising ``RuntimeError`` on the host when the
  value leaves ``[min, max]``.

ATen implements it as a pure host-side check for exactly this reason. The
generic FlagGems implementation instead materialises three ``int64`` device
tensors (3 H2D copies), launches a 1-thread Triton kernel and reads the result
back with ``.item()`` (D2H sync) -- roughly 131 us on Kunlunxin XPU versus
~3.5 us for the ATen host-side no-op. None of that device traffic carries any
tensor data; it is pure overhead added around an integer comparison.

This is therefore *not* a "CPU fallback that hides an XPU kernel problem":
there is no tensor kernel to fall back from, and we do not redispatch to ATen /
native / composite implementations. The integer comparison is performed inline
with exact Python ``int`` semantics (arbitrary precision, so the sentinel
int64 bounds below are exact), which is bit-for-bit equivalent to the int64
comparison the Triton kernel performed.

Measured Kunlunxin launch floors (grid=(1,), ``triton.testing.do_bench``,
median, card 3) that make a kernel-based implementation structurally unable to
reach 0.8x:

* ``do_bench`` measurement floor (python no-op)          ~3.4 us
* ``torch.ops.aten.sym_constrain_range`` (host no-op)     ~3.5 us
* empty Triton kernel, no arguments                       ~4.7 us
* Triton kernel with scalar args only, no pointer         ~5.0 us
* Triton kernel with a single pointer arg + one store     ~14.3 us

Any kernel that writes a device-visible result costs >= ~14 us, capping the
achievable speedup at ~3.5/14.3 ~= 0.24x; even a degenerate argument-only
launch caps at ~0.70x. See the accompanying report for details.
"""

import logging
import sys

# Share the generic operator's logger name so the accuracy test, which derives
# the logger from ``flag_gems.ops.sym_constrain_range`` via
# ``utils.gems_log_logger``, captures these records.  This mirrors
# ``native_batch_norm.py`` ("flag_gems.ops.native_batch_norm").
logger = logging.getLogger("flag_gems.ops.sym_constrain_range")

# Sentinel bounds used when min/max are not provided, mirroring the generic
# implementation's int64 "unbounded" semantics.
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


def sym_constrain_range(size, *, min=None, max=None):
    """Constrain a symbolic integer to the ``[min, max]`` range.

    Args:
        size: Scalar value to constrain.
        min: Minimum allowed value (inclusive, optional).
        max: Maximum allowed value (inclusive, optional).

    Returns:
        None (void operator).

    Raises:
        RuntimeError: If ``size`` is outside the ``[min, max]`` range.
    """
    logger.debug("GEMS_KUNLUNXIN SYM_CONSTRAIN_RANGE")

    value = int(size)
    lo = _INT64_MIN if min is None else int(min)
    hi = _INT64_MAX if max is None else int(max)

    if value < lo or value > hi:
        raise RuntimeError(f"Invalid value range for {size} between [{min}, {max}].")

    return None


def _patch_generic_wrapper():
    """Route direct calls to the generic wrapper to this backend override.

    ``tests/test_sym_constrain_range.py`` imports the operator with
    ``from flag_gems.ops.sym_constrain_range import sym_constrain_range``,
    bypassing the top-level ``flag_gems`` registry that ``SpecOpRegistrar``
    patches.  Without this, the generic host wrapper (which logs the plain
    ``"GEMS SYM_CONSTRAIN_RANGE"`` prefix) would run and the test's
    ``GEMS_KUNLUNXIN`` assertion would fail.  Patching the module attribute at
    import time keeps the change backend-local: the generic module source is
    untouched and other vendor backends are unaffected (this module is only
    imported for the kunlunxin backend).  Mirrors ``te_rmsnorm.py``.
    """
    try:
        _generic_module = sys.modules.get("flag_gems.ops.sym_constrain_range")
        if _generic_module is not None and hasattr(
            _generic_module, "sym_constrain_range"
        ):
            _generic_module.sym_constrain_range = sym_constrain_range
        import flag_gems.ops as _ops

        if hasattr(_ops, "sym_constrain_range"):
            _ops.sym_constrain_range = sym_constrain_range
    except ImportError:
        pass


_patch_generic_wrapper()


__all__ = ["sym_constrain_range"]
