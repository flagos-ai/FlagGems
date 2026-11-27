# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Availability gate for the Ascend TensorView (``tle.make_*_view``) fast paths.

The TensorView constructors lower to block DMAs through HIVM and need a matching
flagtree build *and* CANN toolkit. Neither requirement can be fully checked by
inspecting the Python namespace: a toolkit whose ``bishengir-opt`` disagrees with
the locally built ``triton-mlir-opt`` about bytecode passes every ``hasattr``
check and then fails at compile time with a diagnostic such as

    error: expected mlir::hivm::TCoreTypeAttr, but got: {}

So availability is decided in two stages. ``HAS_TENSORVIEW`` answers the static
question (is the API even importable) and keeps the ``@triton.jit`` kernels from
being defined at all when it is not. ``tensorview_call`` answers the dynamic one
by running the fast path once and permanently degrading to the generic
implementation if it fails to compile.
"""

import logging

import torch

from flag_gems.runtime import device as runtime_device

logger = logging.getLogger(__name__)

IS_ASCEND = runtime_device.vendor_name == "ascend"

# Ascend leaves the vendor-wide TLE flag off, so flag_gems.utils.triton_version_utils
# .HAS_TLE is False here even when TLE is importable. Probe the specific primitives
# instead, the same way ops/glu.py does for its DSA helper.
tle = None
if IS_ASCEND:
    try:
        import triton.experimental.tle as _tle_root
        import triton.experimental.tle.language as _tle_lang  # noqa: F401

        tle = _tle_root
    except ImportError:
        tle = None

_VIEW_CONSTRUCTORS = (
    "make_partition_view",
    "make_strided_view",
    "make_gather_scatter_view",
)


def _has_tile_transfer() -> bool:
    """Check for tile_load/tile_store and the Ascend UB address space.

    ``tile_load``/``tile_store`` live in ``tle.language.tile`` and are
    re-exported from ``tle`` itself, next to the view constructors. They are
    not under ``tle.language.dsa`` -- that package exports the lower level
    ``tile_alloc``/``tile_copy`` family instead. The address space enum is
    the one thing that really does come from ``dsa.ascend``.
    """
    try:
        from triton.experimental.tle import tile_load, tile_store  # noqa: F401
        from triton.experimental.tle.language.dsa.ascend import UB  # noqa: F401
    except ImportError:
        return False
    return True


HAS_TENSORVIEW = bool(
    IS_ASCEND
    and tle is not None
    and all(hasattr(tle, name) for name in _VIEW_CONSTRUCTORS)
    and _has_tile_transfer()
)

# Tri-state, process wide. The fast path is compiled at most once before being
# ruled out, so a toolkit that cannot lower TensorView costs one failed compile
# per process rather than one per call.
_UNKNOWN, _GOOD, _BAD = 0, 1, 2
_state = _UNKNOWN


def _is_compile_error(exc: BaseException) -> bool:
    """Is this Triton failing to compile, as opposed to a genuine bug?

    Matches on the name/module instead of importing the exception types: the
    MLIR-specific ones only exist on the Ascend compiler line, and
    ``flag_gems`` must import on backends that do not have them. Mirrors the
    check in ``_ascend/ops/mul.py``.
    """
    for cls in type(exc).__mro__:
        if not getattr(cls, "__module__", "").startswith("triton."):
            continue
        if cls.__name__ in ("MLIRCompilationError", "CompilationError"):
            return True
    return False


def tensorview_available() -> bool:
    """True unless the fast path has already been ruled out in this process."""
    return HAS_TENSORVIEW and _state is not _BAD


def tensorview_call(op_name, fast_fn, generic_fn, *args, **kwargs):
    """Run ``fast_fn``, falling back to ``generic_fn`` if it cannot compile.

    Only compilation failures are absorbed. Anything else propagates, so a
    numerical or shape bug in a TensorView kernel still surfaces as a test
    failure instead of being quietly masked by the generic implementation.
    """
    global _state

    if not tensorview_available():
        return generic_fn(*args, **kwargs)

    try:
        result = fast_fn(*args, **kwargs)
    except Exception as exc:
        if not _is_compile_error(exc):
            raise
        _state = _BAD
        logger.warning(
            "GEMS_ASCEND %s: the TensorView path does not compile against this "
            "flagtree/CANN combination (%s); using the generic implementation "
            "for the rest of this process.",
            op_name,
            exc,
        )
        return generic_fn(*args, **kwargs)

    _state = _GOOD
    return result


# Unified Buffer size on the supported parts. Tiles are staged here, so the
# per-program footprint has to fit; exceeding it fails at compile time with an
# "ub overflow" diagnostic rather than degrading.
UB_BYTES = 192 * 1024

# Stay at half of UB: the compiler also needs room for double buffering and for
# the intermediates a kernel materializes beyond the tiles counted here.
_UB_BUDGET = UB_BYTES // 2

# A tile whose leading extent is 1 does not lower: the multi-dimensional views
# need at least two rows to form a block DMA.
#
# Callers keep the two questions apart, because clamping the answer to the first
# one would hide the second. ``tile_rows_for_ub`` reports what UB can hold, and
# the eligibility check rejects the shape when that is below MIN_TILE_ROWS; only
# the launch value is then raised to this floor. Clamping inside the helper
# instead would turn "one row fits" into "two rows", and launching those two
# overflows UB at compile time.
MIN_TILE_ROWS = 2


def tile_rows_for_ub(row_bytes, live_tiles, max_rows):
    """Largest power-of-two row count per tile whose staged tiles fit in UB.

    ``row_bytes`` is one row of the payload and ``live_tiles`` is how many
    tiles of that shape are simultaneously live in the kernel. Wide rows get
    fewer rows per tile instead of failing to compile; the result is never
    below 1, so a single row that cannot fit is left to the caller's
    eligibility check -- see ``tile_fits_ub``.
    """
    denom = max(live_tiles, 1) * max(row_bytes, 1)
    budget = _UB_BUDGET // denom
    if budget < 1:
        return 1
    block = 1 << (budget.bit_length() - 1)
    return max(min(max_rows, block), 1)


def tile_fits_ub(row_bytes, live_tiles):
    """Can a tile of MIN_TILE_ROWS rows be staged in UB?

    The eligibility counterpart to ``tile_rows_for_ub``, which floors its
    answer at 1 and so cannot distinguish "one row fits" from "not even one
    row fits". Both of those launch a tile that overflows UB, and UB overflow
    is a compile error: the kernel reports

        [NOTE] Due to Ub overflow, automatically set enable-code-motion to
        false, set enable-auto-multi-buffer to false; compilation still failed.
        error: Failed to run BiShengHIR pipeline

    which ``tensorview_call`` then absorbs, degrading the whole process to the
    generic path for one over-wide shape. Call this with the same
    ``row_bytes`` and ``live_tiles`` the launch passes to
    ``tile_rows_for_ub``, so the two cannot drift apart.
    """
    denom = max(live_tiles, 1) * max(row_bytes, 1)
    return _UB_BUDGET // denom >= MIN_TILE_ROWS


def has_no_duplicates(index, num_rows):
    """Does ``index`` name each row at most once?

    Callers whose kernel does a read-modify-write per tile need this: two tiles
    holding the same row read it before either writes back, so a repeat drops an
    update. ``torch.unique`` answers the same question but sorts, which on NPU
    costs ~0.3 ms and dwarfs the block-DMA kernel it is guarding (~0.012 ms).
    A scatter of ones touches each named row once, so the marks sum to the index
    length exactly when there are no repeats.
    """
    seen = torch.zeros(num_rows, dtype=torch.int32, device=index.device)
    marks = torch.ones(index.numel(), dtype=torch.int32, device=index.device)
    seen.scatter_(0, index.long(), marks)
    return bool(seen.sum().item() == index.numel())


def _reset_state_for_testing():
    """Clear the memoized decision. For tests that exercise both paths."""
    global _state
    _state = _UNKNOWN


__all__ = [
    "HAS_TENSORVIEW",
    "MIN_TILE_ROWS",
    "UB_BYTES",
    "has_no_duplicates",
    "tile_fits_ub",
    "tile_rows_for_ub",
    "tensorview_available",
    "tensorview_call",
    "tle",
]
