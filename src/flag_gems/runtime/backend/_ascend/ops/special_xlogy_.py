"""In-place ``special_xlogy_`` (``out = x * log(y)``) as a fused Ascend TLE kernel.

Semantics (matches FlagGems ``special_xlogy_func`` exactly):

* compute in fp32: ``xf = x.to(fp32)``, ``yf = y.to(fp32)``
* ``y_is_nan = yf != yf``  -> result NaN  (wins over the zero rule)
* ``x_is_zero = xf == 0.0`` -> result 0.0 (wins over ``0 * -inf``)
* otherwise ``xf * log(yf)``
* store back into ``A`` (in place) and return the *same* tensor object.

Kernel shape: 1-D grid capped at the number of physical vector cores; each
program owns one contiguous run of full ``BLOCK`` tiles, so partitioning is
non-interleaved and every program's data is contiguous.  The trailing partial
tile is handled by the last program with a masked ``tl.load``/``tl.store``
(``tle.dsa.copy`` has no mask argument).

Bulk tile traffic uses the TLE DSA staging path (``tle.dsa.alloc`` /
``tle.dsa.copy`` / ``tle.dsa.to_buffer``) for GM -> UB -> compute -> UB -> GM.

The masked tail block is guarded by ``nfull * BLOCK < n`` so a provably
fully-masked tail tile is skipped instead of being executed and discarded on
every grid.

On the large shapes every vector core runs thousands of steady-state tiles, so
the loop's *fixed* per-iteration cost - the scalar-pipe address arithmetic and
the per-tile copy/compute handshakes - is paid once per tile and is a
measurable share of the kernel time.  The steady-state tile is therefore sized
by staged *bytes* (``_TILE_BYTES``) rather than by a fixed element count, so
that every tile moves the same amount of data through UB: the two-byte dtypes
get a wider tile than the four-byte ones and their per-program iteration count
drops, while the per-tile UB byte volume stays UB-safe.  ``_BLOCK`` remains the
four-byte tile and is never shrunk, so no shape ever sees *more* iterations than
before; the small-shape branch is untouched.
"""

import logging

import torch
import torch_npu  # noqa: F401  (must be imported before triton on this target)
import triton
import triton.experimental.tle as tle
import triton.language as tl

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# tunables
# --------------------------------------------------------------------------- #
_BLOCK = 4096  # four-byte large-shape tile; also the small/large shape threshold
_SMALL_BLOCK = 512  # used only when the shape is too small to fill the cores
_SMALL_BLOCK_TILES = 8  # below this many full tiles, shrink BLOCK to spread the work
_TILE_BYTES = 16384  # staged bytes per steady-state tile, at every element width
_DEFAULT_CORES = 40  # only a fallback; the real count is queried at runtime

_CORES_CACHE = None


def _num_vectorcore():
    """Query the physical vector-core count (never hard-code the dev device)."""
    global _CORES_CACHE
    if _CORES_CACHE is None:
        n = 0
        try:
            from triton.runtime import driver

            props = driver.active.utils.get_device_properties(
                torch.npu.current_device()
            )
            n = int(dict(props).get("num_vectorcore", 0) or 0)
        except Exception:  # noqa: BLE001
            n = 0
        _CORES_CACHE = n if n > 0 else _DEFAULT_CORES
    return _CORES_CACHE


# --------------------------------------------------------------------------- #
# kernel
# --------------------------------------------------------------------------- #
@triton.jit
def _xlogy_math(x, y):
    """Reference numerical recipe, computed in fp32, cast back to x's dtype."""
    xf = x.to(tl.float32)
    yf = y.to(tl.float32)
    ynan = yf != yf
    xzero = xf == 0.0
    prod = xf * tl.log(yf)
    r = tl.where(ynan, float("nan"), tl.where(xzero, 0.0, prod))
    return r.to(x.dtype)


@triton.jit
def _xlogy_kernel(a_ptr, b_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)

    nfull = n // BLOCK
    per = tl.cdiv(nfull, nprog)
    t0 = pid * per
    t1 = tl.minimum(t0 + per, nfull)

    offs = tl.arange(0, BLOCK)
    a_ub = tle.dsa.alloc(
        [BLOCK], dtype=a_ptr.dtype.element_ty, mem_addr_space=tle.dsa.ascend.UB
    )
    b_ub = tle.dsa.alloc(
        [BLOCK], dtype=b_ptr.dtype.element_ty, mem_addr_space=tle.dsa.ascend.UB
    )

    for t in range(t0, t1):
        base = t * BLOCK
        tle.dsa.copy(a_ptr + base + offs, a_ub, [BLOCK])
        tle.dsa.copy(b_ptr + base + offs, b_ub, [BLOCK])
        x = tle.dsa.to_tensor(a_ub, writable=False)
        y = tle.dsa.to_tensor(b_ub, writable=False)
        r_ub = tle.dsa.to_buffer(_xlogy_math(x, y), space=tle.dsa.ascend.UB)
        tle.dsa.copy(r_ub, a_ptr + base + offs, [BLOCK])

    if pid == nprog - 1 and nfull * BLOCK < n:
        toffs = nfull * BLOCK + offs
        m = toffs < n
        x = tl.load(a_ptr + toffs, mask=m, other=1.0)
        y = tl.load(b_ptr + toffs, mask=m, other=1.0)
        tl.store(a_ptr + toffs, _xlogy_math(x, y), mask=m)


# --------------------------------------------------------------------------- #
# host entry point
# --------------------------------------------------------------------------- #
_SUPPORTED_DTYPES = (torch.float16, torch.float32, torch.bfloat16)


def special_xlogy_(A, B):
    """In-place ``special_xlogy_``: mutates ``A`` and returns it."""
    logger.debug("GEMS SPECIAL_XLOGY_")
    if A.dtype not in _SUPPORTED_DTYPES:
        from flag_gems.ops.special_xlogy import special_xlogy_ as _generic

        return _generic(A, B)

    """In-place ``xlogy``: mutates ``A`` and returns the very same tensor."""
    n = A.numel()
    if n == 0:
        return A

    a_flat = A.reshape(-1)
    b_flat = B.reshape(-1)

    # Shape-adaptive BLOCK.
    #
    # Small shapes: the grid is capped at the physical vector-core count, so a
    # shape whose full-tile count is below that cap would strand (cores - tiles)
    # idle cores: at n = 4096 with BLOCK = 4096 the whole op runs on 1 of 40
    # cores.  Shrinking BLOCK for such shapes spreads the same elements over
    # more cores.  The small/large decision stays keyed on the four-byte _BLOCK
    # so this branch keeps the exact behaviour that fixed those shapes.
    #
    # Large shapes: hold the staged *bytes* per tile constant instead of the
    # element count.  The per-iteration cost is paid once per tile, so the
    # two-byte dtypes - which make up the weakest workloads - halve their
    # per-program iteration count at an unchanged per-tile byte volume, while
    # the four-byte dtypes keep the previous element tile.  The tile is never
    # narrowed below _BLOCK, so no shape does more iterations than before.
    if triton.cdiv(n, _BLOCK) < _SMALL_BLOCK_TILES:
        block = _SMALL_BLOCK
    else:
        block = max(_BLOCK, _TILE_BYTES // A.element_size())

    grid = min(_num_vectorcore(), triton.cdiv(n, block))
    if grid < 1:
        grid = 1

    _xlogy_kernel[(grid,)](a_flat, b_flat, n, BLOCK=block)
    return A
