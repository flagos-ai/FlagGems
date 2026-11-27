import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

# x ** p is decomposed as exp(p * log(x)). The enflame extern libdevice
# exp2/log2 (tl_extra_shim) lower to slow call sequences: the core
# tl.exp/tl.log elementwise ops are 3.2-3.4x faster for the same math
# (measured 2026-11-18, fp32 2**28 partial kernel: 5758us -> 1677us), and
# tl.exp/tl.log slightly beat tl.exp2/tl.log2 as well.
logger = logging.getLogger(__name__)


BLOCK_SIZE = 8192

# Below this many elements a single program with an internal loop produces
# the scalar in one launch; above it a two-stage reduction is used instead.
# (Crossover measured on H20: a single CTA looping over >= 32k elements is
# slower than paying the extra kernel launch.)
SINGLE_KERNEL_THRESHOLD = 16384

# GCU400: 48 concurrent CTAs saturate the 24-SIP device (add.py / sum.py
# precedent). The two-stage partial kernel is persistent grid-stride, so the
# launch grid is capped here regardless of input size. This both keeps all
# SPs busy on large tensors and keeps grid.x within the 65535 hardware limit
# (the old one-CTA-per-chunk launch needed 65536 CTAs at 2**28 elements and
# failed to launch).
MAX_GRID = 48

# Streaming tile for the persistent two-load partial kernel. Unlike the
# byte-budget model of the pointwise family (add.py), this reduction kernel
# is paced by the per-tile cross-lane tl.sum tree, whose cost scales with the
# ELEMENT count: measured 2026-11-18, a 32768-element tile beats both
# 16384 and 65536 for every dtype at 2**24 / 2**28 (fp16 2**28: 596 vs
# 758 / 916us; fp32 2**28: 931 vs 954 / 1288us), so the tile is fixed at
# 32768 elements (128-256KiB depending on dtype) and only shrinks so small
# inputs can still fill all MAX_GRID programs.
PARTIAL_TILE_ELEMS = 32768

# The capped persistent grid turns the old hard grid.x failure into a
# silent int32 overflow for huge inputs, so tile offsets are explicitly
# widened past this bound (randn.py / heaviside_.py WIDE-arm precedent).
INT32_SAFE_MAX = (1 << 31) - (1 << 16)


@triton.jit
def _load_abs_diff(X, Y, idx, mask):
    acc_dtype = tl.float64 if X.type.element_ty == tl.float64 else tl.float32
    a = tl.load(X + idx, mask=mask, other=0.0).to(acc_dtype)
    b = tl.load(Y + idx, mask=mask, other=0.0).to(acc_dtype)
    return tl.abs(a - b)


@triton.jit
def _load_abs_diff_exact(X, Y, idx):
    acc_dtype = tl.float64 if X.type.element_ty == tl.float64 else tl.float32
    a = tl.load(X + idx).to(acc_dtype)
    b = tl.load(Y + idx).to(acc_dtype)
    return tl.abs(a - b)


@triton.jit
def _pow_1_over_p(total, p):
    return tl.exp(tl.log(total) / p)


# Single-launch path for small inputs: one program loops over the whole
# flattened tensor and writes the final scalar directly.
@libentry()
@triton.jit
def dist_p2_kernel(X, Y, Out, N, BLOCK_SIZE: tl.constexpr):
    acc_dtype = tl.float64 if X.type.element_ty == tl.float64 else tl.float32
    offsets = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros((BLOCK_SIZE,), dtype=acc_dtype)
    for start in range(0, N, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < N
        diff = _load_abs_diff(X, Y, idx, mask)
        acc += tl.where(mask, diff * diff, 0.0)
    tl.store(Out, tl.sqrt(tl.sum(acc)))


@libentry()
@triton.jit
def dist_p1_kernel(X, Y, Out, N, BLOCK_SIZE: tl.constexpr):
    acc_dtype = tl.float64 if X.type.element_ty == tl.float64 else tl.float32
    offsets = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros((BLOCK_SIZE,), dtype=acc_dtype)
    for start in range(0, N, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < N
        diff = _load_abs_diff(X, Y, idx, mask)
        acc += tl.where(mask, diff, 0.0)
    tl.store(Out, tl.sum(acc))


@libentry()
@triton.jit
def dist_p0_kernel(X, Y, Out, N, BLOCK_SIZE: tl.constexpr):
    acc_dtype = tl.float64 if X.type.element_ty == tl.float64 else tl.float32
    offsets = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros((BLOCK_SIZE,), dtype=acc_dtype)
    for start in range(0, N, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < N
        diff = _load_abs_diff(X, Y, idx, mask)
        acc += tl.where(mask, (diff != 0).to(acc_dtype), 0.0)
    tl.store(Out, tl.sum(acc))


@libentry()
@triton.jit
def dist_general_kernel(X, Y, Out, N, p, BLOCK_SIZE: tl.constexpr):
    acc_dtype = tl.float64 if X.type.element_ty == tl.float64 else tl.float32
    p = p.to(acc_dtype)
    offsets = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros((BLOCK_SIZE,), dtype=acc_dtype)
    for start in range(0, N, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < N
        diff = _load_abs_diff(X, Y, idx, mask)
        acc += tl.where(mask, tl.exp(p * tl.log(diff)), 0.0)
    tl.store(Out, _pow_1_over_p(tl.sum(acc), p))


@libentry()
@triton.jit
def dist_max_kernel(X, Y, Out, N, BLOCK_SIZE: tl.constexpr):
    acc_dtype = tl.float64 if X.type.element_ty == tl.float64 else tl.float32
    offsets = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros((BLOCK_SIZE,), dtype=acc_dtype)
    for start in range(0, N, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < N
        diff = _load_abs_diff(X, Y, idx, mask)
        acc = tl.maximum(acc, tl.where(mask, diff, -float("inf")))
    tl.store(Out, tl.max(acc))


@libentry()
@triton.jit
def dist_min_kernel(X, Y, Out, N, BLOCK_SIZE: tl.constexpr):
    acc_dtype = tl.float64 if X.type.element_ty == tl.float64 else tl.float32
    offsets = tl.arange(0, BLOCK_SIZE)
    acc = tl.full((BLOCK_SIZE,), float("inf"), dtype=acc_dtype)
    for start in range(0, N, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < N
        diff = _load_abs_diff(X, Y, idx, mask)
        acc = tl.minimum(acc, tl.where(mask, diff, float("inf")))
    tl.store(Out, tl.min(acc))


# Two-stage path for large inputs. Stage 1 is a persistent grid-stride
# kernel: at most MAX_GRID programs stream over the flattened tensors, each
# accumulating one partial in registers and writing a single Mid entry, so
# the grid never scales with the tensor size. Stage 2 reduces the <= MAX_GRID
# partials to the final scalar.
@libentry()
@triton.jit(do_not_specialize=["N"])
def dist_p2_partial_kernel(
    X, Y, Mid, N, BLOCK: tl.constexpr, EXACT: tl.constexpr, WIDE: tl.constexpr
):
    acc_dtype = tl.float64 if X.type.element_ty == tl.float64 else tl.float32
    pid = tl.program_id(0)
    step = tl.num_programs(0)
    if WIDE:
        num_tiles = (N.to(tl.int64) + BLOCK - 1) // BLOCK
    else:
        num_tiles = (N + BLOCK - 1) // BLOCK
    offsets = tl.arange(0, BLOCK)
    acc = tl.zeros([], dtype=acc_dtype)
    for tile in tl.range(pid, num_tiles, step):
        if WIDE:
            idx = tile.to(tl.int64) * BLOCK + offsets
        else:
            idx = tile * BLOCK + offsets
        if EXACT:
            diff = _load_abs_diff_exact(X, Y, idx)
            acc += tl.sum(diff * diff)
        else:
            mask = idx < N
            diff = _load_abs_diff(X, Y, idx, mask)
            acc += tl.sum(tl.where(mask, diff * diff, 0.0))
    tl.store(Mid + pid, acc)


@libentry()
@triton.jit(do_not_specialize=["N"])
def dist_p1_partial_kernel(
    X, Y, Mid, N, BLOCK: tl.constexpr, EXACT: tl.constexpr, WIDE: tl.constexpr
):
    acc_dtype = tl.float64 if X.type.element_ty == tl.float64 else tl.float32
    pid = tl.program_id(0)
    step = tl.num_programs(0)
    if WIDE:
        num_tiles = (N.to(tl.int64) + BLOCK - 1) // BLOCK
    else:
        num_tiles = (N + BLOCK - 1) // BLOCK
    offsets = tl.arange(0, BLOCK)
    acc = tl.zeros([], dtype=acc_dtype)
    for tile in tl.range(pid, num_tiles, step):
        if WIDE:
            idx = tile.to(tl.int64) * BLOCK + offsets
        else:
            idx = tile * BLOCK + offsets
        if EXACT:
            diff = _load_abs_diff_exact(X, Y, idx)
            acc += tl.sum(diff)
        else:
            mask = idx < N
            diff = _load_abs_diff(X, Y, idx, mask)
            acc += tl.sum(tl.where(mask, diff, 0.0))
    tl.store(Mid + pid, acc)


@libentry()
@triton.jit(do_not_specialize=["N"])
def dist_p0_partial_kernel(
    X, Y, Mid, N, BLOCK: tl.constexpr, EXACT: tl.constexpr, WIDE: tl.constexpr
):
    acc_dtype = tl.float64 if X.type.element_ty == tl.float64 else tl.float32
    pid = tl.program_id(0)
    step = tl.num_programs(0)
    if WIDE:
        num_tiles = (N.to(tl.int64) + BLOCK - 1) // BLOCK
    else:
        num_tiles = (N + BLOCK - 1) // BLOCK
    offsets = tl.arange(0, BLOCK)
    acc = tl.zeros([], dtype=acc_dtype)
    for tile in tl.range(pid, num_tiles, step):
        if WIDE:
            idx = tile.to(tl.int64) * BLOCK + offsets
        else:
            idx = tile * BLOCK + offsets
        if EXACT:
            diff = _load_abs_diff_exact(X, Y, idx)
            acc += tl.sum((diff != 0).to(acc_dtype))
        else:
            mask = idx < N
            diff = _load_abs_diff(X, Y, idx, mask)
            acc += tl.sum(tl.where(mask, (diff != 0).to(acc_dtype), 0.0))
    tl.store(Mid + pid, acc)


@libentry()
@triton.jit(do_not_specialize=["N"])
def dist_general_partial_kernel(
    X, Y, Mid, N, p, BLOCK: tl.constexpr, EXACT: tl.constexpr, WIDE: tl.constexpr
):
    acc_dtype = tl.float64 if X.type.element_ty == tl.float64 else tl.float32
    p = p.to(acc_dtype)
    pid = tl.program_id(0)
    step = tl.num_programs(0)
    if WIDE:
        num_tiles = (N.to(tl.int64) + BLOCK - 1) // BLOCK
    else:
        num_tiles = (N + BLOCK - 1) // BLOCK
    offsets = tl.arange(0, BLOCK)
    acc = tl.zeros([], dtype=acc_dtype)
    for tile in tl.range(pid, num_tiles, step):
        if WIDE:
            idx = tile.to(tl.int64) * BLOCK + offsets
        else:
            idx = tile * BLOCK + offsets
        if EXACT:
            diff = _load_abs_diff_exact(X, Y, idx)
            acc += tl.sum(tl.exp(p * tl.log(diff)))
        else:
            mask = idx < N
            diff = _load_abs_diff(X, Y, idx, mask)
            acc += tl.sum(tl.where(mask, tl.exp(p * tl.log(diff)), 0.0))
    tl.store(Mid + pid, acc)


@libentry()
@triton.jit(do_not_specialize=["N"])
def dist_max_partial_kernel(
    X, Y, Mid, N, BLOCK: tl.constexpr, EXACT: tl.constexpr, WIDE: tl.constexpr
):
    acc_dtype = tl.float64 if X.type.element_ty == tl.float64 else tl.float32
    pid = tl.program_id(0)
    step = tl.num_programs(0)
    if WIDE:
        num_tiles = (N.to(tl.int64) + BLOCK - 1) // BLOCK
    else:
        num_tiles = (N + BLOCK - 1) // BLOCK
    offsets = tl.arange(0, BLOCK)
    acc = tl.full([], -float("inf"), dtype=acc_dtype)
    for tile in tl.range(pid, num_tiles, step):
        if WIDE:
            idx = tile.to(tl.int64) * BLOCK + offsets
        else:
            idx = tile * BLOCK + offsets
        if EXACT:
            diff = _load_abs_diff_exact(X, Y, idx)
            acc = tl.maximum(acc, tl.max(diff))
        else:
            mask = idx < N
            diff = _load_abs_diff(X, Y, idx, mask)
            acc = tl.maximum(acc, tl.max(tl.where(mask, diff, -float("inf"))))
    tl.store(Mid + pid, acc)


@libentry()
@triton.jit(do_not_specialize=["N"])
def dist_min_partial_kernel(
    X, Y, Mid, N, BLOCK: tl.constexpr, EXACT: tl.constexpr, WIDE: tl.constexpr
):
    acc_dtype = tl.float64 if X.type.element_ty == tl.float64 else tl.float32
    pid = tl.program_id(0)
    step = tl.num_programs(0)
    if WIDE:
        num_tiles = (N.to(tl.int64) + BLOCK - 1) // BLOCK
    else:
        num_tiles = (N + BLOCK - 1) // BLOCK
    offsets = tl.arange(0, BLOCK)
    acc = tl.full([], float("inf"), dtype=acc_dtype)
    for tile in tl.range(pid, num_tiles, step):
        if WIDE:
            idx = tile.to(tl.int64) * BLOCK + offsets
        else:
            idx = tile * BLOCK + offsets
        if EXACT:
            diff = _load_abs_diff_exact(X, Y, idx)
            acc = tl.minimum(acc, tl.min(diff))
        else:
            mask = idx < N
            diff = _load_abs_diff(X, Y, idx, mask)
            acc = tl.minimum(acc, tl.min(tl.where(mask, diff, float("inf"))))
    tl.store(Mid + pid, acc)


@libentry()
@triton.jit
def dist_p2_kernel_2(Mid, Out, MID_SIZE, BLOCK_SIZE: tl.constexpr):
    acc_dtype = tl.float64 if Mid.type.element_ty == tl.float64 else tl.float32
    offsets = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros((BLOCK_SIZE,), dtype=acc_dtype)
    for start in range(0, MID_SIZE, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < MID_SIZE
        acc += tl.where(mask, tl.load(Mid + idx, mask=mask, other=0.0), 0.0)
    tl.store(Out, tl.sqrt(tl.sum(acc)))


# Shared final reduction for p = 1 and p = 0 (plain sum, no transform).
@libentry()
@triton.jit
def dist_sum_kernel_2(Mid, Out, MID_SIZE, BLOCK_SIZE: tl.constexpr):
    acc_dtype = tl.float64 if Mid.type.element_ty == tl.float64 else tl.float32
    offsets = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros((BLOCK_SIZE,), dtype=acc_dtype)
    for start in range(0, MID_SIZE, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < MID_SIZE
        acc += tl.where(mask, tl.load(Mid + idx, mask=mask, other=0.0), 0.0)
    tl.store(Out, tl.sum(acc))


@libentry()
@triton.jit
def dist_general_kernel_2(Mid, Out, p, MID_SIZE, BLOCK_SIZE: tl.constexpr):
    acc_dtype = tl.float64 if Mid.type.element_ty == tl.float64 else tl.float32
    p = p.to(acc_dtype)
    offsets = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros((BLOCK_SIZE,), dtype=acc_dtype)
    for start in range(0, MID_SIZE, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < MID_SIZE
        acc += tl.where(mask, tl.load(Mid + idx, mask=mask, other=0.0), 0.0)
    tl.store(Out, _pow_1_over_p(tl.sum(acc), p))


@libentry()
@triton.jit
def dist_max_kernel_2(Mid, Out, MID_SIZE, BLOCK_SIZE: tl.constexpr):
    acc_dtype = tl.float64 if Mid.type.element_ty == tl.float64 else tl.float32
    offsets = tl.arange(0, BLOCK_SIZE)
    acc = tl.full((BLOCK_SIZE,), -float("inf"), dtype=acc_dtype)
    for start in range(0, MID_SIZE, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < MID_SIZE
        m = tl.load(Mid + idx, mask=mask, other=0.0)
        acc = tl.maximum(acc, tl.where(mask, m, -float("inf")))
    tl.store(Out, tl.max(acc))


@libentry()
@triton.jit
def dist_min_kernel_2(Mid, Out, MID_SIZE, BLOCK_SIZE: tl.constexpr):
    acc_dtype = tl.float64 if Mid.type.element_ty == tl.float64 else tl.float32
    offsets = tl.arange(0, BLOCK_SIZE)
    acc = tl.full((BLOCK_SIZE,), float("inf"), dtype=acc_dtype)
    for start in range(0, MID_SIZE, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < MID_SIZE
        m = tl.load(Mid + idx, mask=mask, other=0.0)
        acc = tl.minimum(acc, tl.where(mask, m, float("inf")))
    tl.store(Out, tl.min(acc))


def _choose_partial_config(n, itemsize):
    """Tile/warp ladder for the persistent partial kernel.

    A 32768-element tile is the measured optimum for the large bands (see
    PARTIAL_TILE_ELEMS); it halves on smaller inputs (down to a 4096 floor)
    only while that is needed to fill all MAX_GRID programs. num_warps is
    flat across 1/2/4 on every measured band, so a single value keeps the
    compiled-variant count down.
    """
    block = min(PARTIAL_TILE_ELEMS, triton.next_power_of_2(n))
    if itemsize >= 8:
        block = min(block, PARTIAL_TILE_ELEMS // 4)
    while block > 4096 and triton.cdiv(n, block) < MAX_GRID:
        block //= 2
    return block, 2


def _launch_two_stage(input, other, out, p, n):
    mid_dtype = torch.float64 if input.dtype == torch.float64 else torch.float32
    itemsize = input.element_size()
    block, num_warps = _choose_partial_config(n, itemsize)
    num_tiles = triton.cdiv(n, block)
    grid = min(num_tiles, MAX_GRID)
    mid = torch.empty(grid, dtype=mid_dtype, device=input.device)
    exact = n % block == 0
    wide = n >= INT32_SAFE_MAX
    block2 = triton.next_power_of_2(grid)

    with torch_device_fn.device(input.device):
        # The external llvm converter intermittently fails on cold compiles
        # (nondeterministic; converterResult=-2 family, see P02); retrying
        # the launch recompiles and succeeds (add.py / heaviside_.py
        # precedent). Re-running stage 1 is idempotent (plain overwrite).
        for attempt in range(3):
            try:
                if p == 2:
                    dist_p2_partial_kernel[(grid,)](
                        input,
                        other,
                        mid,
                        n,
                        BLOCK=block,
                        EXACT=exact,
                        WIDE=wide,
                        num_warps=num_warps,
                        num_stages=1,
                    )
                    dist_p2_kernel_2[(1,)](mid, out, grid, BLOCK_SIZE=block2)
                elif p == 1:
                    dist_p1_partial_kernel[(grid,)](
                        input,
                        other,
                        mid,
                        n,
                        BLOCK=block,
                        EXACT=exact,
                        WIDE=wide,
                        num_warps=num_warps,
                        num_stages=1,
                    )
                    dist_sum_kernel_2[(1,)](mid, out, grid, BLOCK_SIZE=block2)
                elif p == 0:
                    dist_p0_partial_kernel[(grid,)](
                        input,
                        other,
                        mid,
                        n,
                        BLOCK=block,
                        EXACT=exact,
                        WIDE=wide,
                        num_warps=num_warps,
                        num_stages=1,
                    )
                    dist_sum_kernel_2[(1,)](mid, out, grid, BLOCK_SIZE=block2)
                elif p == float("inf"):
                    dist_max_partial_kernel[(grid,)](
                        input,
                        other,
                        mid,
                        n,
                        BLOCK=block,
                        EXACT=exact,
                        WIDE=wide,
                        num_warps=num_warps,
                        num_stages=1,
                    )
                    dist_max_kernel_2[(1,)](mid, out, grid, BLOCK_SIZE=block2)
                elif p == float("-inf"):
                    dist_min_partial_kernel[(grid,)](
                        input,
                        other,
                        mid,
                        n,
                        BLOCK=block,
                        EXACT=exact,
                        WIDE=wide,
                        num_warps=num_warps,
                        num_stages=1,
                    )
                    dist_min_kernel_2[(1,)](mid, out, grid, BLOCK_SIZE=block2)
                else:
                    dist_general_partial_kernel[(grid,)](
                        input,
                        other,
                        mid,
                        n,
                        p,
                        BLOCK=block,
                        EXACT=exact,
                        WIDE=wide,
                        num_warps=num_warps,
                        num_stages=1,
                    )
                    dist_general_kernel_2[(1,)](mid, out, p, grid, BLOCK_SIZE=block2)
                return
            except Exception as exc:
                if attempt == 2:
                    raise
                logger.warning(
                    "dist two-stage launch failed (attempt %d): %s",
                    attempt + 1,
                    exc,
                )


def dist(input, other, p=2):
    logger.debug("GEMS_ENFLAME DIST")
    if input.shape != other.shape:
        input, other = torch.broadcast_tensors(input, other)
    if not input.is_contiguous():
        input = input.contiguous()
    if not other.is_contiguous():
        other = other.contiguous()

    n = input.numel()

    # torch returns 0 for finite non-negative p on an empty reduction; for
    # inf / -inf / negative p there is no identity element and torch raises.
    if n == 0:
        if p == float("inf") or p == float("-inf") or p < 0:
            raise RuntimeError(
                f"dist cannot compute the {p} norm on an empty tensor "
                "(no identity element over an empty reduction)"
            )
        return torch.zeros([], dtype=input.dtype, device=input.device)

    out = torch.empty([], dtype=input.dtype, device=input.device)
    p = float(p)

    with torch_device_fn.device(input.device):
        if n <= SINGLE_KERNEL_THRESHOLD:
            block = triton.next_power_of_2(min(n, BLOCK_SIZE))
            if p == 2:
                dist_p2_kernel[(1,)](input, other, out, n, BLOCK_SIZE=block)
            elif p == 1:
                dist_p1_kernel[(1,)](input, other, out, n, BLOCK_SIZE=block)
            elif p == 0:
                dist_p0_kernel[(1,)](input, other, out, n, BLOCK_SIZE=block)
            elif p == float("inf"):
                dist_max_kernel[(1,)](input, other, out, n, BLOCK_SIZE=block)
            elif p == float("-inf"):
                dist_min_kernel[(1,)](input, other, out, n, BLOCK_SIZE=block)
            else:
                dist_general_kernel[(1,)](input, other, out, n, p, BLOCK_SIZE=block)
            return out

    _launch_two_stage(input, other, out, p, n)
    return out
