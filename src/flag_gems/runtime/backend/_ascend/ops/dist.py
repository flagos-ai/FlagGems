import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

exp2 = tl_extra_shim.exp2
log2 = tl_extra_shim.log2
pow = tl_extra_shim.pow
logger = logging.getLogger(__name__)


@triton.jit
def _load_abs_diff(X, Y, idx, mask):
    a = tl.load(X + idx, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(Y + idx, mask=mask, other=0.0).to(tl.float32)
    # Select form instead of tl.abs: the Ascend backend cannot select a
    # scalar fabs on half types (bisheng: "Cannot select: f16 = fabs").
    d = a - b
    return tl.where(d < 0, -d, d)


# UB is 192KB per core, but the compiler double-buffers loads and needs
# several f32 intermediate tiles; measured headroom fits a 4096-element tile
# (8192 overflows UB at ~224KB).
BLOCK_SIZE = 4096

# Large inputs use a persistent two-stage reduction: stage-1 launches at most
# MAX_GRID programs (the chip has < 40 vector cores; more programs than cores
# just serialize), each looping over BLOCK_SIZE tiles with a grid stride, and
# stage-2 reduces the <= MAX_GRID partials.
MAX_GRID = 40

# Below this many elements a single program with an internal loop produces
# the scalar in one launch (measured on 910b; one extra launch costs ~0.1ms).
SINGLE_KERNEL_THRESHOLD = 16384


# Single-launch path for small inputs: one program loops over the whole
# flattened tensor and writes the final scalar directly.
@libentry()
@triton.jit
def dist_p2_kernel(X, Y, Out, N, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for start in range(0, N, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < N
        diff = _load_abs_diff(X, Y, idx, mask)
        acc += tl.where(mask, diff * diff, 0.0)
    tl.store(Out, tl.sqrt(tl.sum(acc)))


@libentry()
@triton.jit
def dist_p1_kernel(X, Y, Out, N, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for start in range(0, N, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < N
        diff = _load_abs_diff(X, Y, idx, mask)
        acc += tl.where(mask, diff, 0.0)
    tl.store(Out, tl.sum(acc))


@libentry()
@triton.jit
def dist_p0_kernel(X, Y, Out, N, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for start in range(0, N, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < N
        diff = _load_abs_diff(X, Y, idx, mask)
        acc += tl.where(mask, (diff != 0).to(tl.float32), 0.0)
    tl.store(Out, tl.sum(acc))


@libentry()
@triton.jit
def dist_general_kernel(X, Y, Out, N, p, BLOCK_SIZE: tl.constexpr):
    p = p.to(tl.float32)
    offsets = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for start in range(0, N, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < N
        diff = _load_abs_diff(X, Y, idx, mask)
        acc += tl.where(mask, exp2(p * log2(diff)), 0.0)
    # libdevice pow keeps extra internal precision in the log domain,
    # ~3x tighter than a raw exp2/log2 roundtrip (measured 1.1e-6 vs
    # 3.0e-6 relative for p = 0.5, s ~ 5e5 in fp32).
    tl.store(Out, pow(tl.sum(acc), 1.0 / p))


@libentry()
@triton.jit
def dist_max_kernel(X, Y, Out, N, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for start in range(0, N, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < N
        diff = _load_abs_diff(X, Y, idx, mask)
        acc = tl.maximum(acc, tl.where(mask, diff, -float("inf")))
    tl.store(Out, tl.max(acc))


@libentry()
@triton.jit
def dist_min_kernel(X, Y, Out, N, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    acc = tl.full((BLOCK_SIZE,), float("inf"), dtype=tl.float32)
    for start in range(0, N, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < N
        diff = _load_abs_diff(X, Y, idx, mask)
        acc = tl.minimum(acc, tl.where(mask, diff, float("inf")))
    tl.store(Out, tl.min(acc))


# Persistent stage-1 kernels: grid is capped at MAX_GRID, each program loops
# over BLOCK_SIZE tiles with a grid stride and writes one partial to Mid.
@libentry()
@triton.jit
def dist_p2_kernel_1(X, Y, Mid, N, GRID, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for start in range(pid * BLOCK_SIZE, N, GRID * BLOCK_SIZE):
        idx = start + offsets
        mask = idx < N
        diff = _load_abs_diff(X, Y, idx, mask)
        acc += tl.where(mask, diff * diff, 0.0)
    tl.store(Mid + pid, tl.sum(acc))


@libentry()
@triton.jit
def dist_p1_kernel_1(X, Y, Mid, N, GRID, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for start in range(pid * BLOCK_SIZE, N, GRID * BLOCK_SIZE):
        idx = start + offsets
        mask = idx < N
        diff = _load_abs_diff(X, Y, idx, mask)
        acc += tl.where(mask, diff, 0.0)
    tl.store(Mid + pid, tl.sum(acc))


@libentry()
@triton.jit
def dist_p0_kernel_1(X, Y, Mid, N, GRID, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for start in range(pid * BLOCK_SIZE, N, GRID * BLOCK_SIZE):
        idx = start + offsets
        mask = idx < N
        diff = _load_abs_diff(X, Y, idx, mask)
        acc += tl.where(mask, (diff != 0).to(tl.float32), 0.0)
    tl.store(Mid + pid, tl.sum(acc))


@libentry()
@triton.jit
def dist_general_kernel_1(X, Y, Mid, N, p, GRID, BLOCK_SIZE: tl.constexpr):
    p = p.to(tl.float32)
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for start in range(pid * BLOCK_SIZE, N, GRID * BLOCK_SIZE):
        idx = start + offsets
        mask = idx < N
        diff = _load_abs_diff(X, Y, idx, mask)
        acc += tl.where(mask, exp2(p * log2(diff)), 0.0)
    tl.store(Mid + pid, tl.sum(acc))


@libentry()
@triton.jit
def dist_max_kernel_1(X, Y, Mid, N, GRID, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    acc = tl.full((BLOCK_SIZE,), -float("inf"), dtype=tl.float32)
    for start in range(pid * BLOCK_SIZE, N, GRID * BLOCK_SIZE):
        idx = start + offsets
        mask = idx < N
        diff = _load_abs_diff(X, Y, idx, mask)
        acc = tl.maximum(acc, tl.where(mask, diff, -float("inf")))
    tl.store(Mid + pid, tl.max(acc))


@libentry()
@triton.jit
def dist_min_kernel_1(X, Y, Mid, N, GRID, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    acc = tl.full((BLOCK_SIZE,), float("inf"), dtype=tl.float32)
    for start in range(pid * BLOCK_SIZE, N, GRID * BLOCK_SIZE):
        idx = start + offsets
        mask = idx < N
        diff = _load_abs_diff(X, Y, idx, mask)
        acc = tl.minimum(acc, tl.where(mask, diff, float("inf")))
    tl.store(Mid + pid, tl.min(acc))


# Stage-2 reduces the <= MAX_GRID partials to the final scalar.
@libentry()
@triton.jit
def dist_p2_kernel_2(Mid, Out, MID_SIZE, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < MID_SIZE
    acc = tl.load(Mid + offsets, mask=mask, other=0.0)
    tl.store(Out, tl.sqrt(tl.sum(acc)))


@libentry()
@triton.jit
def dist_sum_kernel_2(Mid, Out, MID_SIZE, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < MID_SIZE
    acc = tl.load(Mid + offsets, mask=mask, other=0.0)
    tl.store(Out, tl.sum(acc))


@libentry()
@triton.jit
def dist_general_kernel_2(Mid, Out, p, MID_SIZE, BLOCK_SIZE: tl.constexpr):
    p = p.to(tl.float32)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < MID_SIZE
    acc = tl.load(Mid + offsets, mask=mask, other=0.0)
    # libdevice pow keeps extra internal precision in the log domain,
    # ~3x tighter than a raw exp2/log2 roundtrip (measured 1.1e-6 vs
    # 3.0e-6 relative for p = 0.5, s ~ 5e5 in fp32).
    tl.store(Out, pow(tl.sum(acc), 1.0 / p))


@libentry()
@triton.jit
def dist_max_kernel_2(Mid, Out, MID_SIZE, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < MID_SIZE
    acc = tl.load(Mid + offsets, mask=mask, other=-float("inf"))
    tl.store(Out, tl.max(acc))


@libentry()
@triton.jit
def dist_min_kernel_2(Mid, Out, MID_SIZE, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < MID_SIZE
    acc = tl.load(Mid + offsets, mask=mask, other=float("inf"))
    tl.store(Out, tl.min(acc))


def dist(input, other, p=2):
    logger.debug("GEMS_ASCEND DIST")
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

        grid = min(MAX_GRID, triton.cdiv(n, BLOCK_SIZE))
        mid = torch.empty(grid, dtype=torch.float32, device=input.device)
        MID_BLOCK = triton.next_power_of_2(grid)

        if p == 2:
            dist_p2_kernel_1[(grid,)](input, other, mid, n, grid, BLOCK_SIZE=BLOCK_SIZE)
            dist_p2_kernel_2[(1,)](mid, out, grid, BLOCK_SIZE=MID_BLOCK)
        elif p == 1:
            dist_p1_kernel_1[(grid,)](input, other, mid, n, grid, BLOCK_SIZE=BLOCK_SIZE)
            dist_sum_kernel_2[(1,)](mid, out, grid, BLOCK_SIZE=MID_BLOCK)
        elif p == 0:
            dist_p0_kernel_1[(grid,)](input, other, mid, n, grid, BLOCK_SIZE=BLOCK_SIZE)
            dist_sum_kernel_2[(1,)](mid, out, grid, BLOCK_SIZE=MID_BLOCK)
        elif p == float("inf"):
            dist_max_kernel_1[(grid,)](
                input, other, mid, n, grid, BLOCK_SIZE=BLOCK_SIZE
            )
            dist_max_kernel_2[(1,)](mid, out, grid, BLOCK_SIZE=MID_BLOCK)
        elif p == float("-inf"):
            dist_min_kernel_1[(grid,)](
                input, other, mid, n, grid, BLOCK_SIZE=BLOCK_SIZE
            )
            dist_min_kernel_2[(1,)](mid, out, grid, BLOCK_SIZE=MID_BLOCK)
        else:
            dist_general_kernel_1[(grid,)](
                input, other, mid, n, p, grid, BLOCK_SIZE=BLOCK_SIZE
            )
            dist_general_kernel_2[(1,)](mid, out, p, grid, BLOCK_SIZE=MID_BLOCK)

    return out
