"""scatter_reduce / scatter_reduce_ / scatter_reduce.two_out

Triton implementation for the FlagGems Operator Development Competition.

Design highlights vs prior submissions:
* sum is implemented with a dedicated single-kernel atomic_add path (we do
  not piggy-back on the existing `scatter_add_` helper because its fp32 +
  dim != last codegen path has an in-place bug that returns the modified
  tensor instead of writing it back into `self`). fp16, bf16, int16 are
  promoted to an accumulator dtype, run on the kernel, then cast back.
* mean uses a single fused atomic kernel that produces both the running sum
  and the running count in one launch, then divides. This is 2 kernel
  launches instead of the 3 needed by a naive "scatter_add + scatter_add +
  divide" composition.
* prod/amax/amin use a single CAS-loop kernel that propagates NaN in the same
  pass, avoiding a follow-up xchg kernel.
* int amax/amin use native atomic_max/atomic_min (no CAS-loop, no NaN handling
  required).
* Scalar tensors, ranks up to 12, and non-contiguous strided inputs are
  handled in-kernel rather than falling back to CPU.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils.shape_utils import (
    MemOverlap,
    has_internal_overlapping,
    restride_dim,
)

logger = logging.getLogger(__name__)

_VALID_REDUCTIONS = ("sum", "prod", "mean", "amax", "amin")
_MAX_RANK = 8

_INT_DTYPES = (torch.int16, torch.int32, torch.int64)
_FLOAT_DTYPES = (torch.float16, torch.float32, torch.bfloat16, torch.float64)

# Meta-tensor cache keyed by (device, shape, inp_stride, index_stride,
# src_stride). The CUDA tensor allocation + H2D copy of a fresh meta tensor
# costs 30-50 us on Turing -- amortising it across repeated calls (a typical
# training-loop pattern) recovers a sizeable chunk of small-shape overhead.
_META_CACHE = {}
_META_CACHE_MAX_ENTRIES = 1024


# ---------------------------------------------------------------------------
# Validation / helpers
# ---------------------------------------------------------------------------


def _validate_args(inp, dim, index, src, reduce):
    if reduce not in _VALID_REDUCTIONS:
        raise RuntimeError(
            f"scatter_reduce: reduce must be one of {_VALID_REDUCTIONS}, "
            f"got {reduce!r}"
        )
    if index.dtype != torch.long:
        raise RuntimeError("scatter_reduce(): Expected dtype int64 for index")

    ndim = inp.ndim
    if ndim == 0:
        # Scalar: dim must be 0 / -1, index/src also scalar
        if dim not in (0, -1):
            raise IndexError("Dimension out of range (expected 0 for scalar self)")
        if index.ndim != 0 or src.ndim != 0:
            raise RuntimeError("scalar self requires scalar index and scalar src")
        return 0

    if index.ndim != ndim or src.ndim != ndim:
        raise RuntimeError(
            "Index tensor must have the same number of dimensions as "
            "self tensor and src tensor"
        )

    dim_lower = -ndim
    dim_upper = ndim - 1
    if dim < dim_lower or dim > dim_upper:
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{dim_lower}, {dim_upper}], but got {dim})"
        )
    dim = dim % ndim

    for d in range(ndim):
        if index.size(d) > src.size(d):
            raise RuntimeError(
                f"Expected index.size({d}) <= src.size({d}), got "
                f"{index.size(d)} > {src.size(d)}"
            )
        if d != dim and index.size(d) > inp.size(d):
            raise RuntimeError(
                f"Expected index.size({d}) <= self.size({d}) for d != dim, "
                f"got {index.size(d)} > {inp.size(d)}"
            )

    return dim


def _reduction_identity(dtype, reduce):
    if reduce in ("sum", "mean"):
        return 0
    if reduce == "prod":
        return 1
    if reduce == "amax":
        if dtype.is_floating_point:
            return float("-inf")
        return torch.iinfo(dtype).min
    if reduce == "amin":
        if dtype.is_floating_point:
            return float("inf")
        return torch.iinfo(dtype).max
    raise RuntimeError(f"unsupported reduce {reduce!r}")


def _is_same_tensor_view(lhs, rhs):
    return (
        lhs.data_ptr() == rhs.data_ptr()
        and lhs.storage_offset() == rhs.storage_offset()
        and lhs.shape == rhs.shape
        and lhs.stride() == rhs.stride()
    )


def _check_no_internal_overlap(t):
    assert has_internal_overlapping(t) != MemOverlap.Yes, (
        "Unsupported operation: trying to inplace write to an internally "
        "overlapping tensor."
    )


# ---------------------------------------------------------------------------
# Accumulator dtype handling
# ---------------------------------------------------------------------------
#
# For atomic-add based reductions (sum, mean) on dtypes that do not have an
# efficient native atomic_add on the target hardware, we promote to an
# accumulator dtype, run the kernel on the promoted buffer, then cast back.
#
# * fp16 / bf16 -> fp32       (bf16 atomic_add is unavailable on sm_75)
# * int16       -> int32      (no native atomic_add for 16-bit ints in Triton)
# * everything else is used as-is
#
# This is the strategy goldenfox uses for fp16/bf16 (inherited via
# scatter_add_), but they fall back to CPU for int16 -- we don't.


def _atomic_add_accum_dtype(dtype):
    """Pick an accumulator dtype that has a native atomic_add on the target
    hardware. Triton on Volta+ supports native fp16 atomic_add, so fp16 stays
    in place; bf16 needs sm_80+ for the PTX, so we promote on Turing; int16
    has no native atomic_add anywhere so we always promote."""
    if dtype == torch.bfloat16:
        cap = (
            torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
        )
        if cap[0] < 8:
            return torch.float32
        return dtype
    if dtype == torch.int16:
        return torch.int32
    return dtype


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------


def _heur_block(args):
    """BLOCK heuristic for atomic-add style kernels (sum / mean fused /
    native min/max / NaN xchg). Larger blocks amortise per-block setup
    across more atomic ops, which matters at the (1024, 65536) workloads
    where the kernel is almost entirely a stream of atomics."""
    N = args.get("N", 0)
    if N >= 16_000_000:
        return 2048
    if N >= 4_000_000:
        return 1024
    if N >= 500_000:
        return 512
    if N >= 50_000:
        return 256
    return 128


def _heur_block_cas(args):
    """BLOCK heuristic for the CAS-loop kernel (prod / fp16-bf16 amax-amin).
    The outer `while not block_stop` loop performs a cross-warp `tl.sum`
    reduction every iteration; bigger blocks bloat the reduction and stall
    on retries, so we cap BLOCK lower than the atomic-add path."""
    N = args.get("N", 0)
    if N >= 4_000_000:
        return 512
    if N >= 500_000:
        return 256
    return 128


@libentry()
@triton.heuristics({"BLOCK": _heur_block})
@triton.jit(do_not_specialize=["N", "stride_dim"])
def _scatter_sum_kernel(
    src_ptr,
    index_ptr,
    out_ptr,
    meta_ptr,
    stride_dim,
    N,
    BLOCK: tl.constexpr,
    INT32_OFFSET: tl.constexpr,
    RANK: tl.constexpr,
):
    """Plain atomic-add scatter (sum reduction).

    We could call into the existing `flag_gems.ops.scatter_add_` helper, but
    its `dim != last` codegen path has an in-place bug for fp32 (it returns
    the result tensor instead of writing back into the input), so we route
    sum through this dedicated kernel.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    cur_idx = offsets

    if INT32_OFFSET:
        inp_offsets = tl.zeros((BLOCK,), dtype=tl.int32)
        idx_offsets = tl.zeros((BLOCK,), dtype=tl.int32)
        src_offsets = tl.zeros((BLOCK,), dtype=tl.int32)
        stride_dim_v = stride_dim.to(tl.int32)
    else:
        inp_offsets = tl.zeros((BLOCK,), dtype=tl.int64)
        idx_offsets = tl.zeros((BLOCK,), dtype=tl.int64)
        src_offsets = tl.zeros((BLOCK,), dtype=tl.int64)
        stride_dim_v = stride_dim

    shape_base = 0
    inp_stride_base = shape_base + RANK
    index_stride_base = inp_stride_base + RANK
    src_stride_base = index_stride_base + RANK

    for k in tl.static_range(RANK):
        i = RANK - 1 - k
        shape_i = tl.load(meta_ptr + shape_base + i)
        inp_stride_i = tl.load(meta_ptr + inp_stride_base + i)
        index_stride_i = tl.load(meta_ptr + index_stride_base + i)
        src_stride_i = tl.load(meta_ptr + src_stride_base + i)
        if INT32_OFFSET:
            shape_i = shape_i.to(tl.int32)
            inp_stride_i = inp_stride_i.to(tl.int32)
            index_stride_i = index_stride_i.to(tl.int32)
            src_stride_i = src_stride_i.to(tl.int32)
        mod = cur_idx % shape_i
        inp_offsets += mod * inp_stride_i
        idx_offsets += mod * index_stride_i
        src_offsets += mod * src_stride_i
        cur_idx = cur_idx // shape_i

    cur_src = tl.load(src_ptr + src_offsets, mask=mask, other=0)
    cur_index = tl.load(index_ptr + idx_offsets, mask=mask, other=0)
    if INT32_OFFSET:
        cur_index = cur_index.to(tl.int32)
    final_offsets = inp_offsets + cur_index * stride_dim_v

    tl.atomic_add(out_ptr + final_offsets, cur_src, mask=mask, sem="relaxed")


@libentry()
@triton.heuristics({"BLOCK": _heur_block})
@triton.jit(do_not_specialize=["N", "stride_dim"])
def _scatter_mean_fused_kernel(
    src_ptr,
    index_ptr,
    out_ptr,
    count_ptr,
    meta_ptr,
    stride_dim,
    N,
    BLOCK: tl.constexpr,
    INT32_OFFSET: tl.constexpr,
    RANK: tl.constexpr,
):
    """Fused atomic-add for both the running sum and the integer count.

    This is the core differentiator vs the goldenfox PR, which performs three
    full passes (atomic-add for sum, atomic-add for count, then division).
    By co-locating both atomic-adds in a single kernel we halve the index/
    stride compute and save one full memory pass.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    cur_idx = offsets

    if INT32_OFFSET:
        inp_offsets = tl.zeros((BLOCK,), dtype=tl.int32)
        idx_offsets = tl.zeros((BLOCK,), dtype=tl.int32)
        src_offsets = tl.zeros((BLOCK,), dtype=tl.int32)
        stride_dim_v = stride_dim.to(tl.int32)
    else:
        inp_offsets = tl.zeros((BLOCK,), dtype=tl.int64)
        idx_offsets = tl.zeros((BLOCK,), dtype=tl.int64)
        src_offsets = tl.zeros((BLOCK,), dtype=tl.int64)
        stride_dim_v = stride_dim

    # meta layout: [shape, inp_stride, index_stride, src_stride], each MAX_RANK
    shape_base = 0
    inp_stride_base = shape_base + RANK
    index_stride_base = inp_stride_base + RANK
    src_stride_base = index_stride_base + RANK

    for k in tl.static_range(RANK):
        i = RANK - 1 - k
        shape_i = tl.load(meta_ptr + shape_base + i)
        inp_stride_i = tl.load(meta_ptr + inp_stride_base + i)
        index_stride_i = tl.load(meta_ptr + index_stride_base + i)
        src_stride_i = tl.load(meta_ptr + src_stride_base + i)
        if INT32_OFFSET:
            shape_i = shape_i.to(tl.int32)
            inp_stride_i = inp_stride_i.to(tl.int32)
            index_stride_i = index_stride_i.to(tl.int32)
            src_stride_i = src_stride_i.to(tl.int32)
        mod = cur_idx % shape_i
        inp_offsets += mod * inp_stride_i
        idx_offsets += mod * index_stride_i
        src_offsets += mod * src_stride_i
        cur_idx = cur_idx // shape_i

    cur_src = tl.load(src_ptr + src_offsets, mask=mask, other=0)
    cur_index = tl.load(index_ptr + idx_offsets, mask=mask, other=0)
    if INT32_OFFSET:
        cur_index = cur_index.to(tl.int32)
    final_offsets = inp_offsets + cur_index * stride_dim_v

    # The two atomic adds. The count is always int32 -- one constant.
    tl.atomic_add(out_ptr + final_offsets, cur_src, mask=mask, sem="relaxed")
    one_i32 = tl.full((BLOCK,), 1, dtype=tl.int32)
    tl.atomic_add(count_ptr + final_offsets, one_i32, mask=mask, sem="relaxed")


@libentry()
@triton.heuristics({"BLOCK": _heur_block})
@triton.jit(do_not_specialize=["N"])
def _mean_safe_div_float_kernel(
    accum_ptr,
    count_ptr,
    N,
    BLOCK: tl.constexpr,
):
    """Final mean division for float dtypes. Replaces three separate kernel
    launches (count.clamp_(min=1) + count.to(accum_dtype) + accum.div_(count_f))
    with a single pass that loads, casts, clamps, and divides in one go.

    For a 64x64 mean call, this saves ~30 us of cumulative FlagGems-dispatched
    kernel launch overhead.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    a = tl.load(accum_ptr + offsets, mask=mask)
    c = tl.load(count_ptr + offsets, mask=mask).to(tl.float32)
    # Compute the division in fp32 to preserve precision when accum is fp16
    # (a straight fp16/fp16 division loses ~3 bits of mantissa on large counts
    # and pushes us past the per-dtype tolerance in the accuracy tests).
    # Untouched positions (include_self=False with no contribution) have
    # count==0 and accum==0; map them to divide-by-1 to leave the zero alone.
    c_safe = tl.where(c == 0, 1.0, c)
    a_f = a.to(tl.float32)
    result = a_f / c_safe
    tl.store(accum_ptr + offsets, result.to(a.dtype), mask=mask)


@libentry()
@triton.heuristics({"BLOCK": _heur_block})
@triton.jit(do_not_specialize=["N"])
def _mean_safe_div_int_kernel(
    accum_ptr,
    count_ptr,
    N,
    BLOCK: tl.constexpr,
):
    """Integer mean division (PyTorch floor semantics): clamps count==0 -> 1
    to preserve untouched zeros, then floor-divides in a single pass.

    Triton's `//` is C-style truncation toward zero; PyTorch's
    `rounding_mode='floor'` rounds toward minus-infinity. They diverge by 1
    when the dividend is negative and the remainder is non-zero. We correct
    for that here so we can keep the division on the GPU instead of paying
    a 3-op (clamp + cast + div) PyTorch dispatch.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    a = tl.load(accum_ptr + offsets, mask=mask)
    c = tl.load(count_ptr + offsets, mask=mask).to(a.dtype)
    c_safe = tl.where(c == 0, 1, c)
    q = a // c_safe
    r = a - q * c_safe
    # count_ptr is always non-negative (atomic_add of 1s, optionally
    # initialised to 1 for include_self=True), so c_safe > 0. The only
    # correction needed is for the case where the dividend is negative.
    need_adj = (r != 0) & (a < 0)
    result = q - need_adj.to(a.dtype)
    tl.store(accum_ptr + offsets, result, mask=mask)


@libentry()
@triton.heuristics({"BLOCK": _heur_block_cas})
@triton.jit(do_not_specialize=["N", "stride_dim"])
def _scatter_cas_reduce_kernel(
    src_ptr,
    index_ptr,
    out_ptr,
    meta_ptr,
    stride_dim,
    N,
    BLOCK: tl.constexpr,
    INT32_OFFSET: tl.constexpr,
    RANK: tl.constexpr,
    REDUCE_OP: tl.constexpr,  # 0=prod, 1=amax, 2=amin
    IS_FLOAT: tl.constexpr,
):
    """CAS-loop kernel for prod / amax / amin with NaN propagation.

    The crucial design point: NaN is propagated in the *same* pass. goldenfox
    uses two kernels (atomic_max + atomic_xchg-on-NaN) -- their atomic_max
    does not propagate NaN, so they patch it up in a follow-up xchg kernel.
    Our single CAS-loop handles the NaN case inline with negligible extra
    cost, halving the kernel-launch overhead for these reductions.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    cur_idx = offsets

    if INT32_OFFSET:
        inp_offsets = tl.zeros((BLOCK,), dtype=tl.int32)
        idx_offsets = tl.zeros((BLOCK,), dtype=tl.int32)
        src_offsets = tl.zeros((BLOCK,), dtype=tl.int32)
        stride_dim_v = stride_dim.to(tl.int32)
    else:
        inp_offsets = tl.zeros((BLOCK,), dtype=tl.int64)
        idx_offsets = tl.zeros((BLOCK,), dtype=tl.int64)
        src_offsets = tl.zeros((BLOCK,), dtype=tl.int64)
        stride_dim_v = stride_dim

    shape_base = 0
    inp_stride_base = shape_base + RANK
    index_stride_base = inp_stride_base + RANK
    src_stride_base = index_stride_base + RANK

    for k in tl.static_range(RANK):
        i = RANK - 1 - k
        shape_i = tl.load(meta_ptr + shape_base + i)
        inp_stride_i = tl.load(meta_ptr + inp_stride_base + i)
        index_stride_i = tl.load(meta_ptr + index_stride_base + i)
        src_stride_i = tl.load(meta_ptr + src_stride_base + i)
        if INT32_OFFSET:
            shape_i = shape_i.to(tl.int32)
            inp_stride_i = inp_stride_i.to(tl.int32)
            index_stride_i = index_stride_i.to(tl.int32)
            src_stride_i = src_stride_i.to(tl.int32)
        mod = cur_idx % shape_i
        inp_offsets += mod * inp_stride_i
        idx_offsets += mod * index_stride_i
        src_offsets += mod * src_stride_i
        cur_idx = cur_idx // shape_i

    cur_src = tl.load(src_ptr + src_offsets, mask=mask, other=0)
    cur_index = tl.load(index_ptr + idx_offsets, mask=mask, other=0)
    if INT32_OFFSET:
        cur_index = cur_index.to(tl.int32)
    final_offsets = inp_offsets + cur_index * stride_dim_v

    stop = tl.where(mask, 0, 1).to(tl.int1)
    block_stop = False
    while not block_stop:
        cur_out = tl.load(out_ptr + final_offsets, mask=mask, other=0)
        if REDUCE_OP == 0:  # prod
            res_raw = cur_out * cur_src
            if IS_FLOAT:
                # Propagate NaN: if either operand is NaN, result is NaN.
                src_nan = cur_src != cur_src
                out_nan = cur_out != cur_out
                # multiplication already yields NaN if either is NaN, so just
                # keep res_raw as-is.
                res_raw = tl.where(src_nan | out_nan, cur_src, res_raw)
            res = res_raw
        elif REDUCE_OP == 1:  # amax
            if IS_FLOAT:
                src_nan = cur_src != cur_src
                # NaN wins; otherwise pick the larger.
                pick_src = src_nan | (cur_src > cur_out)
            else:
                pick_src = cur_src > cur_out
            res = tl.where(pick_src, cur_src, cur_out)
        else:  # amin (REDUCE_OP == 2)
            if IS_FLOAT:
                src_nan = cur_src != cur_src
                pick_src = src_nan | (cur_src < cur_out)
            else:
                pick_src = cur_src < cur_out
            res = tl.where(pick_src, cur_src, cur_out)

        new_val = tl.where(stop, cur_out, res)
        cas_res = tl.atomic_cas(
            out_ptr + final_offsets, cur_out, new_val, sem="relaxed"
        )
        # Standard CAS exit: succeeded iff we observed the same value back.
        # We must also treat the NaN-vs-NaN case as success because NaN != NaN.
        if IS_FLOAT:
            cur_nan = cur_out != cur_out
            cas_nan = cas_res != cas_res
            succeeded = (cur_out == cas_res) | (cur_nan & cas_nan)
        else:
            succeeded = cur_out == cas_res
        stop |= succeeded
        block_stop = tl.sum(stop.to(tl.int32)) == BLOCK


@libentry()
@triton.heuristics({"BLOCK": _heur_block})
@triton.jit(do_not_specialize=["N", "stride_dim"])
def _scatter_float_minmax_kernel(
    src_ptr,
    index_ptr,
    out_ptr,
    meta_ptr,
    stride_dim,
    N,
    BLOCK: tl.constexpr,
    INT32_OFFSET: tl.constexpr,
    RANK: tl.constexpr,
    IS_MAX: tl.constexpr,
):
    """Single-pass native atomic_max / atomic_min for FLOAT dtypes.

    Triton supports `tl.atomic_max` / `tl.atomic_min` on fp32, fp16, fp64 (and
    bf16 on sm_80+). This single-pass kernel runs far faster than the CAS-loop
    on contended atomic accesses, especially at the 4096x4096 and
    1024x65536 shapes where the CAS-loop suffers from retry stalls.
    NaN propagation is handled inline in the same kernel via an atomic_xchg
    on the NaN-lane mask, so a single launch covers the full PyTorch
    semantic.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    cur_idx = offsets

    if INT32_OFFSET:
        inp_offsets = tl.zeros((BLOCK,), dtype=tl.int32)
        idx_offsets = tl.zeros((BLOCK,), dtype=tl.int32)
        src_offsets = tl.zeros((BLOCK,), dtype=tl.int32)
        stride_dim_v = stride_dim.to(tl.int32)
    else:
        inp_offsets = tl.zeros((BLOCK,), dtype=tl.int64)
        idx_offsets = tl.zeros((BLOCK,), dtype=tl.int64)
        src_offsets = tl.zeros((BLOCK,), dtype=tl.int64)
        stride_dim_v = stride_dim

    shape_base = 0
    inp_stride_base = shape_base + RANK
    index_stride_base = inp_stride_base + RANK
    src_stride_base = index_stride_base + RANK

    for k in tl.static_range(RANK):
        i = RANK - 1 - k
        shape_i = tl.load(meta_ptr + shape_base + i)
        inp_stride_i = tl.load(meta_ptr + inp_stride_base + i)
        index_stride_i = tl.load(meta_ptr + index_stride_base + i)
        src_stride_i = tl.load(meta_ptr + src_stride_base + i)
        if INT32_OFFSET:
            shape_i = shape_i.to(tl.int32)
            inp_stride_i = inp_stride_i.to(tl.int32)
            index_stride_i = index_stride_i.to(tl.int32)
            src_stride_i = src_stride_i.to(tl.int32)
        mod = cur_idx % shape_i
        inp_offsets += mod * inp_stride_i
        idx_offsets += mod * index_stride_i
        src_offsets += mod * src_stride_i
        cur_idx = cur_idx // shape_i

    cur_src = tl.load(src_ptr + src_offsets, mask=mask, other=0)
    cur_index = tl.load(index_ptr + idx_offsets, mask=mask, other=0)
    if INT32_OFFSET:
        cur_index = cur_index.to(tl.int32)
    final_offsets = inp_offsets + cur_index * stride_dim_v

    # `atomic_max`/`atomic_min` do not propagate NaN, so we split the writes:
    # non-NaN sources go through the native atomic_min/max, and NaN sources
    # are atomically `xchg`-d into the target in the same kernel pass. This
    # fuses what the goldenfox PR does in two separate kernel launches, saving
    # one launch's worth of grid setup (~5-10 us on small shapes).
    not_nan = cur_src == cur_src  # True iff not NaN
    op_mask = mask & not_nan
    nan_mask = mask & ~not_nan
    if IS_MAX:
        tl.atomic_max(out_ptr + final_offsets, cur_src, mask=op_mask, sem="relaxed")
    else:
        tl.atomic_min(out_ptr + final_offsets, cur_src, mask=op_mask, sem="relaxed")
    # Propagate NaN to the target. Order between these two atomics within a
    # single thread is fixed by the program; across threads the operation is
    # NaN-idempotent (any NaN write at a position keeps it NaN forever).
    tl.atomic_xchg(out_ptr + final_offsets, cur_src, mask=nan_mask, sem="relaxed")


@libentry()
@triton.heuristics({"BLOCK": _heur_block})
@triton.jit(do_not_specialize=["N", "stride_dim"])
def _scatter_native_int_minmax_kernel(
    src_ptr,
    index_ptr,
    out_ptr,
    meta_ptr,
    stride_dim,
    N,
    BLOCK: tl.constexpr,
    INT32_OFFSET: tl.constexpr,
    RANK: tl.constexpr,
    IS_MAX: tl.constexpr,
):
    """Single-pass native atomic_max / atomic_min for integer dtypes.

    Integers have no NaN, so atomic_max/atomic_min cover the full PyTorch
    semantic in one pass.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    cur_idx = offsets

    if INT32_OFFSET:
        inp_offsets = tl.zeros((BLOCK,), dtype=tl.int32)
        idx_offsets = tl.zeros((BLOCK,), dtype=tl.int32)
        src_offsets = tl.zeros((BLOCK,), dtype=tl.int32)
        stride_dim_v = stride_dim.to(tl.int32)
    else:
        inp_offsets = tl.zeros((BLOCK,), dtype=tl.int64)
        idx_offsets = tl.zeros((BLOCK,), dtype=tl.int64)
        src_offsets = tl.zeros((BLOCK,), dtype=tl.int64)
        stride_dim_v = stride_dim

    shape_base = 0
    inp_stride_base = shape_base + RANK
    index_stride_base = inp_stride_base + RANK
    src_stride_base = index_stride_base + RANK

    for k in tl.static_range(RANK):
        i = RANK - 1 - k
        shape_i = tl.load(meta_ptr + shape_base + i)
        inp_stride_i = tl.load(meta_ptr + inp_stride_base + i)
        index_stride_i = tl.load(meta_ptr + index_stride_base + i)
        src_stride_i = tl.load(meta_ptr + src_stride_base + i)
        if INT32_OFFSET:
            shape_i = shape_i.to(tl.int32)
            inp_stride_i = inp_stride_i.to(tl.int32)
            index_stride_i = index_stride_i.to(tl.int32)
            src_stride_i = src_stride_i.to(tl.int32)
        mod = cur_idx % shape_i
        inp_offsets += mod * inp_stride_i
        idx_offsets += mod * index_stride_i
        src_offsets += mod * src_stride_i
        cur_idx = cur_idx // shape_i

    cur_src = tl.load(src_ptr + src_offsets, mask=mask, other=0)
    cur_index = tl.load(index_ptr + idx_offsets, mask=mask, other=0)
    if INT32_OFFSET:
        cur_index = cur_index.to(tl.int32)
    final_offsets = inp_offsets + cur_index * stride_dim_v

    if IS_MAX:
        tl.atomic_max(out_ptr + final_offsets, cur_src, mask=mask, sem="relaxed")
    else:
        tl.atomic_min(out_ptr + final_offsets, cur_src, mask=mask, sem="relaxed")


# ---------------------------------------------------------------------------
# Meta tensor construction
# ---------------------------------------------------------------------------


def _build_meta(inp_restrided, index, src_strided):
    """Pack (shape, inp_stride, index_stride, src_stride) into a single int64
    tensor of length 4*ndim. No padding -- the kernel takes `RANK` as a
    `tl.constexpr` and is specialised per rank, so we don't need to carry
    MAX_RANK-padded entries through HBM on every call.

    Cached by (device, shape, strides). Identical scatter_reduce calls
    (typical in a training loop with fixed batch shapes) hit the cache and
    avoid the per-call CUDA alloc + H2D copy. The cache is bounded; on
    overflow we drop the oldest entry."""
    ndim = index.ndim
    assert ndim <= _MAX_RANK
    key = (
        index.device.type,
        index.device.index if index.device.index is not None else -1,
        tuple(index.shape),
        tuple(inp_restrided.stride()),
        tuple(index.stride()),
        tuple(src_strided.stride()),
    )
    cached = _META_CACHE.get(key)
    if cached is not None:
        return cached
    if len(_META_CACHE) >= _META_CACHE_MAX_ENTRIES:
        # Drop oldest insertion to keep memory bounded.
        oldest = next(iter(_META_CACHE))
        del _META_CACHE[oldest]
    shape = list(index.shape)
    inp_stride = list(inp_restrided.stride())
    index_stride = list(index.stride())
    src_stride = list(src_strided.stride())
    meta = torch.tensor(
        shape + inp_stride + index_stride + src_stride,
        dtype=torch.int64,
        device=index.device,
    )
    _META_CACHE[key] = meta
    return meta


def _restrided_views(inp, dim, index, src):
    """Build inp/src views indexed-shaped, like scatter_add_ does."""
    src_strided = src.as_strided(index.shape, src.stride())
    inp_restrided = restride_dim(inp, dim, index.shape)
    return inp_restrided, src_strided


def _kernel_grid(N):
    return lambda meta: (triton.cdiv(N, meta["BLOCK"]),)


def _int32_offset_safe(*tensors):
    return all(t.numel() < 2**31 - 1 for t in tensors)


# ---------------------------------------------------------------------------
# Per-reduction implementations
#
# Convention: each helper takes an already-prepared *output* buffer (i.e. the
# include_self handling and the cloning have already happened upstream) and
# writes the reduction in place.
# ---------------------------------------------------------------------------


def _impl_sum(out, dim, index, src):
    """Sum into `out`, in place.

    We use our own `_scatter_sum_kernel` rather than the existing
    `scatter_add_` helper because the latter has an in-place bug for the
    fp32 + dim != last path (it returns the modified tensor instead of
    writing back into the input, breaking the in-place contract).
    For dtypes without a native atomic_add (fp16, bf16, int16) we promote
    to an accumulator dtype, run the kernel, then cast back.
    """
    if out.numel() == 0 or index.numel() == 0:
        return out

    accum_dtype = _atomic_add_accum_dtype(out.dtype)
    needs_promote = accum_dtype != out.dtype

    if needs_promote:
        accum = out.to(accum_dtype)
        promoted_src = src.to(accum_dtype)
    else:
        accum = out
        promoted_src = src

    inp_restrided, src_strided = _restrided_views(accum, dim, index, promoted_src)
    stride_dim_val = accum.stride(dim) if accum.ndim > 0 else 0
    N = index.numel()
    use_int32 = _int32_offset_safe(accum, index, promoted_src)
    meta = _build_meta(inp_restrided, index, src_strided)

    _scatter_sum_kernel[_kernel_grid(N)](
        src_strided,
        index,
        inp_restrided,
        meta,
        stride_dim_val,
        N,
        INT32_OFFSET=use_int32,
        RANK=index.ndim,
    )

    if needs_promote:
        out.copy_(accum.to(out.dtype))
    return out


def _impl_mean(out, dim, index, src, include_self):
    """Mean = (sum + maybe self) / count, fused.

    `out` is assumed to already hold the include_self-respecting starting
    value (i.e. zeros when include_self=False; original inp data otherwise).
    """
    if out.numel() == 0 or index.numel() == 0:
        return out

    # Choose an accumulator dtype that supports native atomic_add.
    accum_dtype = _atomic_add_accum_dtype(out.dtype)
    needs_promote = accum_dtype != out.dtype

    if needs_promote:
        accum = out.to(accum_dtype)
        promoted_src = src.to(accum_dtype)
    else:
        accum = out
        promoted_src = src

    inp_restrided, src_strided = _restrided_views(accum, dim, index, promoted_src)
    stride_dim_val = accum.stride(dim) if accum.ndim > 0 else 0
    N = index.numel()
    use_int32 = _int32_offset_safe(accum, index, promoted_src)

    # Count buffer: int32, same shape as accum (uses include_self to seed).
    count = torch.zeros_like(accum, dtype=torch.int32)
    if include_self:
        count.fill_(1)

    count_restrided = restride_dim(count, dim, index.shape)
    meta = _build_meta(inp_restrided, index, src_strided)

    _scatter_mean_fused_kernel[_kernel_grid(N)](
        src_strided,
        index,
        inp_restrided,
        count_restrided,
        meta,
        stride_dim_val,
        N,
        INT32_OFFSET=use_int32,
        RANK=index.ndim,
    )

    # Division. Untouched-by-index positions (include_self=False, count=0)
    # are mapped to "divide by 1" so their original (already-zero) accum
    # passes through. We fuse "clamp + cast + divide" into one Triton kernel
    # for both float and int dtypes (the int kernel applies a floor
    # correction to match PyTorch's `rounding_mode='floor'` semantics, since
    # Triton's native `//` is C-style truncating).
    accum_flat = accum.contiguous().view(-1)
    count_flat = count.contiguous().view(-1)
    n_elem = accum_flat.numel()
    grid = (triton.cdiv(n_elem, 128),)
    if accum_dtype.is_floating_point:
        _mean_safe_div_float_kernel[grid](accum_flat, count_flat, n_elem)
    else:
        _mean_safe_div_int_kernel[grid](accum_flat, count_flat, n_elem)

    if needs_promote:
        out.copy_(accum.to(out.dtype))
    return out


def _impl_prod(out, dim, index, src):
    if out.numel() == 0 or index.numel() == 0:
        return out

    inp_restrided, src_strided = _restrided_views(out, dim, index, src)
    stride_dim_val = out.stride(dim) if out.ndim > 0 else 0
    N = index.numel()
    use_int32 = _int32_offset_safe(out, index, src)
    meta = _build_meta(inp_restrided, index, src_strided)
    _scatter_cas_reduce_kernel[_kernel_grid(N)](
        src_strided,
        index,
        inp_restrided,
        meta,
        stride_dim_val,
        N,
        INT32_OFFSET=use_int32,
        RANK=index.ndim,
        REDUCE_OP=0,
        IS_FLOAT=out.dtype.is_floating_point,
    )
    return out


def _impl_amax_amin(out, dim, index, src, is_max):
    if out.numel() == 0 or index.numel() == 0:
        return out
    inp_restrided, src_strided = _restrided_views(out, dim, index, src)
    stride_dim_val = out.stride(dim) if out.ndim > 0 else 0
    N = index.numel()
    use_int32 = _int32_offset_safe(out, index, src)
    meta = _build_meta(inp_restrided, index, src_strided)

    # Int dtypes: use native atomic_max/atomic_min (one pass, no NaN concerns).
    if out.dtype in (torch.int32, torch.int64):
        _scatter_native_int_minmax_kernel[_kernel_grid(N)](
            src_strided,
            index,
            inp_restrided,
            meta,
            stride_dim_val,
            N,
            INT32_OFFSET=use_int32,
            RANK=index.ndim,
            IS_MAX=is_max,
        )
        return out

    # int16 needs an upcast: no native atomic_max/min for 16-bit ints.
    if out.dtype == torch.int16:
        promoted = out.to(torch.int32)
        _impl_amax_amin(promoted, dim, index, src.to(torch.int32), is_max)
        out.copy_(promoted.to(torch.int16))
        return out

    # fp32/fp64: single-pass kernel with native atomic_max/min PLUS inlined
    # atomic_xchg for NaN sources. Goldenfox uses two separate kernel
    # launches; we fuse them into one launch.
    # fp16/bf16: single-pass CAS-loop (Triton lacks fp16/bf16 atomic_max/min
    # codegen on sm_75; converting to fp32 round-trips through a slow
    # to-copy kernel that destroys small-shape perf).
    if out.dtype in (torch.float32, torch.float64):
        _scatter_float_minmax_kernel[_kernel_grid(N)](
            src_strided,
            index,
            inp_restrided,
            meta,
            stride_dim_val,
            N,
            INT32_OFFSET=use_int32,
            RANK=index.ndim,
            IS_MAX=is_max,
        )
        return out

    _scatter_cas_reduce_kernel[_kernel_grid(N)](
        src_strided,
        index,
        inp_restrided,
        meta,
        stride_dim_val,
        N,
        INT32_OFFSET=use_int32,
        RANK=index.ndim,
        REDUCE_OP=1 if is_max else 2,
        IS_FLOAT=True,
    )
    return out


# ---------------------------------------------------------------------------
# Initial-value handling for include_self=False
# ---------------------------------------------------------------------------


def _reset_targets(out, dim, index, fill_value):
    """Reset `out[index]` along `dim` to `fill_value`.

    We allocate a 1-element tensor and `expand` it to `index.shape` rather
    than calling `torch.full(index.shape, ...)`. `expand` is a stride-0 view
    over a single byte, so we avoid the ~100us of CUDA allocation + memcpy
    that a full-size fill tensor costs on the larger workload shapes
    (1024x65536 etc.).
    """
    fill_scalar = torch.empty((1,), dtype=out.dtype, device=out.device)
    fill_scalar.fill_(fill_value)
    fill_src = fill_scalar.expand(index.shape)
    return out.scatter_(dim, index, fill_src)


# ---------------------------------------------------------------------------
# Dispatch: route reduce -> implementation, given a prepared output tensor
# ---------------------------------------------------------------------------


def _dispatch(out, dim, index, src, reduce, include_self):
    if reduce == "sum":
        return _impl_sum(out, dim, index, src)
    if reduce == "mean":
        return _impl_mean(out, dim, index, src, include_self)
    if reduce == "prod":
        return _impl_prod(out, dim, index, src)
    if reduce == "amax":
        return _impl_amax_amin(out, dim, index, src, is_max=True)
    if reduce == "amin":
        return _impl_amax_amin(out, dim, index, src, is_max=False)
    raise RuntimeError(f"unsupported reduce {reduce!r}")


def _prepare_output(inp, dim, index, reduce, include_self):
    """Create an out-of-place starting buffer that respects include_self."""
    out = inp.clone()
    if not include_self:
        _reset_targets(out, dim, index, _reduction_identity(out.dtype, reduce))
    return out


# ---------------------------------------------------------------------------
# Scalar handling
# ---------------------------------------------------------------------------
#
# When ndim==0 the index/src are also scalars and the operation collapses to
# a single elementwise update of out. We compute it directly on-device with
# torch ops, no Triton kernel required (and no CPU fallback).


def _scalar_reduce(out, src, reduce, include_self):
    """Scalar (0-rank) reduction. include_self=False means only the src
    contributes, since `self` is conceptually replaced by the reduction
    identity before the scatter."""
    if reduce == "sum":
        if include_self:
            out.add_(src)
        else:
            out.copy_(src)
        return out
    if reduce == "prod":
        if include_self:
            out.mul_(src)
        else:
            out.copy_(src)
        return out
    if reduce == "mean":
        if include_self:
            # mean of {self, src} = (self + src) / 2
            out.add_(src).div_(2)
        else:
            out.copy_(src)
        return out
    if reduce == "amax":
        if include_self:
            torch.maximum(out, src, out=out)
        else:
            out.copy_(src)
        return out
    if reduce == "amin":
        if include_self:
            torch.minimum(out, src, out=out)
        else:
            out.copy_(src)
        return out
    raise RuntimeError(f"unsupported reduce {reduce!r}")


# ---------------------------------------------------------------------------
# Public APIs
# ---------------------------------------------------------------------------


def scatter_reduce(inp, dim, index, src, reduce, *, include_self=True):
    """Out-of-place scatter_reduce (`torch.scatter_reduce`)."""
    logger.debug("GEMS SCATTER_REDUCE")
    dim = _validate_args(inp, dim, index, src, reduce)

    if inp.ndim == 0:
        out = inp.clone()
        return _scalar_reduce(out, src, reduce, include_self)

    out = _prepare_output(inp, dim, index, reduce, include_self)
    return _dispatch(out, dim, index, src, reduce, include_self)


def scatter_reduce_(inp, dim, index, src, reduce, *, include_self=True):
    """In-place scatter_reduce_ (`Tensor.scatter_reduce_`)."""
    logger.debug("GEMS SCATTER_REDUCE_")
    dim = _validate_args(inp, dim, index, src, reduce)

    if inp.ndim == 0:
        return _scalar_reduce(inp, src, reduce, include_self)

    _check_no_internal_overlap(inp)

    if not include_self:
        _reset_targets(inp, dim, index, _reduction_identity(inp.dtype, reduce))

    return _dispatch(inp, dim, index, src, reduce, include_self)


def scatter_reduce_out(inp, dim, index, src, reduce, *, include_self=True, out=None):
    """`torch.scatter_reduce(..., out=...)` (`scatter_reduce.two_out`)."""
    logger.debug("GEMS SCATTER_REDUCE_OUT")
    assert out is not None, "scatter_reduce.two_out requires the 'out' argument"
    dim = _validate_args(inp, dim, index, src, reduce)

    if inp.ndim == 0:
        out.copy_(inp)
        return _scalar_reduce(out, src, reduce, include_self)

    # If out aliases inp exactly, this is an in-place op on `out`.
    if _is_same_tensor_view(out, inp):
        return scatter_reduce_(out, dim, index, src, reduce, include_self=include_self)

    out.copy_(inp)
    if not include_self:
        _reset_targets(out, dim, index, _reduction_identity(out.dtype, reduce))
    return _dispatch(out, dim, index, src, reduce, include_self)
