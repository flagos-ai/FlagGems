"""Shared multi-tensor (foreach) execution utilities.

This module is the execution layer that PyTorch's ``aten::_foreach_*`` operators
need.  A ``Tensor[]`` is validated, grouped by ``(device, dtype)``, its pointers
and element counts are packed into a single device-side metadata buffer, and one
Triton kernel launch per group processes *all* tensors of that group.  The
per-element math is supplied by the caller as a ``triton.jit`` function passed
through ``tl.constexpr``, so a single executor kernel serves many operators
without an opcode switch.

Design notes
------------
* Launch count scales with the number of ``(device, dtype)`` groups, not with
  ``len(tensors)``.  ``tl.pointer_type`` needs a compile-time dtype, hence the
  grouping.
* Heterogeneous ``numel`` is handled by a *chunk table*: a flat array of
  ``(tensor index, element offset)`` pairs.  ``grid = number of chunks``, so
  every program does a bounded amount of work regardless of tensor sizes.
* Only non-overlapping-and-dense tensors can be addressed with flat indices.
  Gappy or overlapping views (``x[:, ::2]``, ``expand``) are staged through a
  contiguous copy, which costs extra launches; that cost is reported by
  :func:`launch_stats` rather than hidden.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# dtype plumbing
# ---------------------------------------------------------------------------

_TORCH_TO_TL = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
    torch.float64: tl.float64,
    torch.int8: tl.int8,
    torch.int16: tl.int16,
    torch.int32: tl.int32,
    torch.int64: tl.int64,
    torch.uint8: tl.uint8,
    torch.bool: tl.int1,
}

# A complex tensor is addressed as interleaved pairs of real components, so the
# kernel only ever needs the dtype of one component.
_COMPLEX_TO_REAL = {
    torch.complex32: torch.float16,
    torch.complex64: torch.float32,
    torch.complex128: torch.float64,
}


def tl_dtype(dtype: torch.dtype):
    if dtype not in _TORCH_TO_TL:
        raise NotImplementedError(f"foreach executor does not support {dtype}")
    return _TORCH_TO_TL[dtype]


def real_dtype_of(dtype: torch.dtype) -> torch.dtype:
    """Result dtype of a complex-to-real elementwise op such as ``abs``."""
    return _COMPLEX_TO_REAL[dtype] if dtype.is_complex else dtype


def same_dtype(dtype: torch.dtype) -> torch.dtype:
    """Result dtype of a dtype-preserving elementwise op such as ``neg``."""
    return dtype


def int_to_float(dtype: torch.dtype) -> torch.dtype:
    """Result dtype of a transcendental op such as ``sin``.

    Measured against this PyTorch build: every integral and bool input is
    promoted to ``float32`` while floating and complex dtypes pass through.
    ``float16`` stays ``float16`` -- promotion is to the *default* float type
    only for non-float inputs, not to the widest one.
    """
    if dtype.is_floating_point or dtype.is_complex:
        return dtype
    return torch.float32


# ---------------------------------------------------------------------------
# validation, layout classification, grouping
# ---------------------------------------------------------------------------


def check_tensor_list(tensors: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    """Reproduce ATen's ``Tensor[]`` preconditions, including messages."""
    if not isinstance(tensors, (list, tuple)):
        raise TypeError("argument 'self' must be tuple of Tensors")
    if len(tensors) == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")
    for t in tensors:
        if not isinstance(t, torch.Tensor):
            raise TypeError("argument 'self' must be tuple of Tensors")
    return list(tensors)


def is_flat_addressable(t: torch.Tensor) -> bool:
    """Whether ``[data_ptr, data_ptr + numel)`` covers exactly the tensor.

    This is ``is_non_overlapping_and_dense``: the storage span equals ``numel``,
    which is precisely what flat indexing assumes.  A transposed but dense
    tensor qualifies -- its elements are visited in memory order, and the
    output allocated by ``empty_like`` carries the same stride, so the flat
    mapping stays element-wise correct.

    Computed here instead of calling ``torch.ops.aten`` to avoid a dispatcher
    round trip on the hot path; verified to agree with the ATen predicate on
    contiguous / transpose / slice / offset / permute / expand / gappy /
    empty / 0-dim / broadcast-squeezed cases.

    ``is_contiguous`` is checked first because it is a C-level call that answers
    the overwhelmingly common case; the Python stride walk below only runs for
    genuine views.  This ordering matters: the walk showed up as the single
    largest CPU cost when profiling a 128-tensor list.
    """
    if t.is_contiguous():
        return True
    if t.numel() <= 1:
        return True
    expected = 1
    for stride, size in sorted(
        (s, sz) for s, sz in zip(t.stride(), t.shape) if sz != 1
    ):
        if stride != expected:
            return False
        expected *= size
    return True


def has_internal_overlap(t: torch.Tensor) -> bool:
    """True when several logical elements provably alias one memory location.

    ``_debug_has_internal_overlap`` is tri-state: ``0`` = no overlap, ``1`` =
    overlap, ``2`` = too hard to tell.  Only ``1`` must reject an in-place
    write; PyTorch itself accepts gappy-but-disjoint views such as
    ``x[:, ::2]`` (which report ``2``) and rejects only ``expand``-style
    aliasing.
    """
    return torch._debug_has_internal_overlap(t) == 1


def group_by_device_dtype(
    tensors: Sequence[torch.Tensor],
) -> Dict[Tuple[Any, torch.dtype], List[int]]:
    """Group tensor *indices* by ``(device, dtype)``, preserving input order."""
    groups: Dict[Tuple[Any, torch.dtype], List[int]] = {}
    for i, t in enumerate(tensors):
        groups.setdefault((t.device, t.dtype), []).append(i)
    return groups


def _pick_block(numels: Sequence[int], max_programs: int = 1 << 18) -> int:
    """Block size balancing per-program work against the 2D grid size.

    The grid is ``(len(numels), cdiv(max numel, block))``, so tensors smaller
    than the largest one contribute programs that exit on the mask.  Those cost
    almost nothing individually, but a very skewed list still needs a larger
    block to keep the total program count in check.

    The search starts at 1024 rather than at the smallest workable value.  Each
    program pays three scalar ``int64`` metadata loads before it touches any
    data, so a small block amortises that fixed cost over too little work: on
    128 x 1M fp32 a sweep measured 339us at ``BLOCK=1024`` against 434us at 256,
    and the ordering held across every shape tried.  1024 elements is also one
    element per lane at ``num_warps=4``, which is why larger blocks stop helping.
    """
    nt = len(numels)
    max_numel = max(numels)
    for block in (1024, 2048, 4096, 8192, 16384):
        if nt * ((max_numel + block - 1) // block) <= max_programs:
            return block
    return 16384


# ---------------------------------------------------------------------------
# executor kernels
# ---------------------------------------------------------------------------


@triton.jit
def foreach_unary_kernel(
    meta_ptr,
    fn: tl.constexpr,
    NT: tl.constexpr,
    IN_DT: tl.constexpr,
    OUT_DT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One launch, ``NT`` tensors.

    ``meta_ptr`` layout, all int64::

        [0,      NT)     input pointers
        [NT,   2*NT)     output pointers
        [2*NT, 3*NT)     element counts

    The grid is 2D, ``(NT, max chunks per tensor)``: axis 0 selects the tensor
    and axis 1 the chunk within it.  Programs past a tensor's own element count
    are fully masked out, which costs a load and an exit -- cheaper than
    materializing an explicit chunk table on the CPU, which measured as the
    single largest overhead in the end-to-end benchmark.
    """
    t = tl.program_id(0)
    offset = tl.program_id(1) * BLOCK

    in_ptr = tl.load(meta_ptr + t).to(tl.pointer_type(IN_DT))
    out_ptr = tl.load(meta_ptr + NT + t).to(tl.pointer_type(OUT_DT))
    n_elements = tl.load(meta_ptr + 2 * NT + t)

    idx = offset + tl.arange(0, BLOCK)
    mask = idx < n_elements
    x = tl.load(in_ptr + idx, mask=mask, other=0)
    # The value is promoted to the output dtype *before* the math runs, which is
    # what ``pointwise_dynamic`` does for its scalar functions.  Those functions
    # commonly end in ``.to(x.dtype)``, so handing them an unpromoted integer
    # would truncate the result: ``log1p`` on int64 measured 0.693 of error
    # until the promotion was moved ahead of the call.
    tl.store(out_ptr + idx, fn(x.to(OUT_DT)).to(OUT_DT), mask=mask)


@triton.jit
def foreach_unary_c2r_kernel(
    meta_ptr,
    fn: tl.constexpr,
    NT: tl.constexpr,
    R_DT: tl.constexpr,
    ACC_DT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Complex input, real output; ``fn(re, im)`` returns one real value.

    The complex input is read through a real-typed pointer where logical
    element ``i`` occupies slots ``2*i`` (real) and ``2*i + 1`` (imaginary).
    Triton's dtype tables have no complex entries, so this interleaved view is
    what makes complex support possible at all.  Metadata layout and 2D grid
    match :func:`foreach_unary_kernel`; counts are in *complex* elements.
    """
    t = tl.program_id(0)
    offset = tl.program_id(1) * BLOCK

    in_ptr = tl.load(meta_ptr + t).to(tl.pointer_type(R_DT))
    out_ptr = tl.load(meta_ptr + NT + t).to(tl.pointer_type(R_DT))
    n_elements = tl.load(meta_ptr + 2 * NT + t)

    idx = offset + tl.arange(0, BLOCK)
    mask = idx < n_elements
    re = tl.load(in_ptr + 2 * idx, mask=mask, other=0).to(ACC_DT)
    im = tl.load(in_ptr + 2 * idx + 1, mask=mask, other=0).to(ACC_DT)
    tl.store(out_ptr + idx, fn(re, im).to(R_DT), mask=mask)


@triton.jit
def foreach_unary_c2c_kernel(
    meta_ptr,
    fn: tl.constexpr,
    NT: tl.constexpr,
    R_DT: tl.constexpr,
    ACC_DT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Complex input, complex output; ``fn(re, im)`` returns a ``(re, im)`` pair.

    Same interleaved addressing as :func:`foreach_unary_c2r_kernel`, but both
    components are written back.  Complex support has to live here rather than
    be borrowed from the single-tensor operators: those raise
    ``KeyError: 'complex64'`` from Triton's dtype table, yet PyTorch's foreach
    variants of ``exp``/``log``/``sin``/... do accept complex, so parity
    requires a dedicated path.
    """
    t = tl.program_id(0)
    offset = tl.program_id(1) * BLOCK

    in_ptr = tl.load(meta_ptr + t).to(tl.pointer_type(R_DT))
    out_ptr = tl.load(meta_ptr + NT + t).to(tl.pointer_type(R_DT))
    n_elements = tl.load(meta_ptr + 2 * NT + t)

    idx = offset + tl.arange(0, BLOCK)
    mask = idx < n_elements
    re = tl.load(in_ptr + 2 * idx, mask=mask, other=0).to(ACC_DT)
    im = tl.load(in_ptr + 2 * idx + 1, mask=mask, other=0).to(ACC_DT)
    out_re, out_im = fn(re, im)
    tl.store(out_ptr + 2 * idx, out_re.to(R_DT), mask=mask)
    tl.store(out_ptr + 2 * idx + 1, out_im.to(R_DT), mask=mask)


@triton.jit
def foreach_binary_kernel(
    meta_ptr,
    fn: tl.constexpr,
    NT: tl.constexpr,
    A_DT: tl.constexpr,
    B_DT: tl.constexpr,
    OUT_DT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One launch, ``NT`` tensors, two tensor inputs per element.

    ``meta_ptr`` layout, all int64::

        [0,      NT)     first input pointers
        [NT,   2*NT)     second input pointers
        [2*NT, 3*NT)     output pointers
        [3*NT, 4*NT)     element counts

    The second operand is read with its *own* pointer rather than being folded
    into the first, which is what makes ``.List`` correct for tensors whose
    strides differ: the two lists are paired position by position, and each
    element count is that pair's own. A unary operator can get away with flat
    traversal of a non-contiguous tensor because the element *set* is unchanged,
    but pairing two differently-strided tensors by flat index would silently
    match up the wrong elements, so both sides are staged to dense form by the
    caller before they reach here.
    """
    t = tl.program_id(0)
    offset = tl.program_id(1) * BLOCK

    a_ptr = tl.load(meta_ptr + t).to(tl.pointer_type(A_DT))
    b_ptr = tl.load(meta_ptr + NT + t).to(tl.pointer_type(B_DT))
    out_ptr = tl.load(meta_ptr + 2 * NT + t).to(tl.pointer_type(OUT_DT))
    n_elements = tl.load(meta_ptr + 3 * NT + t)

    idx = offset + tl.arange(0, BLOCK)
    mask = idx < n_elements
    a = tl.load(a_ptr + idx, mask=mask, other=0)
    b = tl.load(b_ptr + idx, mask=mask, other=0)
    tl.store(out_ptr + idx, fn(a.to(OUT_DT), b.to(OUT_DT)).to(OUT_DT), mask=mask)


@triton.jit
def foreach_binary_scalar_kernel(
    meta_ptr,
    scalar_ptr,
    fn: tl.constexpr,
    NT: tl.constexpr,
    A_DT: tl.constexpr,
    OUT_DT: tl.constexpr,
    S_DT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """``NT`` tensors against one scalar *each*.

    ``meta_ptr`` layout matches the unary kernel (in / out / counts); the
    per-tensor scalars live in their own ``scalar_ptr`` array indexed by tensor.

    A single ``ScalarList`` array covers both the ``.Scalar`` and
    ``.ScalarList`` overloads -- ``.Scalar`` simply broadcasts one value into
    every slot on the host side. Passing the scalars through device memory
    rather than as kernel arguments keeps one compiled kernel for any list
    length, which is the whole point of the shared executor.
    """
    t = tl.program_id(0)
    offset = tl.program_id(1) * BLOCK

    a_ptr = tl.load(meta_ptr + t).to(tl.pointer_type(A_DT))
    out_ptr = tl.load(meta_ptr + NT + t).to(tl.pointer_type(OUT_DT))
    n_elements = tl.load(meta_ptr + 2 * NT + t)
    s = tl.load(scalar_ptr + t).to(S_DT)

    idx = offset + tl.arange(0, BLOCK)
    mask = idx < n_elements
    a = tl.load(a_ptr + idx, mask=mask, other=0)
    tl.store(out_ptr + idx, fn(a.to(OUT_DT), s.to(OUT_DT)).to(OUT_DT), mask=mask)


@triton.jit
def foreach_ternary_kernel(
    meta_ptr,
    scalar_ptr,
    fn: tl.constexpr,
    NT: tl.constexpr,
    A_DT: tl.constexpr,
    OUT_DT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """``NT`` tensors, three tensor inputs and one per-tensor scalar.

    ``meta_ptr`` layout, all int64::

        [0,      NT)     self pointers
        [NT,   2*NT)     tensor1 pointers
        [2*NT, 3*NT)     tensor2 pointers
        [3*NT, 4*NT)     output pointers
        [4*NT, 5*NT)     element counts

    This serves ``addcmul``/``addcdiv`` (``self + value * t1 <op> t2``) and the
    ``lerp`` weight forms; the scalar array is the ``value``/``weight``
    argument, broadcast on the host when the overload supplies a single value.
    """
    t = tl.program_id(0)
    offset = tl.program_id(1) * BLOCK

    a_ptr = tl.load(meta_ptr + t).to(tl.pointer_type(A_DT))
    b_ptr = tl.load(meta_ptr + NT + t).to(tl.pointer_type(A_DT))
    c_ptr = tl.load(meta_ptr + 2 * NT + t).to(tl.pointer_type(A_DT))
    out_ptr = tl.load(meta_ptr + 3 * NT + t).to(tl.pointer_type(OUT_DT))
    n_elements = tl.load(meta_ptr + 4 * NT + t)
    s = tl.load(scalar_ptr + t)

    idx = offset + tl.arange(0, BLOCK)
    mask = idx < n_elements
    a = tl.load(a_ptr + idx, mask=mask, other=0).to(OUT_DT)
    b = tl.load(b_ptr + idx, mask=mask, other=0).to(OUT_DT)
    c = tl.load(c_ptr + idx, mask=mask, other=0).to(OUT_DT)
    tl.store(out_ptr + idx, fn(a, b, c, s.to(OUT_DT)).to(OUT_DT), mask=mask)


# ---------------------------------------------------------------------------
# launch accounting (used by the launch-count regression test)
# ---------------------------------------------------------------------------


class _LaunchStats:
    """Counters describing how the last call was decomposed.

    Kept as plain integers so a test can assert that executor launches track
    the number of ``(device, dtype)`` groups instead of ``len(tensors)``.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.executor_launches = 0
        self.groups = 0
        self.staged_inputs = 0
        self.writebacks = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "executor_launches": self.executor_launches,
            "groups": self.groups,
            "staged_inputs": self.staged_inputs,
            "writebacks": self.writebacks,
        }


_STATS = _LaunchStats()


def launch_stats() -> Dict[str, int]:
    """Snapshot of the counters from the most recent foreach call."""
    return _STATS.as_dict()


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------


def _launch_group(
    srcs: List[torch.Tensor],
    dsts: List[torch.Tensor],
    fn: Callable,
    in_dtype: torch.dtype,
    device: Any,
    is_complex: bool,
) -> None:
    numels = [t.numel() for t in srcs]
    max_numel = max(numels)
    if max_numel == 0:
        return
    block = _pick_block(numels)

    nt = len(srcs)
    # One flat Python list -> one tensor -> one H2D transfer.  Building this as
    # three separate tensors and slicing them in measured noticeably slower,
    # because each slice assignment is its own copy.
    #
    # A pinned staging pool was tried here and rejected.  In isolation the pinned
    # copy is 1.7-3.8x faster than this pageable one, but reusing pinned buffers
    # without ordering the host refill against the outstanding device read is a
    # real data race (a probe that forced the window open corrupted 21 of 24
    # calls).  Making it safe needs a CUDA event per call, and allocating plus
    # recording that event costs more than the pageable copy saves: end to end it
    # measured 93us per call against 46us for this version.
    meta_cpu = torch.tensor(
        [t.data_ptr() for t in srcs] + [t.data_ptr() for t in dsts] + numels,
        dtype=torch.int64,
    )
    meta = meta_cpu.to(device, non_blocking=True)

    grid = (nt, triton.cdiv(max_numel, block))
    if is_complex:
        real_dtype = _COMPLEX_TO_REAL[in_dtype]
        acc = tl.float32 if real_dtype is not torch.float64 else tl.float64
        kernel = (
            foreach_unary_c2r_kernel
            if dsts[0].dtype.is_floating_point
            else foreach_unary_c2c_kernel
        )
        kernel[grid](meta, fn, nt, tl_dtype(real_dtype), acc, block)
    else:
        foreach_unary_kernel[grid](
            meta, fn, nt, tl_dtype(in_dtype), tl_dtype(dsts[0].dtype), block
        )
    _STATS.executor_launches += 1
    _STATS.groups += 1


def check_dtype_supported(dtype: torch.dtype, allowed: Optional[frozenset]) -> None:
    """Reject dtypes the corresponding ATen operator also rejects.

    ``allowed=None`` means "no restriction".  Each operator's set was measured
    against this PyTorch build rather than assumed, because the exclusions are
    not guessable: ``_foreach_floor`` rejects ``bool`` but accepts every integer
    dtype, ``_foreach_frac`` rejects all integers, and ``_foreach_sign``
    accepts ``bool`` while rejecting complex.
    """
    if allowed is not None and dtype not in allowed:
        raise RuntimeError(
            f'"foreach" not implemented for \'{str(dtype).replace("torch.", "")}\''
        )


def foreach_unary(
    tensors: Sequence[torch.Tensor],
    fn: Callable,
    *,
    complex_fn: Optional[Callable] = None,
    out_dtype_fn: Callable[[torch.dtype], torch.dtype] = same_dtype,
    inplace: bool = False,
    allowed_dtypes: Optional[frozenset] = None,
) -> List[torch.Tensor]:
    """Apply a unary elementwise ``triton.jit`` function to a whole TensorList.

    Args:
        tensors: the ``Tensor[]`` argument as received from ATen.
        fn: ``@triton.jit`` function of one value, used for real dtypes.
        complex_fn: ``@triton.jit`` function of ``(re, im)`` for complex
            inputs.  ``None`` means complex is unsupported by this operator.
        out_dtype_fn: maps input dtype to output dtype.
        inplace: write results back into ``tensors`` and return them.
        allowed_dtypes: dtypes the operator accepts; ``None`` means all.

    Returns:
        A list of result tensors, one per input, in input order.
    """
    tensors = check_tensor_list(tensors)
    _STATS.reset()

    results: List[Optional[torch.Tensor]] = [None] * len(tensors)
    # Buckets keyed by (device, input dtype); values are aligned src/dst lists.
    buckets: Dict[Tuple[Any, torch.dtype], Tuple[List, List]] = {}
    writebacks: List[Tuple[torch.Tensor, torch.Tensor]] = []

    for i, t in enumerate(tensors):
        dtype = t.dtype
        check_dtype_supported(dtype, allowed_dtypes)
        if dtype.is_complex and complex_fn is None:
            raise NotImplementedError(
                f"foreach executor has no complex kernel for {dtype}"
            )
        out_dtype = out_dtype_fn(dtype)
        key = (t.device, dtype)

        if inplace:
            if out_dtype != dtype:
                # Same refusal as ATen: an in-place op may not silently narrow
                # its promoted result back into the input's dtype.
                raise RuntimeError(
                    f"result type {str(out_dtype).replace('torch.', '').capitalize()}"
                    " can't be cast to the desired output type "
                    f"{str(dtype).replace('torch.', '').capitalize()}"
                )
            if has_internal_overlap(t):
                raise RuntimeError(
                    "unsupported operation: more than one element of the "
                    "written-to tensor refers to a single memory location. "
                    "Please clone() the tensor before performing the operation."
                )
            if is_flat_addressable(t):
                src = dst = t
            else:
                # Gappy view: compute on a dense copy, then scatter back.
                src = dst = t.contiguous()
                writebacks.append((t, src))
                _STATS.staged_inputs += 1
            results[i] = t
        else:
            src = t if is_flat_addressable(t) else t.contiguous()
            if src is not t:
                _STATS.staged_inputs += 1
            # empty_like reproduces PyTorch's foreach output stride policy: it
            # keeps a dense tensor's stride (so transposed inputs give
            # transposed outputs) and normalizes everything else to contiguous.
            dst = torch.empty_like(src, dtype=out_dtype)
            results[i] = dst

        bucket = buckets.get(key)
        if bucket is None:
            bucket = buckets[key] = ([], [])
        bucket[0].append(src)
        bucket[1].append(dst)

    for (device, dtype), (srcs, dsts) in buckets.items():
        kernel_fn = complex_fn if dtype.is_complex else fn
        _launch_group(srcs, dsts, kernel_fn, dtype, device, dtype.is_complex)

    for dst_view, staged in writebacks:
        dst_view.copy_(staged)
        _STATS.writebacks += 1

    return results  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# multi-input entry points
# ---------------------------------------------------------------------------


def _dense(t: torch.Tensor) -> torch.Tensor:
    """A flat-addressable view of ``t``, copying only when necessary.

    Unlike the unary path, a merely *dense* tensor is not enough for the paired
    kernels: two tensors that are each dense but carry different strides would
    be matched up element-by-element in the wrong order under flat indexing.
    ``contiguous()`` is therefore the condition here, not
    ``is_non_overlapping_and_dense``.
    """
    if t.is_contiguous():
        return t
    _STATS.staged_inputs += 1
    return t.contiguous()


def _check_same_length(a: Sequence[torch.Tensor], b: Sequence[Any], what: str) -> None:
    if len(a) != len(b):
        raise RuntimeError(
            f"Tensor list must have same number of elements as {what}, "
            f"got {len(a)} and {len(b)}"
        )


def _result_dtype(t: torch.Tensor, other: Any) -> torch.dtype:
    """ATen's type promotion for one tensor against a tensor or a number.

    Delegating to ``torch.result_type`` rather than reimplementing the lattice
    is deliberate: it is the same routine ATen consults, so an int64 list plus a
    float scalar promotes to float32 here exactly as ``torch._foreach_add``
    does.
    """
    return torch.result_type(t, other)


def _resolve_scalars(tensors: Sequence[torch.Tensor], scalars: Any) -> List[Any]:
    """Normalise the ``.Scalar`` / ``.ScalarList`` / ``.Tensor`` forms into a list.

    The ``.Tensor`` overloads of ``addcmul``/``addcdiv`` take the values as a
    one-dimensional tensor holding one entry per tensor in the list (ATen
    requires it on the CPU), so it is unpacked here rather than being mistaken
    for a single value.
    """
    if isinstance(scalars, torch.Tensor):
        if scalars.numel() == 1:
            return [scalars.item()] * len(tensors)
        _check_same_length(tensors, scalars, "scalar list")
        return scalars.detach().cpu().tolist()
    if isinstance(scalars, (list, tuple)):
        _check_same_length(tensors, scalars, "scalar list")
        return list(scalars)
    return [scalars] * len(tensors)


def _finish_inplace(
    results: List[torch.Tensor],
    writebacks: List[Tuple[torch.Tensor, torch.Tensor]],
) -> None:
    for dst_view, staged in writebacks:
        dst_view.copy_(staged)
        _STATS.writebacks += 1
    del results


def foreach_binary_list(
    self: Sequence[torch.Tensor],
    other: Sequence[torch.Tensor],
    fn: Callable,
    *,
    inplace: bool = False,
    allowed_dtypes: Optional[frozenset] = None,
    out_dtype_fn: Optional[Callable[[torch.dtype], torch.dtype]] = None,
    out_dtypes: Optional[Sequence[torch.dtype]] = None,
) -> List[torch.Tensor]:
    """``self[i] <op> other[i]`` for every position, one launch per group.

    Both lists are paired position by position, so they must have equal length;
    ATen raises rather than broadcasting across the list.

    ``out_dtypes`` lets the caller pin the result dtype per position. The
    ``.Tensor`` overloads need it: their operand is a scalar tensor expanded to
    the list's shape, and promoting against the *expanded* view would widen an
    fp16 list to fp32, where ATen keeps fp16 because a zero-dimensional tensor
    counts as a scalar.
    """
    self = check_tensor_list(self)
    other = check_tensor_list(other)
    _check_same_length(self, other, "tensor list")
    if out_dtypes is not None:
        _check_same_length(self, out_dtypes, "output dtype list")
    _STATS.reset()

    results: List[torch.Tensor] = []
    buckets: Dict[Tuple[Any, torch.dtype, torch.dtype], Tuple[List, List, List]] = {}
    writebacks: List[Tuple[torch.Tensor, torch.Tensor]] = []

    for i, (a, b) in enumerate(zip(self, other)):
        check_dtype_supported(a.dtype, allowed_dtypes)
        if out_dtypes is not None:
            out_dtype = out_dtypes[i]
        else:
            out_dtype = _result_dtype(a, b)
            if out_dtype_fn is not None:
                out_dtype = out_dtype_fn(out_dtype)
        src_a = _dense(a)
        src_b = _dense(b)
        if inplace:
            if out_dtype != a.dtype:
                raise RuntimeError(
                    f"result type {str(out_dtype).replace('torch.', '').capitalize()}"
                    " can't be cast to the desired output type "
                    f"{str(a.dtype).replace('torch.', '').capitalize()}"
                )
            if has_internal_overlap(a):
                raise RuntimeError(
                    "unsupported operation: more than one element of the "
                    "written-to tensor refers to a single memory location. "
                    "Please clone() the tensor before performing the operation."
                )
            dst = src_a
            if dst is not a:
                writebacks.append((a, dst))
            results.append(a)
        else:
            dst = torch.empty_like(src_a, dtype=out_dtype)
            results.append(dst)
        key = (a.device, src_a.dtype, src_b.dtype)
        bucket = buckets.get(key)
        if bucket is None:
            bucket = buckets[key] = ([], [], [])
        bucket[0].append(src_a)
        bucket[1].append(src_b)
        bucket[2].append(dst)

    for (device, a_dt, b_dt), (a_list, b_list, d_list) in buckets.items():
        _launch_binary_group(a_list, b_list, d_list, fn, a_dt, b_dt, device)

    _finish_inplace(results, writebacks)
    return results


def _launch_binary_group(
    a_list: List[torch.Tensor],
    b_list: List[torch.Tensor],
    d_list: List[torch.Tensor],
    fn: Callable,
    a_dt: torch.dtype,
    b_dt: torch.dtype,
    device: Any,
) -> None:
    numels = [t.numel() for t in a_list]
    max_numel = max(numels)
    if max_numel == 0:
        return
    block = _pick_block(numels)
    nt = len(a_list)
    meta = torch.tensor(
        [t.data_ptr() for t in a_list]
        + [t.data_ptr() for t in b_list]
        + [t.data_ptr() for t in d_list]
        + numels,
        dtype=torch.int64,
    ).to(device, non_blocking=True)
    grid = (nt, triton.cdiv(max_numel, block))
    foreach_binary_kernel[grid](
        meta, fn, nt, tl_dtype(a_dt), tl_dtype(b_dt), tl_dtype(d_list[0].dtype), block
    )
    _STATS.executor_launches += 1
    _STATS.groups += 1


def foreach_binary_scalar(
    self: Sequence[torch.Tensor],
    scalars: Any,
    fn: Callable,
    *,
    inplace: bool = False,
    allowed_dtypes: Optional[frozenset] = None,
    out_dtype_fn: Optional[Callable[[torch.dtype], torch.dtype]] = None,
) -> List[torch.Tensor]:
    """``self[i] <op> scalar`` -- serves the ``.Scalar`` and ``.ScalarList`` forms.

    ``scalars`` is either one number (broadcast to every tensor) or a sequence
    of one number per tensor.
    """
    self = check_tensor_list(self)
    values = _resolve_scalars(self, scalars)
    _STATS.reset()

    results: List[torch.Tensor] = []
    buckets: Dict[Tuple[Any, torch.dtype, torch.dtype], Tuple[List, List, List]] = {}
    writebacks: List[Tuple[torch.Tensor, torch.Tensor]] = []

    for t, v in zip(self, values):
        check_dtype_supported(t.dtype, allowed_dtypes)
        out_dtype = _result_dtype(t, v)
        if out_dtype_fn is not None:
            out_dtype = out_dtype_fn(out_dtype)
        src = _dense(t)
        if inplace:
            if out_dtype != t.dtype:
                raise RuntimeError(
                    f"result type {str(out_dtype).replace('torch.', '').capitalize()}"
                    " can't be cast to the desired output type "
                    f"{str(t.dtype).replace('torch.', '').capitalize()}"
                )
            if has_internal_overlap(t):
                raise RuntimeError(
                    "unsupported operation: more than one element of the "
                    "written-to tensor refers to a single memory location. "
                    "Please clone() the tensor before performing the operation."
                )
            dst = src
            if dst is not t:
                writebacks.append((t, dst))
            results.append(t)
        else:
            dst = torch.empty_like(src, dtype=out_dtype)
            results.append(dst)
        key = (t.device, src.dtype, out_dtype)
        bucket = buckets.get(key)
        if bucket is None:
            bucket = buckets[key] = ([], [], [])
        bucket[0].append(src)
        bucket[1].append(dst)
        bucket[2].append(v)

    for (device, in_dt, out_dt), (srcs, dsts, vals) in buckets.items():
        _launch_scalar_group(srcs, dsts, vals, fn, in_dt, out_dt, device)

    _finish_inplace(results, writebacks)
    return results


def _launch_scalar_group(
    srcs: List[torch.Tensor],
    dsts: List[torch.Tensor],
    vals: List[Any],
    fn: Callable,
    in_dt: torch.dtype,
    out_dt: torch.dtype,
    device: Any,
) -> None:
    numels = [t.numel() for t in srcs]
    max_numel = max(numels)
    if max_numel == 0:
        return
    block = _pick_block(numels)
    nt = len(srcs)
    meta = torch.tensor(
        [t.data_ptr() for t in srcs] + [t.data_ptr() for t in dsts] + numels,
        dtype=torch.int64,
    ).to(device, non_blocking=True)
    # The scalars ride in a device array of the *output* dtype, so integer
    # operators keep integer semantics instead of going through a float.
    #
    # The dtype is settled on the host before the transfer rather than by casting
    # the device tensor afterwards: that extra ``.to(dtype)`` costs a kernel and
    # an allocation, and measured 0.32ms per call -- on its own more than
    # PyTorch's entire foreach call.
    s_dtype = out_dt if out_dt.is_floating_point else torch.float32
    scalar_buf = torch.tensor([float(v) for v in vals], dtype=s_dtype).to(
        device, non_blocking=True
    )
    grid = (nt, triton.cdiv(max_numel, block))
    foreach_binary_scalar_kernel[grid](
        meta,
        scalar_buf,
        fn,
        nt,
        tl_dtype(in_dt),
        tl_dtype(out_dt),
        tl_dtype(scalar_buf.dtype),
        block,
    )
    _STATS.executor_launches += 1
    _STATS.groups += 1


def foreach_ternary(
    self: Sequence[torch.Tensor],
    tensor1: Sequence[torch.Tensor],
    tensor2: Sequence[torch.Tensor],
    scalars: Any,
    fn: Callable,
    *,
    inplace: bool = False,
    allowed_dtypes: Optional[frozenset] = None,
) -> List[torch.Tensor]:
    """``fn(self[i], tensor1[i], tensor2[i], scalar[i])`` for every position.

    Serves ``addcmul`` / ``addcdiv``, whose value argument is either one number
    or one per tensor.
    """
    self = check_tensor_list(self)
    tensor1 = check_tensor_list(tensor1)
    tensor2 = check_tensor_list(tensor2)
    _check_same_length(self, tensor1, "tensor list")
    _check_same_length(self, tensor2, "tensor list")
    values = _resolve_scalars(self, scalars)
    _STATS.reset()

    results: List[torch.Tensor] = []
    buckets: Dict[Tuple[Any, torch.dtype, torch.dtype], Tuple[List, ...]] = {}
    writebacks: List[Tuple[torch.Tensor, torch.Tensor]] = []

    for a, b, c, v in zip(self, tensor1, tensor2, values):
        check_dtype_supported(a.dtype, allowed_dtypes)
        # ``result_type`` pairs two tensors or a tensor and a number, so the
        # three-way promotion is folded through zero-element probes of the
        # intermediate dtype rather than by chaining dtypes directly.
        out_dtype = torch.result_type(a, b)
        out_dtype = torch.result_type(
            torch.empty(0, dtype=out_dtype, device="meta"), c.to("meta")
        )
        if not out_dtype.is_floating_point and not out_dtype.is_complex:
            out_dtype = torch.result_type(a, v)
        src_a, src_b, src_c = _dense(a), _dense(b), _dense(c)
        if inplace:
            if out_dtype != a.dtype:
                raise RuntimeError(
                    f"result type {str(out_dtype).replace('torch.', '').capitalize()}"
                    " can't be cast to the desired output type "
                    f"{str(a.dtype).replace('torch.', '').capitalize()}"
                )
            if has_internal_overlap(a):
                raise RuntimeError(
                    "unsupported operation: more than one element of the "
                    "written-to tensor refers to a single memory location. "
                    "Please clone() the tensor before performing the operation."
                )
            dst = src_a
            if dst is not a:
                writebacks.append((a, dst))
            results.append(a)
        else:
            dst = torch.empty_like(src_a, dtype=out_dtype)
            results.append(dst)
        key = (a.device, src_a.dtype, out_dtype)
        bucket = buckets.get(key)
        if bucket is None:
            bucket = buckets[key] = ([], [], [], [], [])
        bucket[0].append(src_a)
        bucket[1].append(src_b)
        bucket[2].append(src_c)
        bucket[3].append(dst)
        bucket[4].append(v)

    for (device, in_dt, out_dt), (a_l, b_l, c_l, d_l, v_l) in buckets.items():
        _launch_ternary_group(a_l, b_l, c_l, d_l, v_l, fn, in_dt, out_dt, device)

    _finish_inplace(results, writebacks)
    return results


def _launch_ternary_group(
    a_l: List[torch.Tensor],
    b_l: List[torch.Tensor],
    c_l: List[torch.Tensor],
    d_l: List[torch.Tensor],
    v_l: List[Any],
    fn: Callable,
    in_dt: torch.dtype,
    out_dt: torch.dtype,
    device: Any,
) -> None:
    numels = [t.numel() for t in a_l]
    max_numel = max(numels)
    if max_numel == 0:
        return
    block = _pick_block(numels)
    nt = len(a_l)
    meta = torch.tensor(
        [t.data_ptr() for t in a_l]
        + [t.data_ptr() for t in b_l]
        + [t.data_ptr() for t in c_l]
        + [t.data_ptr() for t in d_l]
        + numels,
        dtype=torch.int64,
    ).to(device, non_blocking=True)
    s_dtype = out_dt if out_dt.is_floating_point else torch.float32
    scalar_buf = torch.tensor([float(v) for v in v_l], dtype=s_dtype).to(
        device, non_blocking=True
    )
    grid = (nt, triton.cdiv(max_numel, block))
    foreach_ternary_kernel[grid](
        meta, scalar_buf, fn, nt, tl_dtype(in_dt), tl_dtype(out_dt), block
    )
    _STATS.executor_launches += 1
    _STATS.groups += 1
