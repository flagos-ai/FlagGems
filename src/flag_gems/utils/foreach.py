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

import array
import logging
import math
import numbers
import struct
from functools import reduce
from operator import or_
from types import BuiltinFunctionType
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl
from triton import knobs
from triton.runtime import driver as _triton_driver

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


# The per-element Python ``for`` loop costs more than the check it performs:
# over sixteen tensors the loop measured 0.66us and this bound metaclass entry
# point 0.31us.  ``isinstance`` dispatches to exactly this method, so the answer
# is identical, and every foreach call runs it once per operand list.
_IS_TENSOR = torch.Tensor.__instancecheck__


def check_tensor_list(tensors: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    """Reproduce ATen's ``Tensor[]`` preconditions, including messages."""
    # Fast path for the form the dispatcher and every benchmark call use: a
    # plain list of plain tensors, checked with one class-identity pass instead
    # of an ``isinstance`` call per element.  Two lists of sixteen measured
    # 0.58us here against 1.34us for the ``isinstance`` loop.  Anything that is
    # not exactly a ``torch.Tensor`` (a subclass such as ``Parameter``, or a
    # non-tensor) falls through to the general path, which raises the exact
    # ATen error for each rejected shape.
    if type(tensors) is list and tensors:
        tensor_type = torch.Tensor
        for t in tensors:
            if t.__class__ is not tensor_type:
                break
        else:
            return list(tensors)
    if not isinstance(tensors, (list, tuple)):
        raise TypeError("argument 'self' must be tuple of Tensors")
    if len(tensors) == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")
    if not all(map(_IS_TENSOR, tensors)):
        # Only the rejecting path pays for the walk that names the slot.
        for t in tensors:
            if not isinstance(t, torch.Tensor):
                raise TypeError("argument 'self' must be tuple of Tensors")
    return list(tensors)


def _ternary_operand_lists(
    a: Any, b: Any, c: Any
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """``check_tensor_list`` on a ternary op's three operand lists at once.

    The answers and the exceptions are those of three :func:`check_tensor_list`
    calls followed by the two ``_check_same_length`` calls: each list's type,
    emptiness and elements are examined in turn, and only then do the lengths
    have to agree.  The passing path is three ``isinstance`` tests, three
    ``map``s and three ``list`` copies instead of five calls re-testing the
    same things; a failing test hands that list back to the original function
    so the message is unchanged.
    """
    if not (
        isinstance(a, (list, tuple))
        and isinstance(b, (list, tuple))
        and isinstance(c, (list, tuple))
    ):
        for lst in (a, b, c):
            check_tensor_list(lst)
    if not a or not all(map(_IS_TENSOR, a)):
        check_tensor_list(a)
    if not b or not all(map(_IS_TENSOR, b)):
        check_tensor_list(b)
    if not c or not all(map(_IS_TENSOR, c)):
        check_tensor_list(c)
    if len(a) != len(b) or len(a) != len(c):
        _check_same_length(a, b, "tensor list")
        _check_same_length(a, c, "tensor list")
    return list(a), list(b), list(c)


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


def _pick_block(
    numels: Sequence[int], max_programs: int = 1 << 18, start: int = 1024
) -> int:
    """Block size balancing per-program work against the 2D grid size.

    The grid is ``(len(numels), cdiv(max numel, block))``, so tensors smaller
    than the largest one contribute programs that exit on the mask.  Those cost
    almost nothing individually, but a very skewed list still needs a larger
    block to keep the total program count in check.

    The search starts at ``start`` rather than at the smallest workable value.
    Each program pays three scalar ``int64`` metadata loads before it touches
    any data, so a small block amortises that fixed cost over too little work:
    on 128 x 1M fp32 a sweep measured 339us at ``BLOCK=1024`` against 434us at
    256, and the ordering held across every shape tried.

    1024 is the right floor for four-byte data, where it is 4 KiB per program.
    Callers whose payload is two bytes pass ``start=2048`` to get the same 4 KiB
    of stores per program: the packed zero kernel measured 15.06us at
    ``BLOCK=1024`` against 13.15us at 2048 on 16 x 1M fp16 (full call, median of
    three alternating runs), and 15.10 against 13.22 on bf16, while fp32 was
    unchanged by the larger block.
    """
    nt = len(numels)
    max_numel = max(numels)
    for block in (1024, 2048, 4096, 8192, 16384):
        if block < start:
            continue
        if nt * ((max_numel + block - 1) // block) <= max_programs:
            return block
    return 16384


_PACK_DTYPES = frozenset((torch.float16, torch.bfloat16))

# Below this element count the acceptance latency is host-bound, so the guard
# below (one Python ``reduce`` over the metadata values, ~1.2us at this list
# size) would show up as a regression rather than being hidden behind the
# kernel.  At and above it the launch is bandwidth-bound and the guard is free
# next to the access pattern it enables.
_PACK_MIN_ELEMS = 1 << 20

# The binary path measured the crossover directly instead of inheriting the
# constant above.  With a sixteen-tensor list the packed kernel is 0.9us faster
# at 256^2 (2^16 elements per tensor) and 2.8us faster at 512^2, and it is
# never slower down to 2^10, where both kernels are launch-bound and the
# difference is noise.
#
# That holds for an in-place group, and only for an in-place group.  The
# out-of-place wrapper allocates sixteen destinations per call, which makes the
# call host-bound (58us of host submission against a 24us kernel), so a faster
# kernel cannot show there while the guard below can: accepting a group pays
# the ``reduce`` over the metadata values, ~1.6us at this list size, and the
# measured effect was +2.4 to +3.3us on every out-of-place row at 256^2 and
# 512^2 while the identical kernel launch into a fixed destination was 2.9us
# faster.  An in-place call has no allocation, is GPU-bound (31us of host
# against a 16us kernel at 512^2), and hides the guard -- which is why the
# threshold below is raised for it alone.
_PACK_MIN_ELEMS_BINARY_INPLACE = 1 << 16


def _pack_ok(
    dtypes: Sequence[torch.dtype],
    max_numel: int,
    values: Sequence[int],
    min_elems: int = _PACK_MIN_ELEMS,
) -> bool:
    """Whether the four-wide packed kernel can serve this group.

    The packed kernels read every tensor through an ``int64`` pointer, so each
    pointer must be 8-byte aligned and each element count a multiple of four --
    a view with a non-zero element offset, or a trailing remainder, falls back
    to the scalar kernels.  Both conditions are one bitwise-or over the metadata
    values that are built for the transfer anyway: any value with a low bit set
    rules the packed path out.
    """
    if max_numel < min_elems:
        return False
    for dt in dtypes:
        if dt not in _PACK_DTYPES:
            return False
    return not (reduce(or_, values, 0) & 7)


# ---------------------------------------------------------------------------
# executor kernels
# ---------------------------------------------------------------------------


@triton.jit
def _unpack4(v, DT: tl.constexpr):
    """Split one ``int64`` into four bit-identical ``DT`` values."""
    return (
        (v & 0xFFFF).to(tl.uint16).to(DT, bitcast=True),
        ((v >> 16) & 0xFFFF).to(tl.uint16).to(DT, bitcast=True),
        ((v >> 32) & 0xFFFF).to(tl.uint16).to(DT, bitcast=True),
        ((v >> 48) & 0xFFFF).to(tl.uint16).to(DT, bitcast=True),
    )


@triton.jit
def _pack4(a, b, c, d):
    """Inverse of :func:`_unpack4`; the four values must be 16 bits wide."""
    return (
        a.to(tl.uint16, bitcast=True).to(tl.int64)
        | (b.to(tl.uint16, bitcast=True).to(tl.int64) << 16)
        | (c.to(tl.uint16, bitcast=True).to(tl.int64) << 32)
        | (d.to(tl.uint16, bitcast=True).to(tl.int64) << 48)
    )


@triton.jit
def _alpha_scale(v, ALPHA: tl.constexpr, OUT_DT: tl.constexpr):
    """``ALPHA`` scaling, identical to the expression in the scalar kernels."""
    return (v * ALPHA).to(OUT_DT) if ALPHA != 1 else v.to(OUT_DT)


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

    The alignment split matters more than it looks.  A pointer read out of the
    metadata buffer is an ``int64`` the compiler knows nothing about, so it
    assumes one-element alignment and picks a one-element-per-lane layout,
    emitting one ``ld.global.b16``/``st.global.b16`` per element.  Measured at
    512^2 fp16 x 16 tensors that costs 1.3us on a copy and 2.6us on ``log``.
    Declaring the divisibility we have just checked, *and* giving the compiler
    a mask-free chunk, lets it choose an eight-element layout and emit
    ``ld.global.v4.b32``: 128 bits per lane.  The two changes are only useful
    together -- the hint without the mask-free chunk widens the layout while
    the access stays 16-bit, which measured 2.2x *slower* than the plain loop.
    """
    t = tl.program_id(0)
    offset = tl.program_id(1) * BLOCK

    in_v = tl.load(meta_ptr + t)
    out_v = tl.load(meta_ptr + NT + t)
    n_elements = tl.load(meta_ptr + 2 * NT + t)

    idx = offset + tl.arange(0, BLOCK)
    if (in_v & 15) == 0 and (out_v & 15) == 0:
        in_ptr = tl.multiple_of(in_v.to(tl.pointer_type(IN_DT)), 16)
        out_ptr = tl.multiple_of(out_v.to(tl.pointer_type(OUT_DT)), 16)
        if offset + BLOCK <= n_elements:
            x = tl.load(in_ptr + idx)
            tl.store(out_ptr + idx, fn(x.to(OUT_DT)).to(OUT_DT))
        else:
            # Last chunk of a tensor: keep the masked path.  Programs past a
            # shorter tensor's count land here too and touch nothing.
            mask = idx < n_elements
            x = tl.load(in_ptr + idx, mask=mask, other=0)
            tl.store(out_ptr + idx, fn(x.to(OUT_DT)).to(OUT_DT), mask=mask)
    else:
        # Sliced or otherwise unaligned tensor: no vector access is available,
        # so run exactly the loop this kernel has always run.
        in_ptr = in_v.to(tl.pointer_type(IN_DT))
        out_ptr = out_v.to(tl.pointer_type(OUT_DT))
        mask = idx < n_elements
        x = tl.load(in_ptr + idx, mask=mask, other=0)
        # The value is promoted to the output dtype *before* the math runs,
        # which is what ``pointwise_dynamic`` does for its scalar functions.
        # Those functions commonly end in ``.to(x.dtype)``, so handing them an
        # unpromoted integer would truncate the result: ``log1p`` on int64
        # measured 0.693 of error until the promotion was moved ahead of the
        # call.
        tl.store(out_ptr + idx, fn(x.to(OUT_DT)).to(OUT_DT), mask=mask)


@triton.jit
def foreach_unary_kernel_p4(
    meta_ptr,
    fn: tl.constexpr,
    NT: tl.constexpr,
    IN_DT: tl.constexpr,
    OUT_DT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Unary executor reading four 16-bit elements per lane.

    A pointer that comes out of the metadata buffer carries no alignment
    information, so Triton compiles the 16-bit loop of the kernel above into one
    scalar ``ld.global.b16``/``st.global.b16`` per element: 2.3 TB/s against the
    3.5 TB/s a 128-bit access reaches at the same occupancy.  Reading the same
    memory through an ``int64`` pointer and splitting it with shifts restores
    the wide access, and costs nothing in accuracy because the split is a pure
    bit operation: every lane computes exactly what the scalar kernel computes
    on its four elements.

    Metadata layout and grid are unchanged from :func:`foreach_unary_kernel`.
    A chunk is still ``BLOCK`` *elements*, which is ``BLOCK // 4`` packed words,
    so the second grid axis keeps its meaning.  The caller only takes this path
    when every element count is a multiple of four and every pointer is 8-byte
    aligned, which is what makes the packed mask cover each tensor exactly.
    """
    t = tl.program_id(0)
    offset = tl.program_id(1) * (BLOCK // 4)

    in_ptr = tl.load(meta_ptr + t).to(tl.pointer_type(tl.int64))
    out_ptr = tl.load(meta_ptr + NT + t).to(tl.pointer_type(tl.int64))
    n_elements = tl.load(meta_ptr + 2 * NT + t)

    idx = offset + tl.arange(0, BLOCK // 4)
    mask = idx < n_elements // 4
    x0, x1, x2, x3 = _unpack4(tl.load(in_ptr + idx, mask=mask, other=0), IN_DT)
    tl.store(
        out_ptr + idx,
        _pack4(
            fn(x0.to(OUT_DT)).to(OUT_DT),
            fn(x1.to(OUT_DT)).to(OUT_DT),
            fn(x2.to(OUT_DT)).to(OUT_DT),
            fn(x3.to(OUT_DT)).to(OUT_DT),
        ),
        mask=mask,
    )


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
def _binary_unpack4(v, DT: tl.constexpr):
    """Split one ``int64`` word into four bit-identical ``DT`` values.

    Private to the binary executor even though :func:`_unpack4` has the same
    body: the split order is part of the packed-access contract, and a private
    definition keeps this lane from having to edit a helper the unary and
    ternary executors also use.
    """
    return (
        (v & 0xFFFF).to(tl.uint16).to(DT, bitcast=True),
        ((v >> 16) & 0xFFFF).to(tl.uint16).to(DT, bitcast=True),
        ((v >> 32) & 0xFFFF).to(tl.uint16).to(DT, bitcast=True),
        ((v >> 48) & 0xFFFF).to(tl.uint16).to(DT, bitcast=True),
    )


@triton.jit
def _binary_pack4(a, b, c, d):
    """Inverse of :func:`_binary_unpack4`; the four values are 16 bits wide."""
    return (
        a.to(tl.uint16, bitcast=True).to(tl.int64)
        | (b.to(tl.uint16, bitcast=True).to(tl.int64) << 16)
        | (c.to(tl.uint16, bitcast=True).to(tl.int64) << 32)
        | (d.to(tl.uint16, bitcast=True).to(tl.int64) << 48)
    )


@triton.jit
def _binary_wide_body(
    meta_ptr,
    t,
    fn: tl.constexpr,
    NT: tl.constexpr,
    A_DT: tl.constexpr,
    B_DT: tl.constexpr,
    OUT_DT: tl.constexpr,
    ALPHA: tl.constexpr,
    B_SCALAR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Four 16-bit elements per lane; same semantics as the narrow body."""
    offset = tl.program_id(1) * (BLOCK // 4)

    a_ptr = tl.load(meta_ptr + t).to(tl.pointer_type(tl.int64))
    b_raw = tl.load(meta_ptr + NT + t)
    out_ptr = tl.load(meta_ptr + 2 * NT + t).to(tl.pointer_type(tl.int64))
    n_elements = tl.load(meta_ptr + 3 * NT + t)

    idx = offset + tl.arange(0, BLOCK // 4)
    mask = idx < n_elements // 4
    a0, a1, a2, a3 = _binary_unpack4(tl.load(a_ptr + idx, mask=mask, other=0), A_DT)
    if B_SCALAR:
        b = _alpha_scale(tl.load(b_raw.to(tl.pointer_type(B_DT))), ALPHA, OUT_DT)
        r0 = fn(a0.to(OUT_DT), b).to(OUT_DT)
        r1 = fn(a1.to(OUT_DT), b).to(OUT_DT)
        r2 = fn(a2.to(OUT_DT), b).to(OUT_DT)
        r3 = fn(a3.to(OUT_DT), b).to(OUT_DT)
    else:
        b_ptr = b_raw.to(tl.pointer_type(tl.int64))
        b0, b1, b2, b3 = _binary_unpack4(tl.load(b_ptr + idx, mask=mask, other=0), B_DT)
        b0 = _alpha_scale(b0, ALPHA, OUT_DT)
        b1 = _alpha_scale(b1, ALPHA, OUT_DT)
        b2 = _alpha_scale(b2, ALPHA, OUT_DT)
        b3 = _alpha_scale(b3, ALPHA, OUT_DT)
        r0 = fn(a0.to(OUT_DT), b0).to(OUT_DT)
        r1 = fn(a1.to(OUT_DT), b1).to(OUT_DT)
        r2 = fn(a2.to(OUT_DT), b2).to(OUT_DT)
        r3 = fn(a3.to(OUT_DT), b3).to(OUT_DT)
    tl.store(out_ptr + idx, _binary_pack4(r0, r1, r2, r3), mask=mask)


@triton.jit
def _binary_narrow_body(
    meta_ptr,
    t,
    fn: tl.constexpr,
    NT: tl.constexpr,
    A_DT: tl.constexpr,
    B_DT: tl.constexpr,
    OUT_DT: tl.constexpr,
    ALPHA: tl.constexpr,
    B_SCALAR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """The element-indexed body: one operand element per lane."""
    offset = tl.program_id(1) * BLOCK

    a_ptr = tl.load(meta_ptr + t).to(tl.pointer_type(A_DT))
    b_ptr = tl.load(meta_ptr + NT + t).to(tl.pointer_type(B_DT))
    out_ptr = tl.load(meta_ptr + 2 * NT + t).to(tl.pointer_type(OUT_DT))
    n_elements = tl.load(meta_ptr + 3 * NT + t)

    idx = offset + tl.arange(0, BLOCK)
    mask = idx < n_elements
    a = tl.load(a_ptr + idx, mask=mask, other=0)
    if B_SCALAR:
        b = tl.load(b_ptr)
    else:
        b = tl.load(b_ptr + idx, mask=mask, other=0)
    b = (b * ALPHA).to(OUT_DT) if ALPHA != 1 else b.to(OUT_DT)
    tl.store(out_ptr + idx, fn(a.to(OUT_DT), b).to(OUT_DT), mask=mask)


@triton.jit
def foreach_binary_kernel(
    meta_ptr,
    fn: tl.constexpr,
    NT: tl.constexpr,
    A_DT: tl.constexpr,
    B_DT: tl.constexpr,
    OUT_DT: tl.constexpr,
    ALPHA: tl.constexpr,
    B_SCALAR: tl.constexpr,
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

    ``B_SCALAR`` is the ``.Tensor`` broadcast case: the second operand holds a
    single element, so it is loaded once per program and splatted by the
    element-wise function instead of being materialised into a full-size copy
    on the host.  Reading it here is a uniform device load; reading it back on
    the host with ``.item()`` would synchronise and drain every queued launch.

    ``ALPHA`` scales the second operand before the element-wise function runs;
    it is a ``constexpr`` so the multiply compiles away for the default value.

    The narrow body issues one ``ld.global.b16`` per element because a pointer
    read out of the metadata buffer carries no alignment information, so Triton
    cannot widen the access.  That costs 3.1us on the 16 tensor, 512^2 fp16
    group -- the exact cell the acceptance median is taken from -- whereas the
    same group through an ``int64`` pointer measures 12.5us against 15.6us.  The
    packed kernel already does this from ``_PACK_MIN_ELEMS`` upward, but below
    that count only this kernel is selected.  Rather than raise the shared
    threshold, the kernel checks the two conditions the packed access actually
    needs -- every pointer 8 byte aligned and every count a multiple of four --
    on the metadata values it has already loaded, and takes the four-wide body
    when they hold.  Views with an odd element offset and lists with a short
    tail keep the element-indexed body, so nothing here depends on the caller.
    """
    t = tl.program_id(0)
    a_raw = tl.load(meta_ptr + t)
    b_raw = tl.load(meta_ptr + NT + t)
    out_raw = tl.load(meta_ptr + 2 * NT + t)
    n_elements = tl.load(meta_ptr + 3 * NT + t)

    # Bit-splitting a word is only well defined when all three dtypes are 16
    # bits wide; for anything else this folds to the narrow body at compile time.
    IS16: tl.constexpr = (
        (A_DT.primitive_bitwidth == 16)
        & (B_DT.primitive_bitwidth == 16)
        & (OUT_DT.primitive_bitwidth == 16)
    )
    if IS16:
        if (((a_raw | b_raw | out_raw) & 7) == 0) & ((n_elements & 3) == 0):
            _binary_wide_body(
                meta_ptr, t, fn, NT, A_DT, B_DT, OUT_DT, ALPHA, B_SCALAR, BLOCK
            )
        else:
            _binary_narrow_body(
                meta_ptr, t, fn, NT, A_DT, B_DT, OUT_DT, ALPHA, B_SCALAR, BLOCK
            )
    else:
        _binary_narrow_body(
            meta_ptr, t, fn, NT, A_DT, B_DT, OUT_DT, ALPHA, B_SCALAR, BLOCK
        )


@triton.jit
def foreach_binary_kernel_p4(
    meta_ptr,
    fn: tl.constexpr,
    NT: tl.constexpr,
    A_DT: tl.constexpr,
    B_DT: tl.constexpr,
    OUT_DT: tl.constexpr,
    ALPHA: tl.constexpr,
    B_SCALAR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Four 16-bit elements per lane; :func:`foreach_binary_kernel` semantics.

    Same metadata layout, same grid, same element-wise function and the same
    ``ALPHA``/``B_SCALAR`` handling as :func:`foreach_binary_kernel`; the only
    difference is that each operand is read as packed ``int64`` words and split
    with :func:`_unpack4`, so the access is wide instead of one 16-bit
    transaction per element.  See :func:`foreach_unary_kernel_p4` for why the
    metadata-buffer pointer loses the alignment that would otherwise let Triton
    widen the load by itself.

    The broadcast operand of the ``.Tensor`` case is a single element, so only
    its first entry is read -- as a plain ``B_DT`` load, not as a packed word,
    because it is not part of a run of four.
    """
    t = tl.program_id(0)
    offset = tl.program_id(1) * (BLOCK // 4)

    a_ptr = tl.load(meta_ptr + t).to(tl.pointer_type(tl.int64))
    b_raw = tl.load(meta_ptr + NT + t)
    out_ptr = tl.load(meta_ptr + 2 * NT + t).to(tl.pointer_type(tl.int64))
    n_elements = tl.load(meta_ptr + 3 * NT + t)

    idx = offset + tl.arange(0, BLOCK // 4)
    mask = idx < n_elements // 4
    a0, a1, a2, a3 = _unpack4(tl.load(a_ptr + idx, mask=mask, other=0), A_DT)
    if B_SCALAR:
        b = _alpha_scale(tl.load(b_raw.to(tl.pointer_type(B_DT))), ALPHA, OUT_DT)
        r0 = fn(a0.to(OUT_DT), b).to(OUT_DT)
        r1 = fn(a1.to(OUT_DT), b).to(OUT_DT)
        r2 = fn(a2.to(OUT_DT), b).to(OUT_DT)
        r3 = fn(a3.to(OUT_DT), b).to(OUT_DT)
    else:
        b_ptr = b_raw.to(tl.pointer_type(tl.int64))
        b0, b1, b2, b3 = _unpack4(tl.load(b_ptr + idx, mask=mask, other=0), B_DT)
        b0 = _alpha_scale(b0, ALPHA, OUT_DT)
        b1 = _alpha_scale(b1, ALPHA, OUT_DT)
        b2 = _alpha_scale(b2, ALPHA, OUT_DT)
        b3 = _alpha_scale(b3, ALPHA, OUT_DT)
        r0 = fn(a0.to(OUT_DT), b0).to(OUT_DT)
        r1 = fn(a1.to(OUT_DT), b1).to(OUT_DT)
        r2 = fn(a2.to(OUT_DT), b2).to(OUT_DT)
        r3 = fn(a3.to(OUT_DT), b3).to(OUT_DT)
    tl.store(out_ptr + idx, _pack4(r0, r1, r2, r3), mask=mask)


@triton.jit
def _binary_scalar_narrow_body(
    meta_ptr,
    scalar_ptr,
    scalar_value: tl.float64,
    fn: tl.constexpr,
    NT: tl.constexpr,
    A_DT: tl.constexpr,
    OUT_DT: tl.constexpr,
    S_DT: tl.constexpr,
    ALPHA: tl.constexpr,
    SCALAR_BY_VALUE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """The element-indexed body: one operand element per lane."""
    t = tl.program_id(0)
    offset = tl.program_id(1) * BLOCK

    a_ptr = tl.load(meta_ptr + t).to(tl.pointer_type(A_DT))
    out_ptr = tl.load(meta_ptr + NT + t).to(tl.pointer_type(OUT_DT))
    n_elements = tl.load(meta_ptr + 2 * NT + t)
    if SCALAR_BY_VALUE:
        s = scalar_value.to(S_DT)
    else:
        s = tl.load(scalar_ptr + t)
    s = (s * ALPHA).to(S_DT) if ALPHA != 1 else s
    s = s.to(OUT_DT)

    idx = offset + tl.arange(0, BLOCK)
    mask = idx < n_elements
    a = tl.load(a_ptr + idx, mask=mask, other=0)
    tl.store(out_ptr + idx, fn(a.to(OUT_DT), s.to(OUT_DT)).to(OUT_DT), mask=mask)


@triton.jit
def foreach_binary_scalar_kernel(
    meta_ptr,
    scalar_ptr,
    scalar_value: tl.float64,
    fn: tl.constexpr,
    NT: tl.constexpr,
    A_DT: tl.constexpr,
    OUT_DT: tl.constexpr,
    S_DT: tl.constexpr,
    ALPHA: tl.constexpr,
    SCALAR_BY_VALUE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """``NT`` tensors against one scalar *each*.

    ``meta_ptr`` layout matches the unary kernel (in / out / counts); the
    per-tensor scalars live in their own ``scalar_ptr`` array indexed by tensor.

    ``SCALAR_BY_VALUE`` selects the ``.Scalar`` path: when every tensor shares
    the same value there is nothing to index, so it travels as an ordinary
    kernel argument and no device buffer is built.  ``scalar_value`` is declared
    ``float64`` so the ``.Scalar`` path keeps the exact value the device buffer
    would have held; the cast to ``S_DT`` below reproduces the host-side
    ``torch.tensor(..., dtype=S_DT)`` rounding bit for bit.  ``.ScalarList``
    still needs the per-tensor array.

    ``ALPHA`` scales the scalar before the element-wise function runs; it is a
    ``constexpr`` so the multiply compiles away for the default value.

    A pointer read out of ``meta_ptr`` carries no alignment information, so
    Triton cannot widen the access and the element-indexed body issues one
    ``ld.global.b16`` per element.  ``foreach_binary_scalar_kernel_p4`` already
    reads four 16-bit elements per lane through an ``int64`` pointer, but the
    caller only selects it from ``_PACK_MIN_ELEMS`` upward, so every 512^2
    16-bit group -- the cell the acceptance median is taken from -- took the
    narrow path.  Rather than raise the shared threshold, the kernel checks the
    two conditions the packed access needs -- both pointers 8 byte aligned and
    the count a multiple of four -- on metadata values it has already loaded,
    and inlines the packed body when they hold.  Unaligned views and ragged
    counts keep the element-indexed body, so nothing here depends on the
    caller.
    """
    t = tl.program_id(0)
    a_raw = tl.load(meta_ptr + t)
    out_raw = tl.load(meta_ptr + NT + t)
    n_elements = tl.load(meta_ptr + 2 * NT + t)

    # Packing a word only round-trips when both the operand and the result are
    # 16 bits wide; for anything else this folds to the narrow body at compile
    # time.
    IS16: tl.constexpr = (A_DT.primitive_bitwidth == 16) & (
        OUT_DT.primitive_bitwidth == 16
    )
    if IS16:
        if (((a_raw | out_raw) & 7) == 0) & ((n_elements & 3) == 0):
            foreach_binary_scalar_kernel_p4(
                meta_ptr,
                scalar_ptr,
                scalar_value,
                fn,
                NT,
                A_DT,
                OUT_DT,
                S_DT,
                ALPHA,
                SCALAR_BY_VALUE,
                BLOCK,
            )
        else:
            _binary_scalar_narrow_body(
                meta_ptr,
                scalar_ptr,
                scalar_value,
                fn,
                NT,
                A_DT,
                OUT_DT,
                S_DT,
                ALPHA,
                SCALAR_BY_VALUE,
                BLOCK,
            )
    else:
        _binary_scalar_narrow_body(
            meta_ptr,
            scalar_ptr,
            scalar_value,
            fn,
            NT,
            A_DT,
            OUT_DT,
            S_DT,
            ALPHA,
            SCALAR_BY_VALUE,
            BLOCK,
        )


@triton.jit
def foreach_binary_scalar_kernel_p4(
    meta_ptr,
    scalar_ptr,
    scalar_value: tl.float64,
    fn: tl.constexpr,
    NT: tl.constexpr,
    A_DT: tl.constexpr,
    OUT_DT: tl.constexpr,
    S_DT: tl.constexpr,
    ALPHA: tl.constexpr,
    SCALAR_BY_VALUE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Four 16-bit elements per lane; :func:`foreach_binary_scalar_kernel` semantics."""
    t = tl.program_id(0)
    offset = tl.program_id(1) * (BLOCK // 4)

    a_ptr = tl.load(meta_ptr + t).to(tl.pointer_type(tl.int64))
    out_ptr = tl.load(meta_ptr + NT + t).to(tl.pointer_type(tl.int64))
    n_elements = tl.load(meta_ptr + 2 * NT + t)
    if SCALAR_BY_VALUE:
        s = scalar_value.to(S_DT)
    else:
        s = tl.load(scalar_ptr + t)
    s = (s * ALPHA).to(S_DT) if ALPHA != 1 else s
    s = s.to(OUT_DT)

    idx = offset + tl.arange(0, BLOCK // 4)
    mask = idx < n_elements // 4
    a0, a1, a2, a3 = _unpack4(tl.load(a_ptr + idx, mask=mask, other=0), A_DT)
    tl.store(
        out_ptr + idx,
        _pack4(
            fn(a0.to(OUT_DT), s.to(OUT_DT)).to(OUT_DT),
            fn(a1.to(OUT_DT), s.to(OUT_DT)).to(OUT_DT),
            fn(a2.to(OUT_DT), s.to(OUT_DT)).to(OUT_DT),
            fn(a3.to(OUT_DT), s.to(OUT_DT)).to(OUT_DT),
        ),
        mask=mask,
    )


@triton.jit
def _ternary_narrow_body(
    meta_ptr,
    scalar_ptr,
    scalar_value: tl.float64,
    fn: tl.constexpr,
    NT: tl.constexpr,
    A_DT: tl.constexpr,
    OUT_DT: tl.constexpr,
    S_DT: tl.constexpr,
    SCALAR_BY_VALUE: tl.constexpr,
    BLOCK: tl.constexpr,
    SWAP: tl.constexpr,
):
    """The element-indexed body: one operand element per lane.

    ``SWAP`` transposes the grid so the tile index travels first; see
    :func:`foreach_ternary_kernel_p4` for why.
    """
    if SWAP:
        t = tl.program_id(1)
        offset = tl.program_id(0) * BLOCK
    else:
        t = tl.program_id(0)
        offset = tl.program_id(1) * BLOCK

    a_ptr = tl.load(meta_ptr + t).to(tl.pointer_type(A_DT))
    b_ptr = tl.load(meta_ptr + NT + t).to(tl.pointer_type(A_DT))
    c_ptr = tl.load(meta_ptr + 2 * NT + t).to(tl.pointer_type(A_DT))
    out_ptr = tl.load(meta_ptr + 3 * NT + t).to(tl.pointer_type(OUT_DT))
    n_elements = tl.load(meta_ptr + 4 * NT + t)
    if SCALAR_BY_VALUE:
        s = scalar_value.to(S_DT)
    else:
        s = tl.load(scalar_ptr + t)

    idx = offset + tl.arange(0, BLOCK)
    mask = idx < n_elements
    a = tl.load(a_ptr + idx, mask=mask, other=0).to(OUT_DT)
    b = tl.load(b_ptr + idx, mask=mask, other=0).to(OUT_DT)
    c = tl.load(c_ptr + idx, mask=mask, other=0).to(OUT_DT)
    tl.store(out_ptr + idx, fn(a, b, c, s.to(OUT_DT)).to(OUT_DT), mask=mask)


@triton.jit
def foreach_ternary_kernel(
    meta_ptr,
    scalar_ptr,
    scalar_value: tl.float64,
    fn: tl.constexpr,
    NT: tl.constexpr,
    A_DT: tl.constexpr,
    OUT_DT: tl.constexpr,
    S_DT: tl.constexpr,
    SCALAR_BY_VALUE: tl.constexpr,
    BLOCK: tl.constexpr,
    SWAP: tl.constexpr,
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

    ``SCALAR_BY_VALUE`` selects the ``.Scalar`` form: a single value that every
    tensor shares travels as an ordinary kernel argument instead of a device
    buffer.  ``scalar_value`` is declared ``float64`` so nothing is lost on the
    way in; the cast to ``S_DT`` reproduces the buffer's rounding exactly.

    A pointer read out of ``meta_ptr`` carries no alignment information, so
    Triton cannot widen the access and the element-indexed body issues one
    ``ld.global.b16`` per element.  ``foreach_ternary_kernel_p4`` already reads
    four 16-bit elements per lane through an ``int64`` pointer, but the caller
    only selects it from ``_PACK_MIN_ELEMS`` upward, so every 512^2 16-bit
    group took the narrow path; that group is where the acceptance median
    lives.  Rather than raise the shared threshold, the kernel checks the two
    conditions the packed access needs -- all four pointers 8 byte aligned and
    the count a multiple of four -- on metadata values it has already loaded,
    and inlines the packed body when they hold.  Unaligned views and ragged
    counts keep the element-indexed body, so nothing here depends on the
    caller.
    """
    t = tl.program_id(0)
    a_raw = tl.load(meta_ptr + t)
    b_raw = tl.load(meta_ptr + NT + t)
    c_raw = tl.load(meta_ptr + 2 * NT + t)
    out_raw = tl.load(meta_ptr + 3 * NT + t)
    n_elements = tl.load(meta_ptr + 4 * NT + t)

    # The packed body splits each loaded word into 16-bit halves, which is only
    # well defined when both the operands and the result are 16 bits wide; for
    # anything else this folds to the narrow body at compile time.
    IS16: tl.constexpr = (A_DT.primitive_bitwidth == 16) & (
        OUT_DT.primitive_bitwidth == 16
    )
    if IS16:
        if (((a_raw | b_raw | c_raw | out_raw) & 7) == 0) & ((n_elements & 3) == 0):
            foreach_ternary_kernel_p4(
                meta_ptr,
                scalar_ptr,
                scalar_value,
                fn,
                NT,
                A_DT,
                OUT_DT,
                S_DT,
                SCALAR_BY_VALUE,
                BLOCK,
                SWAP,
            )
        else:
            _ternary_narrow_body(
                meta_ptr,
                scalar_ptr,
                scalar_value,
                fn,
                NT,
                A_DT,
                OUT_DT,
                S_DT,
                SCALAR_BY_VALUE,
                BLOCK,
                SWAP,
            )
    else:
        _ternary_narrow_body(
            meta_ptr,
            scalar_ptr,
            scalar_value,
            fn,
            NT,
            A_DT,
            OUT_DT,
            S_DT,
            SCALAR_BY_VALUE,
            BLOCK,
            SWAP,
        )


@triton.jit
def foreach_ternary_kernel_p4(
    meta_ptr,
    scalar_ptr,
    scalar_value: tl.float64,
    fn: tl.constexpr,
    NT: tl.constexpr,
    A_DT: tl.constexpr,
    OUT_DT: tl.constexpr,
    S_DT: tl.constexpr,
    SCALAR_BY_VALUE: tl.constexpr,
    BLOCK: tl.constexpr,
    SWAP: tl.constexpr,
):
    """Four 16-bit elements per lane; :func:`foreach_ternary_kernel` semantics.

    ``SWAP`` puts the tile index on ``program_id(0)`` and the tensor index on
    ``program_id(1)``.  The hardware issues blocks in linear program-id order,
    so with the tensor index first the sixteen tiles of one tensor are adjacent
    and the grid walks tensor by tensor; transposed it walks tile by tile, which
    at 1024^2 spreads the live addresses across the list instead of over one
    16 MiB buffer at a time and measured 0.7us less on the 128 MiB ternary
    traffic.  Below that size the launch is too short for the difference to
    survive and the transposed order was 0.07us slower at 512^2, so the caller
    enables it only from ``_TERNARY_SWAP_MIN_ELEMS`` up (and only for the
    packed kernel, whose layout this was measured on).
    """
    if SWAP:
        t = tl.program_id(1)
        offset = tl.program_id(0) * (BLOCK // 4)
    else:
        t = tl.program_id(0)
        offset = tl.program_id(1) * (BLOCK // 4)

    a_ptr = tl.load(meta_ptr + t).to(tl.pointer_type(tl.int64))
    b_ptr = tl.load(meta_ptr + NT + t).to(tl.pointer_type(tl.int64))
    c_ptr = tl.load(meta_ptr + 2 * NT + t).to(tl.pointer_type(tl.int64))
    out_ptr = tl.load(meta_ptr + 3 * NT + t).to(tl.pointer_type(tl.int64))
    n_elements = tl.load(meta_ptr + 4 * NT + t)
    if SCALAR_BY_VALUE:
        s = scalar_value.to(S_DT)
    else:
        s = tl.load(scalar_ptr + t)
    s = s.to(OUT_DT)

    idx = offset + tl.arange(0, BLOCK // 4)
    mask = idx < n_elements // 4
    a0, a1, a2, a3 = _unpack4(tl.load(a_ptr + idx, mask=mask, other=0), A_DT)
    b0, b1, b2, b3 = _unpack4(tl.load(b_ptr + idx, mask=mask, other=0), A_DT)
    c0, c1, c2, c3 = _unpack4(tl.load(c_ptr + idx, mask=mask, other=0), A_DT)
    tl.store(
        out_ptr + idx,
        _pack4(
            fn(a0.to(OUT_DT), b0.to(OUT_DT), c0.to(OUT_DT), s).to(OUT_DT),
            fn(a1.to(OUT_DT), b1.to(OUT_DT), c1.to(OUT_DT), s).to(OUT_DT),
            fn(a2.to(OUT_DT), b2.to(OUT_DT), c2.to(OUT_DT), s).to(OUT_DT),
            fn(a3.to(OUT_DT), b3.to(OUT_DT), c3.to(OUT_DT), s).to(OUT_DT),
        ),
        mask=mask,
    )


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


# Every argument of the executor kernels other than the metadata pointer is a
# ``tl.constexpr``, so ``(kernel, NT, fn, dtype, dtype, BLOCK)`` identifies the
# compiled variant exactly and a replay only has to hand the same values to the
# kernel's launcher.  Triton's Python dispatch between that lookup and the
# actual launch -- binder, cache-key hash, launch-metadata and hook checks --
# measured ~4us per launch, about a tenth of the small-shape budget, and none of
# it can change the result.  The hit rate is high by construction: a benchmark
# loop launches the same variant thousands of times.
#
# Only the *launcher* is cached.  The metadata tensor is built by
# ``_device_meta``, which memoises it per exact value list and stream (see the
# comment there); the kernel only reads it, so a repeat call with the same
# pointers gets the same bytes.
#
# The only thing Triton specializes on here is the metadata pointer's 16-byte
# alignment, so that bit is part of the key and an unusually aligned allocation
# gets a variant of its own rather than a mismatch.  A miss, an installed launch
# hook (proton and other instrumentation install one), or a launcher that raises
# all take the normal path, which fills the cache.
_LAUNCH_CACHE: Dict[Tuple[Any, ...], Tuple[Any, Any, Any]] = {}

# Replay accounting.  ``replay`` counts calls served by the cached launcher,
# ``dispatch`` counts calls that went through Triton's dispatcher (first call
# per key, or a fallback).  Together they equal the number of group launches,
# which is what makes a "the cache is write-only" bug visible without a probe.
_LAUNCH_STATS: Dict[str, int] = {"replay": 0, "dispatch": 0}

# One-hit memo in front of ``_LAUNCH_CACHE``.  The key tuple holds a device and
# two Triton dtypes, and building plus hashing it measured 1.1us a call at
# sixteen positions; the identity compares below ask the same question in a
# handful of pointer comparisons.  A repeated call shape presents the same metadata tensor object
# (``_device_meta`` hands the cached one back), the same kernel, function and
# constexprs, and the same grid, so every field of the dict key is covered:
# ``kernel``, ``nt``, ``fn``, both dtypes, ``block`` and ``grid`` are compared
# directly, and ``meta.device`` plus the alignment bit follow from ``meta is
# last[0]`` (a tensor's device never changes and the bit is a property of its
# ``data_ptr``).  The element at index 8 is a replay closure that captures the
# launcher and reads the stream per call.
#
# The entry is only ever the launcher for a key that already replayed
# successfully, and a memo miss falls through to the dict, which is still the
# authority.  A dispatch clears it, since Triton may have replaced the compiled
# entry.
_LAST_UNARY_REPLAY: List[Any] = [None]


def launch_cache_stats() -> Dict[str, int]:
    """Cumulative hits/misses of the executor launch cache (diagnostics only).

    Deliberately separate from :func:`launch_stats`, which is a per-call
    snapshot used by the tests.
    """
    return dict(_LAUNCH_STATS)


def _bind_executor_launcher(
    entry: Tuple[Any, Any, Any],
    grid: Tuple[int, int],
    nt: int,
    meta: torch.Tensor,
    fn: Callable,
    dt_in: Any,
    dt_out: Any,
    block: int,
) -> Callable[[], None]:
    """Bind a cached launcher into a zero-argument replay closure.

    The closure cells are read instead of indexing a tuple on every call, and
    the stream is deliberately *not* captured: ``get_current_stream`` is read
    at launch time so a replay issued under a different ``torch.cuda.stream``
    still lands on the current stream.

    ``CudaLauncher.__call__`` does two ``allocate_scratch`` calls and two
    attribute reads before forwarding its ``*args`` to ``self.launch``.  When
    there is nothing to allocate and ``self.launch`` is the bare C function --
    true for every executor kernel, whose signature has no ``tensordesc``
    entry -- calling it with the full positional list skips that wrapper, which
    measured 0.58us of the 4.3us replay at sixteen positions.  Anything else
    takes the public call, exactly as before.
    """
    launcher, function, packed = entry
    idx = meta.device.index
    g0, g1 = grid
    raw = launcher.launch
    if (
        isinstance(raw, BuiltinFunctionType)
        and launcher.global_scratch_size == 0
        and launcher.profile_scratch_size == 0
    ):
        coop = launcher.launch_cooperative_grid
        pdl = launcher.launch_pdl

        def replay() -> None:
            raw(
                g0,
                g1,
                1,
                _triton_driver.active.get_current_stream(idx),
                function,
                coop,
                pdl,
                None,
                None,
                packed,
                None,
                None,
                None,
                meta,
                fn,
                nt,
                dt_in,
                dt_out,
                block,
            )

        return replay

    def replay() -> None:
        launcher(
            g0,
            g1,
            1,
            _triton_driver.active.get_current_stream(idx),
            function,
            packed,
            None,
            None,
            None,
            meta,
            fn,
            nt,
            dt_in,
            dt_out,
            block,
        )

    return replay


def _launch_executor_kernel(
    kernel: Callable,
    grid: Tuple[int, int],
    nt: int,
    meta: torch.Tensor,
    fn: Callable,
    dt_in: Any,
    dt_out: Any,
    block: int,
) -> None:
    last = _LAST_UNARY_REPLAY[0]
    if (
        last is not None
        and last[0] is meta
        and last[1] is kernel
        and last[2] == nt
        and last[3] is fn
        and last[4] is dt_in
        and last[5] is dt_out
        and last[6] == block
        and last[7] == grid
        and not knobs.runtime.launch_enter_hook.calls
        and not knobs.runtime.launch_exit_hook.calls
    ):
        try:
            last[8]()
        except Exception:
            # Fall through: the dict path below retries once, then dispatches.
            _LAST_UNARY_REPLAY[0] = None
        else:
            _LAUNCH_STATS["replay"] += 1
            return
    # ``meta.device`` is part of the key: a compiled kernel belongs to one
    # device's module, so replaying device A's launcher for device B would use
    # the wrong context.
    key = (
        kernel,
        meta.device,
        nt,
        fn,
        dt_in,
        dt_out,
        block,
        not (meta.data_ptr() & 15),
    )
    entry = _LAUNCH_CACHE.get(key)
    runtime = knobs.runtime
    # ``launch_enter_hook`` is a ``HookChain`` in triton 3.6, never ``None``:
    # the chain exists whether or not anyone subscribed.  ``.calls`` is the
    # list of registered callbacks, so an empty list is the real "no hooks"
    # state.  Proton and other instrumentation add to it, and a replay would
    # then skip their notifications.
    if (
        entry is not None
        and not runtime.launch_enter_hook.calls
        and not runtime.launch_exit_hook.calls
    ):
        # Binding is inside the ``try`` as well: a Triton build whose launcher
        # has no ``launch_pdl`` or a different raw signature raises here, and
        # that must fall back to dispatch like any other unusable entry rather
        # than fail every call after the first.
        try:
            replay = _bind_executor_launcher(
                entry, grid, nt, meta, fn, dt_in, dt_out, block
            )
            replay()
        except Exception:
            # A launcher that cannot be replayed this way is not a correctness
            # problem: forget the entry and let Triton dispatch as usual.
            _LAUNCH_CACHE.pop(key, None)
        else:
            _LAST_UNARY_REPLAY[0] = (
                meta,
                kernel,
                nt,
                fn,
                dt_in,
                dt_out,
                block,
                grid,
                replay,
            )
            _LAUNCH_STATS["replay"] += 1
            return
    _LAUNCH_STATS["dispatch"] += 1
    compiled = kernel[grid](meta, fn, nt, dt_in, dt_out, block)
    if compiled is not None:
        # A dispatch may have replaced the compiled entry, so a closure bound to
        # the previous one is no longer known to be current.
        _LAST_UNARY_REPLAY[0] = None
        _LAUNCH_CACHE[key] = (
            compiled.run,
            compiled.function,
            compiled.packed_metadata,
        )


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------


# Bound-method handles for the per-tensor walks.  ``[NUMEL(t) for t in xs]``
# binds the method once instead of doing an attribute lookup on every tensor:
# measured 7.8us -> 7.3us over 128 tensors, and the same 0.13us at length 1.
# (``list(map(NUMEL, xs))`` is better still at 128 but costs 0.06us more at
# length 1, and the length-1 rows are the ones with the least to spare.)
_NUMEL = torch.Tensor.numel
_DATA_PTR = torch.Tensor.data_ptr


def _launch_group(
    srcs: List[torch.Tensor],
    dsts: List[torch.Tensor],
    fn: Callable,
    in_dtype: torch.dtype,
    device: Any,
    is_complex: bool,
) -> None:
    numels = [_NUMEL(t) for t in srcs]
    max_numel = max(numels)
    if max_numel == 0:
        return
    block = _pick_block(numels)

    nt = len(srcs)
    # One flat Python list -> one tensor -> one H2D transfer.  Building this as
    # three separate tensors and slicing them in measured noticeably slower,
    # because each slice assignment is its own copy.
    #
    # An in-place call passes the same list object as both sources and
    # destinations, so the pointer run is read once instead of twice.
    #
    # A pinned staging pool was tried here and rejected.  In isolation the pinned
    # copy is 1.7-3.8x faster than this pageable one, but reusing pinned buffers
    # without ordering the host refill against the outstanding device read is a
    # real data race (a probe that forced the window open corrupted 21 of 24
    # calls).  Making it safe needs a CUDA event per call, and allocating plus
    # recording that event costs more than the pageable copy saves: end to end it
    # measured 93us per call against 46us for this version.
    ptrs = [_DATA_PTR(t) for t in srcs]
    if dsts is srcs:
        values = ptrs + ptrs + numels
    else:
        values = ptrs + [_DATA_PTR(t) for t in dsts] + numels
    meta = _device_meta(values, device)

    # ``triton.cdiv`` is a decorated helper and costs ~0.9us at this size;
    # plain ints give the same ceiling division.
    grid = (nt, -(-max_numel // block))
    if is_complex:
        real_dtype = _COMPLEX_TO_REAL[in_dtype]
        acc = tl.float32 if real_dtype is not torch.float64 else tl.float64
        kernel = (
            foreach_unary_c2r_kernel
            if dsts[0].dtype.is_floating_point
            else foreach_unary_c2c_kernel
        )
        _launch_executor_kernel(
            kernel, grid, nt, meta, fn, tl_dtype(real_dtype), acc, block
        )
    else:
        kernel = (
            foreach_unary_kernel_p4
            if _pack_ok((in_dtype, dsts[0].dtype), max_numel, values)
            else foreach_unary_kernel
        )
        _launch_executor_kernel(
            kernel,
            grid,
            nt,
            meta,
            fn,
            tl_dtype(in_dtype),
            tl_dtype(dsts[0].dtype),
            block,
        )
    _STATS.executor_launches += 1
    _STATS.groups += 1


def check_dtype_supported(
    dtype: torch.dtype,
    allowed: Optional[frozenset],
    message: Optional[str] = None,
    error_type: type = RuntimeError,
) -> None:
    """Reject dtypes the corresponding ATen operator also rejects.

    ``allowed=None`` means "no restriction".  Each operator's set was measured
    against this PyTorch build rather than assumed, because the exclusions are
    not guessable: ``_foreach_floor`` rejects ``bool`` but accepts every integer
    dtype, ``_foreach_frac`` rejects all integers, and ``_foreach_sign``
    accepts ``bool`` while rejecting complex.

    ``message`` overrides the generic text for operators whose ATen refusal is
    worded differently, such as ``sub`` on bool.  ``error_type`` matches the
    class ATen raises, which for ``pow`` on bool is ``NotImplementedError``.
    """
    if allowed is not None and dtype not in allowed:
        if message is not None:
            raise error_type(message)
        raise error_type(
            f'"foreach" not implemented for \'{str(dtype).replace("torch.", "")}\''
        )


# ``flag_gems`` overrides ``Tensor.copy_`` on the CUDA key inside
# ``use_gems()``, and ``_device_meta`` runs that copy once per group.  The gems
# wrapper validates its arguments and then re-dispatches this CPU->CUDA copy to
# the composite aten kernel anyway, so address that implementation directly.
# The handle is optional: a build without ``redispatch`` keeps the method.
try:
    _ATEN_COMPOSITE_KEYSET = torch._C.DispatchKeySet(
        torch._C.DispatchKey.CompositeExplicitAutograd
    )
except (AttributeError, RuntimeError):  # pragma: no cover - depends on torch build
    _ATEN_COMPOSITE_KEYSET = None

try:
    _ATEN_COPY_REDISPATCH = torch.ops.aten.copy_.default.redispatch
except (AttributeError, RuntimeError):  # pragma: no cover - depends on torch build
    _ATEN_COPY_REDISPATCH = None


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

    # Fast path: one device, one dtype.  Bucketing exists for the mixed case,
    # and its dict/tuple bookkeeping is the largest per-tensor cost in this
    # walk; every unary operator here is normally called that way, so pay for
    # the general machinery only when the list actually mixes.
    first = tensors[0]
    device = first.device
    dtype = first.dtype
    if allowed_dtypes is not None and dtype not in allowed_dtypes:
        check_dtype_supported(dtype, allowed_dtypes)
    if dtype.is_complex and complex_fn is None:
        raise NotImplementedError(f"foreach executor has no complex kernel for {dtype}")
    out_dtype = out_dtype_fn(dtype)
    if inplace and out_dtype != dtype:
        # Same refusal as ATen: an in-place op may not silently narrow its
        # promoted result back into the input's dtype.
        raise RuntimeError(
            f"result type {str(out_dtype).replace('torch.', '').capitalize()}"
            " can't be cast to the desired output type "
            f"{str(dtype).replace('torch.', '').capitalize()}"
        )
    is_complex = dtype.is_complex
    kernel_fn = complex_fn if is_complex else fn

    # Bulk path: one device, one dtype, every input flat-addressable, real dtypes.
    # ATen's C++ grouping primitive answers the device/dtype question for the
    # whole list in one call -- measured 9.9us against 16.6us for the per-tensor
    # attribute walk over 128 tensors -- and ``map`` allocates the outputs
    # without any per-tensor loop bookkeeping.  Paired against the unchanged
    # module (interleaved, order-rotated, nine do_bench medians per side) that
    # is 40.9us of a 301.6us call at (64,) x 128, 40.2us at 65536 x 128 and
    # 3.8us at (64,) x 16, and inside the timer quantum at length 1.  Mixed lists, gappy inputs that need
    # staging, and complex inputs fall through to the general loop below, which
    # keeps the exact ATen error contract for them: when the grouping call
    # raises or reports more than one group, nothing here has been touched.
    if not is_complex and _GROUP_TENSORS is not None:
        try:
            uniform = len(_GROUP_TENSORS([tensors], False)) == 1
        except Exception:
            uniform = False  # let the general loop below produce the error
        # ``is_flat_addressable`` answers the common case on its first line but
        # costs a Python call frame per tensor.  Asking that same first line in
        # one C-level ``map`` measured 11.2us -> 6.5us over 128 tensors, and a
        # False just falls through to the full predicate, so the boolean is
        # unchanged.  ``_IS_CONTIGUOUS`` is the module-level unbound descriptor
        # defined with the grouping helpers below.
        if uniform and (
            all(map(_IS_CONTIGUOUS, tensors)) or all(map(is_flat_addressable, tensors))
        ):
            if inplace:
                # Flat implies non-overlapping and dense, so the overlap probe
                # cannot change the answer and no staging copy is needed.
                _launch_group(tensors, tensors, kernel_fn, dtype, device, False)
                return tensors
            if out_dtype == dtype:
                # One-argument form: it skips the keyword's dispatch work and
                # allocates the identical tensor whenever the output dtype is
                # the input's, which is the float case.
                dsts = list(map(torch.empty_like, tensors))
            else:
                dsts = [torch.empty_like(t, dtype=out_dtype) for t in tensors]
            _launch_group(tensors, dsts, kernel_fn, dtype, device, False)
            return dsts

    results: List[torch.Tensor] = []
    srcs: List[torch.Tensor] = []
    dsts: List[torch.Tensor] = []
    writebacks: List[Tuple[torch.Tensor, torch.Tensor]] = []

    for t in tensors:
        if t.dtype != dtype or t.device != device:
            # Mixed list: redo the call through the bucketing path.  The
            # counters are reset so staging done above is not counted twice.
            _STATS.reset()
            return _foreach_unary_grouped(
                tensors,
                fn,
                complex_fn=complex_fn,
                out_dtype_fn=out_dtype_fn,
                inplace=inplace,
                allowed_dtypes=allowed_dtypes,
            )
        flat = is_flat_addressable(t)
        if inplace:
            # A flat-addressable tensor is non-overlapping-and-dense by
            # definition, so the dispatcher-backed probe cannot change the
            # answer there; skipping it saves a dispatch per tensor.
            if not flat and has_internal_overlap(t):
                raise RuntimeError(
                    "unsupported operation: more than one element of the "
                    "written-to tensor refers to a single memory location. "
                    "Please clone() the tensor before performing the operation."
                )
            if flat:
                src = t
            else:
                # Gappy view: compute on a dense copy, then scatter back.
                src = t.contiguous()
                writebacks.append((t, src))
                _STATS.staged_inputs += 1
            results.append(t)
        else:
            src = t if flat else t.contiguous()
            if src is not t:
                _STATS.staged_inputs += 1
            # empty_like reproduces PyTorch's foreach output stride policy: it
            # keeps a dense tensor's stride (so transposed inputs give
            # transposed outputs) and normalizes everything else to contiguous.
            # One allocation per tensor, not one packed buffer plus unbind, is
            # also what keeps aliasing observable-equivalent to ATen: no two
            # outputs of a call share a storage (checked on two 4x4,
            # mixed-shape, transposed, zero-dim and one-element lists).
            if out_dtype == src.dtype:
                # The one-argument form is measurably cheaper (it skips the
                # keyword's dispatch work) and allocates the identical tensor
                # when the output dtype is the input's -- the float case.
                dst = torch.empty_like(src)
            else:
                dst = torch.empty_like(src, dtype=out_dtype)
            results.append(dst)
            dsts.append(dst)
        srcs.append(src)

    if inplace:
        # Same list object: ``_launch_group`` then knows the pointers occur
        # twice in a row and reads each ``data_ptr`` once.  Returning the
        # inputs themselves means the only aliasing is what the caller passed
        # in (a tensor repeated twice stays shared), as with ATen.
        dsts = srcs
    _launch_group(srcs, dsts, kernel_fn, dtype, device, is_complex)

    for dst_view, staged in writebacks:
        dst_view.copy_(staged)
        _STATS.writebacks += 1

    return results


def _foreach_unary_grouped(
    tensors: Sequence[torch.Tensor],
    fn: Callable,
    *,
    complex_fn: Optional[Callable] = None,
    out_dtype_fn: Callable[[torch.dtype], torch.dtype] = same_dtype,
    inplace: bool = False,
    allowed_dtypes: Optional[frozenset] = None,
) -> List[torch.Tensor]:
    """Bucketing path for a TensorList that mixes devices or dtypes.

    ``tensors`` is already validated and normalised to a list.
    """
    results: List[Optional[torch.Tensor]] = [None] * len(tensors)
    # Buckets keyed by (device, input dtype); values are aligned src/dst lists.
    buckets: Dict[Tuple[Any, torch.dtype], Tuple[List, List]] = {}
    writebacks: List[Tuple[torch.Tensor, torch.Tensor]] = []

    for i, t in enumerate(tensors):
        dtype = t.dtype
        # The helper is only a message builder; calling it per tensor for the
        # accepted case measured 0.6us on a sixteen-tensor list.
        if allowed_dtypes is not None and dtype not in allowed_dtypes:
            check_dtype_supported(dtype, allowed_dtypes)
        if dtype.is_complex and complex_fn is None:
            raise NotImplementedError(
                f"foreach executor has no complex kernel for {dtype}"
            )
        out_dtype = out_dtype_fn(dtype)
        key = (t.device, dtype)
        flat = is_flat_addressable(t)

        if inplace:
            if out_dtype != dtype:
                # Same refusal as ATen: an in-place op may not silently narrow
                # its promoted result back into the input's dtype.
                raise RuntimeError(
                    f"result type {str(out_dtype).replace('torch.', '').capitalize()}"
                    " can't be cast to the desired output type "
                    f"{str(dtype).replace('torch.', '').capitalize()}"
                )
            # A flat-addressable tensor is non-overlapping-and-dense by
            # definition, so the dispatcher-backed probe cannot change the
            # answer there; skipping it saves a dispatch per tensor.
            if not flat and has_internal_overlap(t):
                raise RuntimeError(
                    "unsupported operation: more than one element of the "
                    "written-to tensor refers to a single memory location. "
                    "Please clone() the tensor before performing the operation."
                )
            if flat:
                src = dst = t
            else:
                # Gappy view: compute on a dense copy, then scatter back.
                src = dst = t.contiguous()
                writebacks.append((t, src))
                _STATS.staged_inputs += 1
            results[i] = t
        else:
            src = t if flat else t.contiguous()
            if src is not t:
                _STATS.staged_inputs += 1
            # empty_like reproduces PyTorch's foreach output stride policy: it
            # keeps a dense tensor's stride (so transposed inputs give
            # transposed outputs) and normalizes everything else to contiguous.
            if out_dtype == src.dtype:
                # The one-argument form is measurably cheaper (it skips the
                # keyword's dispatch work) and allocates the identical tensor
                # when the output dtype is the input's -- the float case.
                dst = torch.empty_like(src)
            else:
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


# Promotion is a pure function of the operand dtypes (plus whether an operand is
# zero-dimensional, which ATen treats as a weak scalar), so the answer is
# memoised: a 16-tensor list asks the same question 16 times.
_PROMOTION_CACHE: Dict[Tuple[Any, ...], torch.dtype] = {}


def _promotion_probe(t: torch.Tensor) -> torch.Tensor:
    """A meta tensor standing in for ``t`` in a ``result_type`` query."""
    if t.dim() == 0:
        return torch.empty((), dtype=t.dtype, device="meta")
    return torch.empty(0, dtype=t.dtype, device="meta")


def _promote_dtypes(
    a: torch.Tensor, b: torch.Tensor, c: Optional[torch.Tensor] = None
) -> torch.dtype:
    """Two- or three-way ``result_type`` fold, memoised on the operand shapes.

    ``result_type`` takes two operands, so a three-way promotion is a fold with
    an intermediate dtype.  The original spelled that intermediate out as a
    fresh ``meta`` tensor per tensor in the list, which measured 54us for a
    16-tensor ``addcmul`` -- more than the kernel it feeds.  Nothing about the
    answer depends on the values, so the dtype tuple keys a dict instead.
    """
    key = (
        a.dtype,
        a.dim() == 0,
        b.dtype,
        b.dim() == 0,
        c.dtype if c is not None else None,
        c is not None and c.dim() == 0,
    )
    out = _PROMOTION_CACHE.get(key)
    if out is None:
        out = torch.result_type(_promotion_probe(a), _promotion_probe(b))
        if c is not None:
            out = torch.result_type(
                torch.empty(0, dtype=out, device="meta"), _promotion_probe(c)
            )
        _PROMOTION_CACHE[key] = out
    return out


def _promote_with_scalar(dtype: torch.dtype, value: Any) -> torch.dtype:
    """``result_type(tensor, scalar)`` memoised on ``(dtype, type(value))``.

    A Python number is a weak scalar, so its *value* never changes the answer --
    only its type (an ``int`` keeps an integral tensor integral, a ``float``
    promotes it to the default float dtype).  A zero-dimensional tensor is also
    weak, so the probe only has to match that one property.
    """
    zero_dim = isinstance(value, torch.Tensor) and value.dim() == 0
    key = ("scalar", dtype, type(value), zero_dim)
    out = _PROMOTION_CACHE.get(key)
    if out is None:
        probe = torch.empty(0, dtype=dtype, device="meta")
        out = torch.result_type(probe, value)
        _PROMOTION_CACHE[key] = out
    return out


def _common_scalar(vals: Sequence[Any]) -> Optional[Any]:
    """The single value every slot shares, or ``None`` when they differ.

    ``==`` on its own would fold ``0.0`` and ``-0.0`` together; the sign bit
    matters because ``div``/``pow`` turn it into ``+inf`` versus ``-inf``.
    NaNs never compare equal, so they fall through to the per-tensor buffer.
    """
    # ``float`` is what the device-buffer path would have applied anyway, and
    # it keeps the by-value argument a plain number even if a caller handed in
    # zero-dimensional tensors.
    first = float(vals[0])
    if first == 0.0:
        sign = math.copysign(1.0, first)
        for v in vals:
            v = float(v)
            if v != first or math.copysign(1.0, v) != sign:
                return None
        return first
    # Outside the zeros the value itself carries the sign, so the explicit
    # sign-bit test is only needed for ±0.0.  ``count`` is the same equality
    # scan driven from C: it measured 0.06us against 0.49us for the Python loop
    # over the sixteen entries of a ``.ScalarList`` call.
    count = getattr(vals, "count", None)
    if count is None:  # pragma: no cover - every call site passes a list
        for v in vals:
            if float(v) != first:
                return None
        return first
    return first if count(first) == len(vals) else None


def _meta_from_values(values: List[int], device: Any) -> torch.Tensor:
    """Build an int64 device tensor holding ``values`` (one pageable H2D copy).

    ``array.array`` + ``frombuffer`` skips the generic sequence-to-tensor path,
    and going through ``empty`` + ``copy_`` sidesteps the dispatcher's
    ``_to_copy``: inside ``use_gems()`` the plain ``torch.tensor(...).to(device)``
    form runs the gems implementation and measured 21.5us against 13.1us here.

    The copy addresses the composite aten implementation directly.  Inside
    ``use_gems()`` the ``Tensor.copy_`` method runs flag_gems' Python wrapper,
    which validates its arguments and then redispatches this CPU->CUDA copy to
    exactly that composite kernel anyway; the wrapper measured ~10us per call
    (P1's cProfile: ``copy_`` cumulative 13.5-14.9us, of which the transfer is
    small).  ``redispatch`` is only an optimisation, so a build without it falls
    back to the regular method.
    """
    host = torch.frombuffer(array.array("q", values), dtype=torch.int64)
    out = torch.empty(len(values), dtype=torch.int64, device=device)
    if _ATEN_COPY_REDISPATCH is None or _ATEN_COMPOSITE_KEYSET is None:
        out.copy_(host, non_blocking=True)
    else:
        # Skip the Python-level copy_ registration and go straight to the
        # CompositeExplicitAutograd implementation.
        _ATEN_COPY_REDISPATCH(_ATEN_COMPOSITE_KEYSET, out, host, True)
    return out


# A pageable H2D copy blocks the host until the stream reaches it, and it is the
# single largest exposed cost of a small foreach call: the request is 100-200
# bytes wide, so the transfer itself is nothing and the wait is everything.
# In a steady loop the request is byte-identical from call to call -- the same
# tensor list, and outputs served from the caching allocator's recycled blocks
# -- so the built tensor can be reused instead of rebuilt.
#
# Reuse is exact, not a heuristic about what the metadata "probably" is: the key
# is the whole value list, so a hit can only return a buffer whose contents are
# the bytes this call would have written.  The buffer is never written again
# after it is built and the kernel only reads it, so an in-flight launch from an
# earlier call cannot observe anything different.  The *stream* is part of the
# key because that ordering argument is per stream: a buffer built on stream A
# is only picked up by calls that run on stream A, where the H2D that produced
# it is ordered before the new launch.  A call on another stream simply misses
# and builds its own.  Nothing is refilled, so this does not reintroduce the
# reuse hazard that sank the pinned staging pool (that pool rewrote the bytes a
# DMA was still reading).
#
# The bound keeps a workload whose addresses keep moving from growing the cache
# without limit; insertion order is the eviction order.
_META_CACHE: Dict[Tuple[Any, int, Tuple[int, ...]], torch.Tensor] = {}
_META_CACHE_LIMIT = 32

# Last-hit shortcut in front of the memo above.  Building the key (``tuple``
# over the 4*16 entries plus its hash) measured 1.07us, while the same
# comparison against a retained copy of the previous value list is a single
# C-level list compare at 0.09us -- a steady-state call re-launches with the
# same addresses, because the caching allocator hands back the blocks the
# previous call released.  [values, stream, device, buffer]
#
# The copy is made so that a caller which mutates a reused list in place is
# still compared against what was actually uploaded.  A tuple needs no copy
# (immutable) and is retained as the caller built it: the ternary launcher
# hands in a freshly built tuple, and comparing it against a stored *list*
# could never be equal, which is what kept this shortcut dead on that path.
_LAST_META: List[Any] = [None, None, None, None]


def _device_meta(values: List[int], device: Any) -> torch.Tensor:
    """A read-only int64 device tensor holding ``values`` (memoised).

    See :func:`_meta_from_values` for how it is built and the cache comment
    above for why a hit is exact.
    """
    # ``device`` is a ``torch.device`` carrying an index at every call site; a
    # bare ``torch.device("cuda")`` means the current device, which is what
    # ``torch.empty`` would have used.
    index = device.index if isinstance(device, torch.device) else None
    if index is None:
        index = torch.cuda.current_device()
    stream = _triton_driver.active.get_current_stream(index)
    last = _LAST_META
    if last[1] is stream and last[2] == device and last[0] == values:
        return last[3]
    key = (device, stream, tuple(values))
    cached = _META_CACHE.get(key)
    if cached is None:
        cached = _meta_from_values(values, device)
        if len(_META_CACHE) >= _META_CACHE_LIMIT:
            _META_CACHE.pop(next(iter(_META_CACHE)))
        _META_CACHE[key] = cached
    last[0] = values if isinstance(values, tuple) else list(values)
    last[1] = stream
    last[2] = device
    last[3] = cached
    return cached


def _resolve_scalars(
    tensors: Sequence[torch.Tensor], scalars: Any
) -> Tuple[List[Any], Optional[float], Optional[type]]:
    """Normalise the ``.Scalar`` / ``.ScalarList`` / ``.Tensor`` forms into a list.

    The ``.Tensor`` overloads of ``addcmul``/``addcdiv`` take the values as a
    one-dimensional tensor holding one entry per tensor in the list (ATen
    requires it on the CPU), so it is unpacked here rather than being mistaken
    for a single value.

    Returns ``(values, common, vclass)``: one entry per tensor, plus the value
    every entry shares as a ``float`` -- or ``None`` when a scan is needed to
    find out, or when the value is not a real number and cannot be converted --
    plus the Python type of every entry when the broadcast form makes it one
    object repeated (``None`` when it has to be looked at per element).  A plain
    broadcast scalar is already its own common value, so the equality scan that
    ``_common_scalar`` performs can be skipped for it.

    ``float`` conversion is limited to ``numbers.Real`` so that an illegal
    complex scalar keeps the error the base implementation produced: the
    launcher's ``_common_scalar`` raises ``TypeError`` for it, and for an
    all-empty tensor list -- where the launch is skipped -- the call returns
    the promoted empty outputs instead of failing early here.
    """
    if type(scalars) is float:
        # A plain float is the common form and is already its own common
        # value; reaching the same answer through the three ``isinstance``
        # tests below (one of them against the ``numbers.Real`` ABC) costs
        # 0.4us a call.
        return [scalars] * len(tensors), scalars, float
    if isinstance(scalars, torch.Tensor):
        if scalars.numel() == 1:
            value = scalars.item()
            # Keep the Python type of the original value in ``values``: an
            # ``int`` scalar promotes an integral tensor differently from a
            # ``float``.  The by-value argument is always the float the device
            # buffer would have held.
            if isinstance(value, numbers.Real):
                return [value] * len(tensors), float(value), type(value)
            return [value] * len(tensors), None, None
        _check_same_length(tensors, scalars, "scalar list")
        # ``tolist`` reads the elements; ``detach`` and ``cpu`` only built two
        # throwaway tensors first (the ATen overload already requires CPU, and
        # ``tolist`` syncs on its own if a build ever lets a device tensor
        # through).  Measured 2.1us of the call at sixteen entries.
        values = scalars.tolist()
        return values, _common_scalar(values), None
    if isinstance(scalars, (list, tuple)):
        _check_same_length(tensors, scalars, "scalar list")
        # The scored ``.ScalarList`` call hands in ``[0.5] * 16``: an exact
        # ``float`` repeated.  ``list.count`` runs the same ``==`` scan
        # ``_common_scalar`` drives from C, so the common value is known
        # without the copy and without the helper -- and ``values`` is only
        # ever read downstream (the general loop, ``_ternary_upload``,
        # ``_common_scalar``), never stored or mutated, so the caller's list
        # can be passed through as it is.  Restricted to a ``list`` whose head
        # is an exact ``float``: a tuple keeps its own type, and the ±0.0 sign
        # case still goes through ``_common_scalar``.
        if type(scalars) is list and scalars:
            first = scalars[0]
            if (
                type(first) is float
                and first != 0.0
                and scalars.count(first) == len(scalars)
            ):
                return scalars, first, None
        values = list(scalars)
        return values, _common_scalar(values), None
    if isinstance(scalars, numbers.Real):
        return [scalars] * len(tensors), float(scalars), type(scalars)
    return [scalars] * len(tensors), None, None


def _finish_inplace(
    results: List[torch.Tensor],
    writebacks: List[Tuple[torch.Tensor, torch.Tensor]],
) -> None:
    for dst_view, staged in writebacks:
        dst_view.copy_(staged)
        _STATS.writebacks += 1
    del results


# ATen's own TensorList grouping primitive: one C++ pass over the lists, keyed
# by ``(device, dtype)``.  The executor only needs the *number* of groups to
# decide whether a call is uniform, and measured 2.8us against 6.3us for the
# per-tensor attribute walk over two sixteen-tensor lists.  Absent on a build
# without it, in which case the walk below is used.
try:
    _GROUP_TENSORS = torch._C._group_tensors_by_device_and_dtype
except AttributeError:  # pragma: no cover - very old builds
    _GROUP_TENSORS = None


def _binary_homogeneous(
    self: List[torch.Tensor],
    other: List[torch.Tensor],
    fn: Callable,
    inplace: bool,
    alpha: float,
    allowed_dtypes: Optional[frozenset],
    dtype_error: Optional[str],
    dtype_error_type: type,
    out_dtype_fn: Optional[Callable[[torch.dtype], torch.dtype]],
    out_dtypes: Optional[Sequence[torch.dtype]],
    scalar_operand: bool,
) -> Optional[List[torch.Tensor]]:
    """Execute a uniform binary list in one bucket, or return ``None``.

    ``None`` means the caller's lists need the general per-tensor bucket loop;
    it is the answer for every kind of heterogeneity that loop exists to
    handle (device, dtype, rank class, per-position output dtype).  The scan
    that decides this is also the only per-tensor work left on the hot path:
    the list arguments, dtype checks and result allocation are comprehensions.

    ``scalar_operand`` lists are one repeated single-element tensor by
    contract, so they are neither scanned nor staged: a one-element tensor is
    always contiguous and flat-addressable.
    """
    a0 = self[0]
    dev = a0.device
    dt_a = a0.dtype
    # When ``out_dtypes`` pins the result the promotion is fixed, so the rank
    # class of an operand can no longer change it and is not checked.
    need_promotion = out_dtypes is None
    b0 = other[0]
    dt_b = b0.dtype
    # Uniform in device *and* dtype is the hot case.  One C++ grouping pass
    # answers it for both lists at once, and the alternative -- walking both
    # lists in Python -- was the single largest host cost left here.  The
    # comment used to argue the two generator passes won on the strength of a
    # standalone timing of the idiom in isolation; measured in situ inside the
    # scored benchmark (source-exact substitution, 16 tensors), the grouping
    # pass is 1.0-1.7us cheaper on ``_foreach_add.List``.  When both lists are
    # uniform the promotion is the identity and the rank class of an operand
    # cannot change the result dtype, so the per-tensor ``dim()`` check below
    # is skipped too.
    # ``scalar_operand`` lists are one repeated single-element tensor, whose
    # dtype may legitimately differ from ``self``'s, so they keep the walk.
    uniform = False
    if not scalar_operand and dt_b is dt_a:
        if _GROUP_TENSORS is not None:
            uniform = len(_GROUP_TENSORS([self + other], False)) == 1
        else:
            uniform = all(t.dtype is dt_a and t.device == dev for t in self) and all(
                t.dtype is dt_a and t.device == dev for t in other
            )
    if not uniform:
        zero_dim = a0.dim() == 0
        for t in self:
            if t.dtype is not dt_a or t.device != dev:
                return None
            if need_promotion and (t.dim() == 0) != zero_dim:
                return None

        if scalar_operand:
            if b0.device != dev:
                return None
        else:
            b_zero_dim = b0.dim() == 0
            for t in other:
                if t.dtype is not dt_b or t.device != dev:
                    return None
                if need_promotion and (t.dim() == 0) != b_zero_dim:
                    return None

    if allowed_dtypes is not None and dt_a not in allowed_dtypes:
        check_dtype_supported(dt_a, allowed_dtypes, dtype_error, dtype_error_type)
    if out_dtypes is not None:
        out_dtype = out_dtypes[0]
        for dt in out_dtypes:
            if dt is not out_dtype:
                return None
    elif dt_a is dt_b:
        out_dtype = dt_a
    else:
        out_dtype = _promote_dtypes(a0, b0)
    if out_dtype_fn is not None:
        out_dtype = out_dtype_fn(out_dtype)

    conts = list(map(_IS_CONTIGUOUS, self))
    if all(conts):
        srcs_a = self
    else:
        srcs_a = [t if c else _dense(t) for t, c in zip(self, conts)]
    if scalar_operand:
        srcs_b = other
    else:
        bconts = list(map(_IS_CONTIGUOUS, other))
        srcs_b = (
            other
            if all(bconts)
            else [t if c else _dense(t) for t, c in zip(other, bconts)]
        )

    if inplace:
        # No allocation: the destination is the source, and a source that had
        # to be staged is written back once the kernel has filled it.
        if out_dtype != dt_a:
            raise RuntimeError(
                f"result type {str(out_dtype).replace('torch.', '').capitalize()}"
                " can't be cast to the desired output type "
                f"{str(dt_a).replace('torch.', '').capitalize()}"
            )
        if srcs_a is not self:
            for t, c in zip(self, conts):
                if not c and has_internal_overlap(t):
                    raise RuntimeError(
                        "unsupported operation: more than one element of the "
                        "written-to tensor refers to a single memory location. "
                        "Please clone() the tensor before performing the operation."
                    )
            writebacks = [(t, s) for t, s in zip(self, srcs_a) if s is not t]
        else:
            writebacks = []
        results = list(self)
        dsts = srcs_a
    else:
        # The one-argument form skips the keyword dispatch the explicit dtype
        # form pays on every one of the sixteen allocations, and ``map`` keeps
        # the per-tensor call inside the C loop: a list comprehension pays the
        # ``torch.empty_like`` lookup and call setup sixteen times, which
        # measured 2-3us more per call (256^2-1024^2, all three dtypes).
        if out_dtype is dt_a:
            results = list(map(torch.empty_like, srcs_a))
        else:
            results = [torch.empty_like(t, dtype=out_dtype) for t in srcs_a]
        dsts = results
        writebacks = []

    _launch_binary_group(
        srcs_a, srcs_b, dsts, fn, dt_a, dt_b, dev, alpha, scalar_operand
    )
    _finish_inplace(results, writebacks)
    return results


def foreach_binary_list(
    self: Sequence[torch.Tensor],
    other: Sequence[torch.Tensor],
    fn: Callable,
    *,
    inplace: bool = False,
    alpha: float = 1,
    allowed_dtypes: Optional[frozenset] = None,
    dtype_error: Optional[str] = None,
    dtype_error_type: type = RuntimeError,
    out_dtype_fn: Optional[Callable[[torch.dtype], torch.dtype]] = None,
    out_dtypes: Optional[Sequence[torch.dtype]] = None,
    copy_semantics: bool = False,
    scalar_operand: bool = False,
) -> List[torch.Tensor]:
    """``self[i] <op> alpha * other[i]`` for every position, one launch per group.

    Both lists are paired position by position, so they must have equal length;
    ATen raises rather than broadcasting across the list.

    ``alpha`` scales the second operand inside the kernel.

    ``copy_semantics`` replaces type promotion with assignment semantics:
    ``_foreach_copy`` keeps the destination dtype and simply casts the source
    into it, in both the functional and the in-place form.

    ``out_dtypes`` lets the caller pin the result dtype per position. The
    ``.Tensor`` overloads need it: their operand is a scalar tensor expanded to
    the list's shape, and promoting against the *expanded* view would widen an
    fp16 list to fp32, where ATen keeps fp16 because a zero-dimensional tensor
    counts as a scalar.

    ``scalar_operand`` marks the second list as one repeated single-element
    tensor (the ``.Tensor`` broadcast); the kernel then reads it as a uniform
    scalar instead of indexing it element by element.
    """
    self = check_tensor_list(self)
    other = check_tensor_list(other)
    _check_same_length(self, other, "tensor list")
    if out_dtypes is not None:
        _check_same_length(self, out_dtypes, "output dtype list")
    _STATS.reset()

    # Uniform lists -- same device, dtype and (when nothing pins the output
    # dtype) the same rank class -- need no bucketing at all: one launch, one
    # metadata buffer.  This is the optimizer hot path, so it is decided and
    # executed by a helper that never touches the per-tensor bookkeeping below.
    if not copy_semantics:
        homogeneous = _binary_homogeneous(
            self,
            other,
            fn,
            inplace,
            alpha,
            allowed_dtypes,
            dtype_error,
            dtype_error_type,
            out_dtype_fn,
            out_dtypes,
            scalar_operand,
        )
        if homogeneous is not None:
            return homogeneous

    results: List[torch.Tensor] = []
    buckets: Dict[Tuple[Any, torch.dtype, torch.dtype], Tuple[List, List, List]] = {}
    writebacks: List[Tuple[torch.Tensor, torch.Tensor]] = []
    prev_key = None
    prev_promotion = None
    prev_out_dtype = None
    for i, (a, b) in enumerate(zip(self, other)):
        if allowed_dtypes is not None and a.dtype not in allowed_dtypes:
            check_dtype_supported(
                a.dtype, allowed_dtypes, dtype_error, dtype_error_type
            )
        if copy_semantics:
            # The destination dtype is the source of truth; the operand is cast
            # into it rather than promoted against it.
            out_dtype = a.dtype
        elif out_dtypes is not None:
            out_dtype = out_dtypes[i]
        else:
            # Two operands of the *same* dtype fold to that dtype whatever their
            # ranks are -- the weak-scalar rule only reorders categories -- so
            # the common same-dtype list skips both the probe and the four
            # ``dim()`` calls that key it.
            a_dtype = a.dtype
            if a_dtype is b.dtype:
                promotion = a_dtype
            else:
                promotion = (a_dtype, a.dim() == 0, b.dtype, b.dim() == 0)
            if promotion != prev_promotion:
                out_dtype = a_dtype if a_dtype is b.dtype else _promote_dtypes(a, b)
                if out_dtype_fn is not None:
                    out_dtype = out_dtype_fn(out_dtype)
                prev_promotion = promotion
                prev_out_dtype = out_dtype
            else:
                out_dtype = prev_out_dtype
        contiguous_a = a.is_contiguous()
        src_a = a if contiguous_a else _dense(a)
        src_b = b if b.is_contiguous() else _dense(b)
        if inplace:
            if not copy_semantics and out_dtype != a.dtype:
                raise RuntimeError(
                    f"result type {str(out_dtype).replace('torch.', '').capitalize()}"
                    " can't be cast to the desired output type "
                    f"{str(a.dtype).replace('torch.', '').capitalize()}"
                )
            if not contiguous_a and has_internal_overlap(a):
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
            # The one-argument form skips the keyword/dtype dispatch that the
            # explicit form pays on every one of the sixteen allocations.
            dst = (
                torch.empty_like(src_a)
                if out_dtype is src_a.dtype
                else torch.empty_like(src_a, dtype=out_dtype)
            )
            results.append(dst)
        key = (a.device, src_a.dtype, src_b.dtype)
        if key != prev_key:
            bucket = buckets.get(key)
            if bucket is None:
                bucket = buckets[key] = ([], [], [])
            prev_key = key
        bucket[0].append(src_a)
        bucket[1].append(src_b)
        bucket[2].append(dst)

    for (device, a_dt, b_dt), (a_list, b_list, d_list) in buckets.items():
        _launch_binary_group(
            a_list, b_list, d_list, fn, a_dt, b_dt, device, alpha, scalar_operand
        )

    _finish_inplace(results, writebacks)
    return results


# Every argument of ``foreach_binary_kernel`` other than the metadata pointer is
# a ``tl.constexpr``, so ``(nt, fn, A_DT, B_DT, OUT_DT, ALPHA, B_SCALAR, BLOCK)``
# identifies the compiled variant and a replay only has to hand those values to
# the kernel's own launcher.  Triton's Python dispatch between that lookup and
# the launch -- binder, cache-key hash, launch-metadata and hook checks --
# measured ~10us per call on the small shapes this executor targets, and is the
# difference between the binary rows and their ATen counterparts.
#
# The cache is keyed on the device because a compiled kernel belongs to one
# device's module, and on the metadata pointer's 16-byte alignment because that
# is the only thing Triton specialises on here.  ``alpha`` carries its Python
# type in the key: ``1`` and ``1.0`` compare (and hash) equal but are different
# specialisations of a ``constexpr``.
#
# Only the launcher is cached; the metadata tensor is rebuilt and uploaded on
# every call, so no device-visible buffer is shared across calls.
_BINARY_LAUNCH_CACHE: Dict[Tuple[Any, ...], Tuple[Any, Any, Any]] = {}
_BINARY_LAUNCH_STATS: Dict[str, int] = {"replay": 0, "dispatch": 0}

# One-hit memos for the two dictionary lookups that still sat on the hot path.
#
# ``_launch_binary_group`` derives the kernel choice, block and grid from the
# metadata it just built, and ``_launch_binary_kernel`` derives the launcher
# from a twelve-element key.  A benchmark loop, and any caller repeating one
# call shape, presents the same arguments on every iteration, so an identity
# compare against the previous call's arguments answers both without hashing.
# The entries hold the metadata tensor (as ``_META_CACHE`` already does), and
# they are consulted only for an *identical* metadata tensor, which by
# construction encodes the pointer and numel run the plan was computed from --
# a different call shape produces a different tensor object and misses.
_LAST_BINARY_PLAN: List[Any] = [None]
_LAST_BINARY_REPLAY: List[Any] = [None]


def _launch_binary_kernel(
    meta: torch.Tensor,
    grid: Tuple[int, int],
    nt: int,
    fn: Callable,
    a_dt: Any,
    b_dt: Any,
    out_dt: Any,
    alpha: float,
    b_scalar: bool,
    block: int,
    kernel: Callable = foreach_binary_kernel,
) -> None:
    # A repeated call shape reaches here with the same arguments every time.
    # The twelve-element key tuple and the dict hash for it measured 1.2us at
    # sixteen positions; the identity compare below answers the same question
    # and the closure it calls has already captured everything but the stream,
    # which is still read per call so a launch issued under a different
    # ``torch.cuda.stream`` lands on the stream current at launch time.
    last = _LAST_BINARY_REPLAY[0]
    runtime = knobs.runtime
    if (
        last is not None
        and last[0] is meta
        and last[1] is kernel
        and last[2] == nt
        and last[3] is fn
        and last[4] is a_dt
        and last[5] is b_dt
        and last[6] is out_dt
        and last[7] is alpha
        and last[8] is b_scalar
        and last[9] == block
        and last[10] == grid
        and not runtime.launch_enter_hook.calls
        and not runtime.launch_exit_hook.calls
    ):
        try:
            last[11]()
        except Exception:
            # A memoised launcher can go stale if Triton invalidates the
            # compiled entry it captured.  Forget it and take the same path a
            # first call takes rather than raising out of the op.
            _LAST_BINARY_REPLAY[0] = None
        else:
            _BINARY_LAUNCH_STATS["replay"] += 1
            return
    key = (
        kernel,
        meta.device,
        nt,
        fn,
        a_dt,
        b_dt,
        out_dt,
        alpha,
        type(alpha),
        b_scalar,
        block,
        not (meta.data_ptr() & 15),
    )
    entry = _BINARY_LAUNCH_CACHE.get(key)
    # ``launch_enter_hook`` is a ``HookChain`` in triton 3.6, never ``None``;
    # ``.calls`` is the list of registered callbacks, so an empty list is the
    # real "no hooks" state.  Proton and other instrumentation add to it, and a
    # replay would then skip their notifications.
    if (
        entry is not None
        and not runtime.launch_enter_hook.calls
        and not runtime.launch_exit_hook.calls
    ):
        launch = entry[0]
        packed = entry[1]
        pmeta = entry[2]
        g0, g1 = grid
        idx = meta.device.index

        def replay():
            launch(
                g0,
                g1,
                1,
                _triton_driver.active.get_current_stream(idx),
                packed,
                pmeta,
                None,
                None,
                None,
                meta,
                fn,
                nt,
                a_dt,
                b_dt,
                out_dt,
                alpha,
                b_scalar,
                block,
            )

        try:
            replay()
        except Exception:
            # A launcher that cannot be replayed this way is not a correctness
            # problem: forget the entry and let Triton dispatch as usual.
            _LAST_BINARY_REPLAY[0] = None
            _BINARY_LAUNCH_CACHE.pop(key, None)
        else:
            _LAST_BINARY_REPLAY[0] = (
                meta,
                kernel,
                nt,
                fn,
                a_dt,
                b_dt,
                out_dt,
                alpha,
                b_scalar,
                block,
                grid,
                replay,
            )
            _BINARY_LAUNCH_STATS["replay"] += 1
            return
    _BINARY_LAUNCH_STATS["dispatch"] += 1
    compiled = kernel[grid](meta, fn, nt, a_dt, b_dt, out_dt, alpha, b_scalar, block)
    if compiled is not None:
        # A dispatch may have replaced the compiled entry, so the closure built
        # from the previous one is no longer known to be current.
        _LAST_BINARY_REPLAY[0] = None
        _BINARY_LAUNCH_CACHE[key] = (
            compiled.run,
            compiled.function,
            compiled.packed_metadata,
        )


# ``_pick_block`` searches five block sizes, and the second grid dimension is a
# ``cdiv`` of the same two numbers, so ``(len(numels), max(numels))`` fully
# determines both.  On the shapes this executor targets the search plus the
# ``cdiv`` measured 1.3us against ~0.1 for the dict hit, and the pair repeats
# from call to call.  The bound exists because a workload whose shapes keep
# moving would otherwise accumulate one entry per shape; insertion order is the
# eviction order, as in the metadata memo.
_GRID_CACHE: Dict[Tuple[int, int], Tuple[int, Tuple[int, int]]] = {}
_GRID_CACHE_LIMIT = 64


def _launch_binary_group(
    a_list: List[torch.Tensor],
    b_list: List[torch.Tensor],
    d_list: List[torch.Tensor],
    fn: Callable,
    a_dt: torch.dtype,
    b_dt: torch.dtype,
    device: Any,
    alpha: float = 1,
    b_scalar: bool = False,
) -> None:
    # One loop over ``a_list`` reads both numbers the launch needs.  Three
    # list comprehensions over the same sixteen tensors measured 3.7us; the
    # loop with two bound ``append`` methods measured 2.7us, and ``max`` over
    # the finished list is a C-level reduction.
    ptrs_a = []
    numels = []
    add_ptr = ptrs_a.append
    add_numel = numels.append
    for t in a_list:
        add_ptr(t.data_ptr())
        add_numel(t.numel())
    max_numel = max(numels)
    if max_numel == 0:
        return
    nt = len(a_list)
    # One flat Python list -> one tensor -> one H2D transfer, the same staging
    # as ``_launch_group``.  The four segments are appended into that list
    # directly instead of being built as four lists and concatenated: the
    # concatenation walked every integer a second time and allocated three
    # intermediate lists, which at sixteen positions was measurable here.  An
    # in-place call writes back into its inputs, so the destination run repeats
    # the pointers read above instead of reading them again.  The broadcast
    # operand is one repeated single-element tensor, and the kernel only ever
    # dereferences its first entry, so its run is one read.
    inplace = d_list is a_list
    values = ptrs_a[:] if inplace else ptrs_a
    if b_scalar:
        values.extend([b_list[0].data_ptr()] * nt)
    else:
        values.extend(map(_DATA_PTR, b_list))
    values.extend(ptrs_a if inplace else map(_DATA_PTR, d_list))
    values.extend(numels)
    meta = _device_meta(values, device)
    # The kernel choice, block and grid are functions of the metadata run just
    # built plus the three dtypes and the pack threshold, so an identical
    # metadata tensor answers the previous call's answer.
    d_dt = d_list[0].dtype
    plan = _LAST_BINARY_PLAN[0]
    if (
        plan is not None
        and plan[0] is meta
        and plan[1] is a_dt
        and plan[2] is b_dt
        and plan[3] is d_dt
        and plan[4] is inplace
        and plan[5] == nt
    ):
        kernel, block, grid = plan[6], plan[7], plan[8]
    else:
        cached_grid = _GRID_CACHE.get((nt, max_numel))
        if cached_grid is None:
            block = _pick_block(numels)
            cached_grid = (block, (nt, triton.cdiv(max_numel, block)))
            if len(_GRID_CACHE) >= _GRID_CACHE_LIMIT:
                _GRID_CACHE.pop(next(iter(_GRID_CACHE)))
            _GRID_CACHE[(nt, max_numel)] = cached_grid
        block, grid = cached_grid
        kernel = (
            foreach_binary_kernel_p4
            if _pack_ok(
                (a_dt, b_dt, d_dt),
                max_numel,
                values,
                _PACK_MIN_ELEMS_BINARY_INPLACE if inplace else _PACK_MIN_ELEMS,
            )
            else foreach_binary_kernel
        )
        _LAST_BINARY_PLAN[0] = (
            meta,
            a_dt,
            b_dt,
            d_dt,
            inplace,
            nt,
            kernel,
            block,
            grid,
        )
    _launch_binary_kernel(
        meta,
        grid,
        nt,
        fn,
        tl_dtype(a_dt),
        tl_dtype(b_dt),
        tl_dtype(d_dt),
        alpha,
        b_scalar,
        block,
        kernel,
    )
    _STATS.executor_launches += 1
    _STATS.groups += 1


def _binary_scalar_homogeneous(
    self: List[torch.Tensor],
    values: List[Any],
    vclass: Optional[type],
    common: Optional[float],
    fn: Callable,
    inplace: bool,
    alpha: float,
    allowed_dtypes: Optional[frozenset],
    dtype_error: Optional[str],
    dtype_error_type: type,
    out_dtype_fn: Optional[Callable[[torch.dtype], torch.dtype]],
) -> Optional[List[torch.Tensor]]:
    """Execute a uniform scalar list in one bucket, or return ``None``.

    The counterpart of ``_binary_homogeneous`` for the ``.Scalar`` /
    ``.ScalarList`` forms.  ``None`` means the caller's per-tensor bucket loop
    has to run: the lists are heterogeneous in device or dtype, or the values
    do not all promote the same way.
    """
    a0 = self[0]
    dev = a0.device
    dt = a0.dtype
    # Uniformity and contiguity are the same walk over the list, so they share
    # one pass: ``is_contiguous`` has no side effect, and the partial answer is
    # simply dropped when the early exit is taken.
    # ``t.device`` builds a device object per tensor just to compare it; the
    # integer index is the same test and skips the construction.  It is only
    # available once the device has an index, which on a CUDA-only build it
    # always has, so the device-object comparison stays as the fallback.
    didx = dev.index
    conts = []
    append = conts.append
    if didx is not None:
        for t in self:
            if t.dtype is not dt or t.get_device() != didx:
                return None
            append(t.is_contiguous())
    else:
        for t in self:
            if t.dtype is not dt or t.device != dev:
                return None
            append(t.is_contiguous())
    # ``_promote_with_scalar`` memoises on ``(dtype, type(value), zero-dim)``,
    # so with one dtype the whole list promotes alike as soon as every entry
    # has the same class.  The broadcast forms already repeat one object, so
    # only a real list needs to be looked at; a plain ``float`` -- the usual
    # case -- needs no tuple build and no ``isinstance`` per entry.
    v0 = values[0]
    if vclass is None:
        if type(v0) is float:
            for v in values:
                if type(v) is not float:
                    return None
        else:
            v_key = (type(v0), isinstance(v0, torch.Tensor) and v0.dim() == 0)
            for v in values:
                if (type(v), isinstance(v, torch.Tensor) and v.dim() == 0) != v_key:
                    return None

    if allowed_dtypes is not None and dt not in allowed_dtypes:
        check_dtype_supported(dt, allowed_dtypes, dtype_error, dtype_error_type)
    out_dtype = _promote_with_scalar(dt, v0)
    if out_dtype_fn is not None:
        out_dtype = out_dtype_fn(out_dtype)

    if all(conts):
        srcs = self
    else:
        srcs = [t if c else _dense(t) for t, c in zip(self, conts)]

    if inplace:
        # No allocation: the destination is the source, and a source that had
        # to be staged is written back once the kernel has filled it.
        if out_dtype != dt:
            raise RuntimeError(
                f"result type {str(out_dtype).replace('torch.', '').capitalize()}"
                " can't be cast to the desired output type "
                f"{str(dt).replace('torch.', '').capitalize()}"
            )
        if srcs is not self:
            for t, c in zip(self, conts):
                if not c and has_internal_overlap(t):
                    raise RuntimeError(
                        "unsupported operation: more than one element of the "
                        "written-to tensor refers to a single memory location. "
                        "Please clone() the tensor before performing the operation."
                    )
            writebacks = [(t, s) for t, s in zip(self, srcs) if s is not t]
        else:
            writebacks = []
        results = list(self)
        dsts = srcs
    else:
        # See ``_binary_homogeneous``: ``map`` keeps the sixteen allocations
        # inside the C loop instead of paying the lookup and call setup sixteen
        # times.
        if out_dtype is dt:
            results = list(map(torch.empty_like, srcs))
        else:
            results = [torch.empty_like(t, dtype=out_dtype) for t in srcs]
        dsts = results
        writebacks = []

    _launch_scalar_group(srcs, dsts, values, fn, dt, out_dtype, dev, alpha, common)
    _finish_inplace(results, writebacks)
    return results


def foreach_binary_scalar(
    self: Sequence[torch.Tensor],
    scalars: Any,
    fn: Callable,
    *,
    inplace: bool = False,
    alpha: float = 1,
    allowed_dtypes: Optional[frozenset] = None,
    dtype_error: Optional[str] = None,
    dtype_error_type: type = RuntimeError,
    out_dtype_fn: Optional[Callable[[torch.dtype], torch.dtype]] = None,
) -> List[torch.Tensor]:
    """``self[i] <op> alpha * scalar`` -- serves the ``.Scalar`` / ``.ScalarList`` forms.

    ``scalars`` is either one number (broadcast to every tensor) or a sequence
    of one number per tensor.  ``alpha`` scales the scalar inside the kernel.
    """
    self = check_tensor_list(self)
    values, common, vclass = _resolve_scalars(self, scalars)
    _STATS.reset()

    homogeneous = _binary_scalar_homogeneous(
        self,
        values,
        vclass,
        common,
        fn,
        inplace,
        alpha,
        allowed_dtypes,
        dtype_error,
        dtype_error_type,
        out_dtype_fn,
    )
    if homogeneous is not None:
        return homogeneous

    results: List[torch.Tensor] = []
    buckets: Dict[Tuple[Any, torch.dtype, torch.dtype], Tuple[List, List, List]] = {}
    writebacks: List[Tuple[torch.Tensor, torch.Tensor]] = []
    prev_key = None
    prev_promotion = None
    prev_out_dtype = None
    for t, v in zip(self, values):
        if allowed_dtypes is not None and t.dtype not in allowed_dtypes:
            check_dtype_supported(
                t.dtype, allowed_dtypes, dtype_error, dtype_error_type
            )
        # ``_promote_with_scalar`` memoises on exactly this triple, so a run of
        # tensors sharing it can reuse the previous answer instead of paying
        # for another lookup.  The broadcast form makes the value's class the
        # same object for every position, so only the dtype is looked at.
        promotion = (
            (t.dtype, vclass)
            if vclass is not None
            else (
                t.dtype,
                type(v),
                isinstance(v, torch.Tensor) and v.dim() == 0,
            )
        )
        if promotion != prev_promotion:
            out_dtype = _promote_with_scalar(t.dtype, v)
            if out_dtype_fn is not None:
                out_dtype = out_dtype_fn(out_dtype)
            prev_promotion = promotion
            prev_out_dtype = out_dtype
        else:
            out_dtype = prev_out_dtype
        contiguous = t.is_contiguous()
        src = t if contiguous else _dense(t)
        if inplace:
            if out_dtype != t.dtype:
                raise RuntimeError(
                    f"result type {str(out_dtype).replace('torch.', '').capitalize()}"
                    " can't be cast to the desired output type "
                    f"{str(t.dtype).replace('torch.', '').capitalize()}"
                )
            if not contiguous and has_internal_overlap(t):
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
            # Same one-argument shortcut as the list path: the explicit dtype
            # form pays for keyword parsing on each of the sixteen allocations.
            dst = (
                torch.empty_like(src)
                if out_dtype is src.dtype
                else torch.empty_like(src, dtype=out_dtype)
            )
            results.append(dst)
        key = (t.device, src.dtype, out_dtype)
        if key != prev_key:
            bucket = buckets.get(key)
            if bucket is None:
                bucket = buckets[key] = ([], [], [])
            prev_key = key
        bucket[0].append(src)
        bucket[1].append(dst)
        bucket[2].append(v)

    for (device, in_dt, out_dt), (srcs, dsts, vals) in buckets.items():
        _launch_scalar_group(srcs, dsts, vals, fn, in_dt, out_dt, device, alpha, common)

    _finish_inplace(results, writebacks)
    return results


# The scalar kernel's launcher is cached exactly like the binary one's: same
# key shape, same hook check, same fail-open behaviour.  The two pointer runs
# (metadata and the per-tensor value buffer) both have to keep their alignment,
# because that is what Triton specialises each one on.
_SCALAR_LAUNCH_CACHE: Dict[Tuple[Any, ...], Tuple[Any, Any, Any]] = {}
_SCALAR_LAUNCH_STATS: Dict[str, int] = {"replay": 0, "dispatch": 0}


def _launch_scalar_kernel(
    meta: torch.Tensor,
    scalar_buf: torch.Tensor,
    scalar_value: float,
    grid: Tuple[int, int],
    nt: int,
    fn: Callable,
    a_dt: Any,
    out_dt: Any,
    s_dt: Any,
    alpha: float,
    by_value: bool,
    block: int,
    kernel: Callable = foreach_binary_scalar_kernel,
) -> None:
    key = (
        kernel,
        meta.device,
        nt,
        fn,
        a_dt,
        out_dt,
        s_dt,
        alpha,
        type(alpha),
        by_value,
        block,
        not (meta.data_ptr() & 15),
        not (scalar_buf.data_ptr() & 15),
    )
    entry = _SCALAR_LAUNCH_CACHE.get(key)
    runtime = knobs.runtime
    if (
        entry is not None
        and not runtime.launch_enter_hook.calls
        and not runtime.launch_exit_hook.calls
    ):
        try:
            entry[0](
                grid[0],
                grid[1],
                1,
                _triton_driver.active.get_current_stream(meta.device.index),
                entry[1],
                entry[2],
                None,
                None,
                None,
                meta,
                scalar_buf,
                scalar_value,
                fn,
                nt,
                a_dt,
                out_dt,
                s_dt,
                alpha,
                by_value,
                block,
            )
            _SCALAR_LAUNCH_STATS["replay"] += 1
            return
        except Exception:
            _SCALAR_LAUNCH_CACHE.pop(key, None)
    _SCALAR_LAUNCH_STATS["dispatch"] += 1
    compiled = kernel[grid](
        meta,
        scalar_buf,
        scalar_value,
        fn,
        nt,
        a_dt,
        out_dt,
        s_dt,
        alpha,
        by_value,
        block,
    )
    if compiled is not None:
        _SCALAR_LAUNCH_CACHE[key] = (
            compiled.run,
            compiled.function,
            compiled.packed_metadata,
        )


def _launch_scalar_group(
    srcs: List[torch.Tensor],
    dsts: List[torch.Tensor],
    vals: List[Any],
    fn: Callable,
    in_dt: torch.dtype,
    out_dt: torch.dtype,
    device: Any,
    alpha: float = 1,
    common: Optional[float] = None,
) -> None:
    numels = list(map(_NUMEL, srcs))
    max_numel = max(numels)
    if max_numel == 0:
        return
    nt = len(srcs)
    # The same pair that keys the binary grid memo keys this one: both paths
    # derive their block from ``_pick_block(numels)`` and their second
    # dimension from a ceiling division of the same two numbers, so the entry
    # is interchangeable.
    cached_grid = _GRID_CACHE.get((nt, max_numel))
    if cached_grid is None:
        block = _pick_block(numels)
        cached_grid = (block, (nt, -(-max_numel // block)))
        if len(_GRID_CACHE) >= _GRID_CACHE_LIMIT:
            _GRID_CACHE.pop(next(iter(_GRID_CACHE)))
        _GRID_CACHE[(nt, max_numel)] = cached_grid
    block, grid = cached_grid
    # An in-place call writes back into its inputs, so the destination run
    # repeats the pointer run already read above instead of reading it again.
    ptrs = list(map(_DATA_PTR, srcs))
    values = ptrs + (ptrs if dsts is srcs else list(map(_DATA_PTR, dsts))) + numels
    meta = _device_meta(values, device)
    # The scalars ride in a device array of the *output* dtype, so integer
    # operators keep integer semantics instead of going through a float.
    #
    # The dtype is settled on the host before the transfer rather than by casting
    # the device tensor afterwards: that extra ``.to(dtype)`` costs a kernel and
    # an allocation, and measured 0.32ms per call -- on its own more than
    # PyTorch's entire foreach call.
    s_dtype = out_dt if out_dt.is_floating_point else torch.float32
    # ``.Scalar`` broadcasts one value into every slot, so when the values are
    # all equal there is nothing for an array to index: the value travels as a
    # kernel argument instead.  That drops a host->device copy and a device
    # allocation (~9us) from every such call.  ``common`` is already known for
    # the broadcast forms; only a real list needs the equality scan.
    if common is None:
        common = _common_scalar(vals)
    by_value = common is not None
    if by_value:
        scalar_buf = meta  # unused by the kernel; keeps the pointer slot valid
        scalar_value = common
    else:
        scalar_buf = torch.tensor([float(v) for v in vals], dtype=s_dtype).to(
            device, non_blocking=True
        )
        scalar_value = 0.0
    kernel = (
        foreach_binary_scalar_kernel_p4
        if _pack_ok((in_dt, out_dt), max_numel, values)
        else foreach_binary_scalar_kernel
    )
    _launch_scalar_kernel(
        meta,
        scalar_buf,
        scalar_value,
        grid,
        nt,
        fn,
        tl_dtype(in_dt),
        tl_dtype(out_dt),
        tl_dtype(s_dtype),
        alpha,
        by_value,
        block,
        kernel,
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
    # ``lerp``'s scalar-weight forms hand the same sequence in as the second
    # and third operand; the identity is what lets the homogeneity check and
    # the pointer run skip the repeated third.
    same_bc = tensor1 is tensor2
    self, tensor1, tensor2 = _ternary_operand_lists(self, tensor1, tensor2)
    values, common, vclass = _resolve_scalars(self, scalars)
    _STATS.reset()

    # Homogeneous fast path: one floating dtype, one device, all dense.  The
    # general loop below still handles mixed dtypes (each promotion is folded
    # once and reused), views (``_dense``), and disallowed dtypes (which raise
    # from ``check_dtype_supported`` inside the loop).
    homogeneous, operands = _ternary_homogeneous(
        self, tensor1, tensor2, allowed_dtypes, same_bc
    )
    if homogeneous is not None:
        device = self[0].device
        dsts = self if inplace else list(map(torch.empty_like, self))
        _launch_ternary_group(
            self,
            tensor1,
            tensor2,
            dsts,
            values,
            fn,
            homogeneous,
            homogeneous,
            device,
            common,
            operands,
            same_bc,
        )
        return list(self) if inplace else dsts

    results: List[torch.Tensor] = []
    buckets: Dict[Tuple[Any, torch.dtype, torch.dtype], Tuple[List, ...]] = {}
    writebacks: List[Tuple[torch.Tensor, torch.Tensor]] = []
    prev_key = None
    prev_promotion = None
    prev_out_dtype = None
    for a, b, c, v in zip(self, tensor1, tensor2, values):
        if allowed_dtypes is not None and a.dtype not in allowed_dtypes:
            check_dtype_supported(a.dtype, allowed_dtypes)
        # ``result_type`` pairs two tensors or a tensor and a number, so the
        # three-way promotion is folded through a zero-element probe of the
        # intermediate dtype rather than by chaining dtypes directly.  The
        # integer fallback below also looks at the value's type, so that joins
        # the memo key; the tuple is the same one ``_promote_dtypes`` and
        # ``_promote_with_scalar`` already key on.
        #
        # Three operands of the *same* dtype fold to that dtype whatever their
        # ranks are, so the common case needs neither the probe nor the six
        # ``dim()`` calls that key it.
        if vclass is not None:
            vkey = vclass
        else:
            vkey = type(v)
            if vkey is torch.Tensor:
                vkey = (torch.Tensor, v.dim() == 0)
        a_dtype = a.dtype
        same = a_dtype is b.dtype and a_dtype is c.dtype
        if same:
            promotion = (a_dtype, vkey)
        else:
            promotion = (
                a_dtype,
                a.dim() == 0,
                b.dtype,
                b.dim() == 0,
                c.dtype,
                c.dim() == 0,
                vkey,
            )
        if promotion != prev_promotion:
            out_dtype = a_dtype if same else _promote_dtypes(a, b, c)
            if not out_dtype.is_floating_point and not out_dtype.is_complex:
                out_dtype = _promote_with_scalar(a_dtype, v)
            prev_promotion = promotion
            prev_out_dtype = out_dtype
        else:
            out_dtype = prev_out_dtype
        contiguous_a = a.is_contiguous()
        src_a = a if contiguous_a else _dense(a)
        src_b = b if b.is_contiguous() else _dense(b)
        src_c = c if c.is_contiguous() else _dense(c)
        if inplace:
            if out_dtype != a.dtype:
                raise RuntimeError(
                    f"result type {str(out_dtype).replace('torch.', '').capitalize()}"
                    " can't be cast to the desired output type "
                    f"{str(a.dtype).replace('torch.', '').capitalize()}"
                )
            if not contiguous_a and has_internal_overlap(a):
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
            # The one-argument form skips the keyword/dtype dispatch that the
            # explicit form pays on every one of the sixteen allocations.
            dst = (
                torch.empty_like(src_a)
                if out_dtype is src_a.dtype
                else torch.empty_like(src_a, dtype=out_dtype)
            )
            results.append(dst)
        key = (a.device, src_a.dtype, out_dtype)
        if key != prev_key:
            bucket = buckets.get(key)
            if bucket is None:
                bucket = buckets[key] = ([], [], [], [], [])
            prev_key = key
        bucket[0].append(src_a)
        bucket[1].append(src_b)
        bucket[2].append(src_c)
        bucket[3].append(dst)
        bucket[4].append(v)

    for (device, in_dt, out_dt), (a_l, b_l, c_l, d_l, v_l) in buckets.items():
        _launch_ternary_group(
            a_l, b_l, c_l, d_l, v_l, fn, in_dt, out_dt, device, common
        )

    _finish_inplace(results, writebacks)
    return results


_TERNARY_UPLOAD_TYPES = {
    torch.float32: ("f", torch.float32),
    torch.float64: ("d", torch.float64),
}

# ``array`` has no half-precision typecode, so the value list of an fp16 call
# is packed with ``struct`` instead.  bfloat16 has no ``struct`` code either:
# the float32 blob is rounded through the same cast the generic path used.
_TERNARY_PACK_CODES = {torch.float16: "e"}


def _ternary_upload(values: List[Any], dtype: torch.dtype, device: Any) -> torch.Tensor:
    """Upload one scalar per tensor to ``device`` as a ``len(values)`` buffer.

    ``torch.tensor(values).to(device)`` pays the Python ``Tensor.to`` /
    ``copy_`` registrations on top of the transfer, and ``to`` blocks on the
    stream: measured inside a full call it cost 32us of host time against 14us
    for an explicit non-blocking ``aten::copy_``.  Building the host side with
    ``array`` / ``struct`` + ``frombuffer`` keeps it out of the dispatcher, and
    redispatching ``aten::copy_`` at ``CompositeExplicitAutograd`` skips the
    Python registration.  The host buffer stays pageable and freshly built, so
    the staging copy is ordered before the caller can reuse it.
    """
    if _ATEN_COPY_REDISPATCH is None or _ATEN_COMPOSITE_KEYSET is None:
        return torch.tensor([float(v) for v in values], dtype=dtype).to(
            device, non_blocking=True
        )
    spec = _TERNARY_UPLOAD_TYPES.get(dtype)
    if spec is not None:
        code, host_dtype = spec
        host = torch.frombuffer(
            array.array(code, [float(v) for v in values]), dtype=host_dtype
        )
    elif dtype is torch.bfloat16:
        host = torch.tensor(values, dtype=torch.bfloat16)
        host_dtype = dtype
    else:
        code = _TERNARY_PACK_CODES.get(dtype)
        if code is None:
            return torch.tensor([float(v) for v in values], dtype=dtype).to(
                device, non_blocking=True
            )
        try:
            host = torch.frombuffer(
                struct.pack(f"<{len(values)}{code}", *values), dtype=dtype
            )
        except (struct.error, TypeError):
            # A non-real entry: keep the conversion error the generic path
            # raised rather than ``struct.error``.
            return torch.tensor([float(v) for v in values], dtype=dtype).to(
                device, non_blocking=True
            )
        host_dtype = dtype
    out = torch.empty(len(values), dtype=host_dtype, device=device)
    _ATEN_COPY_REDISPATCH(_ATEN_COMPOSITE_KEYSET, out, host, True)
    return out


# ATen's own TensorList grouping primitive: one C++ pass over the lists, keyed
# by ``(device, dtype)``.  Only the *number* of groups is needed to decide
# whether a call is uniform.  One call over the concatenated 48 tensors
# measured 4.1us against 8.3us for the per-tensor dtype/device/contiguity walk,
# and this helper drops from 8.5us to 7.0us in situ (16 x 256^2, one process).
# Absent on a build without it, in which case the walk below is used.
try:
    _GROUP_TENSORS = torch._C._group_tensors_by_device_and_dtype
except AttributeError:  # pragma: no cover - very old builds
    _GROUP_TENSORS = None

# ``is_contiguous`` is the only per-tensor predicate the grouping call cannot
# answer, so it stays a ``map`` over the unbound descriptor (2.8us over 48
# tensors; the bound-method lookup per tensor is what that avoids).
_IS_CONTIGUOUS = torch.Tensor.is_contiguous
# Same reasoning for the two scalar accessors the launch plans read per tensor:
# the unbound descriptor saves a bound-method lookup on every element.
_NUMEL = torch.Tensor.numel
_DATA_PTR = torch.Tensor.data_ptr


def _ternary_homogeneous(
    a_l: Sequence[torch.Tensor],
    b_l: Sequence[torch.Tensor],
    c_l: Sequence[torch.Tensor],
    allowed_dtypes: Optional[frozenset],
    same_bc: bool = False,
) -> Tuple[Optional[torch.dtype], Optional[List[torch.Tensor]]]:
    """The shared dtype when every operand agrees, else ``None``.

    Three operands of one floating dtype on one device, all dense, need no
    promotion, no densifying and no bucketing: the whole per-tensor
    bookkeeping in :func:`foreach_ternary` is then pure overhead.

    Uniformity of ``(device, dtype)`` is answered in one C++ pass.  The operand
    lists are concatenated into a *single* nested list: the grouping key of a
    tensor in the one list is that tensor's own ``(device, dtype)``, so a
    single group means all of them agree on both.  (With the lists passed
    separately the primitive keys each index off the first list and
    deliberately tolerates cross-list float/double/bfloat16 and CPU-float
    operands for optimizer state tensors, which would not be uniformity here
    and, for a genuinely mixed index, raises instead of returning two groups.)
    Contiguity is the one thing that call does not cover and stays a ``map``
    over the unbound predicate.  Both are exact: any heterogeneity the old
    walk rejected still falls through to the general path.

    ``same_bc`` says the caller was handed the *same* sequence as the second
    and third operand, which is what ``lerp``'s scalar-weight forms do; the
    third copy then repeats a third of both walks for an answer already in
    hand, so only the first pair is examined.  The concatenation is returned
    so the launcher can read the pointer run off it instead of rebuilding it.
    """
    dtype = a_l[0].dtype
    if not dtype.is_floating_point:
        return None, None
    if allowed_dtypes is not None and dtype not in allowed_dtypes:
        return None, None
    if _GROUP_TENSORS is not None:
        operands = a_l + b_l if same_bc else a_l + b_l + c_l
        if len(_GROUP_TENSORS([operands], False)) != 1:
            return None, None
        if all(map(_IS_CONTIGUOUS, operands)):
            return dtype, operands
        return None, None
    dev = a_l[0].get_device()
    for a, b, c in zip(a_l, b_l, c_l):
        if a.dtype is not dtype or b.dtype is not dtype or c.dtype is not dtype:
            return None, None
        if a.get_device() != dev or b.get_device() != dev or c.get_device() != dev:
            return None, None
        if not (a.is_contiguous() and b.is_contiguous() and c.is_contiguous()):
            return None, None
    return dtype, None


_TERNARY_LAUNCH_CACHE: Dict[Tuple[Any, ...], Tuple[Any, Any, Any]] = {}

# ``_pick_block`` searches five block sizes and re-takes the max of the numel
# list the caller already has, yet its answer and the grid it implies depend
# only on the length of the list and its largest element, and both repeat from
# call to call on a fixed workload.  The cache holds the pair; a hit is one
# dict lookup.  The bound exists because a workload whose shapes keep moving
# would otherwise accumulate one entry per shape; insertion order is the
# eviction order, as in the binary executor's grid memo.
_TERNARY_GRID_CACHE: Dict[Tuple[int, int, bool], Tuple[int, Tuple[int, int]]] = {}
_TERNARY_GRID_CACHE_LIMIT = 64

# Below this element count the transposed grid costs more than it saves (the
# crossed threshold was measured on the packed ternary kernel: -0.70us at
# 1024^2 against +0.07us at 512^2, which is 2^18 elements per tensor).
_TERNARY_SWAP_MIN_ELEMS = 1 << 20

# One-entry memo in front of ``_TERNARY_LAUNCH_CACHE``.  A fixed workload
# reuses the same metadata tensor object (``_device_meta`` hands the cached one
# back), the same function and the same constexpr set, so the ten-field key is
# answered by identity instead of being built and hashed: the key holds a
# tensor, a dtype and a function, and that costs ~1.0us a call against ~0.3us
# for the comparisons.
#
# What is memoised is the *launcher entry* ``(run, function, packed_metadata)``,
# never a result or a buffer.  That entry is a pure function of the dict key
# above, and every field of that key is covered here: ``kernel``, ``nt``, ``fn``,
# the three dtypes, ``by_value`` and ``block`` are compared directly; ``grid``
# is compared as part of the same tuple; ``meta.device`` and the alignment bit
# follow from ``meta is last[0]``, since a tensor's device never changes and the
# bit is a property of its ``data_ptr``.  ``numels`` are not keyed separately --
# they reach the kernel as data through ``meta`` and also fix ``block``/``grid``,
# which are compared.  The stream and ``scalar_value`` are read from the call
# and never cached.  So a changed operand list, list length, dtype, block or
# device all miss, and an in-place mutation of the metadata tensor's contents
# cannot matter: the kernel reads that same pointer at run time.
#
# Only the by-value form is memoised: it is the one whose ``scalar_buf`` is the
# metadata tensor itself, whereas the uploaded form hands in a fresh buffer
# every call and would only pay the comparisons.  A miss falls through to the
# dict cache, which repopulates the memo.  Measured +1.6us on the 16-tensor
# ternary oop .ScalarList/.Scalar rows and +1.4us in-place, no cost on the
# uploaded row.
_TERNARY_LAST: List[Any] = [None] * 8


def _launch_ternary_kernel(
    grid: Tuple[int, int],
    nt: int,
    meta: torch.Tensor,
    scalar_buf: torch.Tensor,
    scalar_value: float,
    fn: Callable,
    dt_in: Any,
    dt_out: Any,
    s_dt: Any,
    by_value: bool,
    block: int,
    swap: bool,
    kernel: Callable = foreach_ternary_kernel,
) -> None:
    """``foreach_ternary_kernel[grid](...)`` with a cached launcher.

    Everything except the metadata pointer, the scalar buffer and the scalar
    value is a ``tl.constexpr``, so the key identifies the compiled variant
    exactly and a replay only hands over the runtime arguments.  Same contract
    as ``_launch_executor_kernel``: no device-visible buffer is cached, and a
    miss, an installed launch hook, or a launcher that raises all take
    Triton's normal dispatch.
    """
    runtime = knobs.runtime
    last = _TERNARY_LAST
    key = None
    if (
        by_value
        and last[0] is meta
        and last[1] is scalar_buf
        and last[2] is fn
        and last[3] is kernel
        and last[4] == (grid, nt, dt_in, dt_out, s_dt, by_value, block, swap)
        and not runtime.launch_enter_hook.calls
        and not runtime.launch_exit_hook.calls
    ):
        entry = (last[5], last[6], last[7])
    else:
        key = (
            kernel,
            meta.device,
            nt,
            fn,
            dt_in,
            dt_out,
            s_dt,
            by_value,
            block,
            swap,
            not (meta.data_ptr() & 15),
        )
        entry = _TERNARY_LAUNCH_CACHE.get(key)
        if entry is not None and (
            runtime.launch_enter_hook.calls or runtime.launch_exit_hook.calls
        ):
            entry = None
    if entry is not None:
        try:
            entry[0](
                grid[0],
                grid[1],
                1,
                _triton_driver.active.get_current_stream(meta.device.index),
                entry[1],
                entry[2],
                None,
                None,
                None,
                meta,
                scalar_buf,
                scalar_value,
                fn,
                nt,
                dt_in,
                dt_out,
                s_dt,
                by_value,
                block,
                swap,
            )
            if key is not None and by_value:
                last[0] = meta
                last[1] = scalar_buf
                last[2] = fn
                last[3] = kernel
                last[4] = (grid, nt, dt_in, dt_out, s_dt, by_value, block, swap)
                last[5] = entry[0]
                last[6] = entry[1]
                last[7] = entry[2]
            return
        except Exception:
            last[0] = None
            if key is not None:
                _TERNARY_LAUNCH_CACHE.pop(key, None)
    compiled = kernel[grid](
        meta,
        scalar_buf,
        scalar_value,
        fn,
        nt,
        dt_in,
        dt_out,
        s_dt,
        by_value,
        block,
        swap,
    )
    if compiled is not None:
        _TERNARY_LAUNCH_CACHE[key] = (
            compiled.run,
            compiled.function,
            compiled.packed_metadata,
        )


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
    common: Optional[float] = None,
    operands: Optional[List[torch.Tensor]] = None,
    same_bc: bool = False,
) -> None:
    numels = list(map(torch.Tensor.numel, a_l))
    max_numel = max(numels)
    if max_numel == 0:
        return
    nt = len(a_l)
    # One tuple display with star-unpacking.  Four ``list``+``map`` calls and
    # the concatenations between them measured 4.28us over four sixteen-tensor
    # lists against 3.77us for this form, for the same pointer run.  A tuple
    # also spares ``_device_meta`` its own ``tuple(values)`` conversion, and
    # when the caller's homogeneity check already concatenated the operands
    # (``operands``) the first three runs are read off it rather than rebuilt.
    ptr = torch.Tensor.data_ptr
    if operands is None:
        prun = (*map(ptr, a_l), *map(ptr, b_l), *map(ptr, c_l))
    elif same_bc:
        prun = (*map(ptr, operands), *map(ptr, b_l))
    else:
        prun = (*map(ptr, operands),)
    # An in-place call writes into its inputs, so the destination run repeats
    # the first pointer run instead of reading it a second time.
    values = (
        (*prun, *prun[:nt], *numels) if d_l is a_l else (*prun, *map(ptr, d_l), *numels)
    )
    kernel = (
        foreach_ternary_kernel_p4
        if _pack_ok((in_dt, out_dt), max_numel, values)
        else foreach_ternary_kernel
    )
    # Transposed grid for the packed kernel at the large end; see the kernel's
    # docstring for the measurement and for why the narrow path keeps the
    # tensor-major order.
    swap = kernel is foreach_ternary_kernel_p4 and max_numel >= _TERNARY_SWAP_MIN_ELEMS
    grid_key = (nt, max_numel, swap)
    cached_grid = _TERNARY_GRID_CACHE.get(grid_key)
    if cached_grid is None:
        block = _pick_block(numels)
        # ``triton.cdiv`` is a decorated helper and is far from free at this
        # size; plain ints give the same ceiling division.
        tiles = -(-max_numel // block)
        cached_grid = (block, (tiles, nt) if swap else (nt, tiles))
        if len(_TERNARY_GRID_CACHE) >= _TERNARY_GRID_CACHE_LIMIT:
            _TERNARY_GRID_CACHE.pop(next(iter(_TERNARY_GRID_CACHE)))
        _TERNARY_GRID_CACHE[grid_key] = cached_grid
    block, grid = cached_grid
    meta = _device_meta(values, device)
    s_dtype = out_dt if out_dt.is_floating_point else torch.float32
    # Same treatment as the binary scalar path: the ``.Scalar`` form supplies one
    # value for the whole list, so it travels as a kernel argument instead of a
    # device buffer.  The ``.ScalarList`` form still needs the array.
    if common is None:
        common = _common_scalar(v_l)
    by_value = common is not None
    if by_value:
        scalar_buf = meta  # unused by the kernel; keeps the pointer slot valid
        scalar_value = common
    else:
        scalar_buf = _ternary_upload(v_l, s_dtype, device)
        scalar_value = 0.0
    _launch_ternary_kernel(
        grid,
        nt,
        meta,
        scalar_buf,
        scalar_value,
        fn,
        tl_dtype(in_dt),
        tl_dtype(out_dt),
        tl_dtype(s_dtype),
        by_value,
        block,
        swap,
        kernel,
    )
    _STATS.executor_launches += 1
    _STATS.groups += 1
