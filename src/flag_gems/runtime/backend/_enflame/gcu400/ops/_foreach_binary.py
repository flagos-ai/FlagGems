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

"""Registration table for the multi-input ``aten::_foreach_*`` family.

Twelve operators with four overload shapes each share the multi-input executor
in :mod:`flag_gems.utils.foreach`.  As in :mod:`flag_gems.ops._foreach_unary`,
every operator is a table row rather than a copy of the wrapper; what differs
between rows is only the element-wise math and the accepted dtypes.

Two facts about this family, measured against the pinned PyTorch build rather
than inferred from the names:

* ``_foreach_add`` and friends have **no** ``default`` overload -- only
  ``Scalar`` / ``List`` / ``ScalarList`` / ``Tensor``.  Registering the bare
  name would never dispatch and would raise no error (CLAUDE.md 7.1 class B),
  so each overload is registered under its full name.
* The ``.Tensor`` overload broadcasts *one* tensor across the whole list, while
  ``.List`` pairs the two lists position by position.  They are different
  operations and get different wrappers.

The ``.out`` overloads are deliberately absent: for this family they decompose
to the functional form, so registering them would add keys that are never
consulted.  Verified by calling ``torch.ops.aten._foreach_sqrt.out`` and
observing the functional kernel's debug record.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
import triton
import triton.language as tl
from triton import knobs
from triton.runtime import driver as _triton_driver

from flag_gems.utils import libentry, tl_extra_shim

from ..utils.foreach import (
    _FAST_MATH,
    _STATS,
    _device_meta,
    _promote_dtypes,
    _promote_with_scalar,
    _resolve_scalars,
    foreach_binary_list,
    foreach_binary_scalar,
    foreach_ternary,
    tl_dtype,
)
from ._foreach_addcdiv_flat import FALLBACK as _ADDC_DIV_FLAT_FALLBACK
from ._foreach_addcdiv_flat import run_addcdiv_tensor
from ._foreach_flat import FALLBACK as _COPY_FLAT_FALLBACK
from ._foreach_flat import _empty_output_list, run_copy
from ._foreach_lerp_flat import FALLBACK as _LERP_FLAT_FALLBACK
from ._foreach_lerp_flat import run_lerp_scalar_list

# The same device function ``ops/pow.py`` uses; ``tl.math.pow`` does not exist
# in this Triton version.
_pow = tl_extra_shim.pow

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# element-wise math
# ---------------------------------------------------------------------------
#
# These are written here rather than borrowed from the single-tensor operators
# because those carry extra machinery the executor cannot use: ``add_func``
# takes an ``alpha`` argument bound by ``pointwise_dynamic``, and several of
# them cast to ``float32`` internally, which would silently change the result
# dtype of an integer foreach call.  Each function below is the bare two- or
# three-argument form the executor's ``tl.constexpr`` slot expects.


@triton.jit
def _add_fn(a, b):
    # The paired kernel hands over the raw int1 operand, and arithmetic on int1
    # in this Triton backend behaves like ``and`` (``True + True`` is False).
    # ATen defines bool addition as logical or, so the operand is combined
    # explicitly instead.
    if a.dtype == tl.int1:
        return a | b
    return a + b


@triton.jit
def _sub_fn(a, b):
    return a - b


@triton.jit
def _mul_fn(a, b):
    # Multiplication on int1 already coincides with logical and, but the operand
    # is combined explicitly so the bool path does not depend on that.
    if a.dtype == tl.int1:
        return a & b
    return a * b


@triton.jit
def _div_fn(a, b):
    # 16-bit division goes through the non-IEEE path: GCU400 has no native
    # fp16 divide, so ``a / b`` is a full-precision fp32 divide per lane and
    # dominated the op (16 paired tensors at 1024^2: 78us against 50us for the
    # multiply).  ``ieee_rounding=False`` lets the compiler use the reciprocal
    # approximation plus one Newton step, which measured 58us (+-2 ulp, far
    # inside the 1e-2 tolerance of a 16-bit divide).  fp32 keeps the exact
    # divide: there the approximation buys nothing (104.6us against 104.3us)
    # while the fp32 tolerance is ~11 ulp.
    if a.dtype == tl.float16 or a.dtype == tl.bfloat16:
        # ``fdiv`` answers in the operand width while ``/`` widens to fp32; both
        # arms of the ``if`` must agree, and fp32 is the wider of the two.  The
        # value is rounded to the operand width by ``fdiv`` itself and handed on
        # exactly, so widening afterwards costs a cast and changes no bit.
        return tl.fdiv(a, b, ieee_rounding=False).to(tl.float32)
    return a / b


@triton.jit
def _maximum_fn(a, b):
    return tl.maximum(a, b)


@triton.jit
def _minimum_fn(a, b):
    return tl.minimum(a, b)


@triton.jit
def _pow_fn(a, b):
    # The shim's ``pow`` needs floating point operands; integral foreach pow is
    # promoted by the executor before the call, so the cast here only affects
    # the intermediate and not the stored dtype.  A bool base takes the same
    # path: ATen computes ``bool ** float`` in floating point (``False ** -1``
    # is ``inf``), which the plain power already gives once the operand is
    # promoted.
    af = a.to(tl.float32)
    bf = b.to(tl.float32)
    if _FAST_MATH and (a.dtype == tl.float16 or a.dtype == tl.bfloat16):
        # 16-bit output: ``powf`` is about half the kernel time, and the
        # fixups a negative base needs are exactly what a *block-wide* guard
        # would avoid -- but on data with mixed signs that guard never fires
        # (one negative base rejects the whole block), so the full ``powf``
        # runs on every element.  Pay for the closed form per element
        # instead, cheap enough that the guard is not worth keeping:
        #
        #   |a| ** b = exp2(fast_log2f(|a|) * b)
        #
        # and supply the sign separately: a negative base is ``(-1)**b`` for
        # an integral ``b`` and NaN otherwise.  The 16-bit output tolerance
        # (fp16 rtol 1e-3, bf16 0.016) absorbs the ~2**-22 relative error of
        # ``lg2.approx``; fp32/fp64 keep the exact path below bit for bit, so
        # their ltol 1.3e-6 result is unchanged.
        s = tl.abs(af)
        if a.dtype == tl.bfloat16:
            # A bf16 subnormal *is* an fp32 subnormal, and ``lg2.approx``
            # flushes those to zero (``pow(2**-133, -0.5)`` would come back
            # inf).  Scale them into the normal range first; the scale is a
            # power of two, so the multiply is exact and the logarithm only
            # shifts by a constant.  fp16 subnormals are normal in fp32 and
            # skip this.
            sub = s < 1.1754943508222875e-38
            lg = tl_extra_shim.fast_log2f(tl.where(sub, s * 16777216.0, s)) - tl.where(
                sub, 24.0, 0.0
            )
        else:
            lg = tl_extra_shim.fast_log2f(s)
        # ``|a| == 1`` is pinned to 1: the magnitude is 1 for every exponent,
        # while ``exp2(0 * inf)`` is NaN and ``pow(±1, ±inf)`` is 1.
        r = tl.where(s == 1.0, 1.0, tl.exp2(lg * bf))
        # Negative base: ``(-1)**b`` when ``b`` is integral, NaN otherwise.
        # Parity is tested through ``b/2`` because ``bi % 2`` is exact only for
        # ``|b| < 2**24``; ``b == 2**24`` is even and ``2**24 - 1`` is odd, and
        # ``floor`` keeps both halves of the test on representable values.
        bi = tl.floor(bf)
        odd = (bi * 0.5) != tl.floor(bi * 0.5)
        # ``signbit`` rather than ``af < 0`` so that ``-0.0`` counts as
        # negative: ``pow(-0.0, -1)`` is ``-inf``, not ``+inf``.
        neg = tl_extra_shim.signbit(af) != 0
        r = tl.where(
            neg,
            tl.where(
                bf == bi,
                tl.where(odd, -r, r),
                # A zero or infinite magnitude is never NaN however fractional
                # ``b`` is: ATen gives ``pow(-0.0, 0.5) == -0.0`` and
                # ``pow(-inf, 0.5) == inf``, and ``exp2`` of ``-inf * b`` /
                # ``inf * b`` already produces the right 0 or inf.
                tl.where((s == 0.0) | (s == float("inf")), r, float("nan")),
            ),
            r,
        )
        # ``b == 0`` is pinned to 1 because ``exp2(-inf * 0)`` is NaN while
        # ``pow(0, 0)`` and ``pow(nan, 0)`` are 1.
        r = tl.where(bf == 0.0, 1.0, r)
    elif a.dtype.is_floating():
        # ``powf`` dominates the kernel time and is itself ``exp2(y * log2 x)``
        # plus the fixups a negative base needs.  On a block of normal, positive
        # bases those fixups are dead code, so evaluate the cheap form instead.
        # The guard is a block-wide ``tl.min`` (one reduction per program); a
        # per-element ``tl.where`` executes both branches and measured 35-60%
        # slower than the plain power.  The bound is the smallest normal fp32:
        # a subnormal base stays > 0 but ``log2`` flushes it to zero, and
        # ``pow(2**-149, -0.5)`` came back as ``2.7e22`` instead of ``1.3e19``
        # (rel 2.0e3) when the fast path was allowed to see it.
        # ``a == 1`` is pinned to 1 because ``exp2(log2(1) * inf)`` is NaN while
        # ``pow(1, inf)`` is 1; ``b == 0`` is pinned for the same reason on an
        # infinite base (``exp2(inf * 0)`` is NaN, ``pow(inf, 0)`` is 1).
        if tl.min(af) >= 1.1754943508222875e-38:
            r = tl.where((af == 1.0) | (bf == 0.0), 1.0, tl.exp2(tl.log2(af) * bf))
        else:
            r = _pow(af, bf)
    else:
        # Integral results use ``powf`` unchanged: the last-bit difference
        # changes the value once truncated to an integer.
        r = _pow(af, bf)
    return r


@triton.jit
def _rpow_fn(a, b):
    # ``pow.ScalarAndTensor``: the scalar is the base, so the power is reversed
    # relative to ``_pow_fn`` (``a`` is the exponent, ``b`` the base).  Kept at
    # module scope like every other kernel: defining it inside the wrapper
    # rebuilt the ``JITFunction`` and its caches on every call.
    af = a.to(tl.float32)
    bf = b.to(tl.float32)
    # Same positive-base fast path as ``_pow_fn``.  The base is the scalar
    # here, so the guard is the scalar condition itself -- ``tl.min`` rejects
    # a 0-d operand in this Triton.  The smallest-normal bound and the
    # ``exponent == 0`` pin carry over unchanged.
    if b.dtype.is_floating():
        if bf >= 1.1754943508222875e-38:
            r = tl.where((bf == 1.0) | (af == 0.0), 1.0, tl.exp2(tl.log2(bf) * af))
        else:
            r = _pow(bf, af)
    else:
        r = _pow(bf, af)
    return r


@triton.jit
def _copy_fn(a, b):
    # ``_foreach_copy`` overwrites self with src; the first operand is read and
    # discarded so the executor's paired addressing still applies.
    return b


@triton.jit
def _lerp_tensor_weight_fn(a, b, w, _unused):
    # ``lerp.List``: the weight is the third tensor operand.
    return a + w * (b - a)


@triton.jit
def _lerp_scalar_weight_fn(a, b, _c, w):
    # ``lerp.Scalar`` / ``.ScalarList``: the weight arrives as the scalar; the
    # third tensor slot is a copy of ``b`` and is ignored.
    return a + w * (b - a)


@triton.jit
def _addcmul_fn(a, b, c, v):
    return a + v * b * c


@triton.jit
def _addcdiv_fn(a, b, c, v):
    return a + v * (b / c)


# ---------------------------------------------------------------------------
# GCU400 flat launches for the grid.y hardware cap
# ---------------------------------------------------------------------------
#
# The common paired/ternary executors launch on the 2D grid
# ``(nt, cdiv(max_numel, BLOCK))``, and GCU400 caps ``grid.y`` at 255: a
# uniform float group whose largest tensor needs 256+ tiles -- sixteen 512^2
# fp32 tensors at ``BLOCK=1024`` is exactly 256 -- raises ``OutOfResources``
# at launch and the operator aborts.  The scalar overloads already run on the
# grid-stride kernel in ``utils/foreach.py`` (axis 1 capped at
# ``MAX_GRID_DIM``); the two kernels below extend that scheme to the paired
# and ternary bodies, and the ``_flat_*`` hosts replay the common executors'
# uniform-list semantics in front of them.
#
# A flat path is taken *only* when the common launch would exceed the
# hardware limits, so every shape that fits keeps the common executors and
# their tuned launch caches untouched.  Anything the flat host cannot serve
# exactly -- mixed dtypes, non-contiguous or overlapping views, integral or
# bool operands, heterogeneous output dtypes -- also answers
# ``_FLAT_FALLBACK`` and keeps the common error wording.

_FLAT_FALLBACK = object()

_MAX_GRID_X = 65535

_FLAT_FLOAT = (torch.float16, torch.float32, torch.bfloat16)

# --- TUNING PARAMS START ---
# Set by optimization iterations from family precedent and this module's sweeps; compile/launch
# time config only, no semantic change. 2026-11-18 resweep (grid became NT*parts after the
# missing-tile fix): ternary at 1024^2 in fp32/fp16/bf16 is best with parts=3 (grid=48),
# BLOCK=8192, nw=4, st=1 (fp32 116.2us / 16-bit 62.0us, at the 4-stream probe ceiling).
# 2026-11-19 binary round (maximum_List 11-18 precedent): FIXED_GRID 24->48 (a 16-tensor list
# goes from parts 2 to 3; paired lists always prefer 48 CTAs, while parts>3 only adds CTA-issue
# serialization); the mask-free EXACT arm tiers BLOCK by dtype -- 8192 for fp32, 16384 for
# 16-bit ((dtype, tile width) coupling: EXACT@8192 gives -17.4/-9.6/-4.7% on the three fp32
# tiers, while bf16 regresses at 8192 tiles and turns positive at 16384); ragged lists (any
# numel not divisible by the chosen BLOCK) keep 16384 + mask, bit-identical to history.
_BINARY_FIXED_GRID = 48
_BINARY_BLOCK = 16384
_BINARY_BLOCK_FP32 = 8192
_BINARY_EXACT = True
_TERNARY_FIXED_GRID = 48
_TERNARY_BLOCK = 8192
_NUM_WARPS = 4
_NUM_STAGES = 1
# Promote 16-bit operands to fp32 for computation, then store back to the original dtype (aligned with ATen opmath
# semantics).
FP32_PROMOTE = True
_SCALAR_CACHE_LIMIT = 64
# --- TUNING PARAMS END ---

# Second-operand addressing modes for the binary kernel.
_B_PAIRED = 0  # element-wise from a paired second list
_B_VALUE = 1  # one Python number for every tensor (constexpr slot)
_B_PER_TENSOR = 2  # one number per tensor, from a cached device array
_B_BROADCAST = 3  # single-element tensor operand, read once per program

# Read-only device buffers cached by content: the kernels re-read the tensor
# data every call, so reusing a scalar table only skips its H2D upload.
_SCALAR_CACHE: Dict[Any, torch.Tensor] = {}

# Replay cache for ``_foreach_binary_flat_kernel``: fully specialized launch ->
# ``(CompiledKernel.run, kernel.function, packed_metadata)``.
_BINARY_FLAT_LAUNCH_CACHE: Dict[Any, Any] = {}


def _cached_scalar_buffer(values, dtype, device):
    """A device scalar table (read-only), cached by ``(device, values, dtype)``."""
    key = (str(device), tuple(values), dtype)
    cached = _SCALAR_CACHE.get(key)
    if cached is not None:
        return cached
    if len(_SCALAR_CACHE) >= _SCALAR_CACHE_LIMIT:
        _SCALAR_CACHE.clear()
    cached = torch.tensor(values, dtype=dtype, device=device)
    _SCALAR_CACHE[key] = cached
    return cached


def _binary_launch_shape(dtype, numels):
    """``(BLOCK, EXACT)`` for the flat binary kernel, keyed on family tuning.

    The per-tile bounds mask is the largest kernel-side lever on the paired
    list kernel (the ``maximum_List`` 11-18 judgment): fp32 takes the maskless
    ``EXACT`` body at an 8192 tile while 16-bit wants the wider 16384 tile
    (``EXACT`` at 8192 regresses bf16), and any ragged count keeps the
    historical masked 16384 body bit for bit.
    """
    if _BINARY_EXACT:
        block = _BINARY_BLOCK_FP32 if dtype is torch.float32 else _BINARY_BLOCK
        if all(n % block == 0 for n in numels):
            return block, True
    return _BINARY_BLOCK, False


def _flat_grid(num_tensors, max_numel, fixed_grid, block):
    """One launch axis laid out as ``(tensor, part)`` with full tile coverage.

    The kernel decomposes ``program_id`` into ``(tensor_id = pid % NT,
    part_id = pid // NT)`` and strides tiles by ``cdiv(num_programs, NT)``,
    so a tile of tensor ``t`` is visited by worker ``(t, part)`` exactly when
    ``tile % num_parts == part``.  Every tile is therefore reached only when
    the grid is an exact multiple of ``NT``: a plain ``min(fixed_grid,
    NT * tiles)`` cap left the tiles of the trailing ``grid % NT`` tensors
    unvisited whenever ``NT * tiles`` exceeded the cap without dividing it
    (sixteen tensors at 1024^2 with a 24-program cap dropped every odd tile
    of tensors 8..15, half of each tensor's elements).  ``parts`` is derived
    from the cap by ceiling division and clamped to the tile count, so
    ``grid = NT * parts`` covers every tile of every tensor by construction
    while staying at or just above the cap.
    """
    blocks_per_tensor = -(-max_numel // block)
    parts = min(-(-fixed_grid // num_tensors), blocks_per_tensor)
    return (num_tensors * parts,)


# ``.Tensor`` overloads hand in the per-tensor values as a small CPU tensor and
# ``_resolve_scalars`` unpacks it with ``tolist()`` on every call (~5us of the
# ~30us host path).  The unpacked list is memoised on the tensor's identity and
# version counter: ``_version`` bumps on any in-place mutation, and the cache
# keeps a strong reference so the id cannot be recycled while cached.
_SCALAR_TENSOR_MEMO: Dict[Any, Any] = {}


def _resolve_scalars_memo(tensors, scalars):
    """``_resolve_scalars`` with the ``.Tensor`` unpack memoised."""
    if type(scalars) is not torch.Tensor or scalars.numel() < 2:
        return _resolve_scalars(tensors, scalars)
    key = (id(scalars), scalars._version, scalars.numel(), len(tensors))
    hit = _SCALAR_TENSOR_MEMO.get(key)
    if hit is not None and hit[0] is scalars:
        return hit[1], None, None
    values, common, vclass = _resolve_scalars(tensors, scalars)
    if len(_SCALAR_TENSOR_MEMO) > 8:
        _SCALAR_TENSOR_MEMO.clear()
    _SCALAR_TENSOR_MEMO[key] = (scalars, values)
    return values, common, vclass


@libentry()
@triton.jit
def _foreach_binary_flat_kernel(
    meta_ptr,
    scalar_ptr,
    fn: tl.constexpr,
    NT: tl.constexpr,
    A_DT: tl.constexpr,
    B_DT: tl.constexpr,
    OUT_DT: tl.constexpr,
    ALPHA: tl.constexpr,
    B_MODE: tl.constexpr,
    B_SCALAR: tl.constexpr,
    BLOCK: tl.constexpr,
    EXACT: tl.constexpr,
):
    """Paired-list math on one linearized launch.

    ``meta_ptr`` keeps the common layout -- first-input, second-input and
    output pointers plus element counts, ``NT`` each, int64 -- while the grid
    is a single axis laid out as ``(tensor, part)``: each program takes the
    tiles ``part, part + num_parts, ...`` of its tensor.

    ``B_MODE`` selects how the second operand is addressed; the scalar modes
    keep the executor's arithmetic order (scale, cast to the output dtype,
    then ``fn``).

    ``EXACT`` drops the per-tile bounds mask when every element count divides
    ``BLOCK``; ragged tails keep the masked body.  The two arms each hold a
    complete load-compute-store body because a value defined inside one
    ``if`` and consumed inside another fails to compile (scf.if boundary).
    """
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)
    tensor_id = program_id % NT
    part_id = program_id // NT
    num_parts = tl.cdiv(num_programs, NT)

    a_ptr = tl.load(meta_ptr + tensor_id).to(tl.pointer_type(A_DT))
    b_ptr = tl.load(meta_ptr + NT + tensor_id).to(tl.pointer_type(B_DT))
    out_ptr = tl.load(meta_ptr + 2 * NT + tensor_id).to(tl.pointer_type(OUT_DT))
    n_elements = tl.load(meta_ptr + 3 * NT + tensor_id)
    if B_MODE == 2:  # _B_PER_TENSOR: one scalar per tensor
        b_scalar = tl.load(scalar_ptr + tensor_id)
    else:
        b_scalar = tl.full((), B_SCALAR, OUT_DT)

    arange = tl.arange(0, BLOCK)
    # The tile count stays in i32 to match ``part_id`` and keep the address
    # arithmetic narrow, as in the common executors.
    num_tiles = tl.cdiv(n_elements, BLOCK).to(tl.int32)
    for block_id in tl.range(part_id, num_tiles, num_parts):
        idx = block_id * BLOCK + arange
        if EXACT:
            a = tl.load(a_ptr + idx)
            if B_MODE == 0:  # _B_PAIRED: element-wise from the second list
                b = tl.load(b_ptr + idx)
            elif B_MODE == 3:  # _B_BROADCAST: single-element tensor operand
                b = tl.load(b_ptr)
            else:
                b = b_scalar
            if ALPHA != 1:
                b = (b * ALPHA).to(OUT_DT)
            else:
                b = b.to(OUT_DT)
            tl.store(out_ptr + idx, fn(a.to(OUT_DT), b).to(OUT_DT))
        else:
            mask = idx < n_elements
            a = tl.load(a_ptr + idx, mask=mask, other=0)
            if B_MODE == 0:  # _B_PAIRED: element-wise from the second list
                b = tl.load(b_ptr + idx, mask=mask, other=0)
            elif B_MODE == 3:  # _B_BROADCAST: single-element tensor operand
                b = tl.load(b_ptr)
            else:
                b = b_scalar
            if ALPHA != 1:
                b = (b * ALPHA).to(OUT_DT)
            else:
                b = b.to(OUT_DT)
            tl.store(out_ptr + idx, fn(a.to(OUT_DT), b).to(OUT_DT), mask=mask)


@libentry()
@triton.jit
def _foreach_ternary_flat_kernel(
    meta_ptr,
    scalar_ptr,
    fn: tl.constexpr,
    NT: tl.constexpr,
    A_DT: tl.constexpr,
    OUT_DT: tl.constexpr,
    S_DT: tl.constexpr,
    FP32_PROMOTE: tl.constexpr,
    EXACT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Ternary-list math on one linearized launch.

    ``meta_ptr`` layout matches the common ``foreach_ternary_kernel``: self,
    tensor1 and tensor2 pointers, output pointers and element counts, ``NT``
    each, int64; ``scalar_ptr`` holds one value per tensor.  The scalars ride
    in a cached device array of the tensor dtype, keeping the signature
    f64-free exactly like the GCU400 scalar executor.

    ``FP32_PROMOTE`` lifts 16-bit operands into fp32 for the math and stores
    back in the output dtype, matching ATen's ``opmath_t`` accumulation for
    half/bfloat16 element-wise ternary ops.

    ``EXACT`` drops the per-tile bounds mask when every element count divides
    ``BLOCK``; ragged tails keep the masked body.
    """
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)
    tensor_id = program_id % NT
    part_id = program_id // NT
    num_parts = tl.cdiv(num_programs, NT)

    a_ptr = tl.load(meta_ptr + tensor_id).to(tl.pointer_type(A_DT))
    b_ptr = tl.load(meta_ptr + NT + tensor_id).to(tl.pointer_type(A_DT))
    c_ptr = tl.load(meta_ptr + 2 * NT + tensor_id).to(tl.pointer_type(A_DT))
    out_ptr = tl.load(meta_ptr + 3 * NT + tensor_id).to(tl.pointer_type(OUT_DT))
    n_elements = tl.load(meta_ptr + 4 * NT + tensor_id)
    s = tl.load(scalar_ptr + tensor_id)

    arange = tl.arange(0, BLOCK)
    num_tiles = tl.cdiv(n_elements, BLOCK).to(tl.int32)
    if FP32_PROMOTE:
        sv = s.to(tl.float32)
    else:
        sv = s.to(OUT_DT)
    for block_id in tl.range(part_id, num_tiles, num_parts):
        idx = block_id * BLOCK + arange
        if EXACT:
            a = tl.load(a_ptr + idx)
            b = tl.load(b_ptr + idx)
            c = tl.load(c_ptr + idx)
        else:
            mask = idx < n_elements
            a = tl.load(a_ptr + idx, mask=mask, other=0)
            b = tl.load(b_ptr + idx, mask=mask, other=0)
            c = tl.load(c_ptr + idx, mask=mask, other=0)
        if FP32_PROMOTE:
            a = a.to(tl.float32)
            b = b.to(tl.float32)
            c = c.to(tl.float32)
        if EXACT:
            tl.store(out_ptr + idx, fn(a, b, c, sv).to(OUT_DT))
        else:
            tl.store(out_ptr + idx, fn(a, b, c, sv).to(OUT_DT), mask=mask)


# The flat ternary launch replay, mirroring the common executor's
# ``_launch_ternary_kernel``.  Every argument past the two buffers is a
# ``tl.constexpr``, so ``(kernel, nt, fn, dtype, promote, exact, block, grid,
# meta alignment)`` identifies the compiled variant exactly and a replay only
# hands over the runtime buffers.  No device-visible data is cached: ``meta``
# and ``scalar_buf`` are read-only and content-addressed upstream, and a miss,
# an installed launch hook (topsprof et al.) or a replay that raises all take
# Triton's normal dispatch.  The one-hit memo sits in front of the dict the
# same way as ``_TERNARY_LAST`` upstream.
_TERNARY_FLAT_LAUNCH_CACHE: Dict[Any, Any] = {}
_TERNARY_FLAT_LAST: List[Any] = [None] * 8


def _launch_ternary_flat(
    grid,
    nt: int,
    meta: torch.Tensor,
    scalar_buf: torch.Tensor,
    fn: Callable,
    dt: Any,
    promote: bool,
    exact: bool,
    block: int,
) -> None:
    """``_foreach_ternary_flat_kernel[grid](...)`` with a cached launcher."""
    runtime = knobs.runtime
    last = _TERNARY_FLAT_LAST
    if (
        last[0] is meta
        and last[1] is scalar_buf
        and last[2] is fn
        and last[4] == (grid, nt, dt, promote, exact, block)
        and not runtime.launch_enter_hook.calls
        and not runtime.launch_exit_hook.calls
    ):
        try:
            last[5](
                grid[0],
                1,
                1,
                _triton_driver.active.get_current_stream(meta.device.index),
                last[6],
                last[7],
                None,
                None,
                None,
                meta,
                scalar_buf,
                fn,
                nt,
                dt,
                dt,
                dt,
                promote,
                exact,
                block,
            )
            return
        except Exception:
            last[0] = None
    key = (
        _foreach_ternary_flat_kernel,
        meta.device,
        nt,
        fn,
        dt,
        promote,
        exact,
        block,
        grid,
        not (meta.data_ptr() & 15),
    )
    entry = _TERNARY_FLAT_LAUNCH_CACHE.get(key)
    if (
        entry is not None
        and not runtime.launch_enter_hook.calls
        and not runtime.launch_exit_hook.calls
    ):
        try:
            entry[0](
                grid[0],
                1,
                1,
                _triton_driver.active.get_current_stream(meta.device.index),
                entry[1],
                entry[2],
                None,
                None,
                None,
                meta,
                scalar_buf,
                fn,
                nt,
                dt,
                dt,
                dt,
                promote,
                exact,
                block,
            )
            last[0] = meta
            last[1] = scalar_buf
            last[2] = fn
            last[4] = (grid, nt, dt, promote, exact, block)
            last[5], last[6], last[7] = entry
            return
        except Exception:
            _TERNARY_FLAT_LAUNCH_CACHE.pop(key, None)
            last[0] = None
    compiled = _foreach_ternary_flat_kernel[grid](
        meta,
        scalar_buf,
        fn,
        nt,
        dt,
        dt,
        dt,
        promote,
        exact,
        block,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    if compiled is not None:
        kernel = compiled[0] if isinstance(compiled, tuple) else compiled
        _TERNARY_FLAT_LAUNCH_CACHE[key] = (
            kernel.run,
            kernel.function,
            kernel.packed_metadata,
        )


def _launch_binary_flat(
    grid, meta, scalar_arg, fn, nt, a_dt, b_dt, out_dt, alpha, b_mode, b_value, block, exact
):
    """``_foreach_binary_flat_kernel[grid](...)`` with a cached direct launcher.

    The Triton ``kernel[grid]`` entry re-binds the signature, re-derives the
    specialization key and re-checks the launch hooks on every call, and that
    host path costs more than the kernel itself on GCU400 (25.6us against
    4.6us for a replayed launch).  Since the whole operand list already rides
    one launch, that overhead was the dominant term of the ``.List`` /
    ``.Tensor`` overloads.  The replay is keyed by everything selecting a
    compiled variant plus the metadata alignment Triton specializes on; a
    replay that raises, or any installed launch hook (topsprof et al.), falls
    back to the normal dispatch so instrumentation still sees every launch.
    """
    runtime = knobs.runtime
    key = (
        grid,
        nt,
        fn,
        a_dt,
        b_dt,
        out_dt,
        alpha,
        b_mode,
        b_value,
        block,
        exact,
        meta.device,
        not (meta.data_ptr() & 15),
        not (scalar_arg.data_ptr() & 15),
    )
    entry = _BINARY_FLAT_LAUNCH_CACHE.get(key)
    if (
        entry is not None
        and not runtime.launch_enter_hook.calls
        and not runtime.launch_exit_hook.calls
    ):
        try:
            entry[0](
                grid[0],
                1,
                1,
                _triton_driver.active.get_current_stream(meta.device.index),
                entry[1],
                entry[2],
                None,
                None,
                None,
                meta,
                scalar_arg,
                fn,
                nt,
                a_dt,
                b_dt,
                out_dt,
                alpha,
                b_mode,
                b_value,
                block,
                exact,
            )
            return
        except Exception:
            _BINARY_FLAT_LAUNCH_CACHE.pop(key, None)
    compiled = _foreach_binary_flat_kernel[grid](
        meta,
        scalar_arg,
        fn,
        nt,
        a_dt,
        b_dt,
        out_dt,
        alpha,
        b_mode,
        b_value,
        block,
        exact,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    if compiled is not None:
        kernel = compiled[0] if isinstance(compiled, tuple) else compiled
        _BINARY_FLAT_LAUNCH_CACHE[key] = (
            kernel.run,
            kernel.function,
            kernel.packed_metadata,
        )


def _flat_binary_list(
    self,
    other,
    fn,
    *,
    inplace=False,
    alpha=1,
    allowed_dtypes=None,
    out_dtype_fn=None,
    out_dtypes=None,
    copy_semantics=False,
    scalar_operand=False,
):
    """Uniform float paired lists on one cached launch, or ``_FLAT_FALLBACK``.

    Serves the ``.List`` / ``.Tensor`` forms: two lists that pair position by
    position on one flat index (a single-element ``.Tensor`` operand is read
    once per program).  Every other shape answers ``_FLAT_FALLBACK`` and the
    caller hands it to the common executor unchanged, error wording included.
    """
    if not isinstance(self, (list, tuple)) or not self:
        return _FLAT_FALLBACK
    if alpha != 1:
        return _FLAT_FALLBACK
    nt = len(self)
    if nt > _MAX_GRID_X:
        return _FLAT_FALLBACK
    if not isinstance(other, (list, tuple)) or len(other) != nt:
        return _FLAT_FALLBACK
    a0 = self[0]
    if not isinstance(a0, torch.Tensor):
        return _FLAT_FALLBACK
    device = a0.device
    a_dt = a0.dtype
    if a_dt not in _FLAT_FLOAT:
        return _FLAT_FALLBACK
    if allowed_dtypes is not None and a_dt not in allowed_dtypes:
        return _FLAT_FALLBACK
    b0 = other[0]
    if not isinstance(b0, torch.Tensor):
        return _FLAT_FALLBACK
    b_dt = b0.dtype
    if b_dt not in _FLAT_FLOAT:
        return _FLAT_FALLBACK
    # One traversal over the pairs collects the element counts and both address
    # columns while it validates them.  The previous form walked the lists five
    # times (counts, then the second operand, then three address comprehensions)
    # and every extra walk is a visible slice of the launch, because the kernel
    # itself is only a few microseconds on the short shapes.
    numels = []
    a_ptrs = []
    b_ptrs = []
    max_numel = 0
    if scalar_operand:
        if b0.device != device or not b0.is_contiguous() or b0.numel() != 1:
            return _FLAT_FALLBACK
        b_mode = _B_BROADCAST
        for t in self:
            if t.dtype is not a_dt or t.device != device or not t.is_contiguous():
                return _FLAT_FALLBACK
            numel = t.numel()
            if numel == 0:
                return _FLAT_FALLBACK
            numels.append(numel)
            a_ptrs.append(t.data_ptr())
            if numel > max_numel:
                max_numel = numel
        b_ptrs = [b0.data_ptr()] * nt
    else:
        # Paired lists must share the flat index; a broadcast ``.List``
        # operand (per-position scalar or shape broadcast) would read past a
        # smaller allocation, so it falls back to the common executor.
        b_mode = _B_PAIRED
        for tensor, other_tensor in zip(self, other):
            if (
                tensor.dtype is not a_dt
                or tensor.device != device
                or not tensor.is_contiguous()
                or other_tensor.dtype is not b_dt
                or other_tensor.device != device
                or not other_tensor.is_contiguous()
            ):
                return _FLAT_FALLBACK
            numel = tensor.numel()
            if numel == 0 or other_tensor.numel() != numel:
                return _FLAT_FALLBACK
            numels.append(numel)
            a_ptrs.append(tensor.data_ptr())
            b_ptrs.append(other_tensor.data_ptr())
            if numel > max_numel:
                max_numel = numel

    if out_dtypes is not None:
        out_dt = out_dtypes[0]
        if any(dt is not out_dt for dt in out_dtypes):
            return _FLAT_FALLBACK
    elif copy_semantics:
        out_dt = a_dt
    elif b_dt is a_dt:
        out_dt = a_dt
    else:
        out_dt = _promote_dtypes(a0, b0)
    if out_dtypes is None and not copy_semantics and out_dtype_fn is not None:
        out_dt = out_dtype_fn(out_dt)
    if out_dt not in _FLAT_FLOAT:
        return _FLAT_FALLBACK
    if inplace and out_dt is not a_dt:
        return _FLAT_FALLBACK

    _STATS.reset()
    if inplace:
        results = list(self)
    else:
        results = _empty_output_list(self, out_dt)

    out_ptrs = a_ptrs if inplace else [t.data_ptr() for t in results]
    meta = _device_meta(a_ptrs + b_ptrs + out_ptrs + numels, device)

    block, exact = _binary_launch_shape(a_dt, numels)
    grid = _flat_grid(nt, max_numel, _BINARY_FIXED_GRID, block)
    _launch_binary_flat(
        grid,
        meta,
        meta,
        fn,
        nt,
        tl_dtype(a_dt),
        tl_dtype(b_dt),
        tl_dtype(out_dt),
        alpha,
        b_mode,
        0.0,
        block,
        exact,
    )
    _STATS.executor_launches += 1
    _STATS.groups += 1
    return results


def _flat_binary_scalar(
    self,
    scalars,
    fn,
    *,
    inplace=False,
    alpha=1,
    allowed_dtypes=None,
    out_dtype_fn=None,
):
    """Uniform float ``.Scalar`` / ``.ScalarList`` lists on one cached launch.

    The counterpart of :func:`_flat_binary_list` for the scalar overloads: one
    Python number for the whole list or one number per tensor.  A scalar that
    would promote the result away from the tensor dtype answers
    ``_FLAT_FALLBACK`` so the common executor keeps ATen's promotion; a
    broadcast value travels as a compile-time constant and a per-tensor table
    rides the content cache, so neither form uploads anything in steady state.
    """
    if not isinstance(self, (list, tuple)) or not self:
        return _FLAT_FALLBACK
    if alpha != 1:
        return _FLAT_FALLBACK
    nt = len(self)
    if nt > _MAX_GRID_X:
        return _FLAT_FALLBACK
    a0 = self[0]
    if not isinstance(a0, torch.Tensor):
        return _FLAT_FALLBACK
    dtype = a0.dtype
    if dtype not in _FLAT_FLOAT:
        return _FLAT_FALLBACK
    if allowed_dtypes is not None and dtype not in allowed_dtypes:
        return _FLAT_FALLBACK
    device = a0.device
    numels = []
    for t in self:
        if t.dtype is not dtype or t.device != device or not t.is_contiguous():
            return _FLAT_FALLBACK
        numel = t.numel()
        if numel == 0:
            return _FLAT_FALLBACK
        numels.append(numel)
    try:
        values, common, _ = _resolve_scalars(self, scalars)
    except Exception:
        return _FLAT_FALLBACK
    if len(values) != nt:
        return _FLAT_FALLBACK
    # ``bool`` is an ``int`` subclass with its own ATen promotion rules; the
    # common executor keeps them.
    for v in values:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return _FLAT_FALLBACK
        if _promote_with_scalar(dtype, v) is not dtype:
            return _FLAT_FALLBACK
    if out_dtype_fn is not None and out_dtype_fn(dtype) is not dtype:
        return _FLAT_FALLBACK

    _STATS.reset()
    if inplace:
        results = list(self)
        out_ptrs = [t.data_ptr() for t in self]
    else:
        results = _empty_output_list(self, dtype)
        out_ptrs = [t.data_ptr() for t in results]
    in_ptrs = [t.data_ptr() for t in self]
    # The second-pointer slot is unused in the scalar modes; a real device
    # tensor keeps the pointer argument valid without any upload.
    meta = _device_meta(in_ptrs + [0] * nt + out_ptrs + numels, device)

    if common is not None:
        b_mode = _B_VALUE
        scalar_arg = meta
        b_value = float(common)
    else:
        b_mode = _B_PER_TENSOR
        scalar_arg = _cached_scalar_buffer([float(v) for v in values], dtype, device)
        b_value = 0.0

    block, exact = _binary_launch_shape(dtype, numels)
    grid = _flat_grid(nt, max(numels), _BINARY_FIXED_GRID, block)
    _launch_binary_flat(
        grid,
        meta,
        scalar_arg,
        fn,
        nt,
        tl_dtype(dtype),
        tl_dtype(dtype),
        tl_dtype(dtype),
        alpha,
        b_mode,
        b_value,
        block,
        exact,
    )
    _STATS.executor_launches += 1
    _STATS.groups += 1
    return results


def _flat_ternary(
    self,
    tensor1,
    tensor2,
    scalars,
    fn,
    *,
    inplace=False,
    allowed_dtypes=None,
):
    """Uniform float ternary lists on the flat kernel, or ``_FLAT_FALLBACK``.

    The counterpart of :func:`_flat_binary_list` for the ``addcmul`` /
    ``addcdiv`` / ``lerp`` shapes the common ternary executor serves.  The
    uniform-list semantics it replays are the ones of that executor's
    homogeneous path: one floating dtype, one device, all dense.
    """
    if not isinstance(self, (list, tuple)) or not self:
        return _FLAT_FALLBACK
    if not isinstance(tensor1, (list, tuple)) or not isinstance(tensor2, (list, tuple)):
        return _FLAT_FALLBACK
    nt = len(self)
    if len(tensor1) != nt or len(tensor2) != nt:
        return _FLAT_FALLBACK
    if nt > _MAX_GRID_X:
        return _FLAT_FALLBACK
    a0 = self[0]
    if not isinstance(a0, torch.Tensor):
        return _FLAT_FALLBACK
    if not all(isinstance(t, torch.Tensor) for t in self):
        return _FLAT_FALLBACK
    if not all(isinstance(t, torch.Tensor) for t in tensor1):
        return _FLAT_FALLBACK
    same_bc = tensor1 is tensor2
    if not same_bc and not all(isinstance(t, torch.Tensor) for t in tensor2):
        return _FLAT_FALLBACK
    device = a0.device
    dtype = a0.dtype
    if dtype not in _FLAT_FLOAT:
        return _FLAT_FALLBACK
    if allowed_dtypes is not None and dtype not in allowed_dtypes:
        return _FLAT_FALLBACK
    numels = []
    for t in self:
        if t.dtype is not dtype or t.device != device or not t.is_contiguous():
            return _FLAT_FALLBACK
        numel = t.numel()
        if numel == 0:
            return _FLAT_FALLBACK
        numels.append(numel)
    for t in tensor1:
        if t.dtype is not dtype or t.device != device or not t.is_contiguous():
            return _FLAT_FALLBACK
    if not same_bc:
        for t in tensor2:
            if t.dtype is not dtype or t.device != device or not t.is_contiguous():
                return _FLAT_FALLBACK
    # The kernel walks all three operands on one flat index, so every operand
    # list must pair exactly with ``self``; broadcast shapes fall back.
    for t, numel in zip(tensor1, numels):
        if t.numel() != numel:
            return _FLAT_FALLBACK
    if not same_bc:
        for t, numel in zip(tensor2, numels):
            if t.numel() != numel:
                return _FLAT_FALLBACK
    try:
        values, _, _ = _resolve_scalars_memo(self, scalars)
    except Exception:
        return _FLAT_FALLBACK
    if len(values) != nt:
        return _FLAT_FALLBACK
    for v in values:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return _FLAT_FALLBACK

    _STATS.reset()
    if inplace:
        results = list(self)
        out_ptrs = [t.data_ptr() for t in self]
    else:
        results = _empty_output_list(self, dtype)
        out_ptrs = [t.data_ptr() for t in results]
    a_ptrs = [t.data_ptr() for t in self]
    b_ptrs = [t.data_ptr() for t in tensor1]
    c_ptrs = b_ptrs if same_bc else [t.data_ptr() for t in tensor2]
    meta = _device_meta(a_ptrs + b_ptrs + c_ptrs + out_ptrs + numels, device)
    # The common ternary executor casts the scalars into the tensor dtype;
    # the cached buffer keeps the same rounding with zero steady-state H2D.
    scalar_buf = _cached_scalar_buffer([float(v) for v in values], dtype, device)

    grid = _flat_grid(nt, max(numels), _TERNARY_FIXED_GRID, _TERNARY_BLOCK)
    # Bench and optimizer shapes divide the tile evenly; dropping the bounds
    # mask there measured ~2% on the 1024^2 rows (ladder L2 vs production).
    exact = all(n % _TERNARY_BLOCK == 0 for n in numels)
    _launch_ternary_flat(
        grid,
        nt,
        meta,
        scalar_buf,
        fn,
        tl_dtype(dtype),
        FP32_PROMOTE,
        exact,
        _TERNARY_BLOCK,
    )
    _STATS.executor_launches += 1
    _STATS.groups += 1
    return results


# ---------------------------------------------------------------------------
# dtype sets, as measured
# ---------------------------------------------------------------------------

_FLOAT = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_INT = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)

# Arithmetic accepts bool as well; the comparison-style clamps do not reject it
# either.  Complex is excluded here because the paired kernels have no
# interleaved-component path (the unary executor's complex support does not
# generalise to two operands).
ARITH = frozenset(_FLOAT + _INT + (torch.bool,))
# ``lerp`` / ``addcdiv`` are floating point only: an integral lerp would need to
# round the interpolation, and ATen refuses instead.
FLOAT_ONLY = frozenset(_FLOAT)

# bool support is per *operator*, not per family, and two of the exceptions
# depend on the overload of the second operand: ATen defines bool addition as
# logical or but refuses ``bool - bool`` outright, and the clamp family accepts
# a bool tensor operand yet rejects a bool scalar one.
NO_BOOL = frozenset(_FLOAT + _INT)
# ``pow`` has no bool kernel in the .List form but the .Scalar / .ScalarList
# forms do accept it.  ATen reports it as NotImplementedError, not a dtype error.
_POW_BOOL_ERROR = "\"pow\" not implemented for 'Bool'"


def _float_result(dtype: torch.dtype) -> torch.dtype:
    """Promote an integral result to the default float type.

    ``_foreach_div`` on integers returns float, matching ``torch.div``'s true
    division; the executor asks for this through ``out_dtype_fn``.
    """
    if dtype.is_floating_point or dtype.is_complex:
        return dtype
    return torch.float32


class BinaryOp:
    """One row of the multi-input foreach table."""

    __slots__ = (
        "name",
        "fn",
        "allowed",
        "allowed_scalar",
        "dtype_error",
        "dtype_error_type",
        "out_dtype_fn",
        "arity",
    )

    def __init__(
        self,
        name: str,
        fn: Callable,
        allowed: Optional[frozenset] = ARITH,
        out_dtype_fn: Optional[Callable] = None,
        arity: int = 2,
        allowed_scalar: Optional[frozenset] = None,
        dtype_error: Optional[str] = None,
        dtype_error_type: type = RuntimeError,
    ) -> None:
        self.name = name
        self.fn = fn
        self.allowed = allowed
        # The .Scalar / .ScalarList overloads sometimes accept a narrower set
        # than .List; ``None`` means "same as ``allowed``".
        self.allowed_scalar = allowed_scalar
        # Replaces the generic dtype text when ATen words the refusal
        # differently for this operator.
        self.dtype_error = dtype_error
        self.dtype_error_type = dtype_error_type
        self.out_dtype_fn = out_dtype_fn
        self.arity = arity


# ATen's own wording for the one operator that refuses bool with a dedicated
# message rather than the generic "not implemented" text.
_SUB_BOOL_ERROR = (
    "Subtraction, the `-` operator, with two bool tensors is not supported. "
    "Use the `^` or `logical_xor()` operator instead."
)


BINARY_OPS: Dict[str, BinaryOp] = {
    op.name: op
    for op in (
        BinaryOp("add", _add_fn),
        # ATen refuses ``bool - bool`` rather than defining it.
        BinaryOp("sub", _sub_fn, NO_BOOL, dtype_error=_SUB_BOOL_ERROR),
        BinaryOp("mul", _mul_fn),
        BinaryOp("div", _div_fn, ARITH, _float_result),
        BinaryOp("clamp_max", _minimum_fn, allowed_scalar=NO_BOOL),
        BinaryOp("clamp_min", _maximum_fn, allowed_scalar=NO_BOOL),
        BinaryOp("maximum", _maximum_fn, allowed_scalar=NO_BOOL),
        BinaryOp("minimum", _minimum_fn, allowed_scalar=NO_BOOL),
        # ``pow`` rejects a bool tensor operand but accepts a bool scalar; the
        # refusal is NotImplementedError in ATen rather than a dtype error.
        # Its result dtype is plain ``result_type(base, exponent)``, so unlike
        # ``div`` it needs no output-dtype callback.
        BinaryOp(
            "pow",
            _pow_fn,
            NO_BOOL,
            allowed_scalar=ARITH,
            dtype_error=_POW_BOOL_ERROR,
            dtype_error_type=NotImplementedError,
        ),
        # ``copy`` casts the operand into the destination dtype, so every dtype
        # is accepted; no promotion happens.
        BinaryOp("copy", _copy_fn, None),
        BinaryOp("lerp", _lerp_tensor_weight_fn, FLOAT_ONLY, None, 3),
        BinaryOp("addcmul", _addcmul_fn, ARITH, None, 4),
        BinaryOp("addcdiv", _addcdiv_fn, FLOAT_ONLY, None, 4),
    )
}


def _log(name: str, overload: str, inplace: bool) -> None:
    logger.debug("GEMS _FOREACH_%s%s", name.upper(), "_" if inplace else "")
    del overload


def _check_alpha(alpha, dtype: torch.dtype) -> None:
    """ATen refuses a fractional ``alpha`` for integral and bool operands."""
    if dtype in _INT + (torch.bool,) and float(alpha) != int(alpha):
        raise RuntimeError(
            "For integral input tensors, argument alpha must not be "
            "a floating point number."
        )


def _apply_list(name: str, self, other, inplace: bool, alpha=1):
    """The ``.List`` overload: pair the two lists position by position."""
    op = BINARY_OPS[name]
    _log(name, "List", inplace)
    if alpha != 1:
        for t in self:
            _check_alpha(alpha, t.dtype)
    res = _flat_binary_list(
        self,
        other,
        op.fn,
        inplace=inplace,
        alpha=alpha,
        allowed_dtypes=op.allowed,
        out_dtype_fn=op.out_dtype_fn,
    )
    if res is _FLAT_FALLBACK:
        res = foreach_binary_list(
            self,
            other,
            op.fn,
            inplace=inplace,
            alpha=alpha,
            allowed_dtypes=op.allowed,
            dtype_error=op.dtype_error,
            dtype_error_type=op.dtype_error_type,
            out_dtype_fn=op.out_dtype_fn,
        )
    return None if inplace else res


def _apply_scalar(name: str, self, scalars, inplace: bool, alpha=1):
    """The ``.Scalar`` and ``.ScalarList`` overloads."""
    op = BINARY_OPS[name]
    _log(name, "Scalar", inplace)
    if alpha != 1:
        for t in self:
            _check_alpha(alpha, t.dtype)
    allowed = op.allowed_scalar if op.allowed_scalar is not None else op.allowed
    res = _flat_binary_scalar(
        self,
        scalars,
        op.fn,
        inplace=inplace,
        alpha=alpha,
        allowed_dtypes=allowed,
        out_dtype_fn=op.out_dtype_fn,
    )
    if res is _FLAT_FALLBACK:
        res = foreach_binary_scalar(
            self,
            scalars,
            op.fn,
            inplace=inplace,
            alpha=alpha,
            allowed_dtypes=allowed,
            dtype_error=op.dtype_error,
            dtype_error_type=op.dtype_error_type,
            out_dtype_fn=op.out_dtype_fn,
        )
    return None if inplace else res


def _apply_tensor(name: str, self, other: torch.Tensor, inplace: bool, alpha=1):
    """The ``.Tensor`` overload: one tensor broadcast across the whole list.

    ATen accepts any shape here that broadcasts against each element of the list.
    A zero-dimensional tensor is the common case -- it is what optimizers pass --
    and it is routed through the paired kernel with the operand read once per
    program rather than expanded into a full-size staging copy (or read back
    with ``.item()``).

    Reading the value out would be simpler, but ``.item()`` synchronises the
    device: with sixteen kernels already queued it measured 2.11ms per call
    against PyTorch's 0.039ms, because every call drains the pipeline. Expanding
    instead keeps the scalar on the device and the launch asynchronous.
    """
    if not isinstance(other, torch.Tensor):
        raise TypeError("argument 'other' must be a Tensor")
    op = BINARY_OPS[name]
    _log(name, "Tensor", inplace)
    self_list = list(self)
    if alpha != 1:
        for t in self_list:
            _check_alpha(alpha, t.dtype)
    if other.numel() == 1:
        # A single-element operand needs no copy at all: the kernel reads it
        # once per program as a uniform load, so the same tensor is handed to
        # every position and materialising it would only add a full-size copy.
        operands = [other] * len(self_list)
        scalar_operand = True
    else:
        operands = [
            other if other.shape == t.shape else other.expand_as(t) for t in self_list
        ]
        scalar_operand = False
    # Promotion is resolved against the *original* operand, not the expanded
    # view: ATen treats a zero-dimensional tensor as a scalar here, so an fp16
    # list times an fp32 scalar tensor stays fp16, while a genuine fp32 tensor
    # list would promote. Expanding first and asking ``result_type`` afterwards
    # would silently widen the result.
    #
    # ``_promote_dtypes`` keys on (dtype, is-zero-dimensional) for each side, so
    # a list whose entries all share that pair resolves with one lookup rather
    # than sixteen.  A mixed list falls back to the per-tensor form, which is
    # the same function and therefore yields the same dtypes.
    n = len(self_list)
    homogeneous = False
    if n:
        first = self_list[0]
        first_dtype = first.dtype
        first_zero_dim = first.dim() == 0
        homogeneous = True
        for t in self_list:
            if t.dtype is not first_dtype or (t.dim() == 0) != first_zero_dim:
                homogeneous = False
                break
    if homogeneous:
        promoted = _promote_dtypes(first, other)
        if op.out_dtype_fn is not None:
            promoted = op.out_dtype_fn(promoted)
        out_dtypes = [promoted] * n
    else:
        out_dtypes = [_promote_dtypes(t, other) for t in self_list]
        if op.out_dtype_fn is not None:
            out_dtypes = [op.out_dtype_fn(dt) for dt in out_dtypes]
    res = _flat_binary_list(
        self_list,
        operands,
        op.fn,
        inplace=inplace,
        alpha=alpha,
        allowed_dtypes=op.allowed,
        out_dtypes=out_dtypes,
        scalar_operand=scalar_operand,
    )
    if res is _FLAT_FALLBACK:
        res = foreach_binary_list(
            self_list,
            operands,
            op.fn,
            inplace=inplace,
            alpha=alpha,
            allowed_dtypes=op.allowed,
            dtype_error=op.dtype_error,
            dtype_error_type=op.dtype_error_type,
            out_dtypes=out_dtypes,
            scalar_operand=scalar_operand,
        )
    return None if inplace else res


def _apply_ternary(name: str, self, tensor1, tensor2, scalars, inplace: bool):
    """``addcmul`` / ``addcdiv``: three tensor lists and a value."""
    op = BINARY_OPS[name]
    _log(name, "Ternary", inplace)
    # GCU400-specific fast path: the common foreach_ternary 2D grid exceeds the grid.y<=255 limit on large shapes;
    # when the fast-path prerequisites are not met, fall back to the common executor unchanged.
    if name == "addcdiv" and isinstance(scalars, torch.Tensor):
        fast_result = run_addcdiv_tensor(self, tensor1, tensor2, scalars, inplace)
        if fast_result is not _ADDC_DIV_FLAT_FALLBACK:
            return fast_result
    res = _flat_ternary(
        self,
        tensor1,
        tensor2,
        scalars,
        op.fn,
        inplace=inplace,
        allowed_dtypes=op.allowed,
    )
    if res is _FLAT_FALLBACK:
        res = foreach_ternary(
            self,
            tensor1,
            tensor2,
            scalars,
            op.fn,
            inplace=inplace,
            allowed_dtypes=op.allowed,
        )
    return None if inplace else res


# ---------------------------------------------------------------------------
# wrappers, one per registered ATen overload
# ---------------------------------------------------------------------------
#
# The bodies are uniform, so they are generated rather than typed sixty times;
# the module-level names below are what ``_FULL_CONFIG`` and the tests import,
# and each one is a distinct function object so a caplog probe can tell them
# apart.


def _make_list(name, inplace, with_alpha):
    if with_alpha:

        def wrapper(self, other, *, alpha=1):
            return _apply_list(name, self, other, inplace, alpha)

    else:

        def wrapper(self, other):
            return _apply_list(name, self, other, inplace)

    return wrapper


def _make_scalar(name, inplace, with_alpha):
    if with_alpha:

        def wrapper(self, scalar, *, alpha=1):
            return _apply_scalar(name, self, scalar, inplace, alpha)

    else:

        def wrapper(self, scalar):
            return _apply_scalar(name, self, scalar, inplace)

    return wrapper


def _make_tensor(name, inplace, with_alpha):
    if with_alpha:

        def wrapper(self, other, *, alpha=1):
            return _apply_tensor(name, self, other, inplace, alpha)

    else:

        def wrapper(self, other):
            return _apply_tensor(name, self, other, inplace)

    return wrapper


def _make_ternary(name, inplace):
    def wrapper(self, tensor1, tensor2, value=1):
        return _apply_ternary(name, self, tensor1, tensor2, value, inplace)

    return wrapper


# ``alpha`` exists on the .List overload of add/sub and on add's .Tensor form
# only; the other overloads reject it, so handing them the argument would accept
# calls ATen refuses.  Keyed by (operator, overload).
_WITH_ALPHA = {("add", "List"), ("add", "Tensor"), ("sub", "List")}

# Which overloads each operator actually has, from a live schema audit.  Writing
# this table by hand from the ATen names would risk registering a key that does
# not exist (silently dead) or missing one that does.
_OVERLOADS = {
    "add": ("List", "Scalar", "ScalarList", "Tensor"),
    "sub": ("List", "Scalar", "ScalarList"),
    "mul": ("List", "Scalar", "ScalarList", "Tensor"),
    "div": ("List", "Scalar", "ScalarList", "Tensor"),
    "clamp_max": ("List", "Scalar", "ScalarList"),
    "clamp_min": ("List", "Scalar", "ScalarList"),
    "maximum": ("List", "Scalar", "ScalarList"),
    "minimum": ("List", "Scalar", "ScalarList"),
    "pow": ("List", "Scalar", "ScalarList"),
}

# ``lerp`` is ternary in *every* overload -- ``(self, tensors1, weight)`` where
# only the weight varies between List / Scalar / ScalarList -- so it cannot use
# the two-operand wrappers above and gets its own below.
_LERP_OVERLOADS = ("List", "Scalar", "ScalarList")

_globals = globals()

for _name, _ovs in _OVERLOADS.items():
    for _inplace in (False, True):
        _suffix = "_" if _inplace else ""
        for _ov in _ovs:
            _alpha = (_name, _ov) in _WITH_ALPHA
            if _ov == "List":
                _fn = _make_list(_name, _inplace, _alpha)
            elif _ov == "Tensor":
                _fn = _make_tensor(_name, _inplace, _alpha)
            else:
                _fn = _make_scalar(_name, _inplace, _alpha)
            _attr = f"_foreach_{_name}{_suffix}_{_ov}"
            _fn.__name__ = _attr
            _fn.__qualname__ = _attr
            _globals[_attr] = _fn

for _name in ("addcmul", "addcdiv"):
    for _inplace in (False, True):
        _suffix = "_" if _inplace else ""
        for _ov in ("Scalar", "ScalarList", "Tensor"):
            _fn = _make_ternary(_name, _inplace)
            _attr = f"_foreach_{_name}{_suffix}_{_ov}"
            _fn.__name__ = _attr
            _fn.__qualname__ = _attr
            _globals[_attr] = _fn


def _apply_lerp(self, tensors1, weight, inplace: bool):
    """``self + weight * (tensors1 - self)``, weight per-list or per-tensor.

    The ``List`` overload's weight is itself a TensorList, which the ternary
    executor already handles by treating it as the third tensor operand; the
    scalar forms broadcast their weight the same way ``addcmul``'s value does.
    """
    op = BINARY_OPS["lerp"]
    _log("lerp", "lerp", inplace)
    if (
        isinstance(weight, (list, tuple))
        and weight
        and isinstance(weight[0], torch.Tensor)
    ):
        # weights are tensors: fn(self, tensors1, weights, unused_scalar)
        res = _flat_ternary(
            self,
            tensors1,
            weight,
            0.0,
            _lerp_tensor_weight_fn,
            inplace=inplace,
            allowed_dtypes=op.allowed,
        )
        if res is _FLAT_FALLBACK:
            res = foreach_ternary(
                self,
                tensors1,
                weight,
                0.0,
                _lerp_tensor_weight_fn,
                inplace=inplace,
                allowed_dtypes=op.allowed,
            )
    else:
        # GCU400-specific fast path: the common foreach_ternary 2D grid exceeds the grid.y<=255 limit on large shapes;
        # when the fast-path prerequisites are not met, fall back to the common executor unchanged.
        fast_result = run_lerp_scalar_list(self, tensors1, weight, inplace)
        if fast_result is not _LERP_FLAT_FALLBACK:
            return fast_result
        res = _flat_ternary(
            self,
            tensors1,
            tensors1,
            weight,
            _lerp_scalar_weight_fn,
            inplace=inplace,
            allowed_dtypes=op.allowed,
        )
        if res is _FLAT_FALLBACK:
            res = foreach_ternary(
                self,
                tensors1,
                tensors1,
                weight,
                _lerp_scalar_weight_fn,
                inplace=inplace,
                allowed_dtypes=op.allowed,
            )
    return None if inplace else res


def _make_lerp(inplace):
    def wrapper(self, tensors1, weight):
        return _apply_lerp(self, tensors1, weight, inplace)

    return wrapper


for _inplace in (False, True):
    _suffix = "_" if _inplace else ""
    for _ov in _LERP_OVERLOADS:
        _fn = _make_lerp(_inplace)
        _attr = f"_foreach_lerp{_suffix}_{_ov}"
        _fn.__name__ = _attr
        _fn.__qualname__ = _attr
        _globals[_attr] = _fn


def _foreach_copy(self, src, non_blocking=False):
    logger.debug("GEMS_ENFLAME _FOREACH_COPY")
    del non_blocking
    _log("copy", "copy", False)
    res = run_copy(self, src, False)
    if res is _COPY_FLAT_FALLBACK:
        res = _flat_binary_list(
            self,
            src,
            _copy_fn,
            inplace=False,
            copy_semantics=True,
        )
    if res is _FLAT_FALLBACK:
        res = foreach_binary_list(
            self,
            src,
            _copy_fn,
            inplace=False,
            allowed_dtypes=BINARY_OPS["copy"].allowed,
            copy_semantics=True,
        )
    return res


def _foreach_copy_(self, src, non_blocking=False):
    logger.debug("GEMS_ENFLAME _FOREACH_COPY_")
    del non_blocking
    _log("copy", "copy", True)
    res = run_copy(self, src, True)
    if res is _COPY_FLAT_FALLBACK:
        res = _flat_binary_list(
            self,
            src,
            _copy_fn,
            inplace=True,
            copy_semantics=True,
        )
    if res is _FLAT_FALLBACK:
        foreach_binary_list(
            self,
            src,
            _copy_fn,
            inplace=True,
            allowed_dtypes=BINARY_OPS["copy"].allowed,
            copy_semantics=True,
        )
    return None


def _foreach_pow_ScalarAndTensor(self, exponent):
    """``pow.ScalarAndTensor``: the *base* is the scalar, the list is the exponent.

    The argument order is reversed relative to every other overload in this
    family, which is why it cannot share the generated wrappers.
    """
    logger.debug("GEMS_ENFLAME _FOREACH_POW_SCALARANDTENSOR")
    _log("pow", "ScalarAndTensor", False)
    op = BINARY_OPS["pow"]
    for t in exponent:
        # The exponent list carries the tensors here, and ``pow`` has no bool
        # kernel for a bool tensor operand.
        if t.dtype == torch.bool:
            raise NotImplementedError(_POW_BOOL_ERROR)

    res = _flat_binary_scalar(
        exponent,
        self,
        _rpow_fn,
        inplace=False,
        allowed_dtypes=op.allowed,
        out_dtype_fn=_float_result,
    )
    if res is _FLAT_FALLBACK:
        res = foreach_binary_scalar(
            exponent,
            self,
            _rpow_fn,
            allowed_dtypes=op.allowed,
            dtype_error=op.dtype_error,
            dtype_error_type=op.dtype_error_type,
            out_dtype_fn=_float_result,
        )
    return res


def _clear_loop_vars() -> None:
    """Drop the loop variables so they are not exported as module attributes."""
    for _v in (
        "_name",
        "_ovs",
        "_ov",
        "_inplace",
        "_suffix",
        "_fn",
        "_attr",
        "_alpha",
    ):
        _globals.pop(_v, None)


_clear_loop_vars()


def registered_wrappers() -> Dict[str, Any]:
    """Map ATen overload key -> wrapper, used by ``_FULL_CONFIG`` and tests.

    Building the registration list from the same table that generated the
    wrappers keeps the two from drifting: a wrapper with no key, or a key with
    no wrapper, cannot happen by construction.
    """
    out: Dict[str, Any] = {}
    for name, ovs in _OVERLOADS.items():
        for inplace in (False, True):
            suffix = "_" if inplace else ""
            for ov in ovs:
                out[f"_foreach_{name}{suffix}.{ov}"] = _globals[
                    f"_foreach_{name}{suffix}_{ov}"
                ]
    for name in ("addcmul", "addcdiv"):
        for inplace in (False, True):
            suffix = "_" if inplace else ""
            for ov in ("Scalar", "ScalarList", "Tensor"):
                out[f"_foreach_{name}{suffix}.{ov}"] = _globals[
                    f"_foreach_{name}{suffix}_{ov}"
                ]
    for inplace in (False, True):
        suffix = "_" if inplace else ""
        for ov in _LERP_OVERLOADS:
            out[f"_foreach_lerp{suffix}.{ov}"] = _globals[f"_foreach_lerp{suffix}_{ov}"]
    out["_foreach_copy"] = _foreach_copy
    out["_foreach_copy_"] = _foreach_copy_
    out["_foreach_pow.ScalarAndTensor"] = _foreach_pow_ScalarAndTensor
    return out


def _sequence_guard(x: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
    return x
