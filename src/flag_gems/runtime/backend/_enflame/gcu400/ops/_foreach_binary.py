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
from typing import Any, Callable, Dict, Optional, Sequence

import torch
import triton
import triton.language as tl

from flag_gems.utils import tl_extra_shim
from flag_gems.utils.foreach import _TERNARY_SWAP_MIN_ELEMS, _pack_ok

from ..utils.config_utils import MAX_GRID_DIM
from ..utils.foreach import (
    _FAST_MATH,
    _STATS,
    _device_meta,
    _pick_block,
    _promote_dtypes,
    _resolve_scalars,
    foreach_binary_list,
    foreach_binary_scalar,
    foreach_ternary,
    tl_dtype,
)
from ._foreach_addcdiv_flat import FALLBACK as _ADDC_DIV_FLAT_FALLBACK
from ._foreach_addcdiv_flat import run_addcdiv_tensor
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
_MAX_GRID_Y = 255

_FLAT_FLOAT = (torch.float16, torch.float32, torch.bfloat16)


@triton.jit
def _foreach_binary_flat_kernel(
    meta_ptr,
    fn: tl.constexpr,
    NT: tl.constexpr,
    A_DT: tl.constexpr,
    B_DT: tl.constexpr,
    OUT_DT: tl.constexpr,
    ALPHA: tl.constexpr,
    B_SCALAR: tl.constexpr,
    BLOCK: tl.constexpr,
    NCTA,
):
    """Paired-list math on a grid-stride ``(NT, NCTA)`` launch.

    The metadata layout matches the common ``foreach_binary_kernel``:
    first-input, second-input and output pointers plus element counts, ``NT``
    each, int64.  Axis 1 carries at most ``NCTA`` starting tiles and each
    program strides over the remaining ones, so the second axis never grows
    past the GCU400 limit no matter how large the tensors are.
    """
    tensor_id = tl.program_id(0)
    tile_id = tl.program_id(1)

    a_ptr = tl.load(meta_ptr + tensor_id).to(tl.pointer_type(A_DT))
    b_ptr = tl.load(meta_ptr + NT + tensor_id).to(tl.pointer_type(B_DT))
    out_ptr = tl.load(meta_ptr + 2 * NT + tensor_id).to(tl.pointer_type(OUT_DT))
    n_elements = tl.load(meta_ptr + 3 * NT + tensor_id)

    num_tiles = tl.cdiv(n_elements, BLOCK).to(tl.int32)
    for block_id in tl.range(tile_id, num_tiles, NCTA):
        idx = block_id * BLOCK + tl.arange(0, BLOCK)
        mask = idx < n_elements
        a = tl.load(a_ptr + idx, mask=mask, other=0)
        if B_SCALAR:
            b = tl.load(b_ptr)
        else:
            b = tl.load(b_ptr + idx, mask=mask, other=0)
        b = (b * ALPHA).to(OUT_DT) if ALPHA != 1 else b.to(OUT_DT)
        tl.store(out_ptr + idx, fn(a.to(OUT_DT), b).to(OUT_DT), mask=mask)


@triton.jit
def _foreach_ternary_flat_kernel(
    meta_ptr,
    scalar_ptr,
    fn: tl.constexpr,
    NT: tl.constexpr,
    A_DT: tl.constexpr,
    OUT_DT: tl.constexpr,
    S_DT: tl.constexpr,
    BLOCK: tl.constexpr,
    NCTA,
):
    """Ternary-list math on a grid-stride ``(NT, NCTA)`` launch.

    The metadata layout matches the common ``foreach_ternary_kernel``:
    self, tensor1 and tensor2 pointers, output pointers and element counts,
    ``NT`` each, int64; ``scalar_ptr`` holds one value per tensor.  The
    scalars ride in a device array of the output dtype rather than a
    by-value ``float64`` argument, keeping the signature f64-free exactly
    like the GCU400 scalar executor.
    """
    tensor_id = tl.program_id(0)
    tile_id = tl.program_id(1)

    a_ptr = tl.load(meta_ptr + tensor_id).to(tl.pointer_type(A_DT))
    b_ptr = tl.load(meta_ptr + NT + tensor_id).to(tl.pointer_type(A_DT))
    c_ptr = tl.load(meta_ptr + 2 * NT + tensor_id).to(tl.pointer_type(A_DT))
    out_ptr = tl.load(meta_ptr + 3 * NT + tensor_id).to(tl.pointer_type(OUT_DT))
    n_elements = tl.load(meta_ptr + 4 * NT + tensor_id)
    s = tl.load(scalar_ptr + tensor_id)

    num_tiles = tl.cdiv(n_elements, BLOCK).to(tl.int32)
    for block_id in tl.range(tile_id, num_tiles, NCTA):
        idx = block_id * BLOCK + tl.arange(0, BLOCK)
        mask = idx < n_elements
        a = tl.load(a_ptr + idx, mask=mask, other=0).to(OUT_DT)
        b = tl.load(b_ptr + idx, mask=mask, other=0).to(OUT_DT)
        c = tl.load(c_ptr + idx, mask=mask, other=0).to(OUT_DT)
        tl.store(out_ptr + idx, fn(a, b, c, s.to(OUT_DT)).to(OUT_DT), mask=mask)


def _common_grid_overflows(numels) -> bool:
    """Whether the common paired launch grid exceeds the GCU400 limits.

    The paired executor launches ``(nt, cdiv(max_numel, BLOCK))`` with the
    ``BLOCK`` its own search picks, so the answer is computed the same way;
    ``True`` means the launch would raise ``OutOfResources`` on grid.y.
    """
    nt = len(numels)
    max_numel = max(numels)
    if nt == 0 or max_numel == 0:
        return False
    block = _pick_block(numels)
    tiles = -(-max_numel // block)
    return nt > _MAX_GRID_X or tiles > _MAX_GRID_Y


def _common_ternary_grid_overflows(numels, dtype, values) -> bool:
    """Whether the common ternary launch grid exceeds the GCU400 limits.

    The ternary launcher transposes its grid for the packed 16-bit kernel
    above ``2**20`` elements, which hides the tile count on axis 0; every
    other launch puts it on axis 1 where the 255 cap bites.  ``values`` is
    the pointer/numel run the launch would upload, for the same pack test
    the common launcher applies.
    """
    nt = len(numels)
    max_numel = max(numels)
    if nt == 0 or max_numel == 0:
        return False
    block = _pick_block(numels)
    tiles = -(-max_numel // block)
    swap = (
        _pack_ok((dtype, dtype), max_numel, values)
        and max_numel >= _TERNARY_SWAP_MIN_ELEMS
    )
    if swap:
        return tiles > _MAX_GRID_X or nt > _MAX_GRID_Y
    return nt > _MAX_GRID_X or tiles > _MAX_GRID_Y


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
    """Uniform float paired lists on the flat kernel, or ``_FLAT_FALLBACK``.

    Replays the common executor's uniform-list semantics -- dtype refusal,
    promotion, allocation and the in-place pairing -- for exactly the groups
    whose common launch would exceed the GCU400 grid limits.  Every other
    shape answers ``_FLAT_FALLBACK`` and the caller hands it to the common
    executor unchanged, error wording included.
    """
    if not isinstance(self, (list, tuple)) or not self:
        return _FLAT_FALLBACK
    nt = len(self)
    if not isinstance(other, (list, tuple)) or len(other) != nt:
        return _FLAT_FALLBACK
    if nt > _MAX_GRID_X:
        return _FLAT_FALLBACK
    a0 = self[0]
    if not isinstance(a0, torch.Tensor):
        return _FLAT_FALLBACK
    if not all(isinstance(t, torch.Tensor) for t in self):
        return _FLAT_FALLBACK
    device = a0.device
    a_dt = a0.dtype
    if a_dt not in _FLAT_FLOAT:
        return _FLAT_FALLBACK
    if allowed_dtypes is not None and a_dt not in allowed_dtypes:
        return _FLAT_FALLBACK
    numels = [t.numel() for t in self]
    if not _common_grid_overflows(numels):
        return _FLAT_FALLBACK

    b0 = other[0]
    if not isinstance(b0, torch.Tensor):
        return _FLAT_FALLBACK
    b_dt = b0.dtype
    if b_dt not in _FLAT_FLOAT:
        return _FLAT_FALLBACK
    a_zero_dim = a0.dim() == 0
    for t in self:
        if (
            t.dtype is not a_dt
            or t.device != device
            or not t.is_contiguous()
            or (t.dim() == 0) != a_zero_dim
        ):
            return _FLAT_FALLBACK
    if scalar_operand:
        if b0.device != device or not b0.is_contiguous():
            return _FLAT_FALLBACK
    else:
        b_zero_dim = b0.dim() == 0
        for t in other:
            if (
                t.dtype is not b_dt
                or t.device != device
                or not t.is_contiguous()
                or (t.dim() == 0) != b_zero_dim
            ):
                return _FLAT_FALLBACK

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
    elif out_dt is a_dt:
        results = list(map(torch.empty_like, self))
    else:
        results = [torch.empty_like(t, dtype=out_dt) for t in self]

    a_ptrs = [t.data_ptr() for t in self]
    if scalar_operand:
        b_ptrs = [b0.data_ptr()] * nt
    else:
        b_ptrs = [t.data_ptr() for t in other]
    out_ptrs = a_ptrs if inplace else [t.data_ptr() for t in results]
    meta = _device_meta(a_ptrs + b_ptrs + out_ptrs + numels, device)

    block = _pick_block(numels)
    tiles = -(-max(numels) // block)
    ncta = min(tiles, MAX_GRID_DIM)
    _foreach_binary_flat_kernel[(nt, ncta)](
        meta,
        fn,
        nt,
        tl_dtype(a_dt),
        tl_dtype(b_dt),
        tl_dtype(out_dt),
        alpha,
        scalar_operand,
        block,
        ncta,
        num_warps=4,
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
    for t in self:
        if t.dtype is not dtype or t.device != device or not t.is_contiguous():
            return _FLAT_FALLBACK
    for t in tensor1:
        if t.dtype is not dtype or t.device != device or not t.is_contiguous():
            return _FLAT_FALLBACK
    if not same_bc:
        for t in tensor2:
            if t.dtype is not dtype or t.device != device or not t.is_contiguous():
                return _FLAT_FALLBACK
    try:
        values, _, _ = _resolve_scalars(self, scalars)
    except Exception:
        return _FLAT_FALLBACK
    if not all(isinstance(v, (int, float)) for v in values):
        return _FLAT_FALLBACK

    a_ptrs = [t.data_ptr() for t in self]
    b_ptrs = [t.data_ptr() for t in tensor1]
    c_ptrs = b_ptrs if same_bc else [t.data_ptr() for t in tensor2]
    numels = [t.numel() for t in self]
    # The pack test reads alignment off the whole pointer run; for the
    # out-of-place form the freshly allocated destinations are at least as
    # aligned as the inputs, so the inputs answer the same question.
    if not _common_ternary_grid_overflows(
        numels, dtype, a_ptrs + b_ptrs + c_ptrs + a_ptrs + numels
    ):
        return _FLAT_FALLBACK

    _STATS.reset()
    if inplace:
        results = list(self)
        out_ptrs = a_ptrs
    else:
        results = list(map(torch.empty_like, self))
        out_ptrs = [t.data_ptr() for t in results]
    meta = _device_meta(a_ptrs + b_ptrs + c_ptrs + out_ptrs + numels, device)
    scalar_buf = torch.tensor([float(v) for v in values], dtype=dtype).to(
        device, non_blocking=True
    )

    block = _pick_block(numels)
    tiles = -(-max(numels) // block)
    ncta = min(tiles, MAX_GRID_DIM)
    _foreach_ternary_flat_kernel[(nt, ncta)](
        meta,
        scalar_buf,
        fn,
        nt,
        tl_dtype(dtype),
        tl_dtype(dtype),
        tl_dtype(dtype),
        block,
        ncta,
        num_warps=4,
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
    logger.debug(
        "GEMS_ENFLAME_GCU400 _FOREACH_%s%s", name.upper(), "_" if inplace else ""
    )
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
    logger.debug("GEMS_ENFLAME_GCU400 _FOREACH_COPY")
    del non_blocking
    _log("copy", "copy", False)
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
    logger.debug("GEMS_ENFLAME_GCU400 _FOREACH_COPY_")
    del non_blocking
    _log("copy", "copy", True)
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
    logger.debug("GEMS_ENFLAME_GCU400 _FOREACH_POW_SCALARANDTENSOR")
    _log("pow", "ScalarAndTensor", False)
    op = BINARY_OPS["pow"]
    for t in exponent:
        # The exponent list carries the tensors here, and ``pow`` has no bool
        # kernel for a bool tensor operand.
        if t.dtype == torch.bool:
            raise NotImplementedError(_POW_BOOL_ERROR)

    return foreach_binary_scalar(
        exponent,
        self,
        _rpow_fn,
        allowed_dtypes=op.allowed,
        dtype_error=op.dtype_error,
        dtype_error_type=op.dtype_error_type,
        out_dtype_fn=_float_result,
    )


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
