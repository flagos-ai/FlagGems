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
from flag_gems.utils.foreach import (
    foreach_binary_list,
    foreach_binary_scalar,
    foreach_ternary,
)

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
    return a + b


@triton.jit
def _sub_fn(a, b):
    return a - b


@triton.jit
def _mul_fn(a, b):
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
    # the intermediate and not the stored dtype.
    return _pow(a.to(tl.float32), b.to(tl.float32))


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

    __slots__ = ("name", "fn", "allowed", "out_dtype_fn", "arity")

    def __init__(
        self,
        name: str,
        fn: Callable,
        allowed: Optional[frozenset] = ARITH,
        out_dtype_fn: Optional[Callable] = None,
        arity: int = 2,
    ) -> None:
        self.name = name
        self.fn = fn
        self.allowed = allowed
        self.out_dtype_fn = out_dtype_fn
        self.arity = arity


BINARY_OPS: Dict[str, BinaryOp] = {
    op.name: op
    for op in (
        BinaryOp("add", _add_fn),
        BinaryOp("sub", _sub_fn),
        BinaryOp("mul", _mul_fn),
        BinaryOp("div", _div_fn, ARITH, _float_result),
        BinaryOp("clamp_max", _minimum_fn),
        BinaryOp("clamp_min", _maximum_fn),
        BinaryOp("maximum", _maximum_fn),
        BinaryOp("minimum", _minimum_fn),
        BinaryOp("pow", _pow_fn, ARITH, _float_result),
        BinaryOp("copy", _copy_fn),
        BinaryOp("lerp", _lerp_tensor_weight_fn, FLOAT_ONLY, None, 3),
        BinaryOp("addcmul", _addcmul_fn, ARITH, None, 4),
        BinaryOp("addcdiv", _addcdiv_fn, FLOAT_ONLY, None, 4),
    )
}


def _log(name: str, overload: str, inplace: bool) -> None:
    logger.debug("GEMS _FOREACH_%s%s", name.upper(), "_" if inplace else "")
    del overload


def _apply_list(name: str, self, other, inplace: bool, alpha=1):
    """The ``.List`` overload: pair the two lists position by position."""
    op = BINARY_OPS[name]
    _log(name, "List", inplace)
    if alpha != 1:
        # ``alpha`` scales the second operand before the operation.  Folding it
        # in on the host keeps one compiled kernel per operator instead of a
        # second variant carrying an unused argument in the common case.
        other = torch._foreach_mul(other, alpha)
    res = foreach_binary_list(
        self,
        other,
        op.fn,
        inplace=inplace,
        allowed_dtypes=op.allowed,
        out_dtype_fn=op.out_dtype_fn,
    )
    return None if inplace else res


def _apply_scalar(name: str, self, scalars, inplace: bool, alpha=1):
    """The ``.Scalar`` and ``.ScalarList`` overloads."""
    op = BINARY_OPS[name]
    _log(name, "Scalar", inplace)
    if alpha != 1:
        if isinstance(scalars, (list, tuple)):
            scalars = [s * alpha for s in scalars]
        else:
            scalars = scalars * alpha
    res = foreach_binary_scalar(
        self,
        scalars,
        op.fn,
        inplace=inplace,
        allowed_dtypes=op.allowed,
        out_dtype_fn=op.out_dtype_fn,
    )
    return None if inplace else res


def _apply_tensor(name: str, self, other: torch.Tensor, inplace: bool, alpha=1):
    """The ``.Tensor`` overload: one tensor broadcast across the whole list.

    ATen accepts any shape here that broadcasts against each element of the list.
    A zero-dimensional tensor is the common case -- it is what optimizers pass --
    and it is routed through the paired kernel with the operand expanded rather
    than read back with ``.item()``.

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
    expanded = [
        other if other.shape == t.shape else other.expand_as(t) for t in self_list
    ]
    if alpha != 1:
        expanded = torch._foreach_mul(expanded, alpha)
    # Promotion is resolved against the *original* operand, not the expanded
    # view: ATen treats a zero-dimensional tensor as a scalar here, so an fp16
    # list times an fp32 scalar tensor stays fp16, while a genuine fp32 tensor
    # list would promote. Expanding first and asking ``result_type`` afterwards
    # would silently widen the result.
    out_dtypes = [torch.result_type(t, other) for t in self_list]
    if op.out_dtype_fn is not None:
        out_dtypes = [op.out_dtype_fn(dt) for dt in out_dtypes]
    res = foreach_binary_list(
        self_list,
        expanded,
        op.fn,
        inplace=inplace,
        allowed_dtypes=op.allowed,
        out_dtypes=out_dtypes,
    )
    return None if inplace else res


def _apply_ternary(name: str, self, tensor1, tensor2, scalars, inplace: bool):
    """``addcmul`` / ``addcdiv``: three tensor lists and a value."""
    op = BINARY_OPS[name]
    _log(name, "Ternary", inplace)
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


# ``alpha`` exists only on add/sub; giving the other wrappers the argument would
# accept calls ATen rejects.
_WITH_ALPHA = {"add", "sub"}

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
    _alpha = _name in _WITH_ALPHA
    for _inplace in (False, True):
        _suffix = "_" if _inplace else ""
        for _ov in _ovs:
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
    del non_blocking
    return _apply_list("copy", self, src, inplace=False)


def _foreach_copy_(self, src, non_blocking=False):
    del non_blocking
    return _apply_list("copy", self, src, inplace=True)


def _foreach_pow_ScalarAndTensor(self, exponent):
    """``pow.ScalarAndTensor``: the *base* is the scalar, the list is the exponent.

    The argument order is reversed relative to every other overload in this
    family, which is why it cannot share the generated wrappers.
    """
    _log("pow", "ScalarAndTensor", False)
    op = BINARY_OPS["pow"]

    @triton.jit
    def _rpow(a, b):
        return _pow(b.to(tl.float32), a.to(tl.float32))

    return foreach_binary_scalar(
        exponent,
        self,
        _rpow,
        allowed_dtypes=op.allowed,
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
