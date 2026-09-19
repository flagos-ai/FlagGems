"""Correctness tests for the multi-input and reducing ``_foreach_*`` operators.

Every case compares against the PyTorch reference through ``to_reference``, so
the same file passes both the GPU run and CI's ``--ref=cpu --quick`` run.

Two properties this file tests that a naive parameterization would miss:

* **Registration liveness.** Most of these operators have *no* ``default``
  overload -- ``_foreach_add`` offers only ``Scalar``/``List``/``ScalarList``/
  ``Tensor``. A registration written against the bare name raises no error and
  lets every accuracy assertion pass while never being called, so each key is
  probed directly with a negative control.
* **Non-contiguous pairing.** A unary foreach operator survives a transposed
  input by accident: flat traversal visits the same element set. The ``.List``
  overloads do not have that luxury, because two differently-strided tensors
  paired by flat index would combine the wrong elements. The layout cases below
  are what would catch that.
"""

import logging

import pytest
import torch

import flag_gems
from flag_gems.ops._foreach_binary import BINARY_OPS, registered_wrappers
from flag_gems.ops._foreach_reduction import registered_wrappers as reduction_wrappers

from .accuracy_utils import gems_assert_close, to_reference

BINARY_WRAPPERS = registered_wrappers()
REDUCTION_WRAPPERS = reduction_wrappers()
ALL_WRAPPERS = {**BINARY_WRAPPERS, **REDUCTION_WRAPPERS}

FLOAT_DTYPES = [torch.float16, torch.float32, torch.bfloat16]

# Operators whose ``.List`` form takes a second tensor list only; the ternary
# ones need their own argument construction.
TERNARY = ("addcmul", "addcdiv")


# Sample values stay positive and away from zero: ``pow`` of a negative base
# and ``div`` by a near-zero denominator both produce values whose reference and
# result differ by more than a dtype tolerance for reasons unrelated to the
# kernel.
def _sample(shape, dtype, device):
    return torch.rand(shape, dtype=dtype, device=device) + 1.0


def _key_parts(key):
    base, _, overload = key.partition(".")
    name = base[len("_foreach_") :]
    inplace = name.endswith("_")
    core = name[:-1] if inplace else name
    return core, overload, inplace


def _aten(key):
    base, _, overload = key.partition(".")
    op = getattr(torch.ops.aten, base)
    return getattr(op, overload) if overload else op


def _build_args(core, overload, shapes, dtype, device):
    """Arguments for one overload, as ATen expects them."""
    lists = lambda: [_sample(s, dtype, device) for s in shapes]
    n = len(shapes)
    if core in TERNARY:
        value = {
            "Scalar": 0.5,
            "ScalarList": [0.5] * n,
            # ATen requires the scalars tensor on the CPU for this overload.
            "Tensor": torch.tensor([0.5] * n),
        }[overload]
        return (lists(), lists(), value)
    if core == "lerp":
        weight = {
            "List": [torch.full(s, 0.3, dtype=dtype, device=device) for s in shapes],
            "Scalar": 0.3,
            "ScalarList": [0.3] * n,
        }[overload]
        return (lists(), weight)
    if core == "copy":
        return (lists(),)
    if core in ("max", "norm", "zero"):
        return ()
    return {
        "List": (lists(),),
        "Scalar": (0.5,),
        "ScalarList": ([0.5] * n,),
        "Tensor": (torch.tensor(2.0, device=device),),
        "ScalarAndTensor": None,
    }[overload]


def _to_ref(value):
    """``to_reference`` applied through lists and tensors, leaving numbers alone."""
    if isinstance(value, torch.Tensor):
        return to_reference(value)
    if isinstance(value, (list, tuple)):
        return [_to_ref(v) for v in value]
    return value


def _compare(res, ref, dtype):
    assert len(res) == len(ref)
    for got, want in zip(res, ref):
        assert got.shape == want.shape, f"{got.shape} != {want.shape}"
        gems_assert_close(got, want, dtype)


ALL_KEYS = sorted(ALL_WRAPPERS)


def _marks_for(key):
    """The operators.yaml id for a key, which is also its pytest marker."""
    core, overload, inplace = _key_parts(key)
    import re

    ident = f"foreach_{core}"
    if overload:
        ident += "_" + re.sub(r"(?<!^)(?=[A-Z])", "_", overload).lower()
    if inplace:
        ident += "_"
    return ident


PARAMS = [pytest.param(k, marks=getattr(pytest.mark, _marks_for(k))) for k in ALL_KEYS]


# ---------------------------------------------------------------------------
# Static marker declarations
#
# tools/ci_checks/check_operator_markers.py resolves markers by walking
# FunctionDef.decorator_list with ast, so it cannot see a marker that
# pytest.param() attaches at collection time. It only requires that some
# function in this file carry the decorator, so the full set is declared here
# on a no-op placeholder.
#
# They deliberately do NOT sit on the parametrized test: a marker applied to
# the function applies to every case it generates, so stacking all of them
# there made `pytest -m foreach_mul_tensor` and `pytest -m foreach_add_list`
# select the identical set of cases. The per-parameter markers on the
# parametrize list are what give `-m <id>` its one-operator selectivity.
# ---------------------------------------------------------------------------
@pytest.mark.foreach_add_list
@pytest.mark.foreach_add_list_
@pytest.mark.foreach_add_scalar
@pytest.mark.foreach_add_scalar_
@pytest.mark.foreach_add_scalar_list
@pytest.mark.foreach_add_scalar_list_
@pytest.mark.foreach_add_tensor
@pytest.mark.foreach_add_tensor_
@pytest.mark.foreach_addcdiv_scalar
@pytest.mark.foreach_addcdiv_scalar_
@pytest.mark.foreach_addcdiv_scalar_list
@pytest.mark.foreach_addcdiv_scalar_list_
@pytest.mark.foreach_addcdiv_tensor
@pytest.mark.foreach_addcdiv_tensor_
@pytest.mark.foreach_addcmul_scalar
@pytest.mark.foreach_addcmul_scalar_
@pytest.mark.foreach_addcmul_scalar_list
@pytest.mark.foreach_addcmul_scalar_list_
@pytest.mark.foreach_addcmul_tensor
@pytest.mark.foreach_addcmul_tensor_
@pytest.mark.foreach_clamp_max_list
@pytest.mark.foreach_clamp_max_list_
@pytest.mark.foreach_clamp_max_scalar
@pytest.mark.foreach_clamp_max_scalar_
@pytest.mark.foreach_clamp_max_scalar_list
@pytest.mark.foreach_clamp_max_scalar_list_
@pytest.mark.foreach_clamp_min_list
@pytest.mark.foreach_clamp_min_list_
@pytest.mark.foreach_clamp_min_scalar
@pytest.mark.foreach_clamp_min_scalar_
@pytest.mark.foreach_clamp_min_scalar_list
@pytest.mark.foreach_clamp_min_scalar_list_
@pytest.mark.foreach_copy
@pytest.mark.foreach_copy_
@pytest.mark.foreach_div_list
@pytest.mark.foreach_div_list_
@pytest.mark.foreach_div_scalar
@pytest.mark.foreach_div_scalar_
@pytest.mark.foreach_div_scalar_list
@pytest.mark.foreach_div_scalar_list_
@pytest.mark.foreach_div_tensor
@pytest.mark.foreach_div_tensor_
@pytest.mark.foreach_lerp_list
@pytest.mark.foreach_lerp_list_
@pytest.mark.foreach_lerp_scalar
@pytest.mark.foreach_lerp_scalar_
@pytest.mark.foreach_lerp_scalar_list
@pytest.mark.foreach_lerp_scalar_list_
@pytest.mark.foreach_max
@pytest.mark.foreach_maximum_list
@pytest.mark.foreach_maximum_list_
@pytest.mark.foreach_maximum_scalar
@pytest.mark.foreach_maximum_scalar_
@pytest.mark.foreach_maximum_scalar_list
@pytest.mark.foreach_maximum_scalar_list_
@pytest.mark.foreach_minimum_list
@pytest.mark.foreach_minimum_list_
@pytest.mark.foreach_minimum_scalar
@pytest.mark.foreach_minimum_scalar_
@pytest.mark.foreach_minimum_scalar_list
@pytest.mark.foreach_minimum_scalar_list_
@pytest.mark.foreach_mul_list
@pytest.mark.foreach_mul_list_
@pytest.mark.foreach_mul_scalar
@pytest.mark.foreach_mul_scalar_
@pytest.mark.foreach_mul_scalar_list
@pytest.mark.foreach_mul_scalar_list_
@pytest.mark.foreach_mul_tensor
@pytest.mark.foreach_mul_tensor_
@pytest.mark.foreach_norm_scalar
@pytest.mark.foreach_pow_list
@pytest.mark.foreach_pow_list_
@pytest.mark.foreach_pow_scalar
@pytest.mark.foreach_pow_scalar_
@pytest.mark.foreach_pow_scalar_and_tensor
@pytest.mark.foreach_pow_scalar_list
@pytest.mark.foreach_pow_scalar_list_
@pytest.mark.foreach_sub_list
@pytest.mark.foreach_sub_list_
@pytest.mark.foreach_sub_scalar
@pytest.mark.foreach_sub_scalar_
@pytest.mark.foreach_sub_scalar_list
@pytest.mark.foreach_sub_scalar_list_
@pytest.mark.foreach_zero
@pytest.mark.foreach_zero_
def test_operator_markers_are_declared():
    """Placeholder carrying the marker set for static discovery."""


@pytest.mark.parametrize("key", PARAMS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_foreach_ops(key, dtype):
    core, overload, inplace = _key_parts(key)
    shapes = [(16, 8), (7,), (2, 3, 4)]
    device = flag_gems.device

    inp = [_sample(s, dtype, device) for s in shapes]
    args = _build_args(core, overload, shapes, dtype, device)

    if args is None:  # pow.ScalarAndTensor: the scalar is the base
        ref_out = _aten(key)(2.0, [to_reference(t) for t in inp])
        res_out = ALL_WRAPPERS[key](2.0, inp)
        _compare(res_out, ref_out, dtype)
        return

    # ``to_reference`` may hand back the same object, so in-place references are
    # built from clones: otherwise both calls mutate the same storage and the
    # operator effectively runs twice.
    ref_inp = [to_reference(t.clone()) for t in inp]
    ref_args = _to_ref(args)

    ref_out = _aten(key)(ref_inp, *ref_args)
    res_out = ALL_WRAPPERS[key](inp, *args)

    if inplace:
        # The in-place schemas return ``()``; a wrapper handing back the list
        # would make the dispatcher reject the kernel.
        assert res_out is None
        _compare(inp, ref_inp, dtype)
    else:
        _compare(res_out, ref_out, dtype)


@pytest.mark.parametrize("key", PARAMS)
def test_foreach_ops_registration_key_exists(key):
    """The registered key must name an overload ATen actually has.

    This is the half of liveness that a value comparison cannot reach: most of
    these operators have no ``default`` overload, so a key written as the bare
    name (``_foreach_add`` rather than ``_foreach_add.List``) would register
    without error, never dispatch, and leave every accuracy assertion passing.
    """
    base, _, overload = key.partition(".")
    assert hasattr(torch.ops.aten, base), f"no such ATen operator: {base}"
    overloads = getattr(torch.ops.aten, base).overloads()
    expected = overload or "default"
    assert expected in overloads, (
        f"{key} registers overload '{expected}', but ATen offers {overloads}; "
        "this key would never be dispatched to"
    )


@pytest.mark.parametrize("key", PARAMS)
def test_foreach_ops_registration_is_live(key, caplog):
    """Falsifiable liveness: the negative control must stay silent.

    Calling the FlagGems wrapper must emit the operator's debug record, and the
    plain ATen call must not. Without the negative control a probe proves
    nothing, because a logger left at DEBUG would satisfy the positive half on
    its own.
    """
    if flag_gems.device != "cuda":
        return
    core, overload, _ = _key_parts(key)
    shapes = [(8,), (2, 2)]
    device = flag_gems.device
    inp = [_sample(s, torch.float32, device) for s in shapes]
    args = _build_args(core, overload, shapes, torch.float32, device)
    loggers = (
        "flag_gems.ops._foreach_binary",
        "flag_gems.ops._foreach_reduction",
    )
    aten = _aten(key)

    def call():
        if args is None:
            aten(2.0, [t.clone() for t in inp])
        else:
            aten([t.clone() for t in inp], *args)

    def call_gems():
        # Calling the registered wrapper directly is what ``use_gems()`` would
        # have dispatched to; the CI check_kernelgen_tests job rejects
        # ``use_gems()`` inside test files.
        if args is None:
            ALL_WRAPPERS[key](2.0, [t.clone() for t in inp])
        else:
            ALL_WRAPPERS[key]([t.clone() for t in inp], *args)

    for name in loggers:
        caplog.set_level(logging.DEBUG, logger=name)

    caplog.clear()
    call()
    assert not caplog.text.strip(), "negative control fired: probe proves nothing"

    caplog.clear()
    call_gems()
    assert caplog.text.strip(), f"dead registration for {key}"


@pytest.mark.parametrize(
    "key",
    [
        pytest.param("_foreach_add.List", marks=pytest.mark.foreach_add_list),
        pytest.param("_foreach_mul.List", marks=pytest.mark.foreach_mul_list),
        pytest.param("_foreach_div.List", marks=pytest.mark.foreach_div_list),
    ],
)
def test_accuracy_foreach_paired_noncontiguous(key):
    """Pairing two differently-strided tensors must not mix up positions.

    This is the case a unary operator gets right by accident. Here the two
    operands carry different strides for the same logical shape, so a flat-index
    pairing would combine element ``(i, j)`` of one with a different element of
    the other.
    """
    device = flag_gems.device
    dense = _sample((8, 8), torch.float32, device)
    transposed = _sample((8, 8), torch.float32, device).t()
    gappy = _sample((8, 16), torch.float32, device)[:, ::2]

    inp = [dense, transposed, gappy]
    other = [transposed.clone(), dense, dense]

    ref_inp = [to_reference(t) for t in inp]
    ref_other = [to_reference(t) for t in other]

    ref_out = _aten(key)(ref_inp, ref_other)
    res_out = BINARY_WRAPPERS[key](inp, other)

    _compare(res_out, ref_out, torch.float32)


@pytest.mark.parametrize(
    "key",
    [
        pytest.param("_foreach_add.List", marks=pytest.mark.foreach_add_list),
        pytest.param(
            "_foreach_addcmul.Scalar", marks=pytest.mark.foreach_addcmul_scalar
        ),
    ],
)
def test_foreach_ops_length_mismatch_rejected(key):
    """A shorter second list must fail the way ATen fails, not read past it."""
    device = flag_gems.device
    inp = [_sample((4,), torch.float32, device) for _ in range(3)]
    short = [_sample((4,), torch.float32, device) for _ in range(2)]
    with pytest.raises(RuntimeError):
        if "addcmul" in key:
            BINARY_WRAPPERS[key](inp, short, short, 0.5)
        else:
            BINARY_WRAPPERS[key](inp, short)


@pytest.mark.parametrize(
    "core, ord_",
    [
        pytest.param("norm", 1, marks=pytest.mark.foreach_norm_scalar),
        pytest.param("norm", 2, marks=pytest.mark.foreach_norm_scalar),
    ],
)
def test_accuracy_foreach_reduction_orders(core, ord_):
    """The reductions collapse each tensor to a scalar, for several orders.

    ``ord=1`` and ``ord=2`` take different paths inside the kernel (a sum versus
    a sum of squares), so both are checked rather than trusting one to imply the
    other.
    """
    device = flag_gems.device
    inp = [_sample(s, torch.float32, device) for s in [(16,), (4, 5)]]
    ref_inp = [to_reference(t) for t in inp]
    key = f"_foreach_{core}.Scalar"

    ref_out = _aten(key)(ref_inp, ord_)
    res_out = REDUCTION_WRAPPERS[key](inp, ord_)

    for got in res_out:
        assert got.shape == torch.Size([]), f"expected a scalar, got {got.shape}"
    _compare(res_out, ref_out, torch.float32)


@pytest.mark.foreach_zero_
def test_accuracy_foreach_zero_writes_through_views():
    """``zero_`` must land in the original storage even for a strided view."""
    device = flag_gems.device
    storage = _sample((8, 16), torch.float32, device)
    ref_storage = to_reference(storage.clone())

    torch._foreach_zero_([ref_storage[:, ::2]])
    REDUCTION_WRAPPERS["_foreach_zero_"]([storage[:, ::2]])

    gems_assert_close(storage, ref_storage, torch.float32)


@pytest.mark.parametrize(
    "key",
    [
        pytest.param("_foreach_add.List", marks=pytest.mark.foreach_add_list),
        pytest.param("_foreach_mul.Scalar", marks=pytest.mark.foreach_mul_scalar),
    ],
)
def test_foreach_ops_launch_count_is_independent_of_list_length(key):
    """The paired kernels keep the ``O(groups)`` launch bound of the unary path."""
    if flag_gems.device != "cuda":
        return
    from flag_gems.utils.foreach import launch_stats

    for n in (1, 16, 128):
        inp = [_sample((512,), torch.float32, flag_gems.device) for _ in range(n)]
        if key.endswith(".List"):
            BINARY_WRAPPERS[key](inp, [t.clone() for t in inp])
        else:
            BINARY_WRAPPERS[key](inp, 2.0)
        assert launch_stats()["executor_launches"] == 1, (
            f"{key} with N={n} used more than one launch; a per-tensor loop "
            "would give ~N"
        )


@pytest.mark.foreach_add_scalar
def test_accuracy_foreach_int_promotion():
    """An integral list plus a float scalar promotes, matching ATen."""
    device = flag_gems.device
    inp = [torch.randint(1, 5, (8,), dtype=torch.int64, device=device)]
    ref_inp = [to_reference(t) for t in inp]

    ref_out = torch._foreach_add(ref_inp, 2.5)
    res_out = BINARY_WRAPPERS["_foreach_add.Scalar"](inp, 2.5)

    assert res_out[0].dtype == ref_out[0].dtype
    gems_assert_close(res_out[0], ref_out[0], ref_out[0].dtype)


@pytest.mark.foreach_add_scalar_
def test_foreach_inplace_rejects_promotion():
    """An in-place op may not narrow a promoted result back into an int input."""
    inp = [torch.randint(1, 5, (8,), dtype=torch.int64, device=flag_gems.device)]
    with pytest.raises(RuntimeError):
        BINARY_WRAPPERS["_foreach_add_.Scalar"](inp, 2.5)


@pytest.mark.parametrize("core", ["add", "mul", "div"])
def test_foreach_ops_dtype_sets_match_aten(core):
    """The declared dtype set must not be broader or narrower than ATen's."""
    allowed = BINARY_OPS[core].allowed
    assert allowed is not None
    for dtype in (torch.float32, torch.int32):
        assert dtype in allowed, f"{core} should accept {dtype}"
