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


# ---------------------------------------------------------------------------
# parity cases for the dtype and order corners
# ---------------------------------------------------------------------------
#
# These call the registered FlagGems API by name rather than reaching into
# ``flag_gems.ops``: the public entry point is what the dispatcher would reach,
# so a bug in the registration cannot hide behind a direct call. ``to_reference``
# moves the operands to the reference device, which is what makes the same
# assertions valid under ``--ref=cpu``.


def _gems(key):
    """The public FlagGems function an ATen key is registered to.

    Public names spell the overload with an underscore suffix rather than
    ATen's dot (``_foreach_add_List``, not ``_foreach_add.List``), so the key is
    rewritten rather than looked up verbatim.
    """
    return BINARY_WRAPPERS[key] if key in BINARY_WRAPPERS else REDUCTION_WRAPPERS[key]


@pytest.mark.parametrize("core", ["add", "sub", "mul"])
@pytest.mark.parametrize(
    "left",
    [
        pytest.param([True, True, False], id="tt_f"),
        pytest.param([False, False, True], id="ff_t"),
    ],
)
def test_accuracy_foreach_bool(core, left):
    """bool arithmetic matches ATen per operator, including its refusals.

    ATen defines bool ``add`` as logical or and bool ``mul`` as logical and, but
    rejects bool ``sub`` outright. Sharing one dtype set across the family would
    both accept the refused case and get ``True + True`` wrong.
    """
    device = flag_gems.device
    other = [True, False, False]
    inp = [torch.tensor(left, dtype=torch.bool, device=device)]
    ref_inp = [to_reference(t) for t in inp]

    if core == "sub":
        with pytest.raises(RuntimeError):
            _gems("_foreach_sub.List")(inp, [torch.tensor(other, device=device)])
        with pytest.raises(RuntimeError):
            torch._foreach_sub(ref_inp, [torch.tensor(other, device=ref_inp[0].device)])
        return

    # ``add``/``mul`` accept bool here; ``sub`` is the only refusal in this set.
    res = _gems(f"_foreach_{core}.List")(inp, [torch.tensor(other, device=device)])
    ref_operand = torch.tensor(other, device=ref_inp[0].device)
    ref = getattr(torch, f"_foreach_{core}")(ref_inp, [ref_operand])
    gems_assert_close(res[0], ref[0], torch.bool)


@pytest.mark.parametrize("alpha", [1, 2, 2.5, -1.0])
def test_accuracy_foreach_add_alpha(alpha):
    """``alpha`` scales the second operand, applied inside the kernel.

    The scaling must not go through a host-side temporary, which would add a
    second dispatch per call.
    """
    device = flag_gems.device
    inp = [_sample((16,), torch.float32, device)]
    other = [_sample((16,), torch.float32, device)]
    ref_inp = [to_reference(t) for t in inp]
    ref_other = [to_reference(t) for t in other]

    res = _gems("_foreach_add.List")(inp, other, alpha=alpha)
    ref = torch._foreach_add(ref_inp, ref_other, alpha=alpha)
    gems_assert_close(res[0], ref[0], torch.float32)


def test_foreach_alpha_keeps_one_launch():
    """``alpha`` is not paid for with an extra executor launch."""
    if flag_gems.device != "cuda":
        return
    from flag_gems.utils.foreach import launch_stats

    inp = [_sample((512,), torch.float32, flag_gems.device) for _ in range(16)]
    other = [t.clone() for t in inp]
    _gems("_foreach_add.List")(inp, other, alpha=2.5)
    assert launch_stats()["executor_launches"] == 1


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize(
    "src_dtype",
    [torch.float32, torch.float16, torch.int64, torch.bool],
)
def test_accuracy_foreach_copy_mixed_dtype(inplace, src_dtype):
    """``copy`` keeps the destination dtype and casts the source into it.

    This is assignment semantics, not type promotion: an fp16 destination stays
    fp16 when the source is fp32, and the in-place form accepts the narrowing
    that the generic promotion check would reject.
    """
    device = flag_gems.device

    # ``rand`` has no integral or bool kernel, so those sources are built from
    # an integer randint and cast, keeping the values exactly representable in
    # fp16 so the cast back is lossless in the compared region.
    def _make_source(dtype):
        if dtype.is_floating_point:
            return torch.rand((16,), dtype=dtype, device=device)
        return torch.randint(0, 2, (16,), device=device).to(dtype)

    if inplace:
        inp = [torch.ones((16,), dtype=torch.float16, device=device)]
        src = [_make_source(src_dtype)]
        ref_dst = [to_reference(t) for t in inp]
        ref_src = [to_reference(t) for t in src]
        torch._foreach_copy_(ref_dst, ref_src)
        assert _gems("_foreach_copy_")(inp, src) is None
        assert inp[0].dtype == ref_dst[0].dtype
        assert torch.equal(inp[0].cpu(), ref_dst[0].cpu())
    else:
        inp = [torch.ones((16,), dtype=torch.float16, device=device)]
        src = [_make_source(src_dtype)]
        ref = torch.ops.aten._foreach_copy(
            [to_reference(t) for t in inp], [to_reference(t) for t in src]
        )
        res = _gems("_foreach_copy")(inp, src)
        assert res[0].dtype == ref[0].dtype == torch.float16
        assert torch.equal(res[0].cpu(), ref[0].cpu())


@pytest.mark.parametrize("device_dtype", [torch.float64, torch.int64])
def test_accuracy_foreach_max_large_values(device_dtype):
    """``max`` keeps the input dtype and does not round through fp32.

    Values beyond the exact fp32 range expose both failures at once: an fp32
    accumulator overflows an fp64 input, and a value above ``2**53`` loses its
    low bits on the way out.
    """
    device = flag_gems.device
    if device_dtype == torch.float64:
        values = [1e300, -1e300, 1e299]
        in_dtype = torch.float64
    else:
        values = [2**62, 3, -5]
        in_dtype = torch.int64
    inp = [torch.tensor(values, dtype=in_dtype, device=device)]

    res = _gems("_foreach_max")(inp)
    ref = torch._foreach_max([to_reference(inp[0])])

    assert res[0].dtype == ref[0].dtype, f"{res[0].dtype} != {ref[0].dtype}"
    assert res[0].item() == ref[0].item()


@pytest.mark.parametrize(
    "ord_",
    [
        pytest.param(0, id="count_nonzero"),
        pytest.param(float("inf"), id="max_abs"),
        pytest.param(float("-inf"), id="min_abs"),
        pytest.param(1, id="sum_abs"),
        pytest.param(2, id="l2"),
        pytest.param(3, id="l3"),
        pytest.param(0.5, id="half"),
        pytest.param(-2, id="negative"),
    ],
)
def test_accuracy_foreach_norm_orders(ord_):
    """Every norm order matches ATen, including the special forms.

    ``ord=0`` counts nonzero elements and ``ord=±inf`` are the extrema of
    ``|x|``; routing those through the power sum raises ``ZeroDivisionError``
    for ``0`` and silently wrong values for ``inf``.
    """
    device = flag_gems.device
    inp = [torch.tensor([0.0, 2.0, -3.0, 0.5], device=device)]
    ref = torch._foreach_norm([to_reference(t) for t in inp], ord_)

    res = _gems("_foreach_norm.Scalar")(inp, ord_)
    assert res[0].dtype == ref[0].dtype
    gems_assert_close(res[0], ref[0], ref[0].dtype)


def test_accuracy_foreach_norm_all_zero():
    """An all-zero tensor has norm 0 for every order, including the specials."""
    device = flag_gems.device
    inp = [torch.zeros(8, device=device)]
    for ord_ in (0, 1, 2, 3, float("inf"), float("-inf")):
        res = _gems("_foreach_norm.Scalar")(inp, ord_)
        ref = torch._foreach_norm([to_reference(t) for t in inp], ord_)
        assert res[0].item() == ref[0].item(), f"ord={ord_}"


@pytest.mark.foreach_pow_scalar
@pytest.mark.foreach_pow_scalar_list
@pytest.mark.parametrize("exponent", [True, False, 0, 2, 2.0, 0.5, -1.0, float("inf")])
def test_accuracy_foreach_pow_bool_base(exponent):
    """``pow`` of a bool base: the result dtype follows the exponent.

    A bool exponent keeps bool (``a ** True`` is ``a``, ``a ** False`` is
    ``True``), an int exponent gives int64, and a float exponent gives float32.
    The negative and infinite exponents are included because a bool base is a
    zero-or-one value and ``False ** -1`` is ``inf``: a special-cased bool
    branch that returned the base instead would pass the integer cases and fail
    only here.
    """
    device = flag_gems.device
    inp = [torch.tensor([True, True, False], dtype=torch.bool, device=device)]
    ref_inp = [to_reference(t) for t in inp]

    res = _gems("_foreach_pow.Scalar")(inp, exponent)
    ref = torch._foreach_pow(ref_inp, exponent)

    assert res[0].dtype == ref[0].dtype, f"{res[0].dtype} != {ref[0].dtype}"
    assert res[0].cpu().tolist() == ref[0].cpu().tolist()


@pytest.mark.foreach_pow_list
def test_foreach_pow_list_rejects_bool():
    """``pow.List`` with a bool tensor operand is refused, matching ATen."""
    device = flag_gems.device
    inp = [torch.tensor([True, False], dtype=torch.bool, device=device)]
    with pytest.raises(NotImplementedError):
        _gems("_foreach_pow.List")(inp, [torch.tensor([True, True], device=device)])
    with pytest.raises(NotImplementedError):
        torch._foreach_pow([t.cpu() for t in inp], [torch.tensor([True, True])])


@pytest.mark.foreach_clamp_max_scalar
@pytest.mark.foreach_clamp_min_scalar
@pytest.mark.foreach_maximum_scalar
@pytest.mark.foreach_minimum_scalar
@pytest.mark.parametrize("core", ["clamp_max", "clamp_min", "maximum", "minimum"])
def test_foreach_clamp_scalar_rejects_bool(core):
    """The clamp family accepts a bool *tensor* operand but not a bool scalar."""
    device = flag_gems.device
    inp = [torch.tensor([True, False], dtype=torch.bool, device=device)]
    with pytest.raises(RuntimeError):
        _gems(f"_foreach_{core}.Scalar")(inp, True)
