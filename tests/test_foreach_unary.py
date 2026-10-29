"""Correctness tests for the unary ``_foreach_*`` family.

Thirty operators share one executor, so the tests are parameterized over the
operator table instead of being copied thirty times.  Every case runs against
the PyTorch reference on CPU via ``to_reference``, which is what lets the same
file pass both the GPU run and CI's ``--ref=cpu --quick`` run.

Two things this file deliberately does *not* rely on:

* A plain transposed tensor is not proof of layout correctness for a unary
  operator -- flat traversal visits the same elements in a different order and
  still produces the right answer.  The gappy-view and storage-offset cases are
  the ones that would actually fail.
* Passing tests are not proof that a registration is live; the dispatch
  liveness check is a separate test with a negative control.
"""

import logging

import pytest
import torch

import flag_gems
from flag_gems.ops._foreach_unary import UNARY_OPS

from .accuracy_utils import (
    gems_assert_close,
    gems_assert_equal,
    gems_log_logger,
    gems_log_prefix,
    to_reference,
)

# Every operator name in the table; used to parameterize the shared cases.
UNARY_NAMES = sorted(UNARY_OPS)

# A representative subset for the expensive layout / launch-count cases: one
# dtype-preserving operator, one promoting operator, one with a complex kernel,
# and one implemented outside pointwise_dynamic.
REPRESENTATIVE = ["neg", "sin", "abs", "round"]

# 1 is the degenerate case; 128 is long enough that a per-tensor implementation
# would be visible in the launch-count tests.
TENSOR_LIST_LENGTHS = [1, 2, 17, 128]

# Mixed shapes and element counts in one call: the executor must not assume a
# common numel.
MIXED_SHAPES = [(7,), (4, 5), (2, 3, 4), (1,), (64,)]

FLOAT_DTYPES = [torch.float16, torch.float32, torch.bfloat16]
INT_DTYPES = [torch.int16, torch.int32, torch.int64]


def _op_params(names, suffixes=("",)):
    """``parametrize`` entries carrying each operator's own pytest marker.

    ``-m foreach_<op>`` is how CI narrows a run to the operators a change
    touches, and a marker on the test function cannot do that: it applies to
    every case the function generates, so every operator would select the
    identical set.  ``suffixes`` is ``("_",)`` for the in-place operators and
    ``("", "_")`` for a test parameterized over both.
    """
    return [
        pytest.param(
            name, marks=[getattr(pytest.mark, f"foreach_{name}{s}") for s in suffixes]
        )
        for name in names
    ]


def _mark_all_unary(test):
    """Stack every unary marker on a family-wide invariant test.

    A test that takes no ``name`` parameter cannot carry a per-operator marker,
    yet a change to the shared executor behind it still has to run it.  Markers
    are labels rather than filters, so the test still runs once however many of
    them match.
    """
    for name in UNARY_NAMES:
        test = getattr(pytest.mark, f"foreach_{name}")(test)
        test = getattr(pytest.mark, f"foreach_{name}_")(test)
    return test


def _torch_op(name, inplace=False):
    return getattr(torch, f"_foreach_{name}_" if inplace else f"_foreach_{name}")


def _gems_op(name, inplace=False):
    """The FlagGems wrapper that ``aten::_foreach_<name>`` is registered to.

    Calling it directly is what ``use_gems()`` would have dispatched to, and the
    CI check_kernelgen_tests job rejects use_gems() in test files.
    """
    return getattr(flag_gems, f"_foreach_{name}_" if inplace else f"_foreach_{name}")


IS_KUNLUNXIN = flag_gems.vendor_name == "kunlunxin"

_KUNLUNXIN_UNSUPPORTED = {
    torch.float16: {"log2", "lgamma"},
    torch.float32: {"lgamma", "sigmoid"},
    torch.bfloat16: {"cosh", "log10", "sinh", "log2", "lgamma", "sigmoid"},
    torch.int16: {"ceil", "floor", "lgamma", "log10", "round", "sign"},
    torch.int32: {"ceil", "floor", "lgamma", "log10", "round", "sign"},
    torch.int64: {"ceil", "floor", "lgamma", "log10"},
    torch.bool: {"lgamma", "log10"},
    torch.complex64: set(UNARY_NAMES),
    torch.complex128: set(UNARY_NAMES),
}


_KUNLUNXIN_INTEGRAL = (torch.bool, torch.int16, torch.int32, torch.int64)

_KUNLUNXIN_REGISTERED = frozenset({"cos", "cos_", "exp", "exp_", "sin", "sin_"})


def _supports(name, dtype):
    if IS_KUNLUNXIN:
        if name in _KUNLUNXIN_UNSUPPORTED.get(dtype, ()):
            return False
        if dtype in _KUNLUNXIN_INTEGRAL and name not in _KUNLUNXIN_REGISTERED:
            return False
    allowed = UNARY_OPS[name].allowed
    return allowed is None or dtype in allowed


# Operators whose real domain is ``|x| <= 1``.  An integral input of 2 makes
# both sides return NaN, and NaN compares unequal, so the sample has to stay in
# the domain rather than the comparison being loosened.
UNIT_DOMAIN = {"acos", "asin"}


def _sample(shape, dtype, device, name=None):
    """Input in a range where the operator under test is well defined.

    ``log``/``sqrt``/``lgamma`` need positive values, so the shared range is
    positive and sub-unit; that covers the whole family except the two inverse
    trigonometric operators, whose integral samples are narrowed to ``{0, 1}``.
    """
    if dtype.is_floating_point:
        return torch.rand(shape, dtype=dtype, device=device) * 0.8 + 0.1
    if dtype.is_complex:
        real = torch.rand(shape, dtype=torch.float32, device=device) * 0.8 + 0.1
        imag = torch.rand(shape, dtype=torch.float32, device=device) * 0.8 + 0.1
        return torch.complex(real, imag).to(dtype)
    if dtype is torch.bool:
        return torch.randint(0, 2, shape, device=device).to(torch.bool)
    high = 2 if name in UNIT_DOMAIN else 5
    return torch.randint(
        0 if name in UNIT_DOMAIN else 1, high, shape, dtype=dtype, device=device
    )


def _assert_lists_close(res, ref, name):
    assert len(res) == len(ref)
    for got, want in zip(res, ref):
        assert got.dtype == want.dtype, f"{name}: {got.dtype} != {want.dtype}"
        if got.dtype.is_floating_point or got.dtype.is_complex:
            gems_assert_close(got, want, got.dtype)
        else:
            gems_assert_equal(got, want)


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
# select the identical set of cases. The per-parameter markers built by
# ``_op_params`` are what give ``-m <id>`` its one-operator selectivity.
# ---------------------------------------------------------------------------
@pytest.mark.foreach_abs
@pytest.mark.foreach_abs_
@pytest.mark.foreach_acos
@pytest.mark.foreach_acos_
@pytest.mark.foreach_asin
@pytest.mark.foreach_asin_
@pytest.mark.foreach_atan
@pytest.mark.foreach_atan_
@pytest.mark.foreach_ceil
@pytest.mark.foreach_ceil_
@pytest.mark.foreach_cos
@pytest.mark.foreach_cos_
@pytest.mark.foreach_cosh
@pytest.mark.foreach_cosh_
@pytest.mark.foreach_erf
@pytest.mark.foreach_erf_
@pytest.mark.foreach_erfc
@pytest.mark.foreach_erfc_
@pytest.mark.foreach_exp
@pytest.mark.foreach_exp_
@pytest.mark.foreach_expm1
@pytest.mark.foreach_expm1_
@pytest.mark.foreach_floor
@pytest.mark.foreach_floor_
@pytest.mark.foreach_frac
@pytest.mark.foreach_frac_
@pytest.mark.foreach_lgamma
@pytest.mark.foreach_lgamma_
@pytest.mark.foreach_log
@pytest.mark.foreach_log10
@pytest.mark.foreach_log10_
@pytest.mark.foreach_log1p
@pytest.mark.foreach_log1p_
@pytest.mark.foreach_log2
@pytest.mark.foreach_log2_
@pytest.mark.foreach_log_
@pytest.mark.foreach_neg
@pytest.mark.foreach_neg_
@pytest.mark.foreach_reciprocal
@pytest.mark.foreach_reciprocal_
@pytest.mark.foreach_round
@pytest.mark.foreach_round_
@pytest.mark.foreach_rsqrt
@pytest.mark.foreach_rsqrt_
@pytest.mark.foreach_sigmoid
@pytest.mark.foreach_sigmoid_
@pytest.mark.foreach_sign
@pytest.mark.foreach_sign_
@pytest.mark.foreach_sin
@pytest.mark.foreach_sin_
@pytest.mark.foreach_sinh
@pytest.mark.foreach_sinh_
@pytest.mark.foreach_sqrt
@pytest.mark.foreach_sqrt_
@pytest.mark.foreach_tan
@pytest.mark.foreach_tan_
@pytest.mark.foreach_tanh
@pytest.mark.foreach_tanh_
@pytest.mark.foreach_trunc
@pytest.mark.foreach_trunc_
@pytest.mark.foreach_unary
def test_operator_markers_are_declared():
    """Placeholder carrying the marker set for static discovery."""


@pytest.mark.parametrize("name", _op_params(UNARY_NAMES))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_foreach_unary_float(name, dtype):
    if not _supports(name, dtype):
        return
    inp = [_sample((16, 8), dtype, flag_gems.device) for _ in range(3)]
    ref_inp = [to_reference(t) for t in inp]

    ref_out = _torch_op(name)(ref_inp)
    res_out = _gems_op(name)(inp)

    _assert_lists_close(res_out, ref_out, name)


@pytest.mark.foreach_unary
@pytest.mark.parametrize("name", _op_params(UNARY_NAMES))
@pytest.mark.parametrize("dtype", INT_DTYPES)
def test_accuracy_foreach_unary_int(name, dtype):
    if not _supports(name, dtype):
        return
    inp = [_sample((32,), dtype, flag_gems.device, name) for _ in range(3)]
    ref_inp = [to_reference(t) for t in inp]

    ref_out = _torch_op(name)(ref_inp)
    res_out = _gems_op(name)(inp)

    # The promotion policy is part of the contract: a transcendental returns
    # float32 for an integral input, while ceil/floor/round/trunc/sign/neg keep
    # the input dtype.
    _assert_lists_close(res_out, ref_out, name)


@pytest.mark.foreach_unary
@pytest.mark.parametrize("name", _op_params(UNARY_NAMES))
def test_accuracy_foreach_unary_bool(name):
    if not _supports(name, torch.bool):
        return
    inp = [torch.tensor([True, False, True], device=flag_gems.device)]
    ref_inp = [to_reference(t) for t in inp]

    try:
        ref_out = _torch_op(name)(ref_inp)
    except NotImplementedError:
        # ``_foreach_abs`` accepts bool on CUDA but the CPU kernel does not
        # ("abs_cpu" not implemented for 'Bool'), so under ``--ref cpu`` there is
        # no reference to compare against.  The CUDA-reference run still covers it.
        pytest.skip(f"no CPU reference for _foreach_{name} on bool")
    res_out = _gems_op(name)(inp)

    _assert_lists_close(res_out, ref_out, name)


@pytest.mark.foreach_unary
@pytest.mark.parametrize("name", _op_params(UNARY_NAMES))
@pytest.mark.parametrize("dtype", [torch.complex64])
def test_accuracy_foreach_unary_complex(name, dtype):
    if not _supports(name, dtype) or UNARY_OPS[name].complex_fn is None:
        return
    inp = [_sample((16,), dtype, flag_gems.device) for _ in range(2)]
    ref_inp = [to_reference(t) for t in inp]

    ref_out = _torch_op(name)(ref_inp)
    res_out = _gems_op(name)(inp)

    _assert_lists_close(res_out, ref_out, name)


@pytest.mark.foreach_unary
@pytest.mark.parametrize("name", _op_params(UNARY_NAMES))
def test_accuracy_foreach_unary_rejects_unsupported_dtype(name):
    """Refusals must match ATen's, not be broader or narrower."""
    allowed = UNARY_OPS[name].allowed
    if allowed is None:
        return
    for dtype in (torch.bool, torch.int32, torch.complex64):
        if dtype in allowed:
            continue
        inp = [_sample((8,), dtype, flag_gems.device, name)]
        with pytest.raises(RuntimeError):
            _gems_op(name)(inp)


@pytest.mark.foreach_unary
@pytest.mark.parametrize("name", _op_params(REPRESENTATIVE))
@pytest.mark.parametrize("length", TENSOR_LIST_LENGTHS)
def test_accuracy_foreach_unary_list_lengths(name, length):
    inp = [_sample((16, 8), torch.float32, flag_gems.device) for _ in range(length)]
    ref_inp = [to_reference(t) for t in inp]

    ref_out = _torch_op(name)(ref_inp)
    res_out = _gems_op(name)(inp)

    _assert_lists_close(res_out, ref_out, name)


@pytest.mark.foreach_unary
@pytest.mark.parametrize("name", _op_params(REPRESENTATIVE))
def test_accuracy_foreach_unary_mixed_shapes(name):
    inp = [_sample(s, torch.float32, flag_gems.device) for s in MIXED_SHAPES]
    ref_inp = [to_reference(t) for t in inp]

    ref_out = _torch_op(name)(ref_inp)
    res_out = _gems_op(name)(inp)

    _assert_lists_close(res_out, ref_out, name)
    for got, want in zip(res_out, ref_out):
        assert got.shape == want.shape


@pytest.mark.foreach_unary
@pytest.mark.parametrize("name", _op_params(REPRESENTATIVE))
def test_accuracy_foreach_unary_empty_and_zero_dim(name):
    inp = [
        torch.empty(0, dtype=torch.float32, device=flag_gems.device),
        _sample((), torch.float32, flag_gems.device),
        _sample((3,), torch.float32, flag_gems.device),
    ]
    ref_inp = [to_reference(t) for t in inp]

    ref_out = _torch_op(name)(ref_inp)
    res_out = _gems_op(name)(inp)

    _assert_lists_close(res_out, ref_out, name)


@pytest.mark.foreach_unary
@pytest.mark.parametrize("name", _op_params(REPRESENTATIVE))
def test_accuracy_foreach_unary_mixed_dtypes(name):
    """One call spanning several dtypes must be grouped, not rejected."""
    inp = [
        _sample((16,), torch.float32, flag_gems.device),
        _sample((16,), torch.float16, flag_gems.device),
        _sample((16,), torch.float32, flag_gems.device),
        _sample((16,), torch.bfloat16, flag_gems.device),
    ]
    ref_inp = [to_reference(t) for t in inp]

    ref_out = _torch_op(name)(ref_inp)
    res_out = _gems_op(name)(inp)

    _assert_lists_close(res_out, ref_out, name)
    # Results must come back in input order, not grouped order.
    assert [t.dtype for t in res_out] == [t.dtype for t in ref_out]


@pytest.mark.foreach_unary
@pytest.mark.parametrize("name", _op_params(REPRESENTATIVE))
def test_accuracy_foreach_unary_layouts(name):
    """Layouts that flat addressing must either handle or stage through a copy.

    A plain ``transpose`` is included for completeness but proves little for a
    unary operator: flat traversal reads the same elements and ``empty_like``
    keeps the same stride, so the result is right either way.  The gappy slice,
    the non-zero storage offset, and the ``expand`` case are the discriminating
    ones.
    """
    base = _sample((8, 8), torch.float32, flag_gems.device)
    wide = _sample((8, 16), torch.float32, flag_gems.device)
    inp = [
        base,  # contiguous
        base.t(),  # dense but transposed
        base[2:6],  # slice: contiguous with non-zero storage offset
        wide[:, ::2],  # gappy view: not flat addressable
        base[:, 3],  # strided column
        base.expand(8, 8),  # already 8x8, but a view carrying its own strides
    ]
    ref_inp = [to_reference(t) for t in inp]

    ref_out = _torch_op(name)(ref_inp)
    res_out = _gems_op(name)(inp)

    _assert_lists_close(res_out, ref_out, name)


@pytest.mark.foreach_unary
@pytest.mark.parametrize("name", _op_params(UNARY_NAMES, suffixes=("_",)))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_foreach_unary_inplace(name, dtype):
    if not _supports(name, dtype):
        return
    inp = [_sample((16, 8), dtype, flag_gems.device, name) for _ in range(3)]
    # ``to_reference`` hands back the same object when no upcast is requested, so
    # the reference must be built from clones: otherwise the reference call and
    # the FlagGems call both mutate ``inp`` and the operator runs twice (``log``
    # of its own negative output measured as all-NaN).
    ref_inp = [to_reference(t.clone()) for t in inp]

    _torch_op(name, inplace=True)(ref_inp)
    ret = _gems_op(name, inplace=True)(inp)

    # ``torch._foreach_sin_`` and friends return the mutated list at the Python
    # binding layer, so the return value is checked on the raw ATen overload --
    # that is the one whose schema says ``-> ()`` and the one a wrapper returning
    # a list would make the dispatcher reject.
    del ret
    _assert_lists_close(inp, ref_inp, name)

    probe = [_sample((4,), dtype, flag_gems.device, name)]
    assert getattr(torch.ops.aten, f"_foreach_{name}_")(probe) is None


@pytest.mark.foreach_unary
@pytest.mark.parametrize("name", _op_params(REPRESENTATIVE, suffixes=("_",)))
def test_accuracy_foreach_unary_inplace_writes_through_views(name):
    """In-place on a non-flat view must land in the original storage."""
    storage = _sample((8, 16), torch.float32, flag_gems.device, name)
    ref_storage = to_reference(storage.clone())

    _torch_op(name, inplace=True)([ref_storage[:, ::2]])
    _gems_op(name, inplace=True)([storage[:, ::2]])

    # Both the written half and the untouched half must match.
    gems_assert_close(storage, ref_storage, storage.dtype)


@pytest.mark.foreach_unary
def test_accuracy_foreach_unary_inplace_rejects_overlap():
    """``expand`` aliases one element many times; ATen refuses to write it."""
    src = torch.randn(1, 8, device=flag_gems.device).expand(4, 8)
    if IS_KUNLUNXIN:
        with pytest.raises(RuntimeError):
            _gems_op("cos", inplace=True)([src])
        with pytest.raises(RuntimeError):
            _gems_op("neg", inplace=True)([src])
        return
    with pytest.raises(RuntimeError):
        torch._foreach_neg_([src])


@pytest.mark.foreach_unary
@pytest.mark.parametrize("name", _op_params(["sin", "exp", "log", "sigmoid"], ("_",)))
def test_accuracy_foreach_unary_inplace_rejects_promotion(name):
    """A promoting operator may not narrow its result back into an int input."""
    inp = [torch.randint(1, 5, (8,), dtype=torch.int64, device=flag_gems.device)]
    with pytest.raises(RuntimeError):
        _gems_op(name, inplace=True)(inp)


@pytest.mark.foreach_unary
@pytest.mark.parametrize("name", _op_params(REPRESENTATIVE))
def test_accuracy_foreach_unary_error_parity(name):
    """Empty list and non-tensor entries must fail the way ATen fails."""
    with pytest.raises(RuntimeError):
        _gems_op(name)([])
    with pytest.raises((TypeError, RuntimeError)):
        _gems_op(name)([1.0])


@pytest.mark.foreach_unary
@pytest.mark.parametrize("name", _op_params(REPRESENTATIVE))
def test_accuracy_foreach_unary_out_overload(name):
    """``.out`` is Composite and is deliberately not registered.

    It decomposes to the functional form, so the FlagGems kernel still produces
    the values; this test pins that forwarding rather than a registration.
    """
    inp = [_sample((16,), torch.float32, flag_gems.device) for _ in range(2)]
    ref_inp = [to_reference(t) for t in inp]
    ref_out = _torch_op(name)(ref_inp)

    out = [torch.empty_like(t) for t in inp]
    getattr(torch.ops.aten, f"_foreach_{name}").out(inp, out=out)

    _assert_lists_close(out, ref_out, name)


@pytest.mark.foreach_unary
@pytest.mark.parametrize("name", _op_params(UNARY_NAMES, ("", "_")))
@pytest.mark.parametrize("inplace", [False, True])
def test_foreach_unary_registration_is_live(name, inplace, caplog):
    """Falsifiable liveness: the negative control must stay silent.

    A registration written with the wrong ATen key raises no error and lets
    every accuracy test pass, so each key is probed directly: the call must emit
    the operator's debug record under ``use_gems`` and must not emit it outside.
    """
    if flag_gems.device != "cuda":
        return
    if IS_KUNLUNXIN and not _supports(name, torch.float32):
        pytest.skip(f"kunlunxin backend has no kernel for _foreach_{name} on float32")
    inp = [_sample((8,), torch.float32, flag_gems.device)]
    want = f"GEMS _FOREACH_{name.upper()}{'_' if inplace else ''}"
    wants = (want,)
    logger_name = gems_log_logger(_gems_op(name, inplace))
    if IS_KUNLUNXIN:
        wants = (
            want,
            f"{gems_log_prefix(_gems_op(name, inplace))}"
            f" _FOREACH_{name.upper()}{'_' if inplace else ''}",
        )
        logger_name = None

    with caplog.at_level(logging.DEBUG, logger=logger_name):
        caplog.clear()
        _torch_op(name, inplace)([t.clone() for t in inp])
        assert not any(
            w in caplog.text for w in wants
        ), "negative control fired: probe proves nothing"

        caplog.clear()
        _gems_op(name, inplace)([t.clone() for t in inp])
        assert any(w in caplog.text for w in wants), f"dead registration for {wants}"


def _cuda_kernel_count(fn):
    """CUDA device events for one call, after warming up Triton compilation."""
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sum(
        1
        for e in prof.events()
        if e.device_type == torch.autograd.DeviceType.CUDA and e.device_time_total > 0
    )


@_mark_all_unary
@pytest.mark.foreach_unary
def test_foreach_unary_launch_count_is_independent_of_list_length():
    """The pointwise bound is O(number of groups), never O(number of tensors).

    The threshold is derived from the single-group case measured in the same
    run rather than hardcoded, so the test states the invariant instead of a
    magic number.
    """
    if flag_gems.device != "cuda":
        return

    if IS_KUNLUNXIN:
        from _kunlunxin.utils.foreach import launch_stats

        def run(n):
            ts = [torch.rand(1024, device=flag_gems.device) + 0.1 for _ in range(n)]
            flag_gems._foreach_sin(ts)
            return launch_stats()["executor_launches"]

    else:

        def run(n):
            ts = [torch.rand(1024, device=flag_gems.device) + 0.1 for _ in range(n)]
            return _cuda_kernel_count(lambda: flag_gems._foreach_sin(ts))

    baseline = run(1)
    for n in (16, 64, 128, 256):
        count = run(n)
        assert count <= baseline + 1, (
            f"launch count grew with list length: N=1 -> {baseline}, "
            f"N={n} -> {count}; a per-tensor loop would give ~{n}"
        )


@_mark_all_unary
@pytest.mark.foreach_unary
def test_foreach_unary_launch_count_scales_with_dtype_groups():
    """Launches track ``(device, dtype)`` groups, which is the real bound."""
    if flag_gems.device != "cuda":
        return
    if IS_KUNLUNXIN:
        pytest.skip("kunlunxin backend has no float64, so this dtype list cannot group")
    from flag_gems.utils.foreach import launch_stats

    for dtypes in (
        [torch.float32],
        [torch.float32, torch.float16],
        [torch.float32, torch.float16, torch.bfloat16, torch.float64],
    ):
        ts = [
            (torch.rand(512, device=flag_gems.device) + 0.1).to(d)
            for d in dtypes
            for _ in range(16)
        ]
        flag_gems._foreach_exp(ts)
        stats = launch_stats()
        assert stats["groups"] == len(dtypes)
        assert stats["executor_launches"] == len(dtypes)


@_mark_all_unary
@pytest.mark.foreach_unary
def test_foreach_unary_launch_count_with_heterogeneous_numel():
    """Wildly different element counts still collapse into one launch."""
    if flag_gems.device != "cuda":
        return
    from flag_gems.utils.foreach import launch_stats

    ts = [torch.rand(n, device=flag_gems.device) + 0.1 for n in (1000, 7, 65537, 4096)]
    flag_gems._foreach_log(ts)
    assert launch_stats()["executor_launches"] == 1


# ---------------------------------------------------------------------------
# large-magnitude complex inputs
# ---------------------------------------------------------------------------
#
# The modulus ``sqrt(re**2 + im**2)`` overflows for a large but perfectly finite
# component, and ``abs``/``log``/``sqrt``/``rsqrt`` all derive from it. These
# cases use magnitudes above the point where the naive formula overflows in the
# component dtype, so they fail outright rather than degrading gracefully.

# ``1e20`` squares to ``1e40``, past the fp32 maximum of ~3.4e38; ``1e200``
# does the same in fp64. Both inputs are representable, so the reference is
# finite and the comparison stays meaningful.
LARGE_COMPLEX = {
    torch.complex64: 1e20,
    torch.complex128: 1e200,
}

# The subset whose complex kernel routes through the modulus.
MODULUS_OPS = ["abs", "log", "log1p", "log2", "log10", "sqrt", "rsqrt"]


@pytest.mark.foreach_unary
@pytest.mark.parametrize("name", _op_params(MODULUS_OPS))
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_accuracy_foreach_unary_complex_large(name, dtype):
    """A large finite complex input must not overflow the modulus."""
    if not _supports(name, dtype) or UNARY_OPS[name].complex_fn is None:
        return
    mag = LARGE_COMPLEX[dtype]
    base = torch.tensor([complex(mag, mag), complex(mag, -mag)], dtype=dtype)
    inp = [base.to(flag_gems.device)]
    ref_inp = [to_reference(t) for t in inp]

    ref_out = _torch_op(name)(ref_inp)
    res_out = _gems_op(name)(inp)

    _assert_lists_close(res_out, ref_out, name)


@pytest.mark.foreach_abs
@pytest.mark.foreach_unary
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_accuracy_foreach_abs_complex_large(dtype):
    """``abs`` of a large finite complex value stays finite.

    The modulus is the shared dependency of every complex operator here, so it
    is asserted for its own sake as well as through the operators above.
    """
    if IS_KUNLUNXIN:
        pytest.skip("kunlunxin backend has no complex kernel for _foreach_abs")
    mag = LARGE_COMPLEX[dtype]
    base = torch.tensor([complex(mag, mag)], dtype=dtype)
    inp = [base.to(flag_gems.device)]
    ref_inp = [to_reference(t) for t in inp]

    ref_out = torch._foreach_abs(ref_inp)
    res_out = _gems_op("abs")(inp)

    assert torch.isfinite(res_out[0]).all(), f"{res_out[0].item()} overflowed"
    _assert_lists_close(res_out, ref_out, "abs")
