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

from .accuracy_utils import gems_assert_close, gems_assert_equal, to_reference

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


def _torch_op(name, inplace=False):
    return getattr(torch, f"_foreach_{name}_" if inplace else f"_foreach_{name}")


def _supports(name, dtype):
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


@pytest.mark.foreach_unary
@pytest.mark.parametrize("name", UNARY_NAMES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_foreach_unary_float(name, dtype):
    if not _supports(name, dtype):
        return
    inp = [_sample((16, 8), dtype, flag_gems.device) for _ in range(3)]
    ref_inp = [to_reference(t) for t in inp]

    ref_out = _torch_op(name)(ref_inp)
    with flag_gems.use_gems():
        res_out = _torch_op(name)(inp)

    _assert_lists_close(res_out, ref_out, name)


@pytest.mark.foreach_unary
@pytest.mark.parametrize("name", UNARY_NAMES)
@pytest.mark.parametrize("dtype", INT_DTYPES)
def test_accuracy_foreach_unary_int(name, dtype):
    if not _supports(name, dtype):
        return
    inp = [_sample((32,), dtype, flag_gems.device, name) for _ in range(3)]
    ref_inp = [to_reference(t) for t in inp]

    ref_out = _torch_op(name)(ref_inp)
    with flag_gems.use_gems():
        res_out = _torch_op(name)(inp)

    # The promotion policy is part of the contract: a transcendental returns
    # float32 for an integral input, while ceil/floor/round/trunc/sign/neg keep
    # the input dtype.
    _assert_lists_close(res_out, ref_out, name)


@pytest.mark.foreach_unary
@pytest.mark.parametrize("name", UNARY_NAMES)
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
    with flag_gems.use_gems():
        res_out = _torch_op(name)(inp)

    _assert_lists_close(res_out, ref_out, name)


@pytest.mark.foreach_unary
@pytest.mark.parametrize("name", UNARY_NAMES)
@pytest.mark.parametrize("dtype", [torch.complex64])
def test_accuracy_foreach_unary_complex(name, dtype):
    if not _supports(name, dtype) or UNARY_OPS[name].complex_fn is None:
        return
    inp = [_sample((16,), dtype, flag_gems.device) for _ in range(2)]
    ref_inp = [to_reference(t) for t in inp]

    ref_out = _torch_op(name)(ref_inp)
    with flag_gems.use_gems():
        res_out = _torch_op(name)(inp)

    _assert_lists_close(res_out, ref_out, name)


@pytest.mark.foreach_unary
@pytest.mark.parametrize("name", UNARY_NAMES)
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
            with flag_gems.use_gems():
                _torch_op(name)(inp)


@pytest.mark.foreach_unary
@pytest.mark.parametrize("name", REPRESENTATIVE)
@pytest.mark.parametrize("length", TENSOR_LIST_LENGTHS)
def test_accuracy_foreach_unary_list_lengths(name, length):
    inp = [_sample((16, 8), torch.float32, flag_gems.device) for _ in range(length)]
    ref_inp = [to_reference(t) for t in inp]

    ref_out = _torch_op(name)(ref_inp)
    with flag_gems.use_gems():
        res_out = _torch_op(name)(inp)

    _assert_lists_close(res_out, ref_out, name)


@pytest.mark.foreach_unary
@pytest.mark.parametrize("name", REPRESENTATIVE)
def test_accuracy_foreach_unary_mixed_shapes(name):
    inp = [_sample(s, torch.float32, flag_gems.device) for s in MIXED_SHAPES]
    ref_inp = [to_reference(t) for t in inp]

    ref_out = _torch_op(name)(ref_inp)
    with flag_gems.use_gems():
        res_out = _torch_op(name)(inp)

    _assert_lists_close(res_out, ref_out, name)
    for got, want in zip(res_out, ref_out):
        assert got.shape == want.shape


@pytest.mark.foreach_unary
@pytest.mark.parametrize("name", REPRESENTATIVE)
def test_accuracy_foreach_unary_empty_and_zero_dim(name):
    inp = [
        torch.empty(0, dtype=torch.float32, device=flag_gems.device),
        _sample((), torch.float32, flag_gems.device),
        _sample((3,), torch.float32, flag_gems.device),
    ]
    ref_inp = [to_reference(t) for t in inp]

    ref_out = _torch_op(name)(ref_inp)
    with flag_gems.use_gems():
        res_out = _torch_op(name)(inp)

    _assert_lists_close(res_out, ref_out, name)


@pytest.mark.foreach_unary
@pytest.mark.parametrize("name", REPRESENTATIVE)
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
    with flag_gems.use_gems():
        res_out = _torch_op(name)(inp)

    _assert_lists_close(res_out, ref_out, name)
    # Results must come back in input order, not grouped order.
    assert [t.dtype for t in res_out] == [t.dtype for t in ref_out]


@pytest.mark.foreach_unary
@pytest.mark.parametrize("name", REPRESENTATIVE)
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
    with flag_gems.use_gems():
        res_out = _torch_op(name)(inp)

    _assert_lists_close(res_out, ref_out, name)


@pytest.mark.foreach_unary
@pytest.mark.parametrize("name", UNARY_NAMES)
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
    with flag_gems.use_gems():
        ret = _torch_op(name, inplace=True)(inp)

    # ``torch._foreach_sin_`` and friends return the mutated list at the Python
    # binding layer, so the return value is checked on the raw ATen overload --
    # that is the one whose schema says ``-> ()`` and the one a wrapper returning
    # a list would make the dispatcher reject.
    del ret
    _assert_lists_close(inp, ref_inp, name)

    probe = [_sample((4,), dtype, flag_gems.device, name)]
    with flag_gems.use_gems():
        assert getattr(torch.ops.aten, f"_foreach_{name}_")(probe) is None


@pytest.mark.foreach_unary
@pytest.mark.parametrize("name", REPRESENTATIVE)
def test_accuracy_foreach_unary_inplace_writes_through_views(name):
    """In-place on a non-flat view must land in the original storage."""
    storage = _sample((8, 16), torch.float32, flag_gems.device, name)
    ref_storage = to_reference(storage.clone())

    _torch_op(name, inplace=True)([ref_storage[:, ::2]])
    with flag_gems.use_gems():
        _torch_op(name, inplace=True)([storage[:, ::2]])

    # Both the written half and the untouched half must match.
    gems_assert_close(storage, ref_storage, storage.dtype)


@pytest.mark.foreach_unary
def test_accuracy_foreach_unary_inplace_rejects_overlap():
    """``expand`` aliases one element many times; ATen refuses to write it."""
    src = torch.randn(1, 8, device=flag_gems.device).expand(4, 8)
    with pytest.raises(RuntimeError):
        with flag_gems.use_gems():
            torch._foreach_neg_([src])


@pytest.mark.foreach_unary
@pytest.mark.parametrize("name", ["sin", "exp", "log", "sigmoid"])
def test_accuracy_foreach_unary_inplace_rejects_promotion(name):
    """A promoting operator may not narrow its result back into an int input."""
    inp = [torch.randint(1, 5, (8,), dtype=torch.int64, device=flag_gems.device)]
    with pytest.raises(RuntimeError):
        with flag_gems.use_gems():
            _torch_op(name, inplace=True)(inp)


@pytest.mark.foreach_unary
@pytest.mark.parametrize("name", REPRESENTATIVE)
def test_accuracy_foreach_unary_error_parity(name):
    """Empty list and non-tensor entries must fail the way ATen fails."""
    with pytest.raises(RuntimeError):
        with flag_gems.use_gems():
            _torch_op(name)([])
    with pytest.raises((TypeError, RuntimeError)):
        with flag_gems.use_gems():
            _torch_op(name)([1.0])


@pytest.mark.foreach_unary
@pytest.mark.parametrize("name", REPRESENTATIVE)
def test_accuracy_foreach_unary_out_overload(name):
    """``.out`` is Composite and is deliberately not registered.

    It decomposes to the functional form, so the FlagGems kernel still produces
    the values; this test pins that forwarding rather than a registration.
    """
    inp = [_sample((16,), torch.float32, flag_gems.device) for _ in range(2)]
    ref_inp = [to_reference(t) for t in inp]
    ref_out = _torch_op(name)(ref_inp)

    out = [torch.empty_like(t) for t in inp]
    with flag_gems.use_gems():
        getattr(torch.ops.aten, f"_foreach_{name}").out(inp, out=out)

    _assert_lists_close(out, ref_out, name)


@pytest.mark.foreach_unary
@pytest.mark.parametrize("name", UNARY_NAMES)
@pytest.mark.parametrize("inplace", [False, True])
def test_foreach_unary_registration_is_live(name, inplace, caplog):
    """Falsifiable liveness: the negative control must stay silent.

    A registration written with the wrong ATen key raises no error and lets
    every accuracy test pass, so each key is probed directly: the call must emit
    the operator's debug record under ``use_gems`` and must not emit it outside.
    """
    if flag_gems.device != "cuda":
        return
    inp = [_sample((8,), torch.float32, flag_gems.device)]
    want = f"GEMS _FOREACH_{name.upper()}{'_' if inplace else ''}"
    logger_name = "flag_gems.ops._foreach_unary"

    with caplog.at_level(logging.DEBUG, logger=logger_name):
        caplog.clear()
        _torch_op(name, inplace)([t.clone() for t in inp])
        assert want not in caplog.text, "negative control fired: probe proves nothing"

        caplog.clear()
        with flag_gems.use_gems():
            _torch_op(name, inplace)([t.clone() for t in inp])
        assert want in caplog.text, f"dead registration for {want}"


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


@pytest.mark.foreach_unary
def test_foreach_unary_launch_count_is_independent_of_list_length():
    """The pointwise bound is O(number of groups), never O(number of tensors).

    The threshold is derived from the single-group case measured in the same
    run rather than hardcoded, so the test states the invariant instead of a
    magic number.
    """
    if flag_gems.device != "cuda":
        return

    def run(n):
        ts = [torch.rand(1024, device=flag_gems.device) + 0.1 for _ in range(n)]
        with flag_gems.use_gems():
            return _cuda_kernel_count(lambda: torch._foreach_sin(ts))

    baseline = run(1)
    for n in (16, 64, 128, 256):
        count = run(n)
        assert count <= baseline + 1, (
            f"launch count grew with list length: N=1 -> {baseline}, "
            f"N={n} -> {count}; a per-tensor loop would give ~{n}"
        )


@pytest.mark.foreach_unary
def test_foreach_unary_launch_count_scales_with_dtype_groups():
    """Launches track ``(device, dtype)`` groups, which is the real bound."""
    if flag_gems.device != "cuda":
        return
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
        with flag_gems.use_gems():
            torch._foreach_exp(ts)
        stats = launch_stats()
        assert stats["groups"] == len(dtypes)
        assert stats["executor_launches"] == len(dtypes)


@pytest.mark.foreach_unary
def test_foreach_unary_launch_count_with_heterogeneous_numel():
    """Wildly different element counts still collapse into one launch."""
    if flag_gems.device != "cuda":
        return
    from flag_gems.utils.foreach import launch_stats

    ts = [torch.rand(n, device=flag_gems.device) + 0.1 for n in (1000, 7, 65537, 4096)]
    with flag_gems.use_gems():
        torch._foreach_log(ts)
    assert launch_stats()["executor_launches"] == 1
