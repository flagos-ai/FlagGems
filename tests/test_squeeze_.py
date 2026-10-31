import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# squeeze_ is an in-place metadata operator: it removes dimensions of size 1 by
# rebinding the same tensor object, so the shapes below are chosen for the
# squeeze itself, not for elementwise work. The three overloads are covered by
# their own test functions and marker, each with its own argument shapes. ATen
# spells the overloads squeeze_.dim / squeeze_.dims; pytest marks cannot contain
# a dot, so the dot-free ids squeeze__dim / squeeze__dims stand in for them
# (same convention as all.dim -> all_dim), and edge-case helpers keep the
# canonical @pytest.mark.squeeze_ mark.
SQUEEZE_CASES = [
    # default overload: every size-1 dimension goes away in one call
    (),  # 0-d scalar, nothing to squeeze
    (1,),  # all size-1 dims -> 0-d
    (2, 1, 3),  # interior size-1 dim -> (2, 3)
    (1, 1, 5),  # leading size-1 dims -> (5,)
    (8, 1, 1),  # trailing size-1 dims -> (8,)
    (1, 1, 1),  # everything removed -> 0-d
    (1, 20, 320, 1, 15),  # interleaved size-1 dims -> (20, 320, 15)
    (1024, 1024),  # no size-1 dim, no-op path
]


# Default overload only: the fast-path group in quick mode.
QUICK_SQUEEZE_CASES = [
    (1, 2, 19, 1, 7),
    (2, 19, 7),
    (1,),
    (),
]


# The dim overload needs a valid dim, so it is parametrized separately.
SQUEEZE_DIM_CASES = [
    ((2, 1, 3), 1),
    ((2, 1, 3), -2),
    ((1, 20, 320, 1, 15), 0),
    ((1, 20, 320, 1, 15), -1),  # size is 15, so this stays a no-op
    ((2, 19, 7), 0),  # size 2, no-op
    ((1,), 0),
    ((), 0),  # 0-d accepts dim 0 and -1
    ((), -1),
]


# The dims overload covers list/tuple/empty/unsorted forms and no-op entries.
SQUEEZE_DIMS_CASES = [
    ((2, 1, 3), [1]),
    ((1, 20, 320, 1, 15), [0, 3]),
    ((1, 20, 320, 1, 15), [4, 0, 3]),  # unsorted
    ((1, 20, 320, 1, 15), (-5, -2)),  # tuple of negatives
    ((1, 20, 320, 1, 15), []),  # empty list, no-op
    ((2, 19, 7), [0, 1]),  # neither dim has size 1, no-op
    ((), [0]),
]


_QUICK_DIM_CASES = [((2, 1, 3), 1)]


_QUICK_DIMS_CASES = [((2, 19, 7), [0, 1])]


def _snapshot(tensor):
    return (
        tensor.untyped_storage().data_ptr(),
        tensor.data_ptr(),
        tensor.storage_offset(),
        tensor._version,
    )


def _assert_variant(shape, dtype, call_gems, call_ref):
    if dtype.is_floating_point:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(0, 100, shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_version = ref_inp._version
    ref_out = call_ref(ref_inp)
    ref_delta = ref_inp._version - ref_version

    before = _snapshot(inp)
    res_out = call_gems(inp)

    _assert_squeeze_inplace(res_out, inp, ref_out, ref_inp, before, ref_delta)


def _assert_squeeze_inplace(res_out, inp, ref_out, ref_inp, before, ref_delta):
    # The operator mutates the operand itself: it returns the same object, keeps
    # the same storage, and bumps the version counter exactly like aten does.
    storage_ptr, data_ptr, offset, version = before
    assert res_out is inp
    assert inp.shape == ref_inp.shape
    assert inp.stride() == ref_inp.stride()
    assert inp.storage_offset() == offset == ref_inp.storage_offset()
    assert inp.untyped_storage().data_ptr() == storage_ptr
    assert inp.data_ptr() == data_ptr
    assert inp._version - version == ref_delta
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.squeeze_
@pytest.mark.parametrize(
    "shape", (QUICK_SQUEEZE_CASES if utils.QUICK_MODE else SQUEEZE_CASES)
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_squeeze_(shape, dtype):
    _assert_variant(
        shape,
        dtype,
        lambda t: flag_gems.squeeze_(t),
        lambda t: torch.ops.aten.squeeze_(t),
    )


@pytest.mark.squeeze_
@pytest.mark.parametrize("dtype", utils.INT_DTYPES)
def test_squeeze__int_dtypes(dtype):
    # squeeze_ is dtype agnostic; the integer dtypes clinch that the metadata
    # path never round-trips through floating point.
    _assert_variant(
        (2, 1, 3),
        dtype,
        lambda t: flag_gems.squeeze_(t),
        lambda t: torch.ops.aten.squeeze_(t),
    )


@pytest.mark.squeeze_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_squeeze__non_contiguous(dtype):
    # A transposed input keeps its original strides; squeezing must reuse them
    # instead of compacting the result into a contiguous tensor.
    base = torch.randn((3, 1, 5), dtype=dtype, device=flag_gems.device)
    ref_base = utils.to_reference(base)
    inp = base.transpose(0, 2)
    ref_inp = ref_base.transpose(0, 2)

    ref_version = ref_inp._version
    ref_out = torch.ops.aten.squeeze_(ref_inp)
    ref_delta = ref_inp._version - ref_version

    before = _snapshot(inp)
    res_out = flag_gems.squeeze_(inp)

    _assert_squeeze_inplace(res_out, inp, ref_out, ref_inp, before, ref_delta)
    assert res_out.shape == (5, 3)
    assert not res_out.is_contiguous()


@pytest.mark.squeeze_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_squeeze__expanded_stride_zero(dtype):
    # An expanded dimension has stride 0; the squeezed view must keep it.
    row = torch.randn((1,), dtype=dtype, device=flag_gems.device)
    ref_row = utils.to_reference(row)
    inp = row.expand(5, 1)
    ref_inp = ref_row.expand(5, 1)

    ref_version = ref_inp._version
    ref_out = torch.ops.aten.squeeze_(ref_inp)
    ref_delta = ref_inp._version - ref_version

    before = _snapshot(inp)
    res_out = flag_gems.squeeze_(inp)

    _assert_squeeze_inplace(res_out, inp, ref_out, ref_inp, before, ref_delta)
    assert res_out.shape == (5,)
    assert res_out.stride() == (0,)


@pytest.mark.squeeze_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_squeeze__storage_offset(dtype):
    # Slicing leaves a nonzero storage offset that squeeze must preserve.
    base = torch.randn((4, 12), dtype=dtype, device=flag_gems.device)
    ref_base = utils.to_reference(base)
    inp = base[:, 3:9].view(4, 1, 6)
    ref_inp = ref_base[:, 3:9].view(4, 1, 6)

    ref_version = ref_inp._version
    ref_out = torch.ops.aten.squeeze_(ref_inp)
    ref_delta = ref_inp._version - ref_version

    before = _snapshot(inp)
    res_out = flag_gems.squeeze_(inp)

    _assert_squeeze_inplace(res_out, inp, ref_out, ref_inp, before, ref_delta)
    assert res_out.shape == (4, 6)
    assert res_out.storage_offset() == 3


# Negative cases are edge-case helpers, so they keep the canonical squeeze_
# mark instead of a per-overload one.


@pytest.mark.squeeze_
@pytest.mark.parametrize("dim", [5, -6])
def test_squeeze__rejects_out_of_range_dim(dim):
    inp = torch.randn((1, 2, 19, 7), device=flag_gems.device)
    with pytest.raises(IndexError):
        flag_gems.squeeze__dim(inp, dim)


@pytest.mark.squeeze_
@pytest.mark.parametrize("dims", [[9], [0, 9], [-6]])
def test_squeeze__rejects_out_of_range_dims(dims):
    inp = torch.randn((1, 2, 19, 7), device=flag_gems.device)
    with pytest.raises(IndexError):
        flag_gems.squeeze__dims(inp, dims)


@pytest.mark.squeeze_
@pytest.mark.parametrize("dims", [[0, 0], [1, 1, 0], [-4, 0]])
def test_squeeze__rejects_duplicate_dims(dims):
    inp = torch.randn((1, 2, 19, 7), device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.squeeze__dims(inp, dims)


@pytest.mark.squeeze_
def test_squeeze__rejects_inplace_on_grad_leaf():
    # Autograd forbids in-place mutation of a leaf; rebinding metadata must not
    # bypass that check.
    inp = torch.randn((1, 2, 19, 7), device=flag_gems.device).requires_grad_(True)
    with pytest.raises(RuntimeError):
        flag_gems.squeeze_(inp)


@pytest.mark.squeeze_
@pytest.mark.parametrize("shape", [(1, 5, 1), (1, 4, 1)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_squeeze__backward(shape, dtype):
    # The native op rejects a grad-requiring leaf, so it is applied in place to
    # a non-leaf clone and differentiated through the original leaf.
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    leaf = inp.detach().clone().requires_grad_(True)
    ref_leaf = ref_inp.detach().clone().requires_grad_(True)

    res_out = flag_gems.squeeze_(leaf.clone())
    ref_out = torch.ops.aten.squeeze_(ref_leaf.clone())
    utils.gems_assert_equal(res_out, ref_out)

    # Squeeze only relabels dimensions, so a non-uniform upstream is relayed
    # exactly.
    upstream = torch.randn(tuple(ref_out.shape), dtype=dtype, device=flag_gems.device)
    ref_upstream = utils.to_reference(upstream)
    (res_grad,) = torch.autograd.grad(res_out, leaf, grad_outputs=upstream)
    (ref_grad,) = torch.autograd.grad(ref_out, ref_leaf, grad_outputs=ref_upstream)

    assert res_grad.shape == ref_grad.shape
    utils.gems_assert_equal(res_grad, ref_grad)


@pytest.mark.squeeze__dim
@pytest.mark.parametrize(
    "shape,dim", (_QUICK_DIM_CASES if utils.QUICK_MODE else SQUEEZE_DIM_CASES)
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_squeeze__dim(shape, dim, dtype):
    _assert_variant(
        shape,
        dtype,
        lambda t: flag_gems.squeeze__dim(t, dim),
        lambda t: torch.ops.aten.squeeze_.dim(t, dim),
    )


@pytest.mark.squeeze__dims
@pytest.mark.parametrize(
    "shape,dims", (_QUICK_DIMS_CASES if utils.QUICK_MODE else SQUEEZE_DIMS_CASES)
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_squeeze__dims(shape, dims, dtype):
    _assert_variant(
        shape,
        dtype,
        lambda t: flag_gems.squeeze__dims(t, list(dims)),
        lambda t: torch.ops.aten.squeeze_.dims(t, list(dims)),
    )


# Negative cases are edge-case helpers, so they keep the canonical squeeze_
# mark instead of a per-overload one.
