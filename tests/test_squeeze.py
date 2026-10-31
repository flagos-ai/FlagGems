import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# squeeze is a metadata-only view (aten::squeeze(Tensor(a) self) -> Tensor(a)):
# it drops size-1 dims and aliases its input. The call forms exercised here are
# squeeze(x), squeeze(x, dim) and squeeze(x, [dim, ...]); there is no .out
# overload, and .dimname is uncallable in this build (named tensors raise
# 'NYI: Named tensors are currently unsupported in TorchScript').
#
# The spec shapes carry no size-1 extent, and a size-1 or empty dim is the only
# thing squeeze reacts to, so singleton/empty boundaries are added next to them.
_SINGLETON_SHAPES = [
    (),  # 0-d: nothing to drop
    (1,),  # all size-1 dims -> 0-d
    (1, 1),  # multiple leading size-1 dims
    (1, 256),  # leading size-1 dim
    (256, 1),  # trailing size-1 dim
    (2, 1, 3),  # interleaved size-1 dim
    (1, 7, 1, 32, 1),  # interleaved size-1 dims, rank 5
    (0,),  # empty tensor
    (0, 1, 3),  # empty tensor with a size-1 dim
    (16, 1, 128, 64, 60),  # large shape with a size-1 dim
]


# (shape, dim): singleton removal, negative dims, a non-singleton dim (a no-op
# that still returns a view) and the last dim of a rank-3 tensor.
_DIM_CASES = [
    ((1,), 0),
    ((1, 256), 0),
    ((256, 1), 1),
    ((2, 1, 3), 1),
    ((2, 1, 3), -2),
    ((2, 1, 3), -1),
    ((2, 1, 3), 2),
    ((1, 7, 1, 32, 1), 0),
]


# (shape, dims): explicit lists, unsorted order (order does not matter), the
# empty list (a no-op) and a list holding a non-singleton entry (also a no-op).
_DIMS_CASES = [
    ((3, 1, 5, 1, 1), [1]),
    ((3, 1, 5, 1, 1), [1, 3, 4]),
    ((3, 1, 5, 1, 1), [-1, -3]),
    ((3, 1, 5, 1, 1), [4, 2]),
    ((3, 1, 5, 1, 1), []),
    ((3, 1, 5, 1, 1), [0, 2]),
]


# Layouts a view has to compose with: a storage offset, an expanded stride-0
# dim, a permutation, a transpose and a step-sliced dim.
_LAYOUTS = ["slice_with_offset", "expanded", "permuted", "transposed", "step_slice"]


# Invalid dims are rejected in both modes.
_NEGATIVE_DIM_CASES = [((6, 1, 8), 3), ((6, 1, 8), -4), ((6,), 1)]


_NEGATIVE_DIMS_CASES = [((2, 1, 3), [0, 5]), ((3, 1, 5, 1, 1), [1, 7])]


_DUPLICATE_DIMS_CASES = [[1, 1], [1, -2], [0, 0, 1]]


def _make_layout_input(kind, dtype):
    if kind == "slice_with_offset":
        return torch.randn((5, 1, 6), dtype=dtype, device=flag_gems.device)[2:4]
    if kind == "expanded":
        return torch.randn((3, 1, 1), dtype=dtype, device=flag_gems.device).expand(
            3, 1, 5
        )
    if kind == "permuted":
        return torch.randn((2, 1, 3, 4), dtype=dtype, device=flag_gems.device).permute(
            2, 0, 1, 3
        )
    if kind == "transposed":
        return torch.randn((1, 3, 4), dtype=dtype, device=flag_gems.device).transpose(
            0, 1
        )
    if kind == "step_slice":
        return torch.randn((2, 1, 6), dtype=dtype, device=flag_gems.device)[:, :, ::2]
    raise AssertionError("unknown layout kind: {}".format(kind))


def _assert_view_metadata(res_out, ref_out, inp, ref_inp):
    """squeeze returns a new view object that aliases inp and keeps the native
    shape/stride/offset/conjunction state, leaving inp metadata untouched."""
    assert res_out is not inp
    assert res_out._is_view()
    assert res_out.untyped_storage().data_ptr() == inp.untyped_storage().data_ptr()
    assert tuple(res_out.shape) == tuple(ref_out.shape)
    assert tuple(res_out.stride()) == tuple(ref_out.stride())
    assert res_out.storage_offset() == ref_out.storage_offset()
    assert res_out.is_conj() == ref_out.is_conj()
    assert tuple(inp.shape) == tuple(ref_inp.shape)
    assert tuple(inp.stride()) == tuple(ref_inp.stride())
    assert inp.storage_offset() == ref_inp.storage_offset()


def _to_reference_layout(inp):
    """Move ``inp`` to the reference device while keeping its view geometry.

    A typed device transfer (``to('cpu')``) compacts strides, resets a slice
    offset or materializes an expanded stride-0 dim, which would make the
    reference describe a different tensor. Rebuilding the view over
    transferred storage keeps the offset, strides and lazy conjugate bit
    identical on either device.
    """
    if not utils.TO_CPU:
        return inp
    # Transfer the raw storage and rebuild the view: this keeps the view
    # geometry (offset/strides/conj bit), which to_reference would lose.
    storage = inp.untyped_storage().cpu().clone()
    reference = torch.empty(0, dtype=inp.dtype, device="cpu").set_(
        storage, inp.storage_offset(), inp.shape, inp.stride()
    )
    if inp.is_conj():
        reference = reference.conj()
    return reference


@pytest.mark.squeeze
@pytest.mark.parametrize("shape", _SINGLETON_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_squeeze_all_size_one_dims(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.squeeze(ref_inp)
    res_out = flag_gems.squeeze(inp)

    utils.gems_assert_equal(res_out, ref_out)
    _assert_view_metadata(res_out, ref_out, inp, ref_inp)


@pytest.mark.squeeze
@pytest.mark.parametrize("shape, dim", _DIM_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_squeeze_dim(shape, dim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.squeeze(ref_inp, dim)
    res_out = flag_gems.squeeze(inp, dim)

    utils.gems_assert_equal(res_out, ref_out)
    _assert_view_metadata(res_out, ref_out, inp, ref_inp)


# ``squeeze.dims`` contains a dot, so the literal @pytest.mark.squeeze.dims
# would chain off the ``squeeze`` MarkDecorator (which has no ``__getattr__``)
# and raise AttributeError; build the overload mark via getattr instead, the
# same way tests/test_or_scalar.py does for the ``or`` operator.


@getattr(pytest.mark, "squeeze.dims")
@pytest.mark.squeeze
@pytest.mark.parametrize("shape, dims", _DIMS_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_squeeze_dims(shape, dims, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.squeeze(ref_inp, dims)
    res_out = flag_gems.squeeze(inp, dims)

    utils.gems_assert_equal(res_out, ref_out)
    _assert_view_metadata(res_out, ref_out, inp, ref_inp)


@pytest.mark.squeeze
@pytest.mark.parametrize("kind", _LAYOUTS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_squeeze_noncontiguous_layout(kind, dtype):
    inp = _make_layout_input(kind, dtype)
    ref_inp = _to_reference_layout(inp)

    ref_out = torch.ops.aten.squeeze(ref_inp)
    res_out = flag_gems.squeeze(inp)

    utils.gems_assert_equal(res_out, ref_out)
    _assert_view_metadata(res_out, ref_out, inp, ref_inp)


@pytest.mark.squeeze
def test_squeeze_keeps_conjugate_bit():
    inp = torch.randn((2, 1, 3), dtype=torch.complex64, device=flag_gems.device).conj()
    ref_inp = _to_reference_layout(inp)

    ref_out = torch.ops.aten.squeeze(ref_inp)
    res_out = flag_gems.squeeze(inp)

    utils.gems_assert_equal(res_out, ref_out)
    assert res_out.is_conj()
    _assert_view_metadata(res_out, ref_out, inp, ref_inp)


@pytest.mark.squeeze
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_squeeze_result_writes_through_to_input(dtype):
    inp = torch.randn((2, 1, 3), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.squeeze(ref_inp)
    res_out = flag_gems.squeeze(inp)

    _assert_view_metadata(res_out, ref_out, inp, ref_inp)
    utils.gems_assert_equal(res_out, ref_out)

    # The result aliases the input, so filling it must be visible through the
    # input as well (this is what a data-copying implementation would break).
    res_out.fill_(1)
    ref_out.fill_(1)
    utils.gems_assert_equal(res_out, ref_out)
    utils.gems_assert_equal(inp, ref_inp)


@pytest.mark.squeeze
@pytest.mark.parametrize(
    "shape, dim", [((1, 4, 1, 3), None), ((1, 4, 1, 3), 2), ((2, 1, 5, 1, 1), [1, 4])]
)
def test_squeeze_backward(shape, dim):
    inp = torch.randn(
        shape, dtype=torch.float32, device=flag_gems.device, requires_grad=True
    )
    ref_inp = utils.to_reference(inp.detach().clone().requires_grad_(True))

    if dim is None:
        res_out = flag_gems.squeeze(inp)
        ref_out = torch.ops.aten.squeeze(ref_inp)
    else:
        res_out = flag_gems.squeeze(inp, dim)
        ref_out = torch.ops.aten.squeeze(ref_inp, dim)

    utils.gems_assert_equal(res_out, ref_out)

    # grad_outputs must match the squeezed shape, not the input shape.
    upstream = torch.randn(tuple(res_out.shape), device=flag_gems.device)
    (res_grad,) = torch.autograd.grad(res_out, inp, grad_outputs=upstream)
    (ref_grad,) = torch.autograd.grad(
        ref_out, ref_inp, grad_outputs=utils.to_reference(upstream)
    )

    assert tuple(res_grad.shape) == tuple(inp.shape)
    utils.gems_assert_equal(res_grad, ref_grad)


@pytest.mark.squeeze
@pytest.mark.parametrize("shape, dim", _NEGATIVE_DIM_CASES)
def test_squeeze_out_of_range_dim_raises(shape, dim):
    inp = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)

    with pytest.raises(IndexError):
        flag_gems.squeeze(inp, dim)


@pytest.mark.squeeze
@pytest.mark.parametrize("shape, dims", _NEGATIVE_DIMS_CASES)
def test_squeeze_out_of_range_dims_raises(shape, dims):
    inp = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)

    with pytest.raises(IndexError):
        flag_gems.squeeze(inp, dims)


@pytest.mark.squeeze
@pytest.mark.parametrize("dims", _DUPLICATE_DIMS_CASES)
def test_squeeze_duplicate_dims_raises(dims):
    inp = torch.randn((6, 1, 8), dtype=torch.float32, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems.squeeze(inp, dims)


@pytest.mark.squeeze
def test_squeeze_rejects_non_tensor():
    with pytest.raises((RuntimeError, TypeError, AttributeError)):
        flag_gems.squeeze([1, 2, 3])


# ``squeeze.dims`` contains a dot, so the literal @pytest.mark.squeeze.dims
# would chain off the ``squeeze`` MarkDecorator (which has no ``__getattr__``)
# and raise AttributeError; build the overload mark via getattr instead, the
# same way tests/test_or_scalar.py does for the ``or`` operator.
