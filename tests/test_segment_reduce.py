import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

FLOAT_DTYPES = [torch.float32] if cfg.QUICK_MODE else utils.ALL_FLOAT_DTYPES
REDUCTIONS = ("sum", "mean", "max", "min", "prod")
INITIALS = (None, 0.5)

LENGTH_CASES = (
    ((5,), 0, [2, 0, 3]),
    ((2, 3), 1, [[1, 1, 1], [1, 1, 1]]),
    ((2, 3, 4), 1, [[1, 2], [2, 1]]),
    ((2, 3, 5), 2, [[[2, 3], [1, 4], [3, 2]], [[5, 0], [2, 3], [4, 1]]]),
)

OFFSET_CASES = (
    ((5,), 0, [0, 2, 5]),
    ((2, 3, 4), 1, [[0, 1, 3], [0, 2, 3]]),
)


def _make_data(shape, dtype, requires_grad=False):
    values = torch.arange(1, 1 + torch.Size(shape).numel(), device=flag_gems.device)
    values = values.reshape(shape).to(torch.float32)
    values = (values.remainder(17) - 8) / 5
    values = values + 0.25
    return values.to(dtype).requires_grad_(requires_grad)


def _make_lengths_kwargs(axis, lengths):
    lengths = torch.tensor(lengths, dtype=torch.int64, device=flag_gems.device)
    return {"lengths": lengths, "axis": axis}


def _make_offsets_kwargs(axis, offsets):
    offsets = torch.tensor(offsets, dtype=torch.int64, device=flag_gems.device)
    return {"offsets": offsets, "axis": axis}


def _to_reference_kwargs(kwargs):
    ref_kwargs = dict(kwargs)
    if "lengths" in ref_kwargs:
        ref_kwargs["lengths"] = utils.to_reference(ref_kwargs["lengths"])
    if "offsets" in ref_kwargs:
        ref_kwargs["offsets"] = utils.to_reference(ref_kwargs["offsets"])
    return ref_kwargs


def _assert_segment_reduce(shape, kwargs, reduce, dtype, initial):
    data = _make_data(shape, dtype)
    ref_data = utils.to_reference(data)
    ref_kwargs = _to_reference_kwargs(kwargs)

    ref_out = torch.segment_reduce(ref_data, reduce, **ref_kwargs, initial=initial)
    with flag_gems.use_gems():
        res_out = torch.segment_reduce(data, reduce, **kwargs, initial=initial)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.segment_reduce
@pytest.mark.parametrize("shape, axis, lengths", LENGTH_CASES)
@pytest.mark.parametrize("reduce", REDUCTIONS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("initial", INITIALS)
def test_segment_reduce_lengths(shape, axis, lengths, reduce, dtype, initial):
    _assert_segment_reduce(
        shape,
        _make_lengths_kwargs(axis, lengths),
        reduce,
        dtype,
        initial,
    )


@pytest.mark.segment_reduce
@pytest.mark.parametrize("shape, axis, offsets", OFFSET_CASES)
@pytest.mark.parametrize("reduce", REDUCTIONS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("initial", INITIALS)
def test_segment_reduce_offsets(shape, axis, offsets, reduce, dtype, initial):
    _assert_segment_reduce(
        shape,
        _make_offsets_kwargs(axis, offsets),
        reduce,
        dtype,
        initial,
    )


@pytest.mark.segment_reduce_backward
@pytest.mark.parametrize("shape, axis, lengths", LENGTH_CASES)
@pytest.mark.parametrize("reduce", REDUCTIONS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("initial", INITIALS)
def test_segment_reduce_backward_lengths(shape, axis, lengths, reduce, dtype, initial):
    data = _make_data(shape, dtype, requires_grad=True)
    kwargs = _make_lengths_kwargs(axis, lengths)
    ref_data = utils.to_reference(data.detach()).requires_grad_()
    ref_kwargs = _to_reference_kwargs(kwargs)

    ref_out = torch.segment_reduce(ref_data, reduce, **ref_kwargs, initial=initial)
    with flag_gems.use_gems():
        res_out = torch.segment_reduce(data, reduce, **kwargs, initial=initial)

    grad = _make_data(tuple(res_out.shape), dtype)
    ref_grad = utils.to_reference(grad)
    (ref_grad_input,) = torch.autograd.grad(ref_out, ref_data, ref_grad)
    with flag_gems.use_gems():
        (res_grad_input,) = torch.autograd.grad(res_out, data, grad)

    utils.gems_assert_close(res_grad_input, ref_grad_input, dtype, equal_nan=True)


@pytest.mark.segment_reduce_backward
@pytest.mark.parametrize("shape, axis, offsets", OFFSET_CASES)
@pytest.mark.parametrize("reduce", REDUCTIONS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_segment_reduce_backward_offsets(shape, axis, offsets, reduce, dtype):
    data = _make_data(shape, dtype)
    kwargs = _make_offsets_kwargs(axis, offsets)
    ref_data = utils.to_reference(data)
    ref_kwargs = _to_reference_kwargs(kwargs)

    with flag_gems.use_gems():
        output = torch.segment_reduce(data, reduce, **kwargs)
    ref_output = torch.segment_reduce(ref_data, reduce, **ref_kwargs)
    grad = _make_data(tuple(output.shape), dtype)
    ref_grad = utils.to_reference(grad)

    ref_grad_input = torch.ops.aten._segment_reduce_backward(
        ref_grad,
        ref_output,
        ref_data,
        reduce,
        **ref_kwargs,
    )
    with flag_gems.use_gems():
        res_grad_input = torch.ops.aten._segment_reduce_backward(
            grad,
            output,
            data,
            reduce,
            **kwargs,
        )

    utils.gems_assert_close(res_grad_input, ref_grad_input, dtype, equal_nan=True)


@pytest.mark.segment_reduce
def test_segment_reduce_indices_unsupported():
    data = torch.arange(4, dtype=torch.float32, device=flag_gems.device)
    indices = torch.tensor([0, 0, 1, 1], dtype=torch.int64, device=flag_gems.device)
    with pytest.raises(RuntimeError, match="indices based reduction is not supported"):
        with flag_gems.use_gems():
            torch.segment_reduce(data, "sum", indices=indices)
