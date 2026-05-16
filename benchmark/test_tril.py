import pytest
import torch

from . import base, consts, utils


def _tril_out_transposed_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    out_shape = (*shape[:-2], shape[-1], shape[-2])
    out = torch.empty(out_shape, dtype=dtype, device=device).transpose(-2, -1)
    yield inp, 0, {"out": out}


def _tril_out_sliced_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    if len(shape) == 2:
        out = torch.empty((shape[0] * 2, shape[1]), dtype=dtype, device=device)[::2]
    else:
        out_shape = (shape[0] * 2, *shape[1:])
        out = torch.empty(out_shape, dtype=dtype, device=device)[::2]
    yield inp, 0, {"out": out}


def _tril_extreme_diagonal_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    n = shape[-1]
    for diagonal in (-n, -(n // 2), -1, 0, 1, n // 2, n):
        yield inp, diagonal


def _torch_tril_inplace(inp, diagonal=0):
    """Wrapper for tensor.tril_() method to match torch op signature."""
    return inp.tril_(diagonal)


@pytest.mark.tril
def test_tril():
    bench = base.GenericBenchmarkExcluse1D(
        input_fn=utils.unary_input_fn,
        op_name="tril",
        torch_op=torch.tril,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.tril
def test_tril_extreme_diagonal():
    bench = base.GenericBenchmarkExcluse1D(
        input_fn=_tril_extreme_diagonal_input_fn,
        op_name="tril_extreme_diagonal",
        torch_op=torch.tril,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.tril
@pytest.mark.tril_out
def test_tril_out_transposed():
    bench = base.GenericBenchmarkExcluse1D(
        input_fn=_tril_out_transposed_input_fn,
        op_name="tril_out_transposed",
        torch_op=torch.tril,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.tril
@pytest.mark.tril_out
def test_tril_out_sliced():
    bench = base.GenericBenchmarkExcluse1D(
        input_fn=_tril_out_sliced_input_fn,
        op_name="tril_out_sliced",
        torch_op=torch.tril,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.tril_
def test_tril_inplace():
    bench = base.GenericBenchmarkExcluse1D(
        input_fn=utils.unary_input_fn,
        op_name="tril_",
        torch_op=_torch_tril_inplace,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
