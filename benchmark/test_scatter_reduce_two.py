import pytest
import torch

from . import base, consts


def _input_fn_factory(reduce):
    def inner(shape, dtype, device):
        inp = torch.randn(shape, dtype=dtype, device=device)
        dim = -1
        size_dim = shape[dim]
        index = torch.randint(0, size_dim, shape, dtype=torch.long, device=device)
        src = torch.randn(shape, dtype=dtype, device=device)
        yield inp, dim, index, src, {"reduce": reduce}

    return inner


@pytest.mark.scatter_reduce_two_
def test_scatter_reduce_two_inplace_sum():
    bench = base.GenericBenchmark2DOnly(
        op_name="scatter_reduce_.sum",
        torch_op=torch.Tensor.scatter_reduce_,
        input_fn=_input_fn_factory("sum"),
        dtypes=consts.FLOAT_DTYPES,
        inplace=True,
    )
    bench.run()


@pytest.mark.scatter_reduce_two_
def test_scatter_reduce_two_inplace_amax():
    bench = base.GenericBenchmark2DOnly(
        op_name="scatter_reduce_.amax",
        torch_op=torch.Tensor.scatter_reduce_,
        input_fn=_input_fn_factory("amax"),
        dtypes=consts.FLOAT_DTYPES,
        inplace=True,
    )
    bench.run()


@pytest.mark.scatter_reduce_two_
def test_scatter_reduce_two_inplace_amin():
    bench = base.GenericBenchmark2DOnly(
        op_name="scatter_reduce_.amin",
        torch_op=torch.Tensor.scatter_reduce_,
        input_fn=_input_fn_factory("amin"),
        dtypes=consts.FLOAT_DTYPES,
        inplace=True,
    )
    bench.run()


@pytest.mark.scatter_reduce_two_
def test_scatter_reduce_two_inplace_mean():
    bench = base.GenericBenchmark2DOnly(
        op_name="scatter_reduce_.mean",
        torch_op=torch.Tensor.scatter_reduce_,
        input_fn=_input_fn_factory("mean"),
        dtypes=consts.FLOAT_DTYPES,
        inplace=True,
    )
    bench.run()
