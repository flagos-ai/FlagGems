import pytest
import torch

from . import base, consts


def input_fn(shape, cur_dtype, device):
    self = torch.empty(shape, dtype=cur_dtype, device=device)
    median = 0.0
    sigma = 1.0
    yield self, median, sigma


@pytest.mark.cauchy_
def test_cauchy_inplace():
    bench = base.GenericBenchmark(
        op_name="cauchy_",
        input_fn=input_fn,
        torch_op=torch.Tensor.cauchy_,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.cauchy
def test_cauchy_out():
    bench = base.GenericBenchmark(
        op_name="cauchy",
        input_fn=input_fn,
        torch_op=torch.ops.aten.cauchy,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
