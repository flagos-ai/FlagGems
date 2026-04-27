import pytest
import torch

from . import base, consts


def input_fn(shape, cur_dtype, device):
    self = torch.randn(shape, dtype=cur_dtype, device=device)
    p = 0.5
    yield self, p


@pytest.mark.bernoulli_
def test_bernoulli_inplace():
    bench = base.GenericBenchmark(
        op_name="bernoulli_",
        input_fn=input_fn,
        torch_op=torch.Tensor.bernoulli_,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
