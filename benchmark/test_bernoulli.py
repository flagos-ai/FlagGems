import pytest
import torch

from . import base, consts


def bernoulli_inplace_input_fn(shape, cur_dtype, device):
    self = torch.randn(shape, dtype=cur_dtype, device=device)
    p = 0.5
    yield self, p


@pytest.mark.bernoulli_
def test_bernoulli_inplace():
    bench = base.GenericBenchmark(
        op_name="bernoulli_",
        input_fn=bernoulli_inplace_input_fn,
        torch_op=torch.Tensor.bernoulli_,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def bernoulli_input_fn(shape, cur_dtype, device):
    prob = torch.rand(shape, dtype=cur_dtype, device=device)
    yield (prob,)


@pytest.mark.bernoulli
def test_bernoulli():
    bench = base.GenericBenchmark(
        input_fn=bernoulli_input_fn,
        op_name="bernoulli",
        torch_op=torch.ops.aten.bernoulli,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def bernoulli_p_input_fn(shape, cur_dtype, device):
    x = torch.rand(shape, dtype=cur_dtype, device=device)
    yield (x, 0.5)


@pytest.mark.bernoulli
def test_bernoulli_p():
    bench = base.GenericBenchmark(
        input_fn=bernoulli_p_input_fn,
        op_name="bernoulli.p",
        torch_op=torch.ops.aten.bernoulli.p,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
