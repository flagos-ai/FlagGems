import pytest
import torch

from . import base, consts


def bernoulli_input_fn(shape, cur_dtype, device):
    yield torch.rand(shape, dtype=cur_dtype, device=device),


@pytest.mark.bernoulli
def test_perf_bernoulli():
    bench = base.GenericBenchmark(
        input_fn=bernoulli_input_fn,
        op_name="bernoulli",
        torch_op=torch.bernoulli,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
