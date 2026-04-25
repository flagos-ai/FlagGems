import pytest
import torch

from . import attri_util as attr_utils
from . import performance_utils as utils


def input_fn(shape, cur_dtype, device):
    self = torch.randn(shape, dtype=cur_dtype, device=device)
    p = 0.5
    yield self, p


@pytest.mark.bernoulli_
def test_bernoulli_inplace():
    bench = utils.GenericBenchmark(
        op_name="bernoulli_",
        input_fn=input_fn,
        torch_op=torch.Tensor.bernoulli_,
        dtypes=attr_utils.FLOAT_DTYPES,
    )
    bench.run()
