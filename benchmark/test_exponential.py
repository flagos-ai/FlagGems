import pytest
import torch

from . import base, consts, utils


@pytest.mark.exponential_
def test_exponential_inplace():
    bench = base.GenericBenchmark(
        op_name="exponential_",
        input_fn=utils.unary_input_fn,
        torch_op=torch.Tensor.exponential_,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
