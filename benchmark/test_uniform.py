import pytest
import torch

from . import base, consts, utils


@pytest.mark.uniform_
def test_uniform_inplace():
    bench = base.GenericBenchmark(
        input_fn=utils.unary_input_fn,
        op_name="uniform_",
        torch_op=torch.Tensor.uniform_,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
