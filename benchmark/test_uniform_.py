import pytest
import torch

from . import base, consts


@pytest.mark.uniform_
def test_uniform_():
    bench = base.GenericBenchmark(
        input_fn=base.unary_input_fn,
        op_name="uniform_",
        torch_op=torch.Tensor.uniform_,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
