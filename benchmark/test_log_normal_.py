import pytest
import torch

from . import base, consts


@pytest.mark.log_normal_
def test_log_normal_():
    bench = base.GenericBenchmark(
        op_name="log_normal_",
        torch_op=torch.Tensor.log_normal_,
        input_fn=base.unary_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
