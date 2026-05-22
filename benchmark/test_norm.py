import pytest
import torch

from . import base, consts, utils


@pytest.mark.norm
def test_norm():
    bench = base.GenericBenchmarkExcluse1D(
        input_fn=utils.unary_input_fn,
        op_name="norm",
        torch_op=torch.norm,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
