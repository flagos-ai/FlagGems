import pytest
import torch

from . import base, consts


@pytest.mark.var_mean
def test_var_mean():
    bench = base.UnaryReductionBenchmark(
        op_name="var_mean", torch_op=torch.var_mean, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()
