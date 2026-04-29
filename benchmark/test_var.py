import pytest
import torch

from . import base, consts


@pytest.mark.var
def test_var():
    bench = base.UnaryReductionBenchmark(
        op_name="var", torch_op=torch.var, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()
